import rclpy
from rclpy.node import Node
import numpy as np
import math
import torch
import cv2
import sys
from pathlib import Path
from rclpy.duration import Duration

# ROS2 核心
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs_py import point_cloud2

# TF2 与 坐标变换
import tf2_ros
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

# AI 与 图像处理
from ultralytics import YOLO

TARGET_DETECTION_SRC = Path(__file__).resolve().parents[2] / "target_detection"
if str(TARGET_DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(TARGET_DETECTION_SRC))

from yolo11_custom import create_yolo_model

class YOLOObstacleFeatureNode(Node):
    def __init__(self):
        super().__init__('yolo_obstacle_feature_node')

        # 参数配置
        self.ray_count = 36          
        self.plane = "x>0"         
        self.max_dist = 2.0         
        self.sample_size = 2000      
        self.conf_threshold = 0.3   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心配置：适配实际的坐标系
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_link'  # 默认值，实际优先使用点云消息 frame_id
        self._last_tf_warn_ns = 0

        # 相机内参（根据实际标定值调整）
        self.fx, self.fy = 854.132209, 826.391640
        self.cx, self.cy = 474.667103, 343.609130

        # 初始化工具 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.model = create_yolo_model("yolo11n-seg.pt")
        
        # 生成斐波那契射线（障碍物特征提取用）
        self.rays_np = self._make_fibonacci_rays_quarter(self.ray_count, self.plane)
        self.rays_t = torch.from_numpy(self.rays_np).to(self.device)

        # ROS 订阅与发布 
        self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.pc_sub = Subscriber(self, PointCloud2, '/camera/camera/depth/color/points')
        
        # 时间同步器（RGB和点云）
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.pc_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

        # 发布话题
        self.feat_pub = self.create_publisher(Float32MultiArray, '/obstacle_features', 10)
        self.mask_pc_pub = self.create_publisher(PointCloud2, '/obstacle_points_base', 10)

        self.get_logger().info(
            f"Obs Feature Node 已启动！device={self.device}, base_frame={self.base_frame}, "
            f"default_camera_frame={self.camera_frame}"
        )
        
    def _make_fibonacci_rays_quarter(self, ray_count, plane):
        """生成指定平面的斐波那契射线（用于障碍物特征投影）"""
        n_try = max(ray_count * 8, 128)
        i = np.arange(0, n_try, dtype=np.float64) + 0.5
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        theta = 2.0 * math.pi * i / phi
        z = 1.0 - 2.0 * i / n_try
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        x, y = r * np.cos(theta), r * np.sin(theta)
        dirs = np.stack([x, y, z], axis=-1)

        # 过滤指定平面的射线
        if plane == "x>0": 
            mask = (dirs[:, 0] >= 0.0) & (dirs[:, 2] >= 0.0)
        elif plane == "y>0": 
            mask = (dirs[:, 1] >= 0.0) & (dirs[:, 2] >= 0.0)
        elif plane == "xy>0": 
            mask = (dirs[:, 0] >= 0.0) & (dirs[:, 1] >= 0.0) & (dirs[:, 2] >= 0.0)
        else: 
            raise ValueError(f"不支持的平面: {plane}")

        dirs = dirs[mask][:ray_count]
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return (dirs / (norms + 1e-8)).astype(np.float32)

    def _project_min_along_rays_torch(self, points_np):
        """沿射线投影计算最小距离特征"""
        if points_np.shape[0] == 0:
            return np.full((self.ray_count,), 1.0, dtype=np.float32)

        # 计算角度阈值
        area_weight = 0.5 if self.plane == "xy>0" else (1.0 if self.plane == "x>0" else 4.0)
        cos_threshold = 1.0 - (area_weight / (2.0 * self.ray_count))
        cos_threshold = 1.0 - (1.0 - cos_threshold) * 1.2

        # 张量计算（加速）
        pts_t = torch.from_numpy(points_np).to(self.device).float()
        dot_products = torch.matmul(pts_t, self.rays_t.T) 
        pts_norm = torch.norm(pts_t, dim=1, keepdim=True) 
        cos_alpha = dot_products / (pts_norm + 1e-8)

        # 过滤有效点
        mask = (dot_products > 0) & (cos_alpha > cos_threshold)
        t_filtered = torch.where(mask, dot_products, torch.full_like(dot_products, float("inf")))
        
        # 计算最小距离
        t_min, _ = torch.min(t_filtered, dim=0)
        res = t_min.cpu().numpy()
        res[np.isinf(res)] = self.max_dist
        
        # 归一化
        return (np.clip(res, 0.0, self.max_dist) / self.max_dist).astype(np.float32)

    def _publish_transformed_pc(self, points, original_header):
        """将numpy点云转换为ROS2 PointCloud2并发布"""
        # 构造header（替换为base_link）
        header = original_header
        header.frame_id = self.base_frame
        
        # 创建PointCloud2消息（仅XYZ）
        pc_msg = point_cloud2.create_cloud_xyz32(header, points.tolist())
        self.mask_pc_pub.publish(pc_msg)

    def callback(self, rgb_msg, pc_msg):
        """核心回调：RGB+点云同步处理"""
        source_frame = pc_msg.header.frame_id or self.camera_frame

        # 1. 获取相机→基座标的TF变换
        try:
            if not self.tf_buffer.can_transform(
                self.base_frame,
                source_frame,
                pc_msg.header.stamp,
                timeout=Duration(seconds=0.2),
            ):
                self._warn_tf_unavailable(source_frame)
                return

            t_tf: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=self.base_frame,
                source_frame=source_frame,
                time=pc_msg.header.stamp,    # 用点云时间戳保证同步
                timeout=Duration(seconds=0.2)
            )
        except Exception as e:
            self._warn_tf_exception(source_frame, e)
            return

        # 2. 解析RGB图像并执行YOLO分割
        try:
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            h, w = cv_img.shape[:2]
            # YOLO推理（关闭verbose，提升速度）
            results = self.model.predict(cv_img, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            self.get_logger().warn(f"图像解析/YOLO推理失败: {e}")
            self.feat_pub.publish(Float32MultiArray(data=[float(1.0)] * self.ray_count))
            return
        
        # 3. 生成障碍物分割掩码
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        has_mask = False
        for r in results:
            if r.masks is not None:
                for m in r.masks.data:
                    # 调整掩码尺寸匹配图像
                    m_scaled = m.cpu().numpy()
                    if m_scaled.shape != (h, w):
                        m_scaled = cv2.resize(m_scaled, (w, h))
                    combined_mask[m_scaled > 0.5] = 1
                    has_mask = True
        
        # 无障碍物时发布全1特征
        if not has_mask:
            self.feat_pub.publish(Float32MultiArray(data=[float(1.0)] * self.ray_count))
            return

        # 4. 解析点云（仅XYZ，过滤NaN）
        try:
            pts_struct = point_cloud2.read_points(
                pc_msg, 
                field_names=("x", "y", "z"), 
                skip_nans=True,
                uvs=None
            )
            # 结构化数组转普通float32数组（适配点云point_step=20的格式）
            pts_full = np.array([(p['x'], p['y'], p['z']) for p in pts_struct], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"点云解析失败: {e}")
            self.feat_pub.publish(Float32MultiArray(data=[float(1.0)] * self.ray_count))
            return

        # 空点云处理
        if pts_full.shape[0] == 0:
            self.feat_pub.publish(Float32MultiArray(data=[float(1.0)] * self.ray_count))
            return

        # 5. 点云投影到图像像素坐标系，筛选掩码内的障碍物点
        z = pts_full[:, 2]
        z_safe = np.where(z == 0, 1e-6, z)  # 避免除0
        # 像素坐标计算（针孔相机模型）
        u_pts = (self.fx * pts_full[:, 0] / z_safe + self.cx).astype(int)
        v_pts = (self.fy * pts_full[:, 1] / z_safe + self.cy).astype(int)

        # 过滤图像边界外的点
        in_bounds = (u_pts >= 0) & (u_pts < w) & (v_pts >= 0) & (v_pts < h)
        u_pts, v_pts, pts_filtered = u_pts[in_bounds], v_pts[in_bounds], pts_full[in_bounds]

        # 筛选掩码内的障碍物点（相机坐标系）
        obs_idx = combined_mask[v_pts, u_pts] == 1
        obstacle_pts_cam = pts_filtered[obs_idx]

        # 无障碍物点处理
        if obstacle_pts_cam.shape[0] == 0:
            self.feat_pub.publish(Float32MultiArray(data=[float(1.0)] * self.ray_count))
            return

        # 6. 坐标变换：相机光学帧 → base_link（逐点变换，保证精度）
        obstacle_pts_base = []
        for pt_cam in obstacle_pts_cam:
            # 构造相机坐标系下的PointStamped
            p_stamped = PointStamped()
            p_stamped.header.frame_id = source_frame
            p_stamped.header.stamp = pc_msg.header.stamp
            # 转换为Python原生float（避免ROS类型错误）
            p_stamped.point.x = float(pt_cam[0])
            p_stamped.point.y = float(pt_cam[1])
            p_stamped.point.z = float(pt_cam[2])
            
            # TF2官方变换方法（自动处理旋转+平移）
            p_base = do_transform_point(p_stamped, t_tf)
            obstacle_pts_base.append([
                float(p_base.point.x), 
                float(p_base.point.y), 
                float(p_base.point.z)
            ])
        
        # 转numpy数组
        obstacle_pts_base = np.array(obstacle_pts_base, dtype=np.float32)

        # 7. 发布变换后的障碍物点云（base_link坐标系）
        self._publish_transformed_pc(obstacle_pts_base, pc_msg.header)

        # 8. 点云采样（控制计算量，保证实时性）
        if obstacle_pts_base.shape[0] > self.sample_size:
            idx = np.random.choice(obstacle_pts_base.shape[0], self.sample_size, replace=False)
            sampled_pts_base = obstacle_pts_base[idx]
        else:
            sampled_pts_base = obstacle_pts_base

        # 9. 计算障碍物特征并发布
        feat = self._project_min_along_rays_torch(sampled_pts_base)
        self.feat_pub.publish(Float32MultiArray(data=feat.tolist()))

    def _warn_tf_unavailable(self, source_frame):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_tf_warn_ns > int(2e9):
            self.get_logger().warn(
                f"TF暂不可用 ({source_frame}→{self.base_frame})，等待TF树稳定后重试。"
            )
            self._last_tf_warn_ns = now_ns

    def _warn_tf_exception(self, source_frame, exc):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_tf_warn_ns > int(2e9):
            self.get_logger().warn(
                f"TF变换失败 ({source_frame}→{self.base_frame}): {exc}"
            )
            self._last_tf_warn_ns = now_ns

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = YOLOObstacleFeatureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断，节点停止！")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

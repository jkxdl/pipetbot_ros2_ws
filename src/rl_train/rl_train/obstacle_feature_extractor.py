#!/usr/bin/env python3
import math
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from tf_transformations import translation_matrix, quaternion_matrix

class ObstacleFeatureExtractorTorch(Node):

    def __init__(self):
        super().__init__("obstacle_feature_extractor_torch")
        # -------- 参数声明 --------
        self.declare_parameter("cloud_topic", "/obstacles")
        self.declare_parameter("output_topic", "/obstacle_features")
        self.declare_parameter("ray_count", 36)
        self.declare_parameter("max_range", 2.0)
        self.declare_parameter("ray_frame", "base_link")
        self.declare_parameter("fibonacci_plane", "y>0")
        self.declare_parameter("device", "cuda") # 默认使用 cuda
        # -------- 读取参数 --------
        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.ray_count = int(self.get_parameter("ray_count").value)
        self.max_range_val = float(self.get_parameter("max_range").value)
        self.ray_frame = str(self.get_parameter("ray_frame").value)
        self.fib_plane = str(self.get_parameter("fibonacci_plane").value)

        # 自动切换设备
        device_str = self.get_parameter("device").value
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # -------- 初始化 Fibonacci 射线 (GPU) --------
        self._init_rays()

        # -------- TF & ROS 通信 --------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_cloud = self.create_subscription(
            PointCloud2, self.cloud_topic, self.cb_cloud, 10
        )

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        self.get_logger().info(
            f"[Init] Device: {self.device} | cloud_topic: {self.cloud_topic} | "
            f"output_topic: {self.output_topic} | ray_frame: {self.ray_frame} | "
            f"Rays: {self.ray_count} | max_range: {self.max_range_val} | Plane: {self.fib_plane}"
        )

    def _init_rays(self):
        """生成射线并预传到 GPU"""
        n_try = max(self.ray_count * 8, 256)
        i = np.arange(0, n_try, dtype=np.float64) + 0.5
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        theta = 2.0 * math.pi * i / phi
        z = 1.0 - 2.0 * i / n_try
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        x, y = r * np.cos(theta), r * np.sin(theta)
        dirs = np.stack([x, y, z], axis=-1)

        if self.fib_plane == "x>0": 
            mask = (dirs[:, 0] >= 0.0) & (dirs[:, 2] >= 0.0)
            area_weight = 1.0
        elif self.fib_plane == "y>0": 
            mask = (dirs[:, 1] >= 0.0) & (dirs[:, 2] >= 0.0)
            area_weight = 1.0
        elif self.fib_plane == "xy>0": 
            mask = (dirs[:, 0] >= 0.0) & (dirs[:, 1] >= 0.0) & (dirs[:, 2] >= 0.0)
            area_weight = 0.5
        else: 
            raise ValueError(f"Unsupported plane: {self.fib_plane}")

        selected_dirs = dirs[mask][:self.ray_count]
        
        # 计算余弦阈值
        cos_thresh = 1.0 - (area_weight / (2.0 * self.ray_count))
        self.cos_threshold = 1.0 - (1.0 - cos_thresh) * 1.2
        
        # 转换为 Tensor 并上传
        self.ray_dirs_t = torch.from_numpy(selected_dirs).float().to(self.device)
        self.ray_dirs_t = self.ray_dirs_t / (torch.norm(self.ray_dirs_t, dim=-1, keepdim=True) + 1e-8)
        
        # 预存常量 Tensor 以加快计算
        self.max_range_t = torch.tensor(self.max_range_val, device=self.device)
        self.inf_t = torch.tensor(float('inf'), device=self.device)

    def _compute_ray_distances_torch(self, pts_np: np.ndarray) -> np.ndarray:
        """使用 PyTorch 进行 GPU 加速的投影计算"""
        if pts_np.shape[0] == 0:
            return np.ones(self.ray_count, dtype=np.float32)
        pts_t = torch.from_numpy(pts_np).to(self.device) # (N, 3)
        pts_norm = torch.norm(pts_t, dim=1, keepdim=True) # (N, 1)
        proj = torch.matmul(pts_t, self.ray_dirs_t.T)
        cos_alpha = proj / (pts_norm + 1e-8)
        mask = (proj > 0.0) & (cos_alpha > self.cos_threshold)
        dist_grid = torch.where(mask, proj, self.inf_t)
        dmin, _ = torch.min(dist_grid, dim=0) # (R,)
        dmin = torch.where(torch.isinf(dmin), self.max_range_t, dmin)
        dmin = torch.clamp(dmin, 0.0, self.max_range_val)
        d_norm = dmin / self.max_range_t

        return d_norm.cpu().numpy().astype(np.float32)


    def cb_cloud(self, msg: PointCloud2):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.ray_frame, msg.header.frame_id, rclpy.time.Time()
            )
        except Exception:
            return

        T = translation_matrix([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        R = quaternion_matrix([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        TF = (T @ R).astype(np.float32)

        # 提取点云 (修复 TypeError 的位置)
        pts_struct = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts_np_orig = np.array(pts_struct)    

        if pts_np_orig.size == 0:
            self._publish(np.ones(self.ray_count, dtype=np.float32))
            return

        pts_raw = pts_np_orig.view(np.float32).reshape(-1, 3)

        pts_h = np.ones((pts_raw.shape[0], 4), dtype=np.float32)
        pts_h[:, :3] = pts_raw
        pts_tf = (pts_h @ TF.T)[:, :3] # (N, 3)

        d_norm = self._compute_ray_distances_torch(pts_tf)
        self._publish(d_norm)


    def _publish(self, data: np.ndarray):
        msg = Float32MultiArray()
        msg.data = data.tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleFeatureExtractorTorch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

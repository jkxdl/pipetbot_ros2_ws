import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2


class YoloPointCloudNode(Node):
    def __init__(self):
        super().__init__('yolo_pointcloud_node')

        # 订阅RGB图像话题
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # 订阅点云话题
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # 初始化变量
        self.bridge = CvBridge()
        self.pointcloud = None
        self.pointcloud_width = None
        self.pointcloud_received = False
        self.model = YOLO('yolo11n.pt')  # 加载YOLO模型

        self.get_logger().info("YOLO PointCloud Node has been started.")

    def pointcloud_callback(self, msg):
        try:
            # 使用 sensor_msgs_py.point_cloud2 解析点云
            point_generator = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(point_generator))  # 转换为 NumPy 数组

            # 检查点云是否为空
            if points.size == 0:
                self.get_logger().warn("Received empty PointCloud data.")
                self.pointcloud = None
                self.pointcloud_received = False
                return

            # 点云解析为二维数组，每行包含 [x, y, z]
            self.pointcloud = points
            self.pointcloud_width = msg.width
            self.pointcloud_received = True

            self.get_logger().info(f"PointCloud shape: {self.pointcloud.shape}, width: {self.pointcloud_width}")
        except Exception as e:
            self.get_logger().error(f"Error processing PointCloud: {e}")
            self.pointcloud = None
            self.pointcloud_received = False

    def image_callback(self, msg):
        try:
            # 将图像消息转换为OpenCV格式
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 检查点云是否已接收
            if not self.pointcloud_received:
                self.get_logger().warn("PointCloud data not yet received.")
                return

            # 使用YOLO进行目标检测
            results = self.model(frame)
            detections = results[0].boxes  # 获取检测框信息

            # 可视化检测结果
            annotated_frame = results[0].plot()

            # 如果没有检测到目标，直接返回
            if len(detections) == 0:
                self.get_logger().info("No detections found.")
                cv2.imshow("YOLO Detection", annotated_frame)
                cv2.waitKey(1)
                return

            # 遍历每个检测目标
            for box in detections:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)

                # 检查ROI范围是否有效
                x_min = max(0, min(x_min, frame.shape[1] - 1))
                x_max = max(0, min(x_max, frame.shape[1] - 1))
                y_min = max(0, min(y_min, frame.shape[0] - 1))
                y_max = max(0, min(y_max, frame.shape[0] - 1))

                if x_min >= x_max or y_min >= y_max:
                    self.get_logger().warn(f"Invalid ROI: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                    continue

                # 获取检测框中心点
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2

                # 提取三维坐标
                coordinates = self.get_3d_coordinates(x_center, y_center)
                if coordinates:
                    x_camera, y_camera, z_camera = coordinates
                    self.get_logger().info(
                        f"Detected {box.cls} at (X: {x_camera:.2f}, Y: {y_camera:.2f}, Z: {z_camera:.2f})"
                    )

                    # 在图像上标注三维坐标
                    cv2.putText(
                        annotated_frame,
                        f"({x_camera:.2f}, {y_camera:.2f}, {z_camera:.2f})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

            # 显示带注释的图像
            cv2.imshow("YOLO Detection", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def get_3d_coordinates(self, x, y):
        """
                  根据图像中的像素坐标 (x, y)，从点云中提取三维坐标。
        """
        try:
            if self.pointcloud is None or self.pointcloud_width is None:
                self.get_logger().warn("PointCloud data is not ready.")
                return None

            # 计算点云线性索引
            index = y * self.pointcloud_width + x

            # 检查索引有效性
            if index >= self.pointcloud.shape[0]:
                self.get_logger().warn(f"Invalid index: {index}, PointCloud size: {self.pointcloud.shape[0]}")
                return None

            # 提取三维坐标
            point = self.pointcloud[index]
            x_camera, y_camera, z_camera = point[:3]

            if np.isfinite(x_camera) and np.isfinite(y_camera) and np.isfinite(z_camera):
                return x_camera, y_camera, z_camera
            else:
                self.get_logger().warn(f"Invalid depth at pixel ({x}, {y}).")
                return None
        except Exception as e:
            self.get_logger().error(f"Error extracting 3D coordinates: {e}")
            return None



def main(args=None):
    rclpy.init(args=args)
    node = YoloPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


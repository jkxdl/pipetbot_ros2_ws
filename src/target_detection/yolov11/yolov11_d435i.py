import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from geometry_msgs.msg import Point
from threading import Lock
import threading
from yolo11_custom import create_yolo_model

# 导入自定义服务和消息
from pipettingrobot_interfaces.srv import GetDetections
from pipettingrobot_interfaces.msg import DetectedObject

class Yolov11_d435i(Node):
    def __init__(self):
        super().__init__('yolov11_d435i')

        # 初始化 CvBridge 和 YOLO 模型
        self.bridge = CvBridge()
        self.model = create_yolo_model(
            '/home/robot/pipetbot_ros2_ws/dataset/runs/train/yolov11n_experiment/weights/best.pt'
        )
        self.latest_rgb_image = None
        self.latest_depth_image = None

        # 定义相机内参
        self.camera_intrinsics = {
            "fx": 854.132209,  # 焦距（像素）
            "fy": 826.391640,
            "cx": 474.667103,  # 主点 x 坐标
            "cy": 343.609130   # 主点 y 坐标
        }

        # 初始化检测结果存储
        self.detected_objects = []

        # 创建线程锁
        self.lock = Lock()

        # 订阅相机的 RGB 图像话题
        self.subscription_rgb = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 相机 RGB 话题
            self.rgb_callback,
            10
        )
        self.subscription_rgb  # 防止未使用变量警告

        # 订阅相机的深度图话题
        self.subscription_depth = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',  # 深度图话题
            self.depth_callback,
            10
        )
        self.subscription_depth  # 防止未使用变量警告

        # 创建服务服务器
        self.srv = self.create_service(GetDetections, 'get_detections', self.handle_get_detections)

        self.get_logger().info("YOLOv11 Node with On-Demand Detection Service has been started")

    def depth_callback(self, msg):
        with self.lock:
            try:
                # 将深度图消息转换为 NumPy 格式
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f"Error processing depth image: {e}")

    def rgb_callback(self, msg):
        with self.lock:
            try:
                # 将 RGB 图像消息转换为 OpenCV 格式
                self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f"Error processing RGB image: {e}")

    def handle_get_detections(self, request, response):
        """
        服务回调函数，当客户端请求目标名称时调用。
        返回该目标的三维坐标。
        """
        target_name = request.target_name.lower()
        self.get_logger().info(f"Received request for target: {target_name}")

        # 使用单独的线程处理检测，以避免阻塞服务回调
        detection_thread = threading.Thread(target=self.perform_detection, args=(target_name, response))
        detection_thread.start()
        detection_thread.join()

        return response

    def perform_detection(self, target_name, response):
        with self.lock:
            rgb_image = self.latest_rgb_image.copy() if self.latest_rgb_image is not None else None
            depth_image = self.latest_depth_image.copy() if self.latest_depth_image is not None else None

        if rgb_image is None or depth_image is None:
            response.success = False
            response.coordinates = Point()
            response.message = "RGB or Depth image not available."
            self.get_logger().warning("RGB or Depth image not available.")
            return

        try:
            # 使用 YOLO 模型进行目标检测
            results = self.model(rgb_image)
            detections = results[0].boxes  # 获取检测框信息
            self.get_logger().info(f"YOLO模型使用的设备: {self.model.device}")


            detected_objects = []

            for box in detections:
                # 提取检测框中心点的像素坐标 (u, v)
                x_center, y_center = box.xywh[0][:2].cpu().numpy().astype(int)

                # 确保像素坐标在深度图范围内
                if (0 <= x_center < depth_image.shape[1] and 
                    0 <= y_center < depth_image.shape[0]):

                    # 从深度图中获取深度值，单位转换为米
                    depth_value = depth_image[y_center, x_center] / 1000.0

                    # 过滤无效深度值
                    if np.isnan(depth_value) or depth_value <= 0:
                        self.get_logger().warning(f"Invalid depth value at ({x_center}, {y_center})")
                        continue

                    # 转换像素坐标与深度值到相机坐标系
                    camera_coordinates = self.pixel_to_camera_coordinates(
                        x_center, y_center, depth_value
                    )

                    # 获取目标名称（假设类别编号对应于名称，可以根据实际情况调整）
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"Class_{class_id}"

                    # 创建 DetectedObject 消息
                    detected_obj = DetectedObject()
                    detected_obj.name = class_name
                    detected_obj.coordinates = Point(
                        x=camera_coordinates[0],
                        y=camera_coordinates[1],
                        z=camera_coordinates[2]
                    )

                    detected_objects.append(detected_obj)

            # 搜索目标名称
            for obj in detected_objects:
                if obj.name.lower() == target_name:
                    response.success = True
                    response.coordinates = obj.coordinates
                    response.message = f"Coordinates of '{target_name}' retrieved successfully."
                    self.get_logger().info(f"Served coordinates for target: {target_name}")
                    return

            # 如果未找到目标
            response.success = False
            response.coordinates = Point()
            response.message = f"Target '{target_name}' not found among detected objects."
            self.get_logger().info(f"Target '{target_name}' not found.")

        except Exception as e:
            response.success = False
            response.coordinates = Point()
            response.message = f"Error during detection: {e}"
            self.get_logger().error(f"Detection error: {e}")

    def pixel_to_camera_coordinates(self, u, v, depth):
        """
        将像素坐标与深度值转换为相机坐标系三维坐标。
        Args:
            u (int): 像素坐标 u（x 坐标）。
            v (int): 像素坐标 v（y 坐标）。
            depth (float): 深度值（米）。
        Returns:
            tuple: 相机坐标系中的三维坐标 (X, Y, Z)。
        """
        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]

        X = (u - cx) * depth / fx  # X 坐标
        Y = (v - cy) * depth / fy  # Y 坐标
        Z = depth                  # Z 坐标
        return (X, Y, Z)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = Yolov11_d435i()
        rclpy.spin(node)
    except Exception as e:
        print(f"Exception in main: {e}")
    finally:
        print("main finished (finally block)")
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

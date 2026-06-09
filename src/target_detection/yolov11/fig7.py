import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from threading import Lock

class Yolov11_d435i(Node):
    def __init__(self):
        super().__init__('yolov11_d435i')

        # 初始化 CvBridge 和 YOLO 模型
        self.bridge = CvBridge()
        self.model = YOLO('/home/robot/aubo_ros2_ws/dataset/runs/train/yolov11n_experiment/weights/best.pt')
        self.latest_rgb_image = None
        self.latest_depth_image = None

        # 定义相机内参
        self.camera_intrinsics = {
            "fx": 647.485496,  # 焦距（像素）
            "fy": 657.347923,
            "cx": 313.411255,  # 主点 x 坐标
            "cy": 253.728786   # 主点 y 坐标
        }

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

        # 添加定时器，定时调用检测与图像显示函数（间隔单位为秒）
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("YOLOv11 Node with visualization has been started")

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

    def timer_callback(self):
        """
        定时回调函数：从最新获取的图像中执行目标检测，
        绘制检测结果、深度信息以及三维坐标，并通过 OpenCV 显示图像。
        """
        with self.lock:
            if self.latest_rgb_image is None or self.latest_depth_image is None:
                return
            rgb_image = self.latest_rgb_image.copy()
            depth_image = self.latest_depth_image.copy()

        try:
            # 使用 YOLO 模型对 RGB 图像进行目标检测
            results = self.model(rgb_image)
            detections = results[0].boxes  # 检测框信息

            # 对深度图进行颜色映射以便可视化
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 遍历所有检测结果
            for box in detections:
                # 获取检测框的左上和右下坐标（转换为整数）
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(rgb_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

                # 计算检测框中心点
                x_center, y_center = box.xywh[0][:2].cpu().numpy().astype(int)
                # 确保中心点在深度图范围内，获取深度值并转换为米
                if 0 <= x_center < depth_image.shape[1] and 0 <= y_center < depth_image.shape[0]:
                    depth_value = depth_image[y_center, x_center] / 1000.0
                else:
                    depth_value = 0

                # 获取目标类别
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"Class_{class_id}"

                # 在 RGB 图像上绘制检测框和文本
                text_rgb = f"{class_name}: {depth_value:.2f}m"
                cv2.putText(rgb_image, text_rgb, (xyxy[0], max(xyxy[1]-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(rgb_image, (x_center, y_center), 5, (0, 0, 255), -1)

                # 计算像素点对应的三维坐标
                camera_coordinates = self.pixel_to_camera_coordinates(x_center, y_center, depth_value)
                text_depth = f"3D: ({camera_coordinates[0]:.2f}, {camera_coordinates[1]:.2f}, {camera_coordinates[2]:.2f})"

                # 在深度图（颜色映射后）上标出三维坐标及中心点
                cv2.putText(depth_colormap, text_depth, (xyxy[0], max(xyxy[1]-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(depth_colormap, (x_center, y_center), 5, (0, 0, 255), -1)

            # 显示检测效果和深度对齐效果
            cv2.imshow("Detection", rgb_image)
            cv2.imshow("Depth", depth_colormap)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error during detection/visualization: {e}")

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

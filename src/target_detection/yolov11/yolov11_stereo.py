import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from yolo11_custom import create_yolo_model

class YoloStereo3DNode(Node):
    def __init__(self):
        super().__init__('yolov11_stereo')
       
        # 订阅复合流图像话题
        self.image_subscription = self.create_subscription(
            Image,
            '/stereo/image_raw',  # 复合流图像话题
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed to /stereo/image_raw")
        
        # 初始化变量
        self.bridge = CvBridge()
        self.model = create_yolo_model('yolo11n.pt')  # 加载YOLO模型
        self.left_image = None
        self.right_image = None
        
        # 初始化标志位
        self.left_received = False
        self.right_received = False
        
        # 相机内参
        self.fx = 615.0  # 焦距 fx
        self.fy = 615.0  # 焦距 fy
        self.cx = 320.0  # 光学中心 cx
        self.cy = 240.0  # 光学中心 cy
        self.baseline = 0.12  # 相机基线（单位：米）
        self.get_logger().info("YOLOv11 Stereo 3D Node has been started")
    
    def image_callback(self, msg):
        try:
            # 将复合流图像消息转换为OpenCV格式
            composite_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 假设复合流图像包含左右图像，并且左右图像的大小相同
            height, width, _ = composite_image.shape
            self.left_image = composite_image[:, :width // 2]
            self.right_image = composite_image[:, width // 2:]
            
            self.left_received = True
            self.right_received = True
            
            # 处理图像
            self.process_images()
        except Exception as e:
            self.get_logger().error(f"Error processing stereo image: {e}")

    def process_images(self):
        if not self.left_received or not self.right_received:
            return  # 如果左右图像尚未都接收到，跳过处理

        # 将左右图像转换为灰度图像
        left_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)
        
        # 计算视差图
        stereo = cv2.StereoSGBM_create(minDisparity=0,
                                        numDisparities=16,
                                        blockSize=15,
                                        P1=8 * 3 * 15 ** 2,
                                        P2=32 * 3 * 15 ** 2,
                                        uniquenessRatio=10,
                                        speckleWindowSize=100,
                                        speckleRange=32)
        disparity = stereo.compute(left_gray, right_gray)

        # 将视差值转换为深度
        depth_map = self.fx * self.baseline / (disparity / 16.0 + 0.01)

        # 使用YOLO进行目标检测
        results = self.model(self.left_image)
        detections = results[0].boxes  # 获取检测框信息
        
        # 如果没有检测到任何目标，跳过后续处理
        if len(detections) == 0:
            self.get_logger().info("No detections found")
            return  # 直接跳过后续的深度和坐标计算

        # 可视化检测结果
        annotated_frame = results[0].plot()

        # 整合检测结果与深度信息
        for box in detections:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
            
            x_min = max(0, min(x_min, depth_map.shape[1] - 1))
            x_max = max(0, min(x_max, depth_map.shape[1] - 1))
            y_min = max(0, min(y_min, depth_map.shape[0] - 1))
            y_max = max(0, min(y_max, depth_map.shape[0] - 1))
            
            if x_min >= x_max or y_min >= y_max:
                self.get_logger().warn(f"Invalid ROI: ({x_min}, {y_min}) - ({x_max}, {y_max})")
                continue

            # 获取检测框区域的深度值
            depth_roi = depth_map[y_min:y_max, x_min:x_max]
            valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
            
            if len(valid_depths) == 0:
                self.get_logger().warn(f"No valid depth value in ROI ({x_min}, {y_min}) - ({x_max}, {y_max})")
                continue

            # 使用中值深度计算三维坐标
            depth_value = np.median(valid_depths)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            
            x_camera = (x_center - self.cx) * depth_value / self.fx
            y_camera = (y_center - self.cy) * depth_value / self.fy
            z_camera = depth_value

            self.get_logger().info(f"Detected {box.cls} at (X: {x_camera:.2f}, Y: {y_camera:.2f}, Z: {z_camera:.2f})")

            # 在图像上显示三维坐标
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

        # 显示检测结果
        cv2.imshow("YOLO Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloStereo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

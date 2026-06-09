#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import numpy as np

# 加载 ultralytics YOLO 模型
from ultralytics import YOLO

class YOLO11DetectionNode(Node):
    def __init__(self):
        super().__init__('yolo11_detection_node')
        self.bridge = CvBridge()

        # 同步订阅 RGB 图像与深度图像
        self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                               queue_size=10, slop=0.1)
        self.ats.registerCallback(self.callback)

        # 加载预训练的 YOLO11n 模型，请确保模型文件路径正确
        self.model = YOLO("yolo11n-seg.pt")

        self.conf_threshold = 0.5

        # 设置摄像头内参
        self.fx = 647.485496
        self.fy = 657.347923
        self.cx = 313.411255
        self.cy = 253.728786

    def callback(self, rgb_msg, depth_msg):
        try:
            # 转换 ROS 图像消息到 OpenCV 格式
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("cv_bridge 转换失败: %s" % str(e))
            return

        # 使用 ultralytics 模型进行目标检测
        results = self.model.predict(cv_rgb, conf=self.conf_threshold)

        # 解析检测结果，假定检测结果格式为 [x1, y1, x2, y2, conf, cls]
        detections = []
        for result in results:
            try:
                boxes = result.boxes.data.cpu().numpy()  # 若使用 GPU 转换到 CPU
            except Exception:
                boxes = result.boxes.data.numpy()
            for det in boxes:
                x1, y1, x2, y2, conf, cls = det
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                detections.append([x, y, w, h, conf, cls])

        # 对每个检测目标计算 3D 位置信息，并在 RGB 图像上绘制立体框
        for det in detections:
            x, y, w, h, conf, cls = det

            # 计算检测框中心像素坐标
            center_u = x + w / 2.0
            center_v = y + h / 2.0

            # 检查中心点是否在深度图范围内
            d_u = int(round(center_u))
            d_v = int(round(center_v))
            if d_u < 0 or d_v < 0 or d_u >= cv_depth.shape[1] or d_v >= cv_depth.shape[0]:
                self.get_logger().warn("检测中心超出深度图范围")
                continue

            depth_value = cv_depth[d_v, d_u]
            if depth_value == 0 or np.isnan(depth_value):
                self.get_logger().warn("检测中心深度值无效")
                continue

            # 利用相机内参将像素坐标和深度转换为 3D 坐标（单位：米）
            X = (center_u - self.cx) * depth_value / self.fx
            Y = (center_v - self.cy) * depth_value / self.fy
            Z = depth_value

            # 简单估计目标在实际空间中的尺寸（基于检测框在图像中的大小）
            box_width  = w * depth_value / self.fx
            box_height = h * depth_value / self.fy
            box_depth  = max(box_width, box_height)

            self.get_logger().info(
                "检测到障碍物: 类别 %s, 置信度 %.2f, 3D 中心 (%.2f, %.2f, %.2f), 尺寸 (宽: %.2f, 高: %.2f, 深: %.2f)" %
                (cls, conf, X, Y, Z, box_width, box_height, box_depth)
            )

            # 计算立体框八个角点（假定框与相机坐标系对齐）
            dx = box_width / 2.0
            dy = box_height / 2.0
            dz = box_depth / 2.0
            # 定义 8 个角点（顺序排列，便于后续连线绘制）
            # 后面者 - z 较小的为“后面”，z 较大的为“前面”
            corners_3d = np.array([
                [X - dx, Y - dy, Z - dz],  # 0: 后面左上
                [X + dx, Y - dy, Z - dz],  # 1: 后面右上
                [X + dx, Y + dy, Z - dz],  # 2: 后面右下
                [X - dx, Y + dy, Z - dz],  # 3: 后面左下
                [X - dx, Y - dy, Z + dz],  # 4: 前面左上
                [X + dx, Y - dy, Z + dz],  # 5: 前面右上
                [X + dx, Y + dy, Z + dz],  # 6: 前面右下
                [X - dx, Y + dy, Z + dz]   # 7: 前面左下
            ])

            # 将 3D 点投影到图像平面（使用针孔模型：u = fx * x/z + cx, v = fy * y/z + cy）
            corners_2d = []
            for point in corners_3d:
                x3d, y3d, z3d = point
                u = int(round(self.fx * (x3d / z3d) + self.cx))
                v = int(round(self.fy * (y3d / z3d) + self.cy))
                corners_2d.append((u, v))
            # 定义立体框的边（索引对应的点连线）
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),   # 后面四边
                (4, 5), (5, 6), (6, 7), (7, 4),   # 前面四边
                (0, 4), (1, 5), (2, 6), (3, 7)    # 连接前后对应顶点
            ]

            # 在 RGB 图像上绘制立体框线条，颜色为绿色，粗细为 2
            for edge in edges:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                cv2.line(cv_rgb, pt1, pt2, (0, 255, 0), 2)

            # 同时也绘制 2D 检测框（蓝色）
            cv2.rectangle(cv_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # 在检测框上打印类别和置信度
            label = f"{cls:.0f}:{conf:.2f}"
            cv2.putText(cv_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # 显示绘制了立体框的 RGB 图像
        cv2.imshow("Detection", cv_rgb)
        cv2.waitKey(1)  # 每一帧等待 1 毫秒

def main(args=None):
    rclpy.init(args=args)
    node = YOLO11DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点中断，正在关闭……")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

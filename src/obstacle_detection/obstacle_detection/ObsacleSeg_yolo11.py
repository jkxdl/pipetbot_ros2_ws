#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import numpy as np

# 加载 ultralytics YOLO 模型（请确保所用模型支持分割任务）
from ultralytics import YOLO

class YOLO11SegmentationNode(Node):
    def __init__(self):
        super().__init__('yolo11_segmentation_node')
        self.bridge = CvBridge()

        # 同步订阅 RGB 图像和深度图像
        self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                               queue_size=10, slop=0.1)
        self.ats.registerCallback(self.callback)

        # 加载预训练的 YOLO 模型
        self.model = YOLO("yolo11n-seg.pt")
        self.conf_threshold = 0.5

        # 设置摄像头内参
        self.fx = 647.485496
        self.fy = 657.347923
        self.cx = 313.411255
        self.cy = 253.728786

    def callback(self, rgb_msg, depth_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error("图像转换失败: %s" % str(e))
            return

        # 使用模型进行分割推理
        results = self.model.predict(cv_rgb, conf=self.conf_threshold)

        # 遍历所有检测结果
        for result in results:
            if not hasattr(result, 'masks') or result.masks is None:
                continue

            # 获取分割 mask 数据，shape 通常为 (N, H, W)
            masks = result.masks.data
            # 遍历每个分割区域
            for i in range(len(masks)):
                # 将 mask 转为 numpy 数组，并二值化（阈值 0.5）
                mask = masks[i].cpu().numpy() if hasattr(masks[i], 'cpu') else np.array(masks[i])
                mask_bin = mask > 0.5

                # 如果分割区域面积太小则跳过
                if np.sum(mask_bin) < 50:
                    continue

                # 计算分割区域的二维包围盒
                indices = np.where(mask_bin)
                if indices[0].size == 0 or indices[1].size == 0:
                    continue
                y_min = int(np.min(indices[0]))
                y_max = int(np.max(indices[0]))
                x_min = int(np.min(indices[1]))
                x_max = int(np.max(indices[1]))
                w = x_max - x_min
                h = y_max - y_min

                # 计算包围盒中心像素坐标
                center_u = (x_min + x_max) / 2.0
                center_v = (y_min + y_max) / 2.0

                # 从深度图中获取该分割区域内的深度值（取中值比较稳健）
                depth_region = cv_depth[mask_bin]
                if depth_region.size == 0:
                    continue
                median_depth = np.median(depth_region)
                if median_depth == 0 or np.isnan(median_depth):
                    continue

                # 利用摄像头内参将图像中心像素坐标和深度转换到 3D 坐标（单位：米）
                X = (center_u - self.cx) * median_depth / self.fx
                Y = (center_v - self.cy) * median_depth / self.fy
                Z = median_depth

                # 利用包围盒尺寸（像素）以及深度信息估计物体在实际空间中的尺寸
                box_width  = w * median_depth / self.fx
                box_height = h * median_depth / self.fy
                # 对于深度方向，这里使用一个简单的启发式（可以根据需要进一步设计）
                box_depth  = max(box_width, box_height)

                # 取出目标类别，否则显示 "object"
                if hasattr(result.boxes, 'cls'):
                    class_id = int(result.boxes.cls[i])
                    if hasattr(self.model, 'names'):
                        label_name = self.model.names.get(class_id, str(class_id))
                    else:
                        label_name = str(class_id)
                else:
                    label_name = "object"

                # 可选：在图像上叠加分割 mask（绿色半透明）
                colored_mask = np.zeros_like(cv_rgb, dtype=np.uint8)
                colored_mask[mask_bin] = (0, 255, 0)
                cv_rgb = cv2.addWeighted(cv_rgb, 1.0, colored_mask, 0.3, 0)

                # 计算 3D 立体框 8 个角点（假设立体框与相机坐标系轴对齐）
                dx = box_width / 2.0
                dy = box_height / 2.0
                dz = box_depth / 2.0
                corners_3d = np.array([
                    [X - dx, Y - dy, Z - dz],  # 后面左上
                    [X + dx, Y - dy, Z - dz],  # 后面右上
                    [X + dx, Y + dy, Z - dz],  # 后面右下
                    [X - dx, Y + dy, Z - dz],  # 后面左下
                    [X - dx, Y - dy, Z + dz],  # 前面左上
                    [X + dx, Y - dy, Z + dz],  # 前面右上
                    [X + dx, Y + dy, Z + dz],  # 前面右下
                    [X - dx, Y + dy, Z + dz]   # 前面左下
                ])

                # 将 3D 点投影到 2D 图像平面（针孔模型：u = fx * x/z + cx, v = fy * y/z + cy）
                corners_2d = []
                for point in corners_3d:
                    x3d, y3d, z3d = point
                    u = int(round(self.fx * (x3d / z3d) + self.cx))
                    v = int(round(self.fy * (y3d / z3d) + self.cy))
                    corners_2d.append((u, v))
                
                # 定义立体框各边的连线（背面、前面及对应连接边）
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),   # 背面边界
                    (4, 5), (5, 6), (6, 7), (7, 4),   # 前面边界
                    (0, 4), (1, 5), (2, 6), (3, 7)    # 前后连接边
                ]
                for edge in edges:
                    pt1 = corners_2d[edge[0]]
                    pt2 = corners_2d[edge[1]]
                    cv2.line(cv_rgb, pt1, pt2, (255, 0, 0), 2)  # 蓝色线条

                # 在分割区域附近显示目标名称和立体框尺寸
                text = f"{label_name} L:{box_width:.2f} H:{box_height:.2f} D:{box_depth:.2f}"
                cv2.putText(cv_rgb, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
                # 同时绘制 2D 包围盒（黄色）
                #cv2.rectangle(cv_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        cv2.imshow("Segmentation 3D Box", cv_rgb)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLO11SegmentationNode()
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

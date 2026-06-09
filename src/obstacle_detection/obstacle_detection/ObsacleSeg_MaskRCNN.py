#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
import numpy as np

class MaskRCNNDetectionNode(Node):
    def __init__(self):
        super().__init__('maskrcnn_detection_node')
        self.bridge = CvBridge()

        # 同步订阅 RGB 图像和深度图像
        self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                               queue_size=10, slop=0.1)
        self.ats.registerCallback(self.callback)

        # 加载预训练的 Mask R-CNN 模型（请确保模型文件和配置文件路径正确）
        modelWeights = "obstacle_detection/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco.pb"
        modelConfig = "obstacle_detection/mask_rcnn_inception_v2_coco.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(modelWeights, modelConfig)

        self.detection_threshold = 0.7  # 检测置信度阈值
        self.mask_threshold = 0.3       # 分割 mask 二值化阈值

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
            self.get_logger().error("cv_bridge 转换失败: %s" % str(e))
            return

        (H, W) = cv_rgb.shape[:2]
        # 构造 blob，并前向推理得到检测框和 mask
        blob = cv2.dnn.blobFromImage(cv_rgb, swapRB=True, crop=False)
        self.net.setInput(blob)
        # 输出名称根据模型而定，此处使用 "detection_out_final" 和 "detection_masks"
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        numDetections = boxes.shape[2]

        for i in range(numDetections):
            score = boxes[0, 0, i, 2]
            if score < self.detection_threshold:
                continue

            class_id = int(boxes[0, 0, i, 1])
            # 检测框为归一化坐标，转换到图像尺寸
            x1 = int(boxes[0, 0, i, 3] * W)
            y1 = int(boxes[0, 0, i, 4] * H)
            x2 = int(boxes[0, 0, i, 5] * W)
            y2 = int(boxes[0, 0, i, 6] * H)

            # 防止检测框超出图像边界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)

            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue

            # 取出当前检测的 mask（模型输出中的 mask shape 为 [N, numClasses, mask_H, mask_W]）
            mask = masks[i, class_id]
            # 将 mask 缩放到检测框大小
            mask = cv2.resize(mask, (box_width, box_height))
            # 二值化 mask
            _, mask_bin = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            mask_bin = mask_bin.astype(np.uint8)

            # 构造与原图同尺寸的全局 mask，将分割结果放置到对应位置
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask_bin

            # 可选：将分割区域以透明绿色叠加在图像上
            colored_mask = np.zeros_like(cv_rgb)
            colored_mask[full_mask == 255] = (0, 255, 0)
            cv_rgb = cv2.addWeighted(cv_rgb, 1.0, colored_mask, 0.5, 0)

            # 根据 full_mask 从深度图中提取有效深度值，并计算对应的 3D 坐标
            pts3d = []
            ys, xs = np.where(full_mask == 255)
            for v, u in zip(ys, xs):
                # 注意：请确保 cv_depth 单位与相机内参匹配（通常为米）
                d = cv_depth[v, u]
                if d == 0 or np.isnan(d):
                    continue
                X = (u - self.cx) * d / self.fx
                Y = (v - self.cy) * d / self.fy
                Z = d
                pts3d.append([X, Y, Z])
            pts3d = np.array(pts3d)
            if pts3d.size == 0:
                continue

            # 计算分割区域内的 3D 点的边界，即轴对齐的立体框
            min_vals = np.amin(pts3d, axis=0)
            max_vals = np.amax(pts3d, axis=0)

            # 计算立体框 8 个角点（按顺序排列，便于连线绘制）
            corners_3d = np.array([
                [min_vals[0], min_vals[1], min_vals[2]],  # 0
                [max_vals[0], min_vals[1], min_vals[2]],  # 1
                [max_vals[0], max_vals[1], min_vals[2]],  # 2
                [min_vals[0], max_vals[1], min_vals[2]],  # 3
                [min_vals[0], min_vals[1], max_vals[2]],  # 4
                [max_vals[0], min_vals[1], max_vals[2]],  # 5
                [max_vals[0], max_vals[1], max_vals[2]],  # 6
                [min_vals[0], max_vals[1], max_vals[2]]   # 7
            ])

            # 利用相机模型投影 3D 点到图像平面：u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
            corners_2d = []
            for point in corners_3d:
                Xp, Yp, Zp = point
                if Zp == 0:
                    continue
                u_proj = int(round(self.fx * (Xp / Zp) + self.cx))
                v_proj = int(round(self.fy * (Yp / Zp) + self.cy))
                corners_2d.append((u_proj, v_proj))
            if len(corners_2d) != 8:
                continue

            # 定义立体框的边（8 个角点的连线关系）
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 底面矩形
                (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面矩形
                (0, 4), (1, 5), (2, 6), (3, 7)   # 侧面
            ]
            for edge in edges:
                pt1 = corners_2d[edge[0]]
                pt2 = corners_2d[edge[1]]
                cv2.line(cv_rgb, pt1, pt2, (0, 0, 255), 2)

            # 同时绘制检测框和置信度信息（可选）
            cv2.rectangle(cv_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID:{class_id} {score:.2f}"
            cv2.putText(cv_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 显示叠加了分割 mask 和 3D 立体框的 RGB 图像
        cv2.imshow("Mask R-CNN 3D Cuboid", cv_rgb)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = MaskRCNNDetectionNode()
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

import rclpy
import threading
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from scipy.optimize import linear_sum_assignment

class StereoCircleMatcher(Node):
    def __init__(self):
        super().__init__('stereo_circle_matcher')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 图像存储
        self.left_image = None
        self.right_image = None
        
        # 初始化参数
        self.gamma_values = {'left': 2.5, 'right': 2.5}
        self.hough_params = {
            'left': {'dp': 1.2, 'minDist': 50, 'param1': 45, 'param2': 40, 'minRadius': 10, 'maxRadius': 40},
            'right': {'dp': 1.2, 'minDist': 50, 'param1': 45, 'param2': 40, 'minRadius': 10, 'maxRadius': 40}
        }
        self.denoise_params = {
            'median_ksize': 3,
            'gaussian_ksize': (3, 3),
            'clahe_clip': 2.0,
            'clahe_grid': (8, 8),
            'bilateral_d': 5,
            'bilateral_sigma': 50
        }

        # 初始化图像订阅（注意此处使用 sensor_msgs.msg.Image 作为消息类型）
        self.setup_image_subscribers()
        
        # 初始化可视化窗口
        cv2.namedWindow("Matched Circles", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matched Circles", 1200, 600)

    def setup_image_subscribers(self):
        """设置左右图像订阅同步器"""
        self.left_sub = Subscriber(self, Image, '/stereo/left/image_raw')
        self.right_sub = Subscriber(self, Image, '/stereo/right/image_raw')
        
        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

    def _hybrid_denoise(self, img):
        """混合中值和高斯滤波"""
        try:
            median = cv2.medianBlur(img, self.denoise_params['median_ksize'])
            return cv2.GaussianBlur(median, self.denoise_params['gaussian_ksize'], 0)
        except cv2.error as e:
            self.get_logger().error(f"混合降噪失败: {str(e)}")
            return img

    def _clahe_enhance(self, img):
        """CLAHE对比度增强"""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.denoise_params['clahe_clip'],
                                    tileGridSize=self.denoise_params['clahe_grid'])
            l_clahe = clahe.apply(l_channel)
            return cv2.cvtColor(cv2.merge([l_clahe, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
        except cv2.error as e:
            self.get_logger().error(f"CLAHE增强失败: {str(e)}")
            return img

    def _adaptive_bilateral(self, img):
        """自适应双边滤波"""
        try:
            return cv2.bilateralFilter(img,
                                       self.denoise_params['bilateral_d'],
                                       self.denoise_params['bilateral_sigma'],
                                       self.denoise_params['bilateral_sigma'])
        except cv2.error as e:
            self.get_logger().error(f"双边滤波失败: {str(e)}")
            return img

    def _gamma_correction(self, img, gamma):
        """Gamma校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def preprocess_image(self, img, gamma):
        """图像预处理流水线"""
        try:
            denoised = self._hybrid_denoise(img)
            enhanced = self._clahe_enhance(denoised)
            filtered = self._adaptive_bilateral(enhanced)
            return self._gamma_correction(filtered, gamma)
        except Exception as e:
            self.get_logger().error(f"预处理失败: {str(e)}")
            return img

    def rank_circles(self, circles):
        """根据坐标对圆进行排序，返回每个圆的x和y排序序号"""
        circles_sorted_x = sorted(enumerate(circles), key=lambda c: c[1][0])
        circles_sorted_y = sorted(enumerate(circles), key=lambda c: c[1][1])

        ranks = {}
        for rank, (idx, _) in enumerate(circles_sorted_x):
            ranks[idx] = {'x_rank': rank}
        for rank, (idx, _) in enumerate(circles_sorted_y):
            ranks[idx]['y_rank'] = rank
        return ranks

    def build_cost_matrix(self, left_circles, right_circles, alpha=0.05):
        left_ranks = self.rank_circles(left_circles)
        right_ranks = self.rank_circles(right_circles)

        num_circles = len(left_circles)
        cost_matrix = np.zeros((num_circles, num_circles))
        for i in range(num_circles):
            xL, yL, _ = left_circles[i]
            for j in range(num_circles):
                xR, yR, _ = right_circles[j]
                rank_cost = abs(left_ranks[i]['x_rank'] - right_ranks[j]['x_rank']) + abs(left_ranks[i]['y_rank'] - right_ranks[j]['y_rank'])
                pixel_cost = abs(xL - xR) + abs(yL - yR)
                cost_matrix[i, j] = rank_cost + alpha * pixel_cost
        return cost_matrix

    def match_circles(self, left_circles, right_circles, alpha=0.05):
        """使用排序匹配策略，对左右图像中的圆进行匹配"""
        cost_matrix = self.build_cost_matrix(left_circles, right_circles, alpha)
        left_indices, right_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        for l_idx, r_idx in zip(left_indices, right_indices):
            matched_pairs.append((l_idx, r_idx))
        return matched_pairs

    def detect_circles(self, image, camera_side):
        """检测图像中的圆环"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            params = self.hough_params[camera_side]
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minRadius'],
                maxRadius=params['maxRadius']
            )
            return np.round(circles[0]).astype("int") if circles is not None else []
        except Exception as e:
            self.get_logger().error(f"圆检测失败: {str(e)}")
            return []

    def image_callback(self, left_msg, right_msg):
        """图像回调：预处理、检测、匹配并可视化"""
        with self.lock:
            try:
                # 图像转换
                raw_left = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
                raw_right = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
                
                # 图像预处理
                proc_left = self.preprocess_image(raw_left, self.gamma_values['left'])
                proc_right = self.preprocess_image(raw_right, self.gamma_values['right'])
                
                # 检测圆环
                left_circles = self.detect_circles(proc_left, 'left')
                right_circles = self.detect_circles(proc_right, 'right')
                
                # 仅在左右图像检测到相同数量的圆时进行匹配
                if len(left_circles) != len(right_circles) or len(left_circles) == 0:
                    self.get_logger().info("左右图像中检测到的圆数量不一致，或未检测到圆。")
                    return
                
                matched_pairs = self.match_circles(left_circles, right_circles)
                
                # 在图像上绘制检测结果和匹配 id（匹配成功的左右圆环赋予相同 id）
                annotated_left = proc_left.copy()
                annotated_right = proc_right.copy()
                
                for common_id, (left_idx, right_idx) in enumerate(matched_pairs):
                    # 左图：绘制圆和共同 id
                    xL, yL, rL = left_circles[left_idx]
                    cv2.circle(annotated_left, (xL, yL), rL, (0, 255, 0), 2)
                    cv2.putText(annotated_left, f"ID:{common_id}", (xL-10, yL-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 右图：绘制圆和共同 id
                    xR, yR, rR = right_circles[right_idx]
                    cv2.circle(annotated_right, (xR, yR), rR, (0, 255, 0), 2)
                    cv2.putText(annotated_right, f"ID:{common_id}", (xR-10, yR-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 拼接左右图像进行展示
                h, w = annotated_left.shape[:2]
                display = np.zeros((h, w*2, 3), dtype=np.uint8)
                display[:, :w] = annotated_left
                display[:, w:] = annotated_right
                
                cv2.imshow("Matched Circles", display)
                cv2.waitKey(1)
                
            except Exception as e:
                self.get_logger().error(f"图像处理错误: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = StereoCircleMatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        cv2.destroyAllWindows()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

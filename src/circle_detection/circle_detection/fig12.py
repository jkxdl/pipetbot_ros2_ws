import rclpy
import threading
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber

class CircleDetectorPipeline(Node):
    def __init__(self):
        super().__init__('circle_detector_pipeline')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 参数设置
        self.gamma_value = 2.5
        self.hough_params = {
            'dp': 1.2,
            'minDist': 50,
            'param1': 65,
            'param2': 60,
            'minRadius': 10,
            'maxRadius': 40
        }
        # 降噪参数：中值滤波 + 高斯滤波
        self.denoise_params = {
            'median_ksize': 3,
            'gaussian_ksize': (3, 3)
        }
        # 滤波参数：双边滤波
        self.filter_params = {
            'bilateral_d': 5,
            'bilateral_sigma': 50
        }
        # CLAHE参数
        self.clahe_params = {
            'clipLimit': 2.0,
            'tileGridSize': (8, 8)
        }
        
        # 只订阅左侧图像
        self.left_sub = Subscriber(self, Image, '/stereo/left/image_raw')
        self.left_sub.registerCallback(self.image_callback)

    def _clahe_enhance(self, img):
        """对图像进行CLAHE增强"""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_params['clipLimit'],
                tileGridSize=self.clahe_params['tileGridSize']
            )
            l_clahe = clahe.apply(l)
            lab_enhanced = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        except cv2.error as e:
            self.get_logger().error(f"CLAHE增强失败: {str(e)}")
            return img

    def _hybrid_denoise(self, img):
        """对图像进行混合降噪：中值滤波 + 高斯滤波"""
        try:
            median = cv2.medianBlur(img, self.denoise_params['median_ksize'])
            return cv2.GaussianBlur(median, self.denoise_params['gaussian_ksize'], 0)
        except cv2.error as e:
            self.get_logger().error(f"混合降噪失败: {str(e)}")
            return img

    def _adaptive_bilateral(self, img):
        """对图像进行双边滤波"""
        try:
            return cv2.bilateralFilter(
                img,
                self.filter_params['bilateral_d'],
                self.filter_params['bilateral_sigma'],
                self.filter_params['bilateral_sigma']
            )
        except cv2.error as e:
            self.get_logger().error(f"双边滤波失败: {str(e)}")
            return img

    def _gamma_correction(self, img, gamma):
        """对图像进行Gamma矫正"""
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                              for i in np.arange(256)]).astype("uint8")
            return cv2.LUT(img, table)
        except cv2.error as e:
            self.get_logger().error(f"Gamma矫正失败: {str(e)}")
            return img

    def detect_circles(self, image):
        """使用霍夫变换检测圆环"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_params['dp'],
                minDist=self.hough_params['minDist'],
                param1=self.hough_params['param1'],
                param2=self.hough_params['param2'],
                minRadius=self.hough_params['minRadius'],
                maxRadius=self.hough_params['maxRadius']
            )
            return np.round(circles[0]).astype("int") if circles is not None else []
        except Exception as e:
            self.get_logger().error(f"圆检测失败: {str(e)}")
            return []

    def draw_circles(self, image, circles):
        """在图像上绘制检测到的圆"""
        disp = image.copy()
        for (x, y, r) in circles:
            cv2.circle(disp, (x, y), r, (0, 255, 0), 2)
        return disp

    def image_callback(self, msg):
        with self.lock:
            try:
                # 将ROS图像转换为OpenCV格式
                raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                
                # 1. 原始图像
                step1 = raw
                
                # 2. 原始图像 + CLAHE增强
                step2 = self._clahe_enhance(raw)
                
                # 3. 原始图像 + CLAHE增强 + 降噪
                step3 = self._hybrid_denoise(step2)
                
                # 4. 原始图像 + CLAHE增强 + 降噪 + 滤波（双边滤波）
                step4 = self._adaptive_bilateral(step3)
                
                # 5. 原始图像 + CLAHE增强 + 降噪 + 滤波 + Gamma矫正
                step5 = self._gamma_correction(step4, self.gamma_value)
                
                # 对每一步的图像分别执行霍夫检测并绘制检测到的圆
                steps = {
                    'Original': step1,
                    'CLAHE': step2,
                    'Denoised': step3,
                    'Filtered': step4,
                    'Gamma': step5
                }
                results = {}
                for name, img in steps.items():
                    circles = self.detect_circles(img)
                    results[name] = self.draw_circles(img, circles)
                
                # 构造显示图像：这里将5个图像拼接为两行三列（最后一个空白）
                h, w = raw.shape[:2]
                canvas = np.zeros((h*2, w*3, 3), dtype=np.uint8)
                canvas[0:h, 0:w] = results['Original']
                canvas[0:h, w:2*w] = results['CLAHE']
                canvas[0:h, 2*w:3*w] = results['Denoised']
                canvas[h:2*h, 0:w] = results['Filtered']
                canvas[h:2*h, w:2*w] = results['Gamma']
                
                # 可选：调整显示窗口大小（例如1200x800）
                cv2.namedWindow("Hough Circle Detection Pipeline", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Hough Circle Detection Pipeline", 1200, 800)
                
                # 添加文字标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(canvas, "Original", (10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(canvas, "CLAHE", (w + 10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(canvas, "CLAHE+Denoised", (2*w + 10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(canvas, "CLAHE+Denoised+Filtered", (10, h + 30), font, 1, (0, 255, 0), 2)
                cv2.putText(canvas, "CLAHE+Denoised+Filtered+Gamma", (w + 10, h + 30), font, 1, (0, 255, 0), 2)
                
                cv2.imshow("Hough Circle Detection Pipeline", canvas)
                cv2.waitKey(1)
                
            except Exception as e:
                self.get_logger().error(f"图像处理错误: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CircleDetectorPipeline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

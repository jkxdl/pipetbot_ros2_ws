import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

class StereoDepthCircleDetector(Node):
    def __init__(self):
        super().__init__('stereo_depth_circle_detector')
        self.bridge = CvBridge()

        # 声明并获取期望圆环数量
        self.declare_parameter('expected_circle_count', 50)
        self.expected_circle_count = self.get_parameter('expected_circle_count').get_parameter_value().integer_value

        self.get_logger().info(f"Expected number of circles: {self.expected_circle_count}")

        # 使用 message_filters 订阅左右图像，并进行同步
        left_sub = Subscriber(self, Image, '/stereo/left/image_raw')
        right_sub = Subscriber(self, Image, '/stereo/right/image_raw')

        # 创建时间同步器，队列大小为10，允许的时间差为0.1秒
        ts = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.synced_image_callback)

        self.left_image = None
        self.right_image = None

        # 相机标定参数
        self.fx = 1390.686382  # 焦距 x
        self.fy = 1377.484516  # 焦距 y
        self.cx = 918.236008    # 光心 x
        self.cy = 514.412701    # 光心 y
        self.baseline = 0.061   # 基线，单位: 米

        # Gamma校正参数
        self.gamma_left = 2.5
        self.gamma_right = 2.5

        # 霍夫圆检测参数
        self.hough_params_left = {
            'dp': 1.2,
            'minDist': 50,
            'param1': 45,
            'param2': 40,
            'minRadius': 10,
            'maxRadius': 40,
        }

        self.hough_params_right = {
            'dp': 1.2,
            'minDist': 50,
            'param1': 45,
            'param2': 40,
            'minRadius': 10,
            'maxRadius': 40,
        }
        
        self.denoise_params = {
            'median_kernel': 3,
            'gauss_kernel': (3, 3),
            'clahe_clip': 2.0,
            'clahe_grid': (8, 8),
            'bilateral_d': 5,
            'bilateral_sigma': 50
        }

        # 初始化卡尔曼滤波器列表，每个圆环对应左右两个滤波器
        self.kalman_filters = []
        for _ in range(self.expected_circle_count):
            # 左图卡尔曼滤波器
            kf_left = cv2.KalmanFilter(4, 2)  # 状态维度4 (x, y, vx, vy)，测量维度2 (x, y)
            kf_left.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            kf_left.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            kf_left.processNoiseCov = cv2.setIdentity(kf_left.processNoiseCov, 1e-5)
            kf_left.measurementNoiseCov = cv2.setIdentity(kf_left.measurementNoiseCov, 1e-4)
            kf_left.errorCovPost = cv2.setIdentity(kf_left.errorCovPost, 1)

            # 右图卡尔曼滤波器
            kf_right = cv2.KalmanFilter(4, 2)
            kf_right.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            kf_right.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            kf_right.processNoiseCov = cv2.setIdentity(kf_right.processNoiseCov, 1e-5)
            kf_right.measurementNoiseCov = cv2.setIdentity(kf_right.measurementNoiseCov, 1e-4)
            kf_right.errorCovPost = cv2.setIdentity(kf_right.errorCovPost, 1)

            self.kalman_filters.append((kf_left, kf_right))

        # 存储最新帧的圆环信息
        self.latest_circle_data = []

    def synced_image_callback(self, left_msg, right_msg):
        try:
            # 转换图像后立即显示原始图像
            raw_left = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            raw_right = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
            
            # 显示原始图像
            cv2.imshow("Raw Left", raw_left)
            #cv2.imshow("Raw Right", raw_right)
            
            # 执行预处理并存储
            self.left_image = self.preprocess_image(raw_left, self.gamma_left)
            self.right_image = self.preprocess_image(raw_right, self.gamma_right)
            
            # 显示预处理结果
            cv2.imshow("Processed Left", self.left_image)
            #cv2.imshow("Processed Right", self.right_image)
            
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")
            return

        self.process_images()

    def gamma_correction(self, image, gamma=1.5):
        """Gamma校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def hybrid_denoise(self, img):
        """混合中值-高斯滤波"""
        median = cv2.medianBlur(img, self.denoise_params['median_kernel'])
        return cv2.GaussianBlur(
            median, 
            self.denoise_params['gauss_kernel'], 
            0
        )

    def clahe_enhance(self, img):
        """CLAHE对比度增强"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.denoise_params['clahe_clip'],
            tileGridSize=self.denoise_params['clahe_grid']
        )
        return cv2.cvtColor(
            cv2.merge([clahe.apply(l), a, b]),
            cv2.COLOR_LAB2BGR
        )

    def adaptive_bilateral(self, img):
        """自适应双边滤波"""
        return cv2.bilateralFilter(
            img,
            self.denoise_params['bilateral_d'],
            self.denoise_params['bilateral_sigma'],
            self.denoise_params['bilateral_sigma']
        )

    def preprocess_image(self, img, gamma):
        """完整的预处理流水线"""
        # 步骤1：混合降噪
        denoised = self.hybrid_denoise(img)
        
        # 步骤2：对比度增强
        enhanced = self.clahe_enhance(denoised)
        
        # 步骤3：自适应双边滤波
        filtered = self.adaptive_bilateral(enhanced)
        
        # 步骤4：Gamma校正
        return self.gamma_correction(filtered, gamma)


    def detect_circles(self, image_bgr, gamma, hough_params):
        """检测图像中的圆"""
        processed = self.preprocess_image(image_bgr, gamma)
        
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=hough_params['dp'],
            minDist=hough_params['minDist'],
            param1=hough_params['param1'],
            param2=hough_params['param2'],
            minRadius=hough_params['minRadius'],
            maxRadius=hough_params['maxRadius'],
        )
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            for (x, y, r) in circles:
                result.append((x, y, r))
        return result

    def process_images(self):
        """处理图像，检测圆环，进行排序，并绘制结果"""
        # 1. 检测左右图像中的圆
        left_circles = self.detect_circles(
            self.left_image,
            gamma=self.gamma_left,
            hough_params=self.hough_params_left
        )

        right_circles = self.detect_circles(
            self.right_image,
            gamma=self.gamma_right,
            hough_params=self.hough_params_right
        )

        # 2. 可视化所有检测到的圆环，并绘制圆心坐标和排序结果
        self.visualize_all_circles(left_circles, right_circles)

        # 3. 判断左右图像的圆环数量是否等于 expected_circle_count
        if len(left_circles) != self.expected_circle_count or len(right_circles) != self.expected_circle_count:
            self.get_logger().warn(
                f"Detected circles (L={len(left_circles)}, R={len(right_circles)}) != expected {self.expected_circle_count}. Skip this frame."
            )
            return

        # 4. 对左图和右图中的圆环分别排序
        left_circles_sorted = sorted(left_circles, key=lambda c: (c[1], c[0]))
        right_circles_sorted = sorted(right_circles, key=lambda c: (c[1], c[0]))

        # 5. 分配物理ID并计算三维坐标（应用卡尔曼滤波）
        self.latest_circle_data = []  # 重置最新圆环数据

        for idx, (left_circle, right_circle) in enumerate(zip(left_circles_sorted, right_circles_sorted)):
            if idx >= self.expected_circle_count:
                break

            xL, yL, rL = left_circle
            xR, yR, rR = right_circle

            # 获取对应的卡尔曼滤波器
            kf_left, kf_right = self.kalman_filters[idx]

            # 预测
            kf_left.predict()
            kf_right.predict()

            # 更新
            measurement_left = np.array([[np.float32(xL)], [np.float32(yL)]])
            measurement_right = np.array([[np.float32(xR)], [np.float32(yR)]])

            corrected_left = kf_left.correct(measurement_left)
            corrected_right = kf_right.correct(measurement_right)

            # 获取滤波后的坐标
            xL_filtered = corrected_left[0, 0]
            yL_filtered = corrected_left[1, 0]
            xR_filtered = corrected_right[0, 0]
            yR_filtered = corrected_right[1, 0]

            # 计算视差和三维坐标
            disparity = xL_filtered - xR_filtered
            if disparity <= 0:
                self.get_logger().warn(f"Invalid disparity for circle pair {idx}: Left x={xL_filtered}, Right x={xR_filtered}")
                continue

            Z = (self.fx * self.baseline) / disparity
            X = (xL_filtered - self.cx) * Z / self.fx
            Y = (yL_filtered - self.cy) * Z / self.fy

            circle_info = {
                'id': idx,
                'left_x': xL_filtered,
                'left_y': yL_filtered,
                'right_x': xR_filtered,
                'right_y': yR_filtered,
                'X': X,
                'Y': Y,
                'Z': Z
            }
            self.latest_circle_data.append(circle_info)

            # 在左图中绘制物理ID和三维坐标（使用滤波后的坐标）
            cv2.putText(self.left_image, f"ID:{idx}", (int(xL_filtered) + 5, int(yL_filtered) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(self.left_image, f"({X:.2f}, {Y:.2f}, {Z:.2f})", (int(xL_filtered) + 5, int(yL_filtered) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 绘制滤波后的圆心（绿色）
            cv2.circle(self.left_image, (int(xL_filtered), int(yL_filtered)), 2, (0, 255, 0), 3)
            cv2.circle(self.right_image, (int(xR_filtered), int(yR_filtered)), 2, (0, 255, 0), 3)

        self.get_logger().info(f"Processed {len(self.latest_circle_data)} circles.")

        # 6. 显示结果
        cv2.imshow("Left Image with Circles", self.left_image)
        cv2.imshow("Right Image with Circles", self.right_image)
        cv2.waitKey(1)

    def visualize_all_circles(self, left_circles, right_circles):
        """在左右图像中绘制所有检测到的圆环，并标注圆心坐标和排序后的序号"""
        # 绘制左图中的所有圆环
        for (x, y, r) in left_circles:
            cv2.circle(self.left_image, (int(x), int(y)), int(r), (255, 0, 0), 2)  # 蓝色圆环
            cv2.circle(self.left_image, (int(x), int(y)), 2, (0, 0, 255), 3)       # 红色圆心
            # 绘制圆心坐标
            cv2.putText(self.left_image, f"({x}, {y})", (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 绘制右图中的所有圆环
        for (x, y, r) in right_circles:
            cv2.circle(self.right_image, (int(x), int(y)), int(r), (255, 0, 0), 2)  # 蓝色圆环
            cv2.circle(self.right_image, (int(x), int(y)), 2, (0, 0, 255), 3)         # 红色圆心
            # 绘制圆心坐标
            cv2.putText(self.right_image, f"({x}, {y})", (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 绘制排序后的序号（绿色小字体）
        # 左图
        for idx, (x, y, r) in enumerate(sorted(left_circles, key=lambda c: (c[1], c[0]))):
            cv2.putText(self.left_image, f"{idx}", (int(x) - 10, int(y) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 右图
        for idx, (x, y, r) in enumerate(sorted(right_circles, key=lambda c: (c[1], c[0]))):
            cv2.putText(self.right_image, f"{idx}", (int(x) - 10, int(y) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthCircleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
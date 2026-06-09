import rclpy
import threading
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from scipy.optimize import linear_sum_assignment

# ======================= AD-Census 相关函数 =======================
PATCH_SIZE = 21  # 图像块尺寸，建议为奇数

def extract_patch(image, center, patch_size=PATCH_SIZE):
    """
    从图像中提取以 center 为中心的 patch_size×patch_size 区域（BGR图像），转换为灰度图。
    若区域超出图像边界，则返回 None。
    """
    half = patch_size // 2
    x, y = center
    h, w = image.shape[:2]
    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return None
    patch = image[y - half:y + half + 1, x - half:x + half + 1]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

def ad_census_cost(left_patch, right_patch, ad_weight=0.5, census_weight=0.5):
    """
    计算 AD-Census 匹配代价：
      - AD 代价：两个图像块灰度绝对差均值；
      - Census 代价：以图像块中心像素为阈值，计算二值描述符后计算汉明距离（归一化）。
    """
    ad_cost = np.mean(np.abs(left_patch.astype(np.float32) - right_patch.astype(np.float32)))
    center_val_left = left_patch[left_patch.shape[0] // 2, left_patch.shape[1] // 2]
    census_left = (left_patch >= center_val_left).astype(np.uint8)
    center_val_right = right_patch[right_patch.shape[0] // 2, right_patch.shape[1] // 2]
    census_right = (right_patch >= center_val_right).astype(np.uint8)
    census_cost = np.sum(census_left != census_right) / left_patch.size
    return ad_weight * ad_cost + census_weight * census_cost

def ad_census_match_circles(left_circles, right_circles, proc_left, proc_right,
                            ad_weight=0.5, census_weight=0.5, high_cost=1e6):
    """
    利用 AD-Census 算法对左右检测到的圆进行匹配：
      1. 对每个圆，从预处理图像中提取固定尺寸的图像块；
      2. 构建代价矩阵（无效区域赋予高代价）；
      3. 利用线性规划求解最优匹配。
    返回匹配对列表，格式为 (左图索引, 右图索引)。
    """
    num_left = len(left_circles)
    num_right = len(right_circles)
    cost_matrix = np.zeros((num_left, num_right))
    left_patches = []
    for circle in left_circles:
        x, y, _ = circle
        left_patches.append(extract_patch(proc_left, (x, y)))
    right_patches = []
    for circle in right_circles:
        x, y, _ = circle
        right_patches.append(extract_patch(proc_right, (x, y)))
    for i in range(num_left):
        for j in range(num_right):
            if left_patches[i] is None or right_patches[j] is None:
                cost_matrix[i, j] = high_cost
            else:
                cost_matrix[i, j] = ad_census_cost(left_patches[i], right_patches[j],
                                                   ad_weight, census_weight)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(i, j) for i, j in zip(row_ind, col_ind)]
    return matches

# ======================= SGM 算法（动态规划） =======================
def sgm_match_circles(left_circles, right_circles, penalty=10):
    """
    利用动态规划（SGM思想）对左右检测到的圆进行匹配：
      1. 分别对左右圆按照 (y, x) 排序；
      2. 构造 DP 表，考虑匹配代价（绝对位置差）和不匹配惩罚 penalty；
      3. 回溯得到匹配对（恢复原始索引）。
    """
    left_sorted = sorted(enumerate(left_circles), key=lambda item: (item[1][1], item[1][0]))
    right_sorted = sorted(enumerate(right_circles), key=lambda item: (item[1][1], item[1][0]))
    if not left_sorted or not right_sorted:
        return []
    left_indices, left_sorted_circles = zip(*left_sorted)
    right_indices, right_sorted_circles = zip(*right_sorted)
    L = len(left_sorted_circles)
    R = len(right_sorted_circles)
    dp = np.zeros((L+1, R+1))
    for i in range(1, L+1):
        dp[i][0] = i * penalty
    for j in range(1, R+1):
        dp[0][j] = j * penalty
    cost_matrix = np.zeros((L, R))
    for i in range(L):
        xL, yL, _ = left_sorted_circles[i]
        for j in range(R):
            xR, yR, _ = right_sorted_circles[j]
            cost = abs(xL - xR) + abs(yL - yR)
            cost_matrix[i][j] = cost
            dp[i+1][j+1] = min(
                dp[i][j] + cost,
                dp[i][j+1] + penalty,
                dp[i+1][j] + penalty
            )
    matches = []
    i, j = L, R
    while i > 0 and j > 0:
        current = dp[i][j]
        cost = cost_matrix[i-1][j-1]
        if current == dp[i-1][j-1] + cost:
            matches.append((left_indices[i-1], right_indices[j-1]))
            i -= 1
            j -= 1
        elif current == dp[i-1][j] + penalty:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches

# ======================= BP 算法 =======================
def bp_match_circles(left_circles, right_circles, max_iter=20, epsilon=1e-3):
    """
    利用简单的 BP（Belief Propagation）思想对左右圆进行匹配：
      1. 构造基于欧氏距离的代价矩阵；
      2. 初始化左右消息，并迭代更新；
      3. 计算信念 belief = cost + m + n，对每个左圆选择信念值最小的右圆作为匹配。
    返回匹配对列表 (左图索引, 右图索引)。
    """
    num_left = len(left_circles)
    num_right = len(right_circles)
    cost = np.zeros((num_left, num_right))
    for i in range(num_left):
        xL, yL, _ = left_circles[i]
        for j in range(num_right):
            xR, yR, _ = right_circles[j]
            cost[i, j] = np.sqrt((xL - xR)**2 + (yL - yR)**2)
    m = np.zeros((num_left, num_right))
    n = np.zeros((num_left, num_right))
    for _ in range(max_iter):
        m_old = m.copy()
        for i in range(num_left):
            for j in range(num_right):
                row = cost[i] + n[i]
                if num_right > 1:
                    row_except_j = np.delete(row, j)
                    min_val = np.min(row_except_j)
                else:
                    min_val = 0
                m[i, j] = cost[i, j] - min_val
        for j in range(num_right):
            for i in range(num_left):
                col = cost[:, j] + m[:, j]
                if num_left > 1:
                    col_except_i = np.delete(col, i)
                    min_val = np.min(col_except_i)
                else:
                    min_val = 0
                n[i, j] = cost[i, j] - min_val
        if np.max(np.abs(m - m_old)) < epsilon:
            break
    belief = cost + m + n
    matches = []
    for i in range(num_left):
        j = int(np.argmin(belief[i]))
        matches.append((i, j))
    return matches

# ======================= ROS2 Node =======================
class StereoCircleMatcher(Node):
    def __init__(self):
        super().__init__('stereo_circle_matcher')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.left_image = None
        self.right_image = None
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
        self.setup_image_subscribers()
        # 显示窗口足够大，可显示 3 行匹配效果（每行左右图并排）
        cv2.namedWindow("Matching Effects", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matching Effects", 1200, 900)

    def setup_image_subscribers(self):
        self.left_sub = Subscriber(self, Image, '/stereo/left/image_raw')
        self.right_sub = Subscriber(self, Image, '/stereo/right/image_raw')
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

    def _hybrid_denoise(self, img):
        try:
            median = cv2.medianBlur(img, self.denoise_params['median_ksize'])
            return cv2.GaussianBlur(median, self.denoise_params['gaussian_ksize'], 0)
        except cv2.error as e:
            self.get_logger().error(f"混合降噪失败: {str(e)}")
            return img

    def _clahe_enhance(self, img):
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.denoise_params['clahe_clip'],
                                    tileGridSize=self.denoise_params['clahe_grid'])
            l_clahe = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
        except cv2.error as e:
            self.get_logger().error(f"CLAHE增强失败: {str(e)}")
            return img

    def _adaptive_bilateral(self, img):
        try:
            return cv2.bilateralFilter(img,
                                       self.denoise_params['bilateral_d'],
                                       self.denoise_params['bilateral_sigma'],
                                       self.denoise_params['bilateral_sigma'])
        except cv2.error as e:
            self.get_logger().error(f"双边滤波失败: {str(e)}")
            return img

    def _gamma_correction(self, img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def preprocess_image(self, img, gamma):
        try:
            denoised = self._hybrid_denoise(img)
            enhanced = self._clahe_enhance(denoised)
            filtered = self._adaptive_bilateral(enhanced)
            return self._gamma_correction(filtered, gamma)
        except Exception as e:
            self.get_logger().error(f"预处理失败: {str(e)}")
            return img

    def detect_circles(self, image, camera_side):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            params = self.hough_params[camera_side]
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=params['dp'],
                                       minDist=params['minDist'], param1=params['param1'],
                                       param2=params['param2'], minRadius=params['minRadius'],
                                       maxRadius=params['maxRadius'])
            return np.round(circles[0]).astype("int") if circles is not None else []
        except Exception as e:
            self.get_logger().error(f"圆检测失败: {str(e)}")
            return []

    def image_callback(self, left_msg, right_msg):
        with self.lock:
            try:
                raw_left = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
                raw_right = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
                proc_left = self.preprocess_image(raw_left, self.gamma_values['left'])
                proc_right = self.preprocess_image(raw_right, self.gamma_values['right'])
                left_circles = self.detect_circles(proc_left, 'left')
                right_circles = self.detect_circles(proc_right, 'right')

                # ---------------- SGM 匹配效果 ----------------
                sgm_matches = sgm_match_circles(left_circles, right_circles, penalty=10)
                sgm_left = proc_left.copy()
                sgm_right = proc_right.copy()
                # 绘制所有圆（蓝色）
                for circle in left_circles:
                    x, y, r = circle
                    cv2.circle(sgm_left, (x, y), r, (255, 0, 0), 2)
                for circle in right_circles:
                    x, y, r = circle
                    cv2.circle(sgm_right, (x, y), r, (255, 0, 0), 2)
                # 对匹配对赋予相同 id（绿色）
                for match_id, (l_idx, r_idx) in enumerate(sgm_matches):
                    xL, yL, _ = left_circles[l_idx]
                    xR, yR, _ = right_circles[r_idx]
                    cv2.putText(sgm_left, f"ID:{match_id}", (xL-10, yL-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(sgm_right, f"ID:{match_id}", (xR-10, yR-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                sgm_row = np.hstack((sgm_left, sgm_right))

                # ---------------- AD-Census 匹配效果 ----------------
                adcensus_matches = ad_census_match_circles(left_circles, right_circles, proc_left, proc_right,
                                                           ad_weight=0.5, census_weight=0.5)
                adcensus_left = proc_left.copy()
                adcensus_right = proc_right.copy()
                for circle in left_circles:
                    x, y, r = circle
                    cv2.circle(adcensus_left, (x, y), r, (255, 0, 0), 2)
                for circle in right_circles:
                    x, y, r = circle
                    cv2.circle(adcensus_right, (x, y), r, (255, 0, 0), 2)
                for match_id, (l_idx, r_idx) in enumerate(adcensus_matches):
                    xL, yL, _ = left_circles[l_idx]
                    xR, yR, _ = right_circles[r_idx]
                    cv2.putText(adcensus_left, f"ID:{match_id}", (xL-10, yL-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(adcensus_right, f"ID:{match_id}", (xR-10, yR-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                adcensus_row = np.hstack((adcensus_left, adcensus_right))

                # ---------------- BP 匹配效果 ----------------
                bp_matches = bp_match_circles(left_circles, right_circles, max_iter=20, epsilon=1e-3)
                bp_left = proc_left.copy()
                bp_right = proc_right.copy()
                for circle in left_circles:
                    x, y, r = circle
                    cv2.circle(bp_left, (x, y), r, (255, 0, 0), 2)
                for circle in right_circles:
                    x, y, r = circle
                    cv2.circle(bp_right, (x, y), r, (255, 0, 0), 2)
                for match_id, (l_idx, r_idx) in enumerate(bp_matches):
                    xL, yL, _ = left_circles[l_idx]
                    xR, yR, _ = right_circles[r_idx]
                    cv2.putText(bp_left, f"ID:{match_id}", (xL-10, yL-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(bp_right, f"ID:{match_id}", (xR-10, yR-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                bp_row = np.hstack((bp_left, bp_right))

                # 将三种匹配结果竖直拼接
                display = np.vstack((sgm_row, adcensus_row, bp_row))
                cv2.imshow("Matching Effects", display)
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

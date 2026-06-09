import rclpy
import asyncio
import threading
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, GoalResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from pipettingrobot_interfaces.action import GetExpectedCirclePosition
from pipettingrobot_interfaces.msg import CirclePosition
from scipy.optimize import linear_sum_assignment

# ros2 action send_goal /get_expected_circlesposition pipettingrobot_interfaces/action/GetExpectedCirclePosition "{expected_circle_count: 50}"

class StereoDepthCircleDetector(Node):
    def __init__(self):
        super().__init__('stereo_depth_circle_detector')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 图像存储
        self.left_image = None
        self.right_image = None
        self.valid_detection = False  # 有效检测标志
        
        # 初始化参数
        self.camera_params = {
            'fx': 1390.686382, 'fy': 1377.484516,
            'cx': 918.236008, 'cy': 514.412701,
            'baseline': 0.061
        }
        self.gamma_values = {'left': 2.5, 'right': 2.5}
        self.hough_params = {
            'left': {'dp':1.2, 'minDist':50, 'param1':45, 'param2':40, 'minRadius':10, 'maxRadius':40},
            'right': {'dp':1.2, 'minDist':50, 'param1':45, 'param2':40, 'minRadius':10, 'maxRadius':40}
        }
        self.denoise_params = {
            'median_ksize': 3,           # 中值滤波核大小
            'gaussian_ksize': (3, 3),     # 高斯滤波核尺寸
            'clahe_clip': 2.0,            # CLAHE对比度限制
            'clahe_grid': (8, 8),         # CLAHE网格尺寸
            'bilateral_d': 5,             # 双边滤波直径
            'bilateral_sigma': 50         # 双边滤波标准差
        }

        self.action_cb_group = ReentrantCallbackGroup()

        # 初始化动作服务器
        self.action_server = ActionServer(
            self,
            GetExpectedCirclePosition,
            'get_expected_circlesposition',
            self.execute_callback,
            callback_group=self.action_cb_group,  # 指定回调组
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback
        )
        
        # 初始化图像订阅
        self.setup_image_subscribers()

        # 初始化可视化窗口
        #cv2.namedWindow("Preprocessing", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("Preprocessing", 1200, 600)

    def setup_image_subscribers(self):
        """设置图像订阅同步器"""
        self.left_sub = Subscriber(self, Image, '/stereo/left/image_raw')
        self.right_sub = Subscriber(self, Image, '/stereo/right/image_raw')
        
        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

    def _hybrid_denoise(self, img):
        """混合中值-高斯滤波"""
        try:
            # 中值滤波去除椒盐噪声
            median = cv2.medianBlur(img, self.denoise_params['median_ksize'])
            # 高斯滤波平滑高斯噪声
            return cv2.GaussianBlur(
                median,
                self.denoise_params['gaussian_ksize'],
                0
            )
        except cv2.error as e:
            self.get_logger().error(f"混合降噪失败: {str(e)}")
            return img

    def _clahe_enhance(self, img):
        """CLAHE对比度受限自适应直方图均衡"""
        try:
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # 应用CLAHE到L通道
            clahe = cv2.createCLAHE(
                clipLimit=self.denoise_params['clahe_clip'],
                tileGridSize=self.denoise_params['clahe_grid']
            )
            l_clahe = clahe.apply(l_channel)
            
            # 合并通道并转换回BGR
            return cv2.cvtColor(
                cv2.merge([l_clahe, a_channel, b_channel]),
                cv2.COLOR_LAB2BGR
            )
        except cv2.error as e:
            self.get_logger().error(f"CLAHE增强失败: {str(e)}")
            return img

    def _adaptive_bilateral(self, img):
        """自适应双边滤波器"""
        try:
            return cv2.bilateralFilter(
                img,
                self.denoise_params['bilateral_d'],
                self.denoise_params['bilateral_sigma'],
                self.denoise_params['bilateral_sigma']
            )
        except cv2.error as e:
            self.get_logger().error(f"双边滤波失败: {str(e)}")
            return img

    def _gamma_correction(self, img, gamma):
        """Gamma校正（修改为私有方法）"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def preprocess_image(self, img, gamma):
        """完整的图像预处理流水线"""
        # 处理流程：降噪 → 增强 → 滤波 → Gamma校正
        try:
            # Step 1: 混合降噪
            denoised = self._hybrid_denoise(img)
            
            # Step 2: 对比度增强
            enhanced = self._clahe_enhance(denoised)
            
            # Step 3: 边缘保留滤波
            filtered = self._adaptive_bilateral(enhanced)
            
            # Step 4: Gamma校正
            return self._gamma_correction(filtered, gamma)
        except Exception as e:
            self.get_logger().error(f"预处理失败: {str(e)}")
            return img
    
    def rank_circles(self, circles):
        # 根据坐标排序，分别获得x、y排序序号
        circles_sorted_x = sorted(enumerate(circles), key=lambda c: c[1][0])
        circles_sorted_y = sorted(enumerate(circles), key=lambda c: c[1][1])

        ranks = {}
        for rank, (idx, _) in enumerate(circles_sorted_x):
            ranks[idx] = {'x_rank': rank}
        for rank, (idx, _) in enumerate(circles_sorted_y):
            ranks[idx]['y_rank'] = rank
        return ranks

    def build_cost_matrix(self, left_circles, right_circles, alpha=0.05):
        left_ranks = self.rank_circles(left_circles) # type: ignore
        right_ranks = self.rank_circles(right_circles)

        num_circles = len(left_circles)
        cost_matrix = np.zeros((num_circles, num_circles))

        for i in range(num_circles):
            xL, yL, _ = left_circles[i]
            for j in range(num_circles):
                xR, yR, _ = right_circles[j]

                rank_cost = abs(left_ranks[i]['x_rank'] - right_ranks[j]['x_rank']) \
                            + abs(left_ranks[i]['y_rank'] - right_ranks[j]['y_rank'])

                pixel_cost = abs(xL - xR) + abs(yL - yR)

                cost_matrix[i, j] = rank_cost + alpha * pixel_cost

        return cost_matrix

    # 匹配主函数
    def match_circles(self,left_circles, right_circles, alpha=0.05):
        cost_matrix = self.build_cost_matrix(left_circles, right_circles, alpha)
        left_indices, right_indices = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        for l_idx, r_idx in zip(left_indices, right_indices):
            matched_pairs.append((l_idx, r_idx))
        return matched_pairs 
    
    def image_callback(self, left_msg, right_msg):
        """修改后的图像回调函数"""
        with self.lock:
            try:
                # 原始图像转换
                raw_left = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
                raw_right = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
                
                # 执行完整预处理流程
                self.left_image = self.preprocess_image(raw_left, self.gamma_values['left'])
                self.right_image = self.preprocess_image(raw_right, self.gamma_values['right'])
                
                # 可视化预处理效果
                h, w = raw_left.shape[:2]
                display = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                
                # 第一行显示原始图像
                display[0:h, 0:w] = raw_left
                display[0:h, w:w*2] = raw_right
                
                # 第二行显示处理后的图像
                display[h:h*2, 0:w] = self.left_image
                display[h:h*2, w:w*2] = self.right_image
                
                # 添加文字说明
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display, "Raw Left", (10,30), font, 1, (0,255,0), 2)
                cv2.putText(display, "Raw Right", (w+10,30), font, 1, (0,255,0), 2)
                cv2.putText(display, "Processed Left", (10,h+30), font, 1, (0,255,0), 2)
                cv2.putText(display, "Processed Right", (w+10,h+30), font, 1, (0,255,0), 2)
                
                #cv2.imshow("Preprocessing", display)
                cv2.waitKey(1)

                # 实时检测圆环数量（基于预处理后的图像）
                left_count = len(self.detect_circles(self.left_image, 'left'))
                right_count = len(self.detect_circles(self.right_image, 'right'))
                self.valid_detection = (left_count == right_count)

            except Exception as e:
                self.get_logger().error(f'图像处理错误: {str(e)}')
                self.valid_detection = False

    def goal_callback(self, goal_request):
        """接受所有目标请求"""
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        """处理新目标请求"""
        with self.lock:
            if hasattr(self, 'current_goal') and self.current_goal.is_active:
                self.current_goal.abort()
            self.current_goal = goal_handle
        goal_handle.execute()

    def detect_circles(self, image, camera_side):
        """改进后的圆检测方法"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 获取当前相机的霍夫参数
            params = self.hough_params[camera_side]
            
            # 执行霍夫圆检测
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

    async def execute_callback(self, goal_handle):
        """异步执行目标处理（完整修复版）"""
        try:
            # ==================== 1. 事件循环管理 ====================
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # ==================== 2. 初始化参数 ======================
            requested_count = goal_handle.request.expected_circle_count
            self.get_logger().info(f"开始处理目标，期望圆环数: {requested_count}")

            # ==================== 3. 等待检测条件 ====================
            start_time = self.get_clock().now()
            timeout_sec = 20.0  # 总超时时间（秒）

            while (self.get_clock().now() - start_time).nanoseconds < timeout_sec * 1e9:
                # 检查是否被取消
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("检测到取消请求")
                    goal_handle.canceled()
                    return GetExpectedCirclePosition.Result()

                # 获取最新有效图像
                with self.lock:
                    if not (self.left_image is not None and self.right_image is not None):
                        await asyncio.sleep(0.1)  # 移除loop参数
                        continue
                    
                    try:
                        left_img = self.left_image.copy()
                        right_img = self.right_image.copy()
                    except Exception as e:
                        self.get_logger().error(f"图像拷贝失败: {str(e)}")
                        await asyncio.sleep(0.1)
                        continue

                # ================= 4. 处理单帧 ====================
                try:
                    # 使用await正确等待异步方法
                    result = await self.process_frame(  # 确保有await
                        left_img,
                        right_img,
                        requested_count,
                        goal_handle
                    )

                    if result and len(result.circles) >= requested_count * 1:
                        goal_handle.succeed()
                        return result

                    await asyncio.sleep(0.1)  # 控制处理频率

                except asyncio.CancelledError:
                    self.get_logger().info("处理被取消")
                    raise
                except Exception as e:
                    self.get_logger().error(f"帧处理异常: {str(e)}", throttle_duration_sec=5)
                    await asyncio.sleep(0.1)

            # 超时处理
            self.get_logger().warn(f"处理超时 ({timeout_sec}s)")
            goal_handle.abort()
            return GetExpectedCirclePosition.Result()

        except Exception as e:
            self.get_logger().error(f"目标处理失败: {str(e)}")
            goal_handle.abort()
            return GetExpectedCirclePosition.Result()

    
    async def process_frame(self, left_img, right_img, expected_count, goal_handle):
        """执行帧处理"""
        try:

            # 检测圆环（带严格数量验证）
            left_circles = self.detect_circles(left_img, 'left')
            right_circles = self.detect_circles(right_img, 'right')
            
            # 最终数量验证
            if len(left_circles) != expected_count or len(right_circles) != expected_count:
                self.get_logger().error(f"最终验证失败: 左{len(left_circles)} 右{len(right_circles)}")
                return None
        
            # 排序匹配策略
            left_sorted = left_circles  # 直接使用原始检测到圆环
            right_sorted = right_circles

            matched_pairs = self.match_circles(left_sorted, right_sorted, alpha=0.05)

            circles_3d = []
            for left_idx, right_idx in matched_pairs:
                xL, yL, rL = left_sorted[left_idx]
                xR, yR, rR = right_sorted[right_idx]

                disparity = xL - xR
                if disparity <= 0:
                    disparity = 0.1  # 或其他合理默认值，防止跳过
                    self.get_logger().warn(f"视差负值 ID:{left_idx}-{right_idx} 调整为 {disparity}")

                Z = (self.camera_params['fx'] * self.camera_params['baseline']) / disparity
                X = (xL - self.camera_params['cx']) * Z / self.camera_params['fx']
                Y = (yL - self.camera_params['cy']) * Z / self.camera_params['fy']

                circles_3d.append(CirclePosition(
                    id=int(left_idx),
                    x=float(X), y=float(Y), z=float(Z)
                ))

            
            # 构建最终结果
            if len(circles_3d) < expected_count * 1:
                self.get_logger().warn(f"有效坐标不足: {len(circles_3d)}/{expected_count}")
                return None
                
            return GetExpectedCirclePosition.Result(circles=circles_3d)
        
        except Exception as e:
            self.get_logger().error(f"帧处理失败: {str(e)}")
            return None


def main(args=None):
    rclpy.init(args=args)
    detector = StereoDepthCircleDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.destroy_node()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

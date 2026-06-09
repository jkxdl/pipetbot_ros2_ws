import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf_transformations
from tf2_ros import StaticTransformBroadcaster
from std_srvs.srv import Trigger

class HandEyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration_node')

        # 参数设置
        self.declare_parameter('chessboard_rows', 6)  # 棋盘格内角点行数
        self.declare_parameter('chessboard_cols', 9)  # 棋盘格内角点列数
        self.declare_parameter('square_size', 0.025)  # 每个格子的边长（单位：米）
        self.declare_parameter('calibration_samples', 12)
        self.declare_parameter('camera_fx', 1502.192807)
        self.declare_parameter('camera_fy', 1484.796154)
        self.declare_parameter('camera_cx', 965.942334)
        self.declare_parameter('camera_cy', 503.32030)
        self.declare_parameter('camera_dist_coeffs', [-0.042219, 0.032766, -0.002505, 0.001575, 0.0])  # 畸变
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'tool0')  # 末端执行器坐标系
        self.declare_parameter('camera_frame', 'camera_in_link')   # 相机坐标系

        # 获取参数
        self.chessboard_rows = self.get_parameter('chessboard_rows').get_parameter_value().integer_value
        self.chessboard_cols = self.get_parameter('chessboard_cols').get_parameter_value().integer_value
        self.square_size = self.get_parameter('square_size').get_parameter_value().double_value
        self.num_samples = self.get_parameter('calibration_samples').get_parameter_value().integer_value
        self.camera_fx = self.get_parameter('camera_fx').get_parameter_value().double_value
        self.camera_fy = self.get_parameter('camera_fy').get_parameter_value().double_value
        self.camera_cx = self.get_parameter('camera_cx').get_parameter_value().double_value
        self.camera_cy = self.get_parameter('camera_cy').get_parameter_value().double_value
        self.camera_dist_coeffs = self.get_parameter('camera_dist_coeffs').get_parameter_value().double_array_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.end_effector_frame = self.get_parameter('end_effector_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 初始化订阅者：图像话题
        self.image_sub = self.create_subscription(
            Image,
            '/stereo/left/image_raw',
            self.image_callback,
            10
        )

        # 初始化 CvBridge
        self.bridge = CvBridge()

        # 初始化 tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 初始化静态变换发布器（用于发布标定完成后的静态变换）
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # 存储样本的姿态列表
        self.camera_poses = []
        self.robot_poses = []
        self.sample_count = 0

        # 用于存储当前相机中检测到的标记姿态（相机到标记）
        self.current_camera_pose = None

        # 创建一个服务，用于手动触发捕获当前样本
        self.capture_service = self.create_service(Trigger, 'capture_pose', self.handle_capture_pose)

        # 初始化 OpenCV 窗口
        cv2.namedWindow("Detected Chessboard", cv2.WINDOW_NORMAL)

        self.get_logger().info("Hand Eye Calibration Node Initialized. Move the robot, then call 'ros2 service call /capture_pose std_srvs/srv/Trigger {}' to capture a sample.")

    def image_callback(self, msg):
        # 图像回调仅用于更新 current_camera_pose，不自动捕获

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge 错误: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # 检测棋盘格角点
        pattern_size = (self.chessboard_cols, self.chessboard_rows)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        self.current_camera_pose = None  # 默认无标记

        if ret:
            # 提高角点的检测精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 计算棋盘格的 3D 世界坐标
            objp = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.chessboard_cols, 0:self.chessboard_rows].T.reshape(-1, 2)
            objp *= self.square_size  # 转换为真实尺寸

            # 使用 solvePnP 计算相机姿态
            ret, rvec, tvec = cv2.solvePnP(
                objp,
                corners_refined,
                self.get_camera_matrix(),
                self.get_dist_coeffs()
            )

            if ret:
                R_cam_marker, _ = cv2.Rodrigues(rvec)
                t_cam_marker = tvec.reshape(3, 1)

                # 构建 4x4 转换矩阵
                T_cam_marker = np.vstack((
                    np.hstack((R_cam_marker, t_cam_marker)),
                    np.array([0, 0, 0, 1])
                ))

                self.current_camera_pose = T_cam_marker

                # 可视化棋盘格
                cv2.drawChessboardCorners(cv_image, pattern_size, corners_refined, ret)

        cv2.imshow("Detected Chessboard", cv_image)
        cv2.waitKey(1)

    def handle_capture_pose(self, request, response):
        # 服务回调：当用户调用 /capture_pose 时尝试捕获当前相机和机器人姿态
        if self.current_camera_pose is None:
            self.get_logger().warn("未检测到标记，请确保标记在相机视野内并清晰可见。")
            response.success = False
            response.message = "No marker detected."
            return response

        # 获取末端执行器位姿
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.end_effector_frame,
                rclpy.time.Time()
            )

            T_robot_ee = self.transform_to_matrix(trans)

            # 末端执行器相对于基座的变换
            #T_base_to_gripper = T_robot_ee

            # 存储样本
            self.camera_poses.append(self.current_camera_pose)
            self.robot_poses.append(T_robot_ee)
            self.sample_count += 1

            self.get_logger().info(f"捕获第 {self.sample_count}/{self.num_samples} 个样本。")
            


            # 如果样本足够，执行标定
            if self.sample_count >= self.num_samples:
                self.perform_calibration()

            response.success = True
            response.message = f"Sample {self.sample_count} captured."
            return response

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"无法获取变换: {e}")
            response.success = False
            response.message = f"Failed to capture robot pose: {e}"
            return response

    def transform_to_matrix(self, transform: TransformStamped):
        # 将 ROS 的 TransformStamped 转换为 4x4 矩阵
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        q = [rotation.x, rotation.y, rotation.z, rotation.w]
        rot_matrix = tf_transformations.quaternion_matrix(q)
        rot_matrix[0:3, 3] = [translation.x, translation.y, translation.z]
        return rot_matrix

    def get_camera_matrix(self):
        # 使用提供的相机内参
        return np.array([
            [self.camera_fx, 0, self.camera_cx],
            [0, self.camera_fy, self.camera_cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_dist_coeffs(self):
        # 使用提供的相机畸变系数
        return np.array(self.camera_dist_coeffs, dtype=np.float64)

    def perform_calibration(self):
        self.get_logger().info("开始手眼标定...")

        if len(self.camera_poses) < self.num_samples or len(self.robot_poses) < self.num_samples:
            self.get_logger().error("标定样本不足")
            return

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for i in range(self.num_samples):
            T_gripper_to_base = self.robot_poses[i]
            T_cam = self.camera_poses[i]

            R_gripper2base.append(T_gripper_to_base[0:3, 0:3].astype(np.float64))
            t_gripper2base.append(T_gripper_to_base[0:3, 3].astype(np.float64))
            R_target2cam.append(T_cam[0:3, 0:3].astype(np.float64))
            t_target2cam.append(T_cam[0:3, 3].astype(np.float64))

        try:
            # 使用 OpenCV 的 calibrateHandEye 函数进行手眼标定
            R_eye_hand, t_eye_hand = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            #self.get_logger().info(f"机器人平移输入:\n{t_gripper2base}")
            #self.get_logger().info(f"相机平移输入:\n{t_target2cam}")

           

            self.get_logger().info("手眼标定成功！")
            self.get_logger().info(f"旋转矩阵:\n{R_eye_hand}")
            self.get_logger().info(f"平移向量:\n{t_eye_hand}")

            # 构建 4x4 转换矩阵
            T_eye_to_hand = np.vstack((
                np.hstack((R_eye_hand, t_eye_hand.reshape(3, 1))),
                np.array([0, 0, 0, 1])
            ))
            

            # 将标定结果转换为 TransformStamped 消息
            transform_msg = TransformStamped()
            transform_msg.header.stamp = self.get_clock().now().to_msg()
            transform_msg.header.frame_id = self.end_effector_frame
            transform_msg.child_frame_id = self.camera_frame

            q = tf_transformations.quaternion_from_matrix(T_eye_to_hand)
            transform_msg.transform.translation.x = T_eye_to_hand[0, 3]
            transform_msg.transform.translation.y = T_eye_to_hand[1, 3]
            transform_msg.transform.translation.z = T_eye_to_hand[2, 3]

            transform_msg.transform.rotation.x = q[0]
            transform_msg.transform.rotation.y = q[1]
            transform_msg.transform.rotation.z = q[2]
            transform_msg.transform.rotation.w = q[3]

            # 发布静态变换
            self.static_tf_broadcaster.sendTransform(transform_msg)
            self.get_logger().info(f"手眼标定静态变换已发布：'{self.camera_frame}' 相对于 '{self.end_effector_frame}'")

        except Exception as e:
            self.get_logger().error(f"标定失败: {e}")

        # 重置样本
        self.camera_poses = []
        self.robot_poses = []
        self.sample_count = 0

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

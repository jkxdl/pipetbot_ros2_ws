import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import glob
import subprocess
from rclpy.executors import MultiThreadedExecutor


class StereoImagePublisher(Node):
    def __init__(self):
        super().__init__('stereo_image_publisher')
        self.capture = None
        self.timer = None
        self.initialized = False

        # 设置目标设备的 ID_PATH
        self.declare_parameter('target_id_path', 'pci-0000:00:14.0-usb-0:5.2:1.0')
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 360)
        self.declare_parameter('fourcc', 'MJPG')
        self.declare_parameter('reopen_after_failures', 3)
        self.target_id_path = self.get_parameter('target_id_path').value
        self.target_fps = float(self.get_parameter('fps').value)
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.fourcc = str(self.get_parameter('fourcc').value)
        self.reopen_after_failures = int(
            self.get_parameter('reopen_after_failures').value
        )
        self.read_failure_count = 0
 
        # 初始化发布器
        self.left_image_publisher = self.create_publisher(Image, '/stereo/left/image_raw', 10)
        self.right_image_publisher = self.create_publisher(Image, '/stereo/right/image_raw', 10)

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        # 查找目标设备的索引
        self.device_path = self.find_device_by_id_path(self.target_id_path)
        if self.device_path is None:
            self.get_logger().error(f"未找到 ID_PATH 为 {self.target_id_path} 的设备")
            return

        self.get_logger().info(f"找到目标设备，路径为 {self.device_path}")

        if not self.open_capture():
            return

        # 创建定时器发布图像
        self.timer = self.create_timer(0.1, self.publish_stereo_images)
        self.initialized = True

    def find_device_by_id_path(self, target_id_path):
        """
        通过 ID_PATH 查找摄像头设备索引
        """
        device_paths = sorted(glob.glob('/dev/video*'))
        if not device_paths:
            self.get_logger().error('当前系统中没有可用的 /dev/video* 设备')
            return None

        for device_path in device_paths:
            try:
                # 使用 udevadm 获取设备详细信息
                cmd = f"udevadm info --query=all --name={device_path}"
                output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
                if f"ID_PATH={target_id_path}" in output:
                    return device_path
            except subprocess.CalledProcessError:
                continue
        return None

    def open_capture(self):
        self.capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.get_logger().error(f"无法打开设备 {self.device_path}")
            self.capture.release()
            self.capture = None
            return False

        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.enable_auto_exposure_and_white_balance()
        self.log_capture_configuration()
        self.read_failure_count = 0
        return True

    def reopen_capture(self):
        self.get_logger().warn(f'尝试重新打开设备 {self.device_path}')
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.open_capture()

    def log_capture_configuration(self):
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
        self.get_logger().info(
            f"摄像头配置: device={self.device_path}, "
            f"requested={self.frame_width}x{self.frame_height}@{self.target_fps} {self.fourcc}, "
            f"actual={actual_width}x{actual_height}@{actual_fps:.2f} {codec or 'unknown'}"
        )

    def enable_auto_exposure_and_white_balance(self):
        """
        启用自动曝光和自动白平衡
        """
        # 启用自动曝光
        # OpenCV中自动曝光的设置可能因操作系统和驱动不同而有所差异
        # 对于大多数USB摄像头，设置 CAP_PROP_AUTO_EXPOSURE 为 1 开启自动曝光
        # 注意：有些摄像头可能需要设置为具体的值，如 0.75
        auto_exposure_set = self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        if not auto_exposure_set:
            self.get_logger().warn("无法设置自动曝光")
        else:
            self.get_logger().info("已启用自动曝光")

        # 启用自动白平衡
        # OpenCV中自动白平衡的设置也可能因摄像头而异
        # 对于大多数摄像头，设置 CAP_PROP_AUTO_WB 为 1 开启自动白平衡
        auto_wb_set = self.capture.set(cv2.CAP_PROP_AUTO_WB, 1)
        if not auto_wb_set:
            self.get_logger().warn("无法设置自动白平衡")
        else:
            self.get_logger().info("已启用自动白平衡")

        

    def publish_stereo_images(self):
        """
        从摄像头读取复合流数据并分割为左右图像后发布
        """
        if self.capture is None:
            return
        ret, frame = self.capture.read()
        if not ret:
            self.read_failure_count += 1
            self.get_logger().error(
                f"无法从摄像头读取数据: {self.device_path} "
                f"({self.frame_width}x{self.frame_height}@{self.target_fps} {self.fourcc})"
            )
            if self.read_failure_count >= self.reopen_after_failures:
                self.reopen_capture()
            return
        self.read_failure_count = 0

        # 分割复合流为左右图像
        height, width, _ = frame.shape
        mid = width // 2  # 中间位置，减少多次计算
        left_image, right_image = frame[:, :mid], frame[:, mid:]  # 同时完成切片

        # 转换为 ROS 图像消息
        now = self.get_clock().now().to_msg()  # 提前生成时间戳，避免重复调用
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='bgr8')
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='bgr8')

        # 添加时间戳并发布
        left_msg.header.stamp = now
        right_msg.header.stamp = now
        self.left_image_publisher.publish(left_msg)
        self.right_image_publisher.publish(right_msg)

    def destroy_node(self):
        """
        节点销毁时释放摄像头
        """
        if getattr(self, 'capture', None) is not None:
            self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoImagePublisher()
    if not node.initialized:
        node.destroy_node()
        rclpy.shutdown()
        return
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("节点已中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

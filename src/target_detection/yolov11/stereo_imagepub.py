import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class StereoCameraNode(Node):
    def __init__(self):
        super().__init__('stereo_camera_node')

        # 获取设备标识符并打开设备
        video_device_path = self.find_camera_by_id("STYT_220117_K_USB_Camera_01.00.00")
        if not video_device_path:
            self.get_logger().error("Camera device not found.")
            return
        
        # 使用 OpenCV 打开复合流设备
        self.cap = cv2.VideoCapture(video_device_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera at {video_device_path}.")
            return

        # 设置图像分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 创建复合流图像的发布话题
        self.image_pub = self.create_publisher(Image, '/stereo/image_raw', 10)

        # 初始化 CvBridge
        self.bridge = CvBridge()

        # 定时器，每 33 毫秒捕获一次图像
        self.timer = self.create_timer(0.033, self.publish_image)

        self.get_logger().info("Stereo Camera Node has been started")

    def find_camera_by_id(self, camera_id):
        """根据设备标识符查找设备路径"""
        base_path = '/dev/v4l/by-id'
        for entry in os.listdir(base_path):
            if camera_id in entry:
                device_path = os.path.join(base_path, entry)
                return '/dev/' + os.readlink(device_path).split('/')[-1]  # 返回设备路径
        return None

    def publish_image(self):
        """捕获并发布复合流图像"""
        ret, frame = self.cap.read()
        if ret:
            # 将图像转换为 ROS 消息
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

            # 发布图像
            self.image_pub.publish(msg)

            #self.get_logger().info("Published composite image")
        else:
            self.get_logger().warn("Failed to capture image")

    def destroy(self):
        """清理资源"""
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


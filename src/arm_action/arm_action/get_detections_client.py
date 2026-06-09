# get_detections_client.py
import rclpy
from rclpy.node import Node
from pipettingrobot_interfaces.srv import GetDetections  
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from functools import partial
import threading
import time

class GetDetectionsClient:
    def __init__(self, node: Node):
        self.node = node
        self.client = self.node.create_client(GetDetections, '/get_detections')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.camera_frame = "camera_link"
        self.target_frame = "base_link"
        self.node.get_logger().info("GetDetectionsClient 已初始化。")
        self.retry_interval = 1.0  # 初始重试间隔（秒）
        self.lock = threading.Lock()

    def send_request(self, target_name, callback):
        def attempt_request():
            with self.lock:
                if not self.client.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().error('服务 /get_detections 不可用，正在重试...')
                    self.schedule_retry(attempt_request)
                    return

                request = GetDetections.Request()
                request.target_name = target_name

                future = self.client.call_async(request)
                future.add_done_callback(partial(self.response_callback, callback, attempt_request))

        attempt_request()

    def response_callback(self, callback, retry_func, future):
        try:
            response = future.result()
            if not response.success:
                self.node.get_logger().error(f"检测失败: {response.message}，正在重试...")
                self.schedule_retry(retry_func)
                return

            # 转换坐标
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = self.node.get_clock().now().to_msg()
            camera_point.point = response.coordinates

            base_point = self.transform_to_base_coords(camera_point)
            if base_point:
                self.node.get_logger().info(
                    f"检测成功 - 相机坐标: ({camera_point.point.x}, {camera_point.point.y}, {camera_point.point.z}) "
                    f"-> 基座坐标: ({base_point.point.x}, {base_point.point.y}, {base_point.point.z})"
                )
                callback(True, base_point.point, "检测成功")
            else:
                self.node.get_logger().error("坐标转换失败，正在重试...")
                self.schedule_retry(retry_func)
        except Exception as e:
            self.node.get_logger().error(f"服务调用失败: {e}，正在重试...")
            self.schedule_retry(retry_func)

    def transform_to_base_coords(self, camera_point):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_point.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            transformed_point = do_transform_point(camera_point, transform)
            return transformed_point
        except Exception as e:
            self.node.get_logger().error(f"坐标转换失败: {e}")
            return None

    def schedule_retry(self, retry_func):
        self.retry_interval = min(self.retry_interval , 60.0)  
        self.node.get_logger().info(
            f'将在 {self.retry_interval} 秒后重试获取检测请求...'
        )
        timer = self.node.create_timer(
            self.retry_interval,
            lambda: (retry_func(), timer.cancel())
        )

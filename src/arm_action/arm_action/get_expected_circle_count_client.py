# get_expected_circle_count_client.py
import rclpy
from rclpy.node import Node
from pipettingrobot_interfaces.srv import GetExpectedCircleCount  
from functools import partial
import threading
import time
#ros2 action send_goal /set_target_pose pipettingrobot_interfaces/action/SetTargetPose "{pose: {position: {x: 0.5, y: 0.0, z: 0.6}, orientation: {x: 0.5, y: 0.5, z: 0.5, w: 0.5}}}"

class GetExpectedCircleCountClient:
    def __init__(self, node: Node):
        self.node = node
        self.client = self.node.create_client(GetExpectedCircleCount, '/get_expected_circle_count')
        self.node.get_logger().info("GetExpectedCircleCountClient 已初始化。")
        self.retry_interval = 1.0  # 初始重试间隔（秒）
        self.lock = threading.Lock()

    def send_request(self, callback):
        def attempt_request():
            with self.lock:
                if not self.client.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().error('服务 /get_expected_circle_count 不可用，正在重试...')
                    self.schedule_retry(attempt_request)
                    return

                request = GetExpectedCircleCount.Request()

                future = self.client.call_async(request)
                future.add_done_callback(partial(self.response_callback, callback, attempt_request))

        attempt_request()

    def response_callback(self, callback, retry_func, future):
        try:
            response = future.result()
            if not response.success:
                self.node.get_logger().error(f"获取预期圆环数量失败: {response.message}，正在重试...")
                self.schedule_retry(retry_func)
                return

            self.node.get_logger().info(f"获取预期圆环数量成功: {response.expected_circle_count} - {response.message}")
            callback(True, response.expected_circle_count, "获取成功")
        except Exception as e:
            self.node.get_logger().error(f"服务调用失败: {e}，正在重试...")
            self.schedule_retry(retry_func)

    def schedule_retry(self, retry_func):
        self.retry_interval = min(self.retry_interval * 2, 60.0)  # 指数退避，最多60秒
        self.node.get_logger().info(
            f'将在 {self.retry_interval} 秒后重试获取预期圆环数量请求...'
        )
        timer = self.node.create_timer(
            self.retry_interval,
            lambda: (retry_func(), timer.cancel())
        )

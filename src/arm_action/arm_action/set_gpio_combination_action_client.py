# set_gpio_combination_action_client.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from pipettingrobot_interfaces.action import SetGPIOCombination  
from functools import partial
import threading
import time

class SetGPIOCombinationActionClient:
    def __init__(self, node: Node):
        self.node = node
        self.action_client = ActionClient(self.node, SetGPIOCombination, '/set_gpio_combination')
        self.node.get_logger().info("SetGPIOCombinationActionClient 已初始化。")
        self.retry_interval = 1.0  # 初始重试间隔（秒）
        self.lock = threading.Lock()

    def send_goal(self, combination_id: int, callback):
        def attempt_goal():
            with self.lock:
                if not self.action_client.wait_for_server(timeout_sec=1.0):
                    self.node.get_logger().error('动作服务器 /set_gpio_combination 不可用，正在重试...')
                    self.schedule_retry(attempt_goal)
                    return

                goal_msg = SetGPIOCombination.Goal()
                goal_msg.combination_id = combination_id

                send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
                send_goal_future.add_done_callback(partial(self.goal_response_callback, callback, attempt_goal))

        attempt_goal()

    def goal_response_callback(self, callback, retry_func, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.node.get_logger().error('GPIO 组合动作请求被拒绝，正在重试...')
                self.schedule_retry(retry_func)
                return

            self.node.get_logger().info('GPIO 组合动作请求已接受，等待结果...')
            goal_handle.get_result_async().add_done_callback(partial(self.result_callback, callback, retry_func))
        except Exception as e:
            self.node.get_logger().error(f'发送 GPIO 组合动作请求失败: {e}，正在重试...')
            self.schedule_retry(retry_func)

    def result_callback(self, callback, retry_func, future):
        try:
            result = future.result().result
            if result.success:
                self.node.get_logger().info(f'GPIO 组合动作成功: {result.message}')
                callback(True, result.message)
            else:
                self.node.get_logger().error(f'GPIO 组合动作失败: {result.message}，正在重试...')
                self.schedule_retry(retry_func)
        except Exception as e:
            self.node.get_logger().error(f'获取 GPIO 组合动作结果失败: {e}，正在重试...')
            self.schedule_retry(retry_func)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback.feedback_message
        self.node.get_logger().info(f"动作反馈: {feedback}")

    def schedule_retry(self, retry_func):
        self.retry_interval = min(self.retry_interval , 60.0)  # 指数退避，最多60秒
        self.node.get_logger().info(
            f'将在 {self.retry_interval} 秒后重试发送 GPIO 组合动作请求...'
        )
        timer = self.node.create_timer(
            self.retry_interval,
            lambda: (retry_func(), timer.cancel())
        )

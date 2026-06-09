# get_circle_coords_client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from pipettingrobot_interfaces.action import GetExpectedCirclePosition
from geometry_msgs.msg import PointStamped, Point
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from functools import partial
import threading
import time

class GetCircleCoordsClient:
    def __init__(self, node: Node):
        self.node = node
        
        # 初始化动作客户端
        self.action_client = ActionClient(
            self.node,
            GetExpectedCirclePosition,
            '/get_expected_circlesposition'
        )
        
        # TF相关初始化
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.camera_frame = "camera_in_link"
        self.target_frame = "base_link"
        
        # 重试机制参数
        self.retry_interval = 1.0
        self.max_retries = 20
        self.lock = threading.Lock()
        
        self.node.get_logger().info("动作客户端已初始化")

    def send_goal(self, expected_count, callback):
        """发送动作目标"""
        def attempt_request(retry_count=0):
            with self.lock:
                if not self.action_client.wait_for_server(timeout_sec=1.0):
                    self.node.get_logger().error("动作服务器不可用，正在重试...")
                    self._schedule_retry(attempt_request, retry_count)
                    return

                # 创建目标请求
                goal_msg = GetExpectedCirclePosition.Goal()
                goal_msg.expected_circle_count = expected_count
                
                # 发送目标
                send_goal_future = self.action_client.send_goal_async(
                    goal_msg,
                    feedback_callback=partial(self.feedback_callback, callback))
                
                send_goal_future.add_done_callback(
                    partial(self.goal_response_callback, callback, attempt_request, retry_count))

        attempt_request()

    def goal_response_callback(self, callback, retry_func, retry_count, future):
        """处理目标响应"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("目标被拒绝")
            self._schedule_retry(retry_func, retry_count)
            return
        
        # 获取结果
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(
            partial(self.result_callback, callback, goal_handle, retry_func, retry_count))

    def result_callback(self, callback, goal_handle, retry_func, retry_count, future):
        """处理最终结果"""
        result = future.result().result
        
        if not result.circles:
            self.node.get_logger().error("未获取到有效坐标")
            self._schedule_retry(retry_func, retry_count)
            return
        
        # 执行坐标转换
        transformed_points = []
        for circle in result.circles:
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = self.node.get_clock().now().to_msg()
            camera_point.point = Point(
                x=circle.x,
                y=circle.y,
                z=circle.z
            )
            
            base_point = self.transform_to_base_coords(camera_point)
            if base_point:
                self.node.get_logger().info(
                    f"圆环 ID {circle.id} - 基座坐标: "
                    f"({base_point.point.x:.3f}, {base_point.point.y:.3f}, {base_point.point.z:.3f})"
                )
                transformed_points.append(base_point.point)
            else:
                self.node.get_logger().error(f"圆环 ID {circle.id} 坐标转换失败")
                self._schedule_retry(retry_func, retry_count)
                return
        
        callback(True, transformed_points, "坐标获取成功")

    def feedback_callback(self, callback, feedback):
        """处理反馈信息"""
        self.node.get_logger().info(f"处理进度: {feedback.feedback.status}")

    def transform_to_base_coords(self, camera_point):
        """坐标转换（与原实现相同）"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_point.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            return do_transform_point(camera_point, transform)
        except Exception as e:
            self.node.get_logger().error(f"坐标转换失败: {e}")
            return None

    def _schedule_retry(self, retry_func, retry_count):
        """重试机制"""
        if retry_count >= self.max_retries:
            self.node.get_logger().error("超过最大重试次数")
            return
            
        self.retry_interval = min(self.retry_interval, 60.0)
        self.node.get_logger().info(f"{self.retry_interval}秒后重试...")
        
        timer = self.node.create_timer(
            self.retry_interval,
            lambda: (retry_func(retry_count + 1), timer.cancel())
            )
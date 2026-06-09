import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from aubo_msgs.srv import SetIO
from pipettingrobot_interfaces.action import SetGPIOCombination  # 替换为你的包名

class PredefinedGPIOController(Node):
    def __init__(self):
        super().__init__('predefined_gpio_controller')
        self.cli = self.create_client(SetIO, '/io_and_status_controller/set_io')

        # 等待服务可用
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /io_and_status_controller/set_io 服务...')

        # 设置工具电压为12V
        self.set_tool_voltage(12)

        # 预定义的GPIO口状态组合
        self.gpio_combinations = {
            1: [(8, 0), (9, 1), (10, 1), (11, 1)],  # 按钮初始位置
            2: [(8, 0), (9, 1), (10, 0), (11, 1)],  # 按钮接触位置
            3: [(8, 0), (9, 1), (10, 1), (11, 0)],  # 按钮第一按动位置
            4: [(8, 0), (9, 1), (10, 0), (11, 0)],  # 按钮第二按动位置
            5: [(8, 1), (9, 0), (10, 1), (11, 1)],  # 180度
            6: [(8, 0), (9, 0), (10, 0), (11, 1)],  # 正向调节
            7: [(8, 0), (9, 0), (10, 1), (11, 0)],  # 反向调节
            8: [(8, 1), (9, 1), (10, 0), (11, 1)],  # 向上
            9: [(8, 1), (9, 0), (10, 1), (11, 0)],  # 0度
            10: [(8, 1), (9, 1), (10, 1), (11, 0)],  # 向下
            11: [(8, 1), (9, 1), (10, 1), (11, 1)],  # 初始
        }

        # 创建动作服务端
        self.action_server = ActionServer(
            self,
            SetGPIOCombination,
            'set_gpio_combination',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info('预定义GPIO控制器动作服务已启动，等待请求...')

    def goal_callback(self, goal_request):
        """处理接收到的目标请求"""
        self.get_logger().info(f'Received goal request with combination_id: {goal_request.combination_id}')
        if 1 <= goal_request.combination_id <= 11:
            self.get_logger().info('接受目标请求')
            return GoalResponse.ACCEPT
        else:
            self.get_logger().warning('拒绝目标请求：无效的 combination_id')
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """处理取消请求"""
        self.get_logger().info('收到取消请求')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """执行目标的回调函数"""
        combination_id = goal_handle.request.combination_id
        self.get_logger().info(f'开始执行组合 {combination_id} 设置...')

        feedback_msg = f'Starting to set combination {combination_id}'
        goal_handle.publish_feedback(SetGPIOCombination.Feedback(feedback_message=feedback_msg))

        success, message = self.execute_combination(combination_id, goal_handle)

        if success:
            result = SetGPIOCombination.Result()
            result.success = True
            result.message = message
            self.get_logger().info(f'组合 {combination_id} 设置成功')
            goal_handle.succeed()
            return result
        else:
            result = SetGPIOCombination.Result()
            result.success = False
            result.message = message
            self.get_logger().error(f'组合 {combination_id} 设置失败: {message}')
            goal_handle.abort()
            return result

    def set_tool_voltage(self, voltage):
        """设置工具电压"""
        self.get_logger().info(f'设置工具电压为 {voltage}V...')
        request = SetIO.Request()
        request.fun = 4  # 设置工具电压
        request.pin = 0  # 无需指定PIN
        request.state = float(voltage)

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('工具电压设置成功')
            else:
                # 假设 SetIO.Response 包含一个 message 字段用于描述错误
                error_message = getattr(future.result(), 'message', '未知错误')
                self.get_logger().error(f'工具电压设置失败: {error_message}')
        else:
            self.get_logger().error('工具电压设置失败: 未收到服务响应')


    def set_gpio_state(self, pin, state):
        """设置单个GPIO口的状态"""
        request = SetIO.Request()
        request.fun = 1  # 数字输出模式
        request.pin = pin
        request.state = float(state)

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f'GPIO {pin} 状态已设置为 {state}')
            else:
                self.get_logger().error(f'无法设置 GPIO {pin} 到状态 {state}')
        else:
            self.get_logger().error(f'未收到 GPIO {pin} 的服务响应')

    def execute_combination(self, combination_id, goal_handle):
        """根据组合ID执行对应的GPIO口状态设置"""
        if combination_id in self.gpio_combinations:
            self.get_logger().info(f'执行组合 {combination_id} 设置...')
            for idx, (pin, state) in enumerate(self.gpio_combinations[combination_id], start=1):
                if goal_handle.is_cancel_requested:
                    self.get_logger().warning('动作被取消，中止执行')
                    goal_handle.canceled()
                    return False, '动作被取消'
                
                self.set_gpio_state(pin, state)
                feedback_msg = f'Set GPIO {pin} to {state} (Step {idx}/{len(self.gpio_combinations[combination_id])})'
                # 发布反馈
                goal_handle.publish_feedback(SetGPIOCombination.Feedback(feedback_message=feedback_msg))
                self.get_logger().info(feedback_msg)
            return True, f'组合 {combination_id} 设置成功'
        else:
            self.get_logger().warning(f'无效的组合ID: {combination_id}')
            return False, f'无效的组合ID: {combination_id}'

def main(args=None):
    rclpy.init(args=args)
    gpio_controller = PredefinedGPIOController()
    try:
        rclpy.spin(gpio_controller)
    except KeyboardInterrupt:
        gpio_controller.get_logger().info('节点已停止')
    finally:
        gpio_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

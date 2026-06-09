import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from threading import Lock

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    RobotState,
    PositionConstraint,
    OrientationConstraint,
    Constraints,
    JointConstraint,
    DisplayTrajectory
)
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState

# === 自定义服务 ===
from pipettingrobot_interfaces.srv import SetTargetPose, SetJointAngles

class MoveArmServer(Node):
    def __init__(self):
        super().__init__('move_arm_server')

        # 规划组的关节名称顺序 (请根据实际机器人修改)
        self.planning_group_joint_names = [
            'shoulder_joint',
            'upperArm_joint',
            'foreArm_joint',
            'wrist1_joint',
            'wrist2_joint',
            'wrist3_joint'
        ]

        self.current_joint_state = {}
        self.lock = Lock()
        self.declare_parameter('use_path_constraints', False)

        # 订阅 /joint_states 话题，获取当前关节信息
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.get_logger().info("已订阅 /joint_states，等待关节状态更新...")

        # 动作客户端：用于规划 (move_action)
        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')
        # 动作客户端：用于执行轨迹 (execute_trajectory)
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        # 等待服务和动作服务可用
        while not self._move_group_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('等待 /move_action 动作服务器...')
        while not self._execute_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('等待 /execute_trajectory 动作服务器...')

        self.get_logger().info('服务和动作服务器已准备就绪。')

        # 创建并启动自定义服务：set_target_pose
        self._pose_service = self.create_service(SetTargetPose, 'set_target_pose', self.set_target_pose_callback)
        self.get_logger().info("服务 [set_target_pose] 已创建，等待客户端请求...")

        # 创建并启动新的自定义服务：set_joint_angles
        self._joint_angles_service = self.create_service(SetJointAngles, 'set_joint_angles', self.set_joint_angles_callback)
        self.get_logger().info("服务 [set_joint_angles] 已创建，等待客户端请求...")

        # 创建 DisplayTrajectory 发布者
        self.display_publisher = self.create_publisher(DisplayTrajectory, 'display_planned_path', 10)

        # 保存规划好的轨迹
        self.planned_trajectory = None

    def joint_states_callback(self, msg):
        """
        处理接收到的关节状态消息，并根据规划组重新排序。
        """
        # 将接收到的关节状态转为字典
        joint_positions = dict(zip(msg.name, msg.position))
        missing_joints = [
            j for j in self.planning_group_joint_names if j not in joint_positions
        ]
        if missing_joints:
            # 有可能还没全部发布齐或非相关关节
            return

        # 按规划组顺序排列关节状态
        ordered_joint_positions = [
            joint_positions[j] for j in self.planning_group_joint_names
        ]
        with self.lock:
            self.current_joint_state = dict(zip(self.planning_group_joint_names, ordered_joint_positions))
        #self.get_logger().debug(f"更新关节状态: {self.current_joint_state}")

    def set_target_pose_callback(self, request, response):
        """
        自定义服务回调函数：客户端发送目标位姿 (geometry_msgs/Pose)，
        本函数规划并执行机械臂到达该位姿后，返回 success 及相关信息。
        """
        self.get_logger().info("收到 set_target_pose 服务请求，开始处理...")

        # 读取目标位姿
        target_pose = request.pose

        # 将 geometry_msgs/Pose 转换为 PoseStamped（注意 frame_id）
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"  # 或者您的机器人基座坐标系
        pose_stamped.pose = target_pose

        # 执行规划（plan_only=True）
        self.get_logger().info("正在规划路径...")
        self.plan_to_target(pose_stamped, plan_only=True)
        
        # 为异步处理添加回调，直接在规划回调中设置服务响应
        # 因此不需要阻塞式等待，响应将在执行完成后设置
        # 需要在规划成功后的回调中设置服务响应
        # 这里简化为假设执行完成后立即设置响应
        # 你需要根据实际执行结果在对应的回调中设置响应
        # 比如在 handle_execution_done 中设置 response.success 和 response.message
        # 由于回调是异步的，无法在这里立即返回响应
        # 所以可以考虑使用其他机制，如存储响应并在执行完成后填充
        # 但这涉及到更复杂的同步机制
        # 简化处理，暂时只返回规划是否成功

        # 修改：仅返回规划结果
        plan_success, plan_message = self.plan_to_target(pose_stamped, plan_only=True)

        if not plan_success:
            response.success = False
            response.message = f"规划失败: {plan_message}"
            return response

        # 规划成功，显示规划轨迹
        #self.get_logger().info(f"规划结果: {plan_message}")

        # 自动执行规划轨迹
        self.get_logger().info("自动执行规划轨迹...")
        exec_success, exec_message = self.plan_to_target(pose_stamped, plan_only=False)
        response.success = exec_success
        response.message = exec_message

        return response  # 响应将在执行完成后返回

    def set_joint_angles_callback(self, request, response):
        """
        自定义服务回调函数：客户端发送目标关节角度 (float64[] joint_angles)，
        本函数规划并执行机械臂到达该关节角度后，返回 success 及相关信息。
        """
        self.get_logger().info("收到 set_joint_angles 服务请求，开始处理...")

        # 读取目标关节角度
        target_joint_angles = request.joint_angles

        if len(target_joint_angles) != len(self.planning_group_joint_names):
            response.success = False
            response.message = f"关节角度数量与规划组关节数量不匹配。规划组有 {len(self.planning_group_joint_names)} 个关节，但收到 {len(target_joint_angles)} 个角度。"
            self.get_logger().error(response.message)
            return response

        # 创建一个 JointState 消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.planning_group_joint_names
        joint_state_msg.position = target_joint_angles

        # 执行规划（plan_only=True）
        self.get_logger().info("正在规划路径到指定关节角度...")
        plan_success, plan_message = self.plan_to_target_joint(joint_state_msg, plan_only=True)

        if not plan_success:
            response.success = False
            response.message = f"规划失败: {plan_message}"
            return response

        # 规划成功，显示规划轨迹
        #self.get_logger().info(f"规划结果: {plan_message}")

        # 自动执行规划轨迹
        self.get_logger().info("自动执行规划轨迹...")
        exec_success, exec_message = self.plan_to_target_joint(joint_state_msg, plan_only=False)
        response.success = exec_success
        response.message = exec_message

        return response

    def plan_to_target(self, target_pose: PoseStamped, plan_only=True):
        """
        规划机械臂到目标位姿。
        如果 plan_only=True，仅进行规划，不执行。
        如果 plan_only=False，执行规划后的轨迹。
        """
        with self.lock:
            if not self.current_joint_state:
                msg = "当前关节状态为空，无法规划。"
                self.get_logger().error(msg)
                return (False, msg)

        # 构建 MoveGroup 动作请求
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'aubo_arm'  # 根据实际机器人修改
        goal_msg.request.planner_id = 'RRTConnectkConfigDefault'
        goal_msg.request.num_planning_attempts = 20
        goal_msg.request.allowed_planning_time = 15.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        if target_pose:
            # 设置目标约束
            goal_msg.request.goal_constraints = self.create_goal_constraints(target_pose)

        # 设置起始状态
        self.set_start_state(goal_msg)

        if target_pose:
            # 设置路径约束
            if bool(self.get_parameter('use_path_constraints').value):
                goal_msg.request.path_constraints = self.create_path_constraints(self.current_joint_state)

        # 设置 plan_only 属性
        goal_msg.planning_options.plan_only = plan_only

        # 发送规划请求
        self._move_group_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback).add_done_callback(
            lambda future: self.handle_plan_result(future, plan_only)
        )

    def plan_to_target_joint(self, target_joint_state: JointState, plan_only=True):
        """
        规划机械臂到目标关节角度。
        如果 plan_only=True，仅进行规划，不执行。
        如果 plan_only=False，执行规划后的轨迹。
        """
        with self.lock:
            if not self.current_joint_state:
                msg = "当前关节状态为空，无法规划。"
                self.get_logger().error(msg)
                return (False, msg)

        # 构建 MoveGroup 动作请求
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'aubo_arm'  # 根据实际机器人修改
        goal_msg.request.planner_id = 'RRTConnectkConfigDefault'
        goal_msg.request.num_planning_attempts = 20
        goal_msg.request.allowed_planning_time = 15.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        # 设置目标关节状态
        goal_msg.request.goal_constraints = self.create_joint_constraints(target_joint_state)

        # 设置起始状态
        self.set_start_state(goal_msg)

        # 设置路径约束
        if bool(self.get_parameter('use_path_constraints').value):
            goal_msg.request.path_constraints = self.create_path_constraints(self.current_joint_state)

        # 设置 plan_only 属性
        goal_msg.planning_options.plan_only = plan_only

        # 发送规划请求
        self._move_group_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback).add_done_callback(
            lambda future: self.handle_plan_result(future, plan_only)
        )

    def handle_plan_result(self, future, plan_only):
        """
        处理规划结果，并根据 plan_only 选择是否执行轨迹。
        """
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                msg = "规划请求被拒绝。"
                self.get_logger().error(msg)
                # 根据需要处理拒绝情况，例如返回或重试
                return

            # 获取规划结果
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(lambda fut: self.handle_plan_done(fut, plan_only))
        except Exception as e:
            self.get_logger().error(f'规划请求失败: {e}')

    def handle_plan_done(self, future, plan_only):
        """
        处理规划完成后的结果。
        """
        try:
            get_result_response = future.result()
            result = get_result_response.result
            error_code = result.error_code

            if error_code.val == error_code.SUCCESS:
                self.get_logger().info("规划成功！轨迹已发布到 RViz。")
                if plan_only:
                    # 仅规划，不执行
                    self.planned_trajectory = result.planned_trajectory
                    self.display_trajectory(result.planned_trajectory)
                    return

                # 执行轨迹
                self.get_logger().info("正在执行规划轨迹...")
                self.execute_trajectory(result.planned_trajectory)
            else:
                self.get_logger().error(f"规划失败，错误码: {error_code.val}")
        except Exception as e:
            self.get_logger().error(f'规划结果处理失败: {e}')

    def execute_trajectory(self, trajectory):
        """
        执行已规划的轨迹。
        """
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # 发送执行请求
        self._execute_client.send_goal_async(goal_msg).add_done_callback(self.handle_execute_result)

    def handle_execute_result(self, future):
        """
        处理执行请求的结果。
        """
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("执行请求被拒绝。")
                return

            self.get_logger().info("执行请求已被接受，等待执行结果...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.handle_execution_done)
        except Exception as e:
            self.get_logger().error(f'执行请求失败: {e}')

    def handle_execution_done(self, future):
        """
        处理执行完成后的结果。
        """
        try:
            exe_result = future.result().result
            if exe_result.error_code.val == exe_result.error_code.SUCCESS:
                self.get_logger().info("成功执行轨迹。")
            else:
                self.get_logger().error(f"执行失败，错误码: {exe_result.error_code.val}")
        except Exception as e:
            self.get_logger().error(f'执行结果处理失败: {e}')

    def set_start_state(self, goal_msg):
        """
        将当前关节状态设置为路径规划的起始状态。
        """
        with self.lock:
            if not self.current_joint_state:
                self.get_logger().error("关节状态为空，无法设置起始状态")
                return

            # 创建 JointState 消息
            joint_state_msg = JointState()
            joint_state_msg.name = list(self.current_joint_state.keys())
            joint_state_msg.position = list(self.current_joint_state.values())

        robot_state = RobotState()
        robot_state.joint_state = joint_state_msg
        goal_msg.request.start_state = robot_state

    def create_goal_constraints(self, target_pose: PoseStamped):
        """
        将目标姿态转化为 MoveIt's goal_constraints。
        """
        constraints = Constraints()

        # 位置约束
        pos_constraint = PositionConstraint()
        pos_constraint.header = target_pose.header
        pos_constraint.link_name = 'tool0'  # 替换为末端执行器的实际链接名
        pos_constraint.weight = 1.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 0.02, 0.02]  # 对目标位置的容忍范围
        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)

        constraints.position_constraints.append(pos_constraint)

        # 姿态约束
        ori_constraint = OrientationConstraint()
        ori_constraint.header = target_pose.header
        ori_constraint.link_name = 'tool0'  # 替换为末端执行器的实际链接名
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        constraints.orientation_constraints.append(ori_constraint)

        return [constraints]

    def create_joint_constraints(self, target_joint_state: JointState):
        """
        将目标关节角度转化为 MoveIt's goal_constraints。
        """
        constraints = Constraints()
        joint_constraints = []

        for i, joint_name in enumerate(target_joint_state.name):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = target_joint_state.position[i]
            jc.tolerance_above = 0.1  # 根据需要调整
            jc.tolerance_below = 0.1
            jc.weight = 1.0
            joint_constraints.append(jc)

        constraints.joint_constraints = joint_constraints
        return [constraints]

    def create_path_constraints(self, joint_positions):
        """
        基于当前关节状态生成路径优化目标函数。
        """
        constraints = Constraints()
        joint_constraints = []

        # ============ 1) shoulder_joint 限制在 -50 度到 50 度之间 ===========#
        tolerance_shoulder = 0.8727  # 50 度转换为弧度
        current_val_shoulder = joint_positions.get("shoulder_joint", 0.0)

        jc_shoulder = JointConstraint()
        jc_shoulder.joint_name = "shoulder_joint"
        jc_shoulder.position = current_val_shoulder
        jc_shoulder.tolerance_above = min(tolerance_shoulder, 50 * (3.1416 / 180))  # 0.8727 弧度
        jc_shoulder.tolerance_below = min(tolerance_shoulder, abs(-50 * (3.1416 / 180)))  # 0.8727 弧度
        jc_shoulder.weight = 1.0
        joint_constraints.append(jc_shoulder)

        # ============ 2) wrist1_joint 尽量保持当前值 ± 1.0 ============#
        tolerance_wrist1 = 1.0  # 根据实际需求调整容忍度
        current_val_wrist1 = joint_positions.get("wrist1_joint", 0.0)

        jc_wrist1 = JointConstraint()
        jc_wrist1.joint_name = "wrist1_joint"
        jc_wrist1.position = current_val_wrist1
        jc_wrist1.tolerance_above = tolerance_wrist1
        jc_wrist1.tolerance_below = tolerance_wrist1
        jc_wrist1.weight = 1.0
        joint_constraints.append(jc_wrist1)

        # ============ 3) wrist3_joint 尽量保持当前值 ± 0.1 ============#
        tolerance_wrist3 = 0.1  # 根据实际需求调整容忍度
        current_val_wrist3 = joint_positions.get("wrist3_joint", 0.0)

        jc_wrist3 = JointConstraint()
        jc_wrist3.joint_name = "wrist3_joint"
        jc_wrist3.position = current_val_wrist3
        jc_wrist3.tolerance_above = tolerance_wrist3
        jc_wrist3.tolerance_below = tolerance_wrist3
        jc_wrist3.weight = 1.0
        joint_constraints.append(jc_wrist3)

        # 将所有关节约束添加到路径约束中
        constraints.joint_constraints = joint_constraints
        return constraints

    def execute_trajectory(self, trajectory):
        """
        执行已规划的轨迹。
        """
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # 发送执行请求
        self._execute_client.send_goal_async(goal_msg).add_done_callback(self.handle_execute_result)

    def handle_execute_result(self, future):
        """
        处理执行请求的结果。
        """
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("执行请求被拒绝。")
                return

            self.get_logger().info("执行请求已被接受，等待执行结果...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.handle_execution_done)
        except Exception as e:
            self.get_logger().error(f'执行请求失败: {e}')

    def handle_execution_done(self, future):
        """
        处理执行完成后的结果。
        """
        try:
            exe_result = future.result().result
            if exe_result.error_code.val == exe_result.error_code.SUCCESS:
                self.get_logger().info("成功执行轨迹。")
            else:
                self.get_logger().error(f"执行失败，错误码: {exe_result.error_code.val}")
        except Exception as e:
            self.get_logger().error(f'执行结果处理失败: {e}')

    def display_trajectory(self, trajectory):
        """
        发布规划轨迹到 RViz。
        """
        display_traj = DisplayTrajectory()

        # 创建 RobotState 消息
        with self.lock:
            joint_state_msg = JointState()
            joint_state_msg.name = list(self.current_joint_state.keys())
            joint_state_msg.position = list(self.current_joint_state.values())

        robot_state = RobotState()
        robot_state.joint_state = joint_state_msg

        # 设置 trajectory_start 为 RobotState
        display_traj.trajectory_start = robot_state

        # 添加规划轨迹
        display_traj.trajectory.append(trajectory)

        # 发布到 RViz
        self.display_publisher.publish(display_traj)
        self.get_logger().info("规划轨迹已发布到 RViz。")

def main(args=None):
    rclpy.init(args=args)
    node = MoveArmServer()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        node.get_logger().info('节点已中断。')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

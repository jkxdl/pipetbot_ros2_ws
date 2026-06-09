import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
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

# === 自定义动作 ===
from pipettingrobot_interfaces.action import SetTargetPose  # 确保已定义 SetTargetPose.action 并生成接口


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

        # 创建并启动自定义动作服务器：SetTargetPose
        self._pose_action_server = ActionServer(
            self,
            SetTargetPose,
            'set_target_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info("动作服务器 [set_target_pose] 已创建，等待客户端请求...")

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

    # --- Action Server Callbacks ---

    def goal_callback(self, goal_request):
        """
        Callback to determine whether to accept or reject a goal.
        """
        self.get_logger().info('收到新的 SetTargetPose 目标请求。')
        # 可以在这里添加验证逻辑
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Callback to handle goal cancellation.
        """
        self.get_logger().info('SetTargetPose 目标被请求取消。')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        执行动作的回调函数。
        """
        self.get_logger().info('开始处理 SetTargetPose 目标...')

        feedback_msg = SetTargetPose.Feedback()
        response_msg = SetTargetPose.Result()

        # 读取目标位姿
        target_pose = goal_handle.request.pose

        # 将 geometry_msgs/Pose 转换为 PoseStamped（注意 frame_id）
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"  # 或者您的机器人基座坐标系
        pose_stamped.pose = target_pose

        # 提供初步反馈
        feedback_msg.feedback = "规划路径中..."
        goal_handle.publish_feedback(feedback_msg)

        # 规划轨迹
        self.get_logger().info("正在规划路径...")
        plan_future = self.plan_to_target(pose_stamped, plan_only=True)

        if plan_future is None:
            response_msg.success = False
            response_msg.message = "规划失败：无法规划轨迹。"
            goal_handle.succeed()
            return response_msg

        # 等待规划结果
        try:
            goal_handle_plan = await plan_future
            if not goal_handle_plan.accepted:
                msg = "规划请求被拒绝"
                self.get_logger().error(msg)
                response_msg.success = False
                response_msg.message = msg
                goal_handle.abort()
                return response_msg

            plan_result = await goal_handle_plan.get_result_async()
            plan_result = plan_result.result

            if plan_result.error_code.val != plan_result.error_code.SUCCESS:
                msg = f"规划失败，错误码: {plan_result.error_code.val}"
                self.get_logger().error(msg)
                response_msg.success = False
                response_msg.message = msg
                goal_handle.abort()
                return response_msg

            # 保存规划轨迹
            self.planned_trajectory = plan_result.planned_trajectory

            # 发布规划轨迹到 RViz
            self.display_trajectory(plan_result.planned_trajectory)

            # 提供反馈
            feedback_msg.feedback = "路径规划成功，开始执行轨迹..."
            goal_handle.publish_feedback(feedback_msg)

            # 执行轨迹
            self.get_logger().info("规划成功，开始执行轨迹...")
            exec_future = self.execute_trajectory(plan_result.planned_trajectory)

            # 等待执行结果
            goal_handle_exec = await exec_future
            if not goal_handle_exec.accepted:
                msg = "执行请求被拒绝"
                self.get_logger().error(msg)
                response_msg.success = False
                response_msg.message = msg
                goal_handle.abort()
                return response_msg

            exe_result = await goal_handle_exec.get_result_async()
            exe_result = exe_result.result

            if exe_result.error_code.val == exe_result.error_code.SUCCESS:
                msg = "成功到达目标位姿。"
                self.get_logger().info(msg)
                response_msg.success = True
                response_msg.message = msg
                goal_handle.succeed()
                return response_msg
            else:
                msg = f"执行失败，错误码: {exe_result.error_code.val}"
                self.get_logger().error(msg)
                response_msg.success = False
                response_msg.message = msg
                goal_handle.abort()
                return response_msg

        except Exception as e:
            self.get_logger().error(f"规划或执行过程中发生异常: {e}")
            response_msg.success = False
            response_msg.message = f"规划或执行过程中发生异常: {e}"
            goal_handle.abort()
            return response_msg

    def plan_to_target(self, target_pose: PoseStamped, plan_only=True):
        """
        规划机械臂到目标位姿。
        如果 plan_only=True，仅进行规划，不执行。
        如果 plan_only=False，执行规划后的轨迹。
        返回 MoveGroup.GoalHandle 或 ExecuteTrajectory.GoalHandle。
        """
        self.get_logger().info("进入 plan_to_target 方法")
        if plan_only and target_pose is None:
            if self.planned_trajectory:
                # 执行已保存的轨迹
                return self.execute_trajectory(self.planned_trajectory)
            else:
                self.get_logger().error("没有规划好的轨迹可执行。")
                return None

        with self.lock:
            if not self.current_joint_state:
                msg = "当前关节状态为空，无法规划。"
                self.get_logger().error(msg)
                return None

        # 构建 MoveGroup 动作请求
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'aubo_arm'  # 根据实际机器人修改
        goal_msg.request.planner_id = 'RRTstarkConfigDefault'
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
                path_constraints = self.create_path_constraints(self.current_joint_state)
                if path_constraints:
                    goal_msg.request.path_constraints = path_constraints

        # 设置 plan_only 属性
        goal_msg.planning_options.plan_only = plan_only

        # 发送规划请求
        self.get_logger().info("发送规划请求")
        send_goal_future = self._move_group_client.send_goal_async(goal_msg)
        return send_goal_future

    def execute_trajectory(self, trajectory):
        """
        执行已规划的轨迹。
        返回 ExecuteTrajectory.GoalHandle。
        """
        self.get_logger().info("进入 execute_trajectory 方法")
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # 发送执行请求
        self.get_logger().info("发送执行请求")
        send_goal_future = self._execute_client.send_goal_async(goal_msg)
        return send_goal_future

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
        box.dimensions = [0.001, 0.001, 0.001]  # 对目标位置的容忍范围
        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)

        constraints.position_constraints.append(pos_constraint)

        # 姿态约束
        ori_constraint = OrientationConstraint()
        ori_constraint.header = target_pose.header
        ori_constraint.link_name = 'tool0'  # 替换为末端执行器的实际链接名
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0

        constraints.orientation_constraints.append(ori_constraint)

        return [constraints]

    def create_path_constraints(self, joint_positions):
        """
        基于当前关节状态生成路径优化目标函数。
        """
        from moveit_msgs.msg import JointConstraint, Constraints

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
        tolerance_wrist1 = 1.2  # 根据实际需求调整容忍度
        current_val_wrist1 = joint_positions.get("wrist1_joint", 0.0)

        jc_wrist1 = JointConstraint()
        jc_wrist1.joint_name = "wrist1_joint"
        jc_wrist1.position = current_val_wrist1
        jc_wrist1.tolerance_above = tolerance_wrist1
        jc_wrist1.tolerance_below = tolerance_wrist1
        jc_wrist1.weight = 1.0
        joint_constraints.append(jc_wrist1)

        # ============ 3) wrist3_joint 尽量保持当前值 ± 0.1 ============#
        tolerance_wrist3 = 0.2  # 根据实际需求调整容忍度
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
    executor = rclpy.executors.MultiThreadedExecutor()  # 使用多线程执行器
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        node.get_logger().info('节点已中断。')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

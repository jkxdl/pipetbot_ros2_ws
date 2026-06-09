import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
import sys

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import ApplyPlanningScene # 修正为服务类型
from moveit_msgs.msg import PlanningScene, CollisionObject, PositionConstraint, OrientationConstraint, Constraints, RobotState
from geometry_msgs.msg import PoseStamped,Pose
from std_srvs.srv import Trigger
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState


class SequenceMoveAndCaptureNode(Node):
    def __init__(self):
        super().__init__('sequence_move_and_capture_node')

        # 规划组的关节名称顺序
        self.planning_group_joint_names = [
            'shoulder_joint',
            'upperArm_joint',
            'foreArm_joint',
            'wrist1_joint',
            'wrist2_joint',
            'wrist3_joint'
        ]
        
        self.current_joint_state = {}

        # 订阅 /joint_states 话题
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10  # 队列长度
        )

        self.get_logger().info("已订阅 /joint_states，等待关节状态更新...")
        # 动作客户端：用于规划 (move_action)
        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')
        # 动作客户端：用于执行轨迹 (execute_trajectory)
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        # 创建/capture_pose服务客户端
        self._capture_client = self.create_client(Trigger, '/capture_pose')

        # 创建/apply_planning_scene服务客户端
        self._planning_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        # 等待服务和动作服务可用
        while not self._capture_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /capture_pose 服务...')
        while not self._planning_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /apply_planning_scene 服务...')
        while not self._move_group_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('等待 /move_action 动作服务器...')
        while not self._execute_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('等待 /execute_trajectory 动作服务器...')

        self.get_logger().info('服务和动作服务器已准备就绪。')

        # 添加障碍物
        self.add_obstacles()

        # 定义标定点列表
        self.calibration_points = self.get_calibration_points()
        self.current_point_index = 0

    
    def joint_states_callback(self, msg):
        """
        处理接收到的关节状态消息，并根据规划组重新排序。
        """
        # 将接收到的关节状态转为字典
        joint_positions = dict(zip(msg.name, msg.position))

        # 检查是否所有规划组的关节都存在于接收到的关节状态中
        missing_joints = [
            joint for joint in self.planning_group_joint_names if joint not in joint_positions
        ]
        if missing_joints:
            self.get_logger().warn(f"以下关节未在 /joint_states 中找到: {missing_joints}")
            return

        # 按规划组的顺序排列关节状态
        ordered_joint_positions = [
            joint_positions[joint] for joint in self.planning_group_joint_names
        ]

        # 更新当前关节状态
        self.current_joint_state = dict(zip(self.planning_group_joint_names, ordered_joint_positions))
        #self.get_logger().info(f"更新关节状态: {self.current_joint_state}")

        # 如果路径规划尚未启动，则启动
        if not hasattr(self, 'planning_started') or not self.planning_started:
            self.planning_started = True
            self.plan_to_next_point(plan_only=True)

    
    def add_obstacles(self):
        """
        向规划场景中添加地面和墙面障碍物。
        """
        self.get_logger().info('正在向规划场景添加地面和墙面障碍物...')
        collision_objects = []

        # 添加地面障碍物
        ground = CollisionObject()
        ground.header.frame_id = "world"
        ground.id = "ground_plane"

        ground_primitive = SolidPrimitive()
        ground_primitive.type = SolidPrimitive.BOX
        ground_primitive.dimensions = [1.0, 1.0, 0.01]  # 地面尺寸
        ground.primitives.append(ground_primitive)

        ground_pose = Pose()
        ground_pose.position.x = 0.0
        ground_pose.position.y = 0.0
        ground_pose.position.z = -0.01  # 地面略低于0
        ground_pose.orientation.w = 1.0
        ground.primitive_poses.append(ground_pose)

        ground.operation = CollisionObject.ADD
        collision_objects.append(ground)

        # 添加墙面障碍物
        wall = CollisionObject()
        wall.header.frame_id = "world"
        wall.id = "wall_obstacle"

        wall_primitive = SolidPrimitive()
        wall_primitive.type = SolidPrimitive.BOX
        wall_primitive.dimensions = [2.0, 0.05, 2.0]  # 墙面尺寸
        wall.primitives.append(wall_primitive)

        wall_pose = Pose()
        wall_pose.position.x = 0.0  # 根据实际需求调整位置
        wall_pose.position.y = 0.5
        wall_pose.position.z = 0.0  # 墙面高度中心
        wall_pose.orientation.w = 1.0
        wall.primitive_poses.append(wall_pose)

        wall.operation = CollisionObject.ADD
        collision_objects.append(wall)

        # 构建规划场景消息
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.extend(collision_objects)

        # 创建/apply_planning_scene服务请求
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene

        # 发送/apply_planning_scene请求
        self._planning_scene_client.call_async(request).add_done_callback(self.handle_planning_scene_response)

    def handle_planning_scene_response(self, future):
        """
        处理/apply_planning_scene服务的响应。
        """
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('地面和墙面障碍物已成功添加到规划场景。')
            else:
                self.get_logger().warn('障碍物添加失败。')
        except Exception as e:
            self.get_logger().error(f'处理/apply_planning_scene响应时发生错误: {e}')


    def get_calibration_points(self):
        """
        定义多个标定点，每个点包含位置和姿态。
        """
        calibration_points = []
        points = [
            {"position": [0.5, -0.12, 0.5], "orientation": [0.5, 0.5, 0.5, 0.5]},
            {"position": [0.52927, 0.14369, 0.53581], "orientation": [0.5, 0.5, 0.5, 0.5]},
            {"position": [0.45093, -0.20759, 0.42544], "orientation": [0.5, 0.5, 0.5, 0.5]},
            {"position": [0.47867, -0.058207, 0.44601], "orientation": [0.5, 0.5, 0.5, 0.5]},
        ]

        for pt in points:
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'  # 根据实际情况修改参考坐标系
            pose.pose.position.x = pt["position"][0]
            pose.pose.position.y = pt["position"][1]
            pose.pose.position.z = pt["position"][2]
            pose.pose.orientation.x = pt["orientation"][0]
            pose.pose.orientation.y = pt["orientation"][1]
            pose.pose.orientation.z = pt["orientation"][2]
            pose.pose.orientation.w = pt["orientation"][3]
            calibration_points.append(pose)

        return calibration_points
    def set_start_state(self, goal_msg):
        """
        将当前关节状态设置为路径规划的起始状态。
        """
        if not self.current_joint_state:
            self.get_logger().error("关节状态为空，无法设置起始状态")
            return

        # 创建 JointState 消息
        joint_state_msg = JointState()
        joint_state_msg.name = list(self.current_joint_state.keys())
        joint_state_msg.position = list(self.current_joint_state.values())

        # 创建 RobotState 消息并赋值
        robot_state = RobotState()
        robot_state.joint_state = joint_state_msg

        # 设置起始状态
        goal_msg.request.start_state = robot_state

    
    def plan_to_next_point(self, plan_only=True):
        """
        对下一个标定点进行规划。
        """
        if self.current_point_index >= len(self.calibration_points):
            self.get_logger().info('所有标定点已完成。')
            return

        target_pose = self.calibration_points[self.current_point_index]
        self.get_logger().info(f"规划到标定点 {self.current_point_index + 1}")

        # 检查关节状态
        if not self.current_joint_state:
            self.get_logger().error("当前关节状态为空，无法进行规划")
            return


        # 创建 MoveGroup 动作请求
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'aubo_arm'  # 修改为实际规划组名
        goal_msg.request.planner_id = 'RRTConnectkConfigDefault'

        # 设置规划参数
        goal_msg.request.num_planning_attempts = 20
        goal_msg.request.allowed_planning_time = 15.0
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1

        # 设置目标位姿为终点 (使用 goal_constraints)
        goal_msg.request.goal_constraints = self.create_goal_constraints(target_pose)

        # 设置起始状态
        self.set_start_state(goal_msg)

        # 添加路径优化约束 (path_constraints)
        goal_msg.request.path_constraints = self.create_path_constraints(self.current_joint_state)

        # 设置 plan_only 属性
        goal_msg.planning_options.plan_only = plan_only

        # 发送路径规划请求
        self._move_group_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback).add_done_callback(
            self.handle_plan_result if plan_only else self.handle_executable_plan_result
        )


    def create_path_constraints(self, joint_positions):
        """
        基于当前关节状态生成路径优化目标函数。
        """
        from moveit_msgs.msg import JointConstraint, Constraints

        constraints = Constraints()
        joint_constraints = []

        # ============ 1) shoulder_joint 限制在 -50 度到 50 度之间 ===========#
        tolerance_shoulder = 0.8727  # 50 度转换为弧度
        # 由于 JointConstraint 是基于期望位置和容忍度来设置的，
        # 为了限制关节在一定范围内，可以设置期望位置为当前值，并调整容忍度
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
    def create_goal_constraints(self, target_pose: PoseStamped):
        """
        根据目标位置和姿态生成 MoveIt 的 goal_constraints。
        """
        constraints = Constraints()

        # 创建位置约束
        pos_constraint = PositionConstraint()
        pos_constraint.header = target_pose.header
        pos_constraint.link_name = 'tool0'  # 替换为末端执行器的实际链接名
        pos_constraint.weight = 1.0

        # 定义允许的空间范围（立方体）
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.1, 0.1, 0.1]  # 目标点的容忍范围
        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)

        # 添加位置约束
        constraints.position_constraints.append(pos_constraint)

        # 创建姿态约束
        ori_constraint = OrientationConstraint()
        ori_constraint.header = target_pose.header
        ori_constraint.link_name = 'tool0'  # 替换为末端执行器的实际链接名
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        # 添加姿态约束
        constraints.orientation_constraints.append(ori_constraint)

        return [constraints]




    def handle_plan_result(self, future):
        """
        处理plan_only规划结果：显示目标模型和轨迹后，等待用户确认。
        """
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('规划请求被拒绝。')
                self.current_point_index += 1
                self.plan_to_next_point(plan_only=True)
                return

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.handle_plan_done)
        except Exception as e:
            self.get_logger().error(f'规划请求失败: {e}')
            self.current_point_index += 1
            self.plan_to_next_point(plan_only=True)

    def handle_plan_done(self, future):
        """
        仅规划（plan_only=True）结果的处理函数，用于显示轨迹和等待用户确认。
        """
        try:
            get_result_response = future.result()
            result = get_result_response.result
            error_code = result.error_code
            if error_code.val == error_code.SUCCESS:
                self.get_logger().info("规划成功！Rviz中已显示目标位姿与规划轨迹。")
                self.get_logger().info("输入 'y' 并回车以执行该轨迹，否则跳过此标定点：")
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == 'y':
                    # 用户确认执行，则再次规划，但这次plan_only=False，以获得可执行轨迹
                    self.plan_to_next_point(plan_only=False)
                else:
                    self.get_logger().info("用户取消执行，跳过此标定点。")
                    self.current_point_index += 1
                    self.plan_to_next_point(plan_only=True)
            else:
                self.get_logger().error(f"规划失败，错误码：{result.error_code.val}")
                self.current_point_index += 1
                self.plan_to_next_point(plan_only=True)
        except Exception as e:
            self.get_logger().error(f'规划结果处理失败: {e}')
            self.current_point_index += 1
            self.plan_to_next_point(plan_only=True)

    def handle_executable_plan_result(self, future):
        """
        当用户确认后再次规划（plan_only=False），获得可执行的轨迹。
        """
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('可执行规划请求被拒绝。')
                self.current_point_index += 1
                self.plan_to_next_point(plan_only=True)
                return

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.handle_executable_plan_done)
        except Exception as e:
            self.get_logger().error(f'可执行规划请求失败: {e}')
            self.current_point_index += 1
            self.plan_to_next_point(plan_only=True)

    def handle_executable_plan_done(self, future):
        """
        处理可执行规划结果，然后通过execute_trajectory动作执行。
        """
        try:
            get_result_response = future.result()
            result = get_result_response.result
            error_code = result.error_code
            if error_code.val == error_code.SUCCESS:
                self.get_logger().info("可执行规划成功，正在发送执行请求...")
                self.execute_trajectory(result.planned_trajectory)
            else:
                self.get_logger().error(f"可执行规划失败，错误码：{error_code.val}")
                self.current_point_index += 1
                self.plan_to_next_point(plan_only=True)
        except Exception as e:
            self.get_logger().error(f'可执行规划结果处理失败: {e}')
            self.current_point_index += 1
            self.plan_to_next_point(plan_only=True)

    def execute_trajectory(self, trajectory):
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory
        self._execute_client.send_goal_async(goal_msg).add_done_callback(self.handle_execute_result)

    def handle_execute_result(self, future):
        """
        执行轨迹的结果处理。
        """
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("执行请求被拒绝。")
                self.current_point_index += 1
                self.plan_to_next_point(plan_only=True)
                return

            self.get_logger().info("执行请求已被接受，等待执行结果...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.handle_execution_done)
        except Exception as e:
            self.get_logger().error(f'执行请求失败: {e}')
            self.current_point_index += 1
            self.plan_to_next_point(plan_only=True)

    def handle_execution_done(self, future):
        """
        执行完成后调用/capture_pose服务进行标定。
        """
        try:
            get_result_response = future.result()
            result = get_result_response.result
            if result.error_code.val == result.error_code.SUCCESS:
                self.get_logger().info(f'成功到达标定点 {self.current_point_index + 1}，调用/capture_pose服务进行标定...')
                self.call_capture_pose_service()
            else:
                self.get_logger().error(f'执行失败，错误码：{result.error_code.val}')
                self.current_point_index += 1
                self.plan_to_next_point(plan_only=True)
        except Exception as e:
            self.get_logger().error(f'执行结果处理失败: {e}')
            self.current_point_index += 1
            self.plan_to_next_point(plan_only=True)

    def call_capture_pose_service(self):
        req = Trigger.Request()
        self._capture_client.call_async(req).add_done_callback(self.handle_capture_response)

    def handle_capture_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'标定成功: {response.message}')
            else:
                self.get_logger().warn(f'标定失败: {response.message}')
        except Exception as e:
            self.get_logger().error(f'调用 /capture_pose 服务异常: {e}')

        # 标定完成后移动到下一个点，重新进入plan_only状态
        self.current_point_index += 1
        self.plan_to_next_point(plan_only=True)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'动作反馈: {feedback}')


def main(args=None):
    rclpy.init(args=args)
    node = SequenceMoveAndCaptureNode()
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
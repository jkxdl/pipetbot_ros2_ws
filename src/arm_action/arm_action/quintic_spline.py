import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from threading import Lock
import builtin_interfaces.msg as bip
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import (
    RobotState,
    DisplayTrajectory
)
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

# === 自定义动作接口 ===
from pipettingrobot_interfaces.action import SetTargetPose  # 确保该接口包已正确生成


class MoveArmServer(Node):
    def __init__(self):
        super().__init__('move_arm_server')
        self.link_offset_shoulder = 0.1215
        self.link_upper_arm = 0.408
        self.link_fore_arm = 0.376
        self.link_wrist_pitch = 0.1025
        self.tool_offset_x = 0.2
        self.tool_offset_y = 0.2374
        self.base_height = 0.122
        self.branch_a_z_offset = -0.0129
        self.branch_b_z_offset = -0.2179
        self.supported_orientation = (0.5, 0.5, 0.5, 0.5)
        self.orientation_tolerance = 5e-3
        # 规划组关节名称（请根据实际机器人修改）
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
        self.declare_parameter('analytic_orientation_tolerance', self.orientation_tolerance)
        self.orientation_tolerance = float(
            self.get_parameter('analytic_orientation_tolerance').value
        )

        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.get_logger().info("已订阅 /joint_states，等待关节状态更新...")

        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        while not self._trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('等待 /joint_trajectory_controller/follow_joint_trajectory 动作服务器...')
        self.get_logger().info('服务和动作服务器已准备就绪。')

        self._pose_action_server = ActionServer(
            self,
            SetTargetPose,
            'set_target_pose_quintic',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info("动作服务器 [set_target_pose_quintic] 已创建，等待客户端请求...")

        self.display_publisher = self.create_publisher(DisplayTrajectory, 'display_planned_path', 10)
        self.planned_trajectory = None

    def joint_states_callback(self, msg):
        """处理关节状态消息，并按规划组顺序保存"""
        joint_positions = dict(zip(msg.name, msg.position))
        missing = [j for j in self.planning_group_joint_names if j not in joint_positions]
        if missing:
            return
        with self.lock:
            self.current_joint_state = {name: joint_positions[name] for name in self.planning_group_joint_names}

    def goal_callback(self, goal_request):
        self.get_logger().info('收到新的 SetTargetPose 目标请求。')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('目标请求取消。')
        return CancelResponse.ACCEPT

    def compute_distance(self, start_angles: list, target_angles: list) -> float:
        # 计算欧氏距离
        diff = [t - s for s, t in zip(start_angles, target_angles)]
        distance = math.sqrt(sum(d ** 2 for d in diff))
        return distance

    def determine_motion_time_and_samples(self, start_angles: list, target_angles: list,
                                        desired_speed: float, control_rate: float):
        """
        desired_speed: 平均运动速度（单位与角度差相关，如 rad/s）
        control_rate: 控制器采样率（Hz）
        """
        distance = self.compute_distance(start_angles, target_angles)
        T_total = distance / desired_speed if desired_speed > 0 else 5.0
        num_samples = int(T_total * control_rate)
        # 为避免样本数太少，可以设定一个下限
        num_samples = max(num_samples, 10)
        return T_total, num_samples


    async def execute_callback(self, goal_handle):
        self.get_logger().info('处理目标请求 (IK + 五次样条轨迹) ...')
        response_msg = SetTargetPose.Result()

        with self.lock:
            if not self.current_joint_state:
                self.get_logger().error("当前关节状态为空！")
                response_msg.success = False
                response_msg.message = "未收到有效的关节状态数据。"
                goal_handle.succeed()
                return response_msg

        # 异步调用 IK 求解（注意 compute_ik 修改为 async）
        target_joint_angles = await self.compute_ik(goal_handle.request.pose)
        if target_joint_angles is None:
            self.get_logger().error("IK 求解失败。")
            response_msg.success = False
            response_msg.message = "IK 求解失败。"
            goal_handle.succeed()
            return response_msg
        self.get_logger().info(f"目标关节解: {target_joint_angles}")

        with self.lock:
            start_joint_angles = [self.current_joint_state[j] for j in self.planning_group_joint_names]
        self.get_logger().info(f"起始关节角: {start_joint_angles}")
        joint_deltas = [t - s for s, t in zip(start_joint_angles, target_joint_angles)]
        max_delta = max(abs(delta) for delta in joint_deltas)
        self.get_logger().info(f"关节角增量: {[round(delta, 4) for delta in joint_deltas]}")
        self.get_logger().info(f"最大关节角变化: {max_delta:.4f} rad")
        if max_delta < 0.01:
            self.get_logger().warn("目标关节解与当前关节状态几乎相同，机械臂可能不会有明显动作。")

        # 动态确定运动时间和采样点数量
        # 例如，假设期望平均速度为 0.4 rad/s，控制频率为 100 Hz
        T_total, num_samples = self.determine_motion_time_and_samples(start_joint_angles, target_joint_angles,
                                                                    desired_speed=0.4, control_rate=100)
        self.get_logger().info(f"动态运动时间 T_total: {T_total:.2f} s, 采样点数量: {num_samples}")
        traj_points = self.generate_quintic_trajectory(start_joint_angles, target_joint_angles, T_total, num_samples)
        self.get_logger().info("轨迹采样生成完成。")

        jt = JointTrajectory()
        jt.joint_names = self.planning_group_joint_names
        sample_dt = T_total / num_samples
        time_from_start = 0.0
        for point in traj_points:
            pt = JointTrajectoryPoint()
            pt.positions = point
            pt.velocities = [0.0] * len(point)
            pt.accelerations = [0.0] * len(point)
            # 先创建一个 Duration 对象
            d = bip.Duration()
            d.sec = int(time_from_start)
            d.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            pt.time_from_start = d
            jt.points.append(pt)
            time_from_start += sample_dt

        if jt.points:
            final_point = jt.points[-1]
            self.get_logger().info(
                f"轨迹终点关节角: {[round(v, 4) for v in final_point.positions]}"
            )
            self.get_logger().info(
                f"轨迹总时长: {final_point.time_from_start.sec + final_point.time_from_start.nanosec / 1e9:.2f} s"
            )

        from moveit_msgs.msg import RobotTrajectory
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = jt

        self.display_trajectory(robot_traj)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = jt
        exec_future = self._trajectory_client.send_goal_async(goal_msg)
        goal_handle_exec = await exec_future
        if not goal_handle_exec.accepted:
            self.get_logger().error("执行请求被拒绝。")
            response_msg.success = False
            response_msg.message = "执行请求被拒绝。"
            goal_handle.succeed()
            return response_msg

        exe_result = await goal_handle_exec.get_result_async()
        exe_result = exe_result.result
        if exe_result.error_code == 0:
            self.get_logger().info("轨迹执行成功。")
            response_msg.success = True
            response_msg.message = "轨迹成功执行。"
        else:
            self.get_logger().error(
                f"执行失败，错误码: {exe_result.error_code}, 信息: {exe_result.error_string}"
            )
            response_msg.success = False
            response_msg.message = f"执行失败，错误码: {exe_result.error_code}"
        goal_handle.succeed()
        return response_msg

    async def compute_ik(self, target_pose: Pose) -> list:
        """使用解析几何法求解当前移液姿态下的逆解。"""
        self.get_logger().info(
            "Analytic IK target pose in base_link: "
            f"pos=({target_pose.position.x:.4f}, {target_pose.position.y:.4f}, {target_pose.position.z:.4f}), "
            f"quat=({target_pose.orientation.x:.4f}, {target_pose.orientation.y:.4f}, "
            f"{target_pose.orientation.z:.4f}, {target_pose.orientation.w:.4f})"
        )

        with self.lock:
            if not self.current_joint_state:
                self.get_logger().error("当前关节状态为空，无法进行解析 IK 计算！")
                return None
            current_angles = [
                self.current_joint_state[name]
                for name in self.planning_group_joint_names
            ]
            self.get_logger().info(
                f"IK seed joints: {dict(zip(self.planning_group_joint_names, [round(v, 4) for v in current_angles]))}"
            )

        if not self._is_supported_orientation(target_pose):
            self.get_logger().error(
                "解析 IK 目前仅支持固定末端姿态 (0.5, 0.5, 0.5, 0.5)。"
            )
            return None

        candidates = self._solve_analytic_candidates(target_pose)
        if not candidates:
            self.get_logger().error("解析 IK 未找到可行解。")
            return None

        best_solution = min(
            candidates,
            key=lambda q: self._wrapped_distance(q, current_angles)
        )
        self.get_logger().info(
            f"解析 IK 候选解数量: {len(candidates)}，选择最接近当前关节状态的解。"
        )
        return best_solution

    def _is_supported_orientation(self, target_pose: Pose) -> bool:
        q = (
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w,
        )
        ref = self.supported_orientation
        same = max(abs(a - b) for a, b in zip(q, ref))
        neg_same = max(abs(a + b) for a, b in zip(q, ref))
        return min(same, neg_same) <= self.orientation_tolerance

    def _solve_analytic_candidates(self, target_pose: Pose) -> list:
        x = target_pose.position.x
        y = target_pose.position.y
        z = target_pose.position.z

        planar_x = x - self.tool_offset_x
        planar_y = y
        shoulder_sq = planar_x * planar_x + planar_y * planar_y - self.link_offset_shoulder ** 2
        if shoulder_sq < -1e-8:
            self.get_logger().error(
                f"解析 IK 失败：目标点超出肩部平面可达范围，shoulder_sq={shoulder_sq:.6f}"
            )
            return []

        radial_abs = math.sqrt(max(0.0, shoulder_sq))
        candidates = []
        for radial in (radial_abs, -radial_abs):
            denom = planar_x * planar_x + planar_y * planar_y
            if denom < 1e-10:
                continue

            sin_q1 = (
                self.link_offset_shoulder * planar_x - radial * planar_y
            ) / denom
            cos_q1 = (
                -radial * planar_x - self.link_offset_shoulder * planar_y
            ) / denom
            q1 = math.atan2(sin_q1, cos_q1)

            branch_a = self._solve_planar_2r(
                radial,
                z - self.branch_a_z_offset,
            )
            for q2, q3 in branch_a:
                q4 = -q2 + q3
                q5 = (math.pi / 2.0) - q1
                q6 = 0.0
                candidates.append([
                    self._wrap_to_pi(q1),
                    self._wrap_to_pi(q2),
                    self._wrap_to_pi(q3),
                    self._wrap_to_pi(q4),
                    self._wrap_to_pi(q5),
                    self._wrap_to_pi(q6),
                ])

            branch_b = self._solve_planar_2r(
                radial,
                z - self.branch_b_z_offset,
            )
            for q2, q3 in branch_b:
                q4 = math.pi - q2 + q3
                q5 = q1 - (math.pi / 2.0)
                q6 = math.pi
                candidates.append([
                    self._wrap_to_pi(q1),
                    self._wrap_to_pi(q2),
                    self._wrap_to_pi(q3),
                    self._wrap_to_pi(q4),
                    self._wrap_to_pi(q5),
                    self._wrap_to_pi(q6),
                ])

        return self._deduplicate_solutions(candidates)

    def _solve_planar_2r(self, radial: float, vertical: float) -> list:
        l2 = self.link_upper_arm
        l3 = self.link_fore_arm
        cos_q3 = (
            radial * radial + vertical * vertical - l2 * l2 - l3 * l3
        ) / (2.0 * l2 * l3)
        if cos_q3 < -1.0 - 1e-8 or cos_q3 > 1.0 + 1e-8:
            return []

        cos_q3 = max(-1.0, min(1.0, cos_q3))
        phi = math.atan2(radial, vertical)
        solutions = []
        for q3 in (math.acos(cos_q3), -math.acos(cos_q3)):
            beta = math.atan2(
                l3 * math.sin(q3),
                l2 + l3 * math.cos(q3)
            )
            q2 = phi + beta
            solutions.append((q2, q3))
        return solutions

    def _deduplicate_solutions(self, solutions: list) -> list:
        unique = []
        for solution in solutions:
            if any(
                max(
                    abs(self._wrap_to_pi(a - b))
                    for a, b in zip(solution, existing)
                ) < 1e-5
                for existing in unique
            ):
                continue
            unique.append(solution)
        return unique

    def _wrapped_distance(self, a: list, b: list) -> float:
        return math.sqrt(
            sum(self._wrap_to_pi(x - y) ** 2 for x, y in zip(a, b))
        )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi




    def generate_quintic_trajectory(self, start_angles: list, target_angles: list, T: float, N: int) -> list:
        dof = len(start_angles)
        traj = []
        for i in range(N + 1):
            t = T * i / N
            sample = []
            for j in range(dof):
                delta = target_angles[j] - start_angles[j]
                a3 = 10.0 * delta / (T ** 3)
                a4 = -15.0 * delta / (T ** 4)
                a5 = 6.0 * delta / (T ** 5)
                q_t = start_angles[j] + a3 * (t ** 3) + a4 * (t ** 4) + a5 * (t ** 5)
                sample.append(q_t)
            traj.append(sample)
        return traj

    def display_trajectory(self, trajectory):
        display_traj = DisplayTrajectory()
        with self.lock:
            joint_state_msg = JointState()
            joint_state_msg.name = list(self.current_joint_state.keys())
            joint_state_msg.position = list(self.current_joint_state.values())
        robot_state = RobotState()
        robot_state.joint_state = joint_state_msg
        display_traj.trajectory_start = robot_state
        display_traj.trajectory.append(trajectory)
        self.display_publisher.publish(display_traj)
        self.get_logger().info("规划轨迹已发布到 RViz。")


def main(args=None):
    rclpy.init(args=args)
    node = MoveArmServer()
    executor = rclpy.executors.MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        node.get_logger().info('节点已中断。')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

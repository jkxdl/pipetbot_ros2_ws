import rclpy
from rclpy.node import Node
from arm_action.get_detections_client import GetDetectionsClient
from arm_action.get_circle_coords_client import GetCircleCoordsClient
from arm_action.set_target_pose_action_client import SetTargetPoseActionClient
from arm_action.set_gpio_combination_action_client import SetGPIOCombinationActionClient
from arm_action.rl_trajectory import RLStrategy
from geometry_msgs.msg import Pose
from functools import partial
from pipettingrobot_interfaces.srv import (
    SetActiveTube,
    SetCircleCount,
    SetTubeVisualState,
    StartPipetting,
)
from pipettingrobot_interfaces.msg import PipettingTaskStatus
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
import time
from collections import Counter

class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')
        
        self.declare_parameter('planning_method', 'rl')  # 'script' 或 'rl'
        
        # 默认权重路径
        self.declare_parameter('rl_checkpoint', '/home/robot/pipetbot_managerbased_rl/logs/skrl/pipetbot_td3/2026-03-15_13-52-45_td3_torch/checkpoints/best_agent.pt')
        # 默认配置路径
        self.declare_parameter('rl_config_path', '/home/robot/pipetbot_managerbased_rl/source/pipetbot_managerbased_rl/pipetbot_managerbased_rl/tasks/manager_based/pipetbot_managerbased_rl/agents/skrl_td3_cfg.yaml')
        self.declare_parameter('enable_terminal_input', False)
        
        self.planning_method = self.get_parameter('planning_method').value
        ckpt_path = self.get_parameter('rl_checkpoint').value
        config_path = self.get_parameter('rl_config_path').value # 获取 yaml 路径
        self.enable_terminal_input = self.get_parameter('enable_terminal_input').value
        self.rl_checkpoint_path = ckpt_path
        self.rl_config_file = config_path

        # === 2. 初始化 RL 策略 ===
        self.rl_strategy = None
        self.rl_running = False
        self.rl_target_pose = None
        self.rl_callback = None
        self.rl_timer = self.create_timer(0.1, self.rl_control_loop)
        self.rl_timer.cancel()
        if self.planning_method == 'rl':
            self.ensure_rl_strategy()
        elif self.planning_method == 'quintic':
            self.get_logger().info("使用 IK + 五次样条 路径规划模式")
        else:
            self.get_logger().info("使用 传统脚本 路径规划模式")
        # 声明参数
        self.declare_parameter('gpio_combination_id_final', 1)  # 末端执行器恢复
        self.declare_parameter('test_tube_rack_correction_x', -0.16)
        self.declare_parameter('test_tube_rack_correction_y', 0.0)
        self.declare_parameter('cup_work_correction_x', -0.14)
        self.declare_parameter('cup_work_correction_y', 0.1)
        self.declare_parameter('observe_correction_z', 0.41)
        self.declare_parameter('cup_work_correction_z', 0.24)
        self.declare_parameter('circle_work_correction_x', -0.112)
        self.declare_parameter('circle_work_correction_y', 0.003)
        self.declare_parameter('circle_work_correction_z', 0.228)

        # 获取参数值
        self.gpio_combination_id_final = self.get_parameter('gpio_combination_id_final').get_parameter_value().integer_value
        self.test_tube_rack_correction_x = self.get_parameter(
            'test_tube_rack_correction_x'
        ).get_parameter_value().double_value
        self.test_tube_rack_correction_y = self.get_parameter(
            'test_tube_rack_correction_y'
        ).get_parameter_value().double_value
        self.cup_work_correction_x = self.get_parameter('cup_work_correction_x').get_parameter_value().double_value
        self.cup_work_correction_y = self.get_parameter('cup_work_correction_y').get_parameter_value().double_value
        self.cup_work_correction_z = self.get_parameter('cup_work_correction_z').get_parameter_value().double_value
        self.observe_correction_z = self.get_parameter('observe_correction_z').get_parameter_value().double_value
        self.circle_work_correction_x = self.get_parameter('circle_work_correction_x').get_parameter_value().double_value
        self.circle_work_correction_y = self.get_parameter('circle_work_correction_y').get_parameter_value().double_value
        self.circle_work_correction_z = self.get_parameter('circle_work_correction_z').get_parameter_value().double_value
        

        # 初始化客户端
        self.set_target_pose_client = SetTargetPoseActionClient(self, '/set_target_pose')
        self.quintic_pose_client = SetTargetPoseActionClient(self, '/set_target_pose_quintic')
        self.get_circle_coords_client = GetCircleCoordsClient(self)
        self.get_detections_client = GetDetectionsClient(self)
        self.set_gpio_combination_client = SetGPIOCombinationActionClient(self)
        
        # 初始化变量
        self.circle_coords = []
        self.cup_coords = None
        self.expected_circle_count = 50
        self.current_loop = 0
        self.current_step = 0
        self.total_steps = 10
        self.dispensing = False  # 标志位，防止多次启动
        self.marker_id_counter = 0
        self.last_error_message = ''
        self.current_phase = 'idle'

        # 创建一个 Marker 发布器
        self.marker_publisher = self.create_publisher(Marker, '/target_coordinates', 10)
        self.reset_scene_client = self.create_client(Trigger, '/pipetting_ui/reset_scene')
        self.set_active_tube_client = self.create_client(SetActiveTube, '/pipetting_ui/set_active_tube')
        self.set_tube_visual_state_client = self.create_client(
            SetTubeVisualState, '/pipetting_ui/set_tube_visual_state'
        )
        self.task_status_publisher = self.create_publisher(
            PipettingTaskStatus, '/pipetting/task_status', 10
        )

        self.create_service(StartPipetting, '/pipetting/start', self.handle_start_pipetting)
        self.create_service(SetCircleCount, '/pipetting/set_circle_count', self.handle_set_circle_count)

        if self.enable_terminal_input:
            self.get_logger().info('终端输入模式已启用。')
            import threading

            self.input_thread = threading.Thread(target=self.listen_for_input, daemon=True)
            self.input_thread.start()

        self.get_logger().info("主节点已初始化。")
        self.publish_task_status(message='ready')

    def listen_for_input(self):
        while rclpy.ok():
            try:
                user_input = input("请输入 'y' 开始移液流程，或输入 'circle_count' 设置试管数量：").strip().lower()
                if user_input == 'y':
                    if not self.dispensing:
                        self.dispensing = True
                        self.get_logger().info("收到 'y' 指令，开始移液流程。")
                        self.begin_pipetting(reset_visual_state=True)
                    else:
                        self.get_logger().warn("分配流程已在执行中。")
                elif user_input.startswith('circle_count'):
                    count_str = user_input.split()[-1]
                    try:
                        self.expected_circle_count = int(count_str)
                        self.get_logger().info(f"设置试管数量为: {self.expected_circle_count}")
                    except ValueError:
                        self.get_logger().error("无效的试管数量，输入无效！")
                else:
                    self.get_logger().warn("无效指令，请输入 'y' 来开始移液流程，或输入 'circle_count <number>' 来设置试管数量。")
            except EOFError:
                # 处理输入流关闭的情况
                break

    def handle_start_pipetting(self, request, response):
        if request.expected_circle_count > 0:
            self.expected_circle_count = request.expected_circle_count

        if request.planning_method:
            success, message = self.configure_planning_method(request.planning_method)
            if not success:
                response.success = False
                response.message = message
                return response

        started, message = self.begin_pipetting(request.reset_visual_state)
        response.success = started
        response.message = message
        return response

    def handle_set_circle_count(self, request, response):
        if request.expected_circle_count <= 0:
            response.success = False
            response.message = 'expected_circle_count must be > 0'
            return response

        self.expected_circle_count = request.expected_circle_count
        self.get_logger().info(f"设置试管数量为: {self.expected_circle_count}")
        response.success = True
        response.message = f'circle_count set to {self.expected_circle_count}'
        return response

    def begin_pipetting(self, reset_visual_state=False):
        if self.dispensing:
            message = '分配流程已在执行中。'
            self.get_logger().warn(message)
            return False, message

        self.dispensing = True
        self.current_loop = 0
        self.current_step = 0
        self.last_error_message = ''
        self.current_phase = 'starting'
        self.circle_coords = []
        self.cup_coords = None

        if reset_visual_state:
            self.reset_visual_scene()

        self.clear_active_tube_visual()
        self.get_logger().info(
            f"开始移液流程，试管数量: {self.expected_circle_count}"
        )
        self.publish_task_status(message='pipetting started')
        self.send_initial_pose()
        return True, 'pipetting started'

    def ensure_rl_strategy(self):
        if self.rl_strategy is None:
            self.get_logger().info("使用 RL 路径规划模式")
            self.get_logger().info(f"Model: {self.rl_checkpoint_path}")
            self.get_logger().info(f"Config: {self.rl_config_file}")
            self.rl_strategy = RLStrategy(self, self.rl_checkpoint_path, self.rl_config_file)

    def configure_planning_method(self, method):
        normalized = method.strip().lower()
        if normalized not in ('rl', 'script', 'quintic'):
            return False, f"unsupported planning method: {method}"

        if normalized == self.planning_method:
            return True, f'planning method already {normalized}'

        if normalized == 'rl':
            self.ensure_rl_strategy()
            self.planning_method = 'rl'
            self.get_logger().info('已切换到 RL 路径规划模式')
            self.publish_task_status(message='planning method: rl')
            return True, 'planning method set to rl'

        if normalized == 'quintic':
            self.planning_method = 'quintic'
            self.get_logger().info('已切换到 IK + 五次样条路径规划模式')
            self.publish_task_status(message='planning method: quintic')
            return True, 'planning method set to quintic'

        self.planning_method = 'script'
        self.get_logger().info('已切换到传统脚本路径规划模式')
        self.publish_task_status(message='planning method: script')
        return True, 'planning method set to script'

    def send_initial_pose(self):
        # 获取 'TestTubeRack' 坐标并发送初始位姿
        self.current_phase = 'detecting_rack'
        self.current_step = 0
        self.publish_task_status(message='detecting rack')
        self.get_logger().info("请求获取 'TestTubeRack' 的位置...")
        self.get_target_coords("TestTubeRack", self.on_get_test_tube_rack_coords)
    
    def get_target_coords(self, target_name, callback):
        """通用的函数，通过目标名称获取目标坐标"""
        self.get_detections_client.send_request(target_name, callback)

    def on_get_test_tube_rack_coords(self, success, test_tube_rack_coords, message):
        if success:
            self.get_logger().info(f"获取 'TestTubeRack' 坐标成功: {message}")
            
            # 在RViz中显示目标坐标
            self.publish_marker(test_tube_rack_coords, "TestTubeRack")

            # 加上默认位姿和安全Z坐标
            pose = Pose()
            pose.position.x = test_tube_rack_coords.x + self.test_tube_rack_correction_x
            pose.position.y = test_tube_rack_coords.y + self.test_tube_rack_correction_y
            pose.position.z = test_tube_rack_coords.z + self.observe_correction_z  
            pose.orientation.x = 0.5
            pose.orientation.y = 0.5
            pose.orientation.z = 0.5
            pose.orientation.w = 0.5  # 单位四元数
            print("Initial Pose:", pose)
            self.move_to_pose(pose, self.on_initial_pose_result)
        else:
            self.get_logger().error(f"获取试管架坐标失败: {message}")
            self.fail_current_run(message)

    def on_initial_pose_result(self, success, message):
        if success:
            self.get_logger().info(f"试管检测位姿发送成功: {message}")
            self.get_circle_coords()
        else:
            self.get_logger().error(f"试管检测发送失败: {message}")
            self.fail_current_run(message)

    def get_circle_coords(self):
        self.current_phase = 'detecting_tubes'
        self.publish_task_status(message='detecting tube openings')
        self.get_logger().info("请求获取所有试管口坐标...")
        self.get_circle_coords_client.send_goal(
            expected_count=self.expected_circle_count,  
            callback=self.on_get_circle_coords_result
        )

    def on_get_circle_coords_result(self, success, circle_coords, message):
        if success:
            self.get_logger().info(f"获取试管口坐标成功: {message}")
            self.circle_coords = circle_coords
            # === 提取所有z值并保留两位小数 ===
            z_rounded_list = [round(circle.z, 2) for circle in self.circle_coords]

            # === 统计众数 ===
            z_counter = Counter(z_rounded_list)
            most_common_z, count = z_counter.most_common(1)[0]
            self.get_logger().info(f"试管口Z坐标众数为: {most_common_z}（出现 {count} 次）")
            # === 替换所有 z 坐标为众数 ===
            for circle in self.circle_coords:
                circle.z = most_common_z
            # 显示它们的坐标和名字
            for idx, circle in enumerate(self.circle_coords):
                # 坐标合法性检查
                if any([
                    not isinstance(circle.x, float),
                    not isinstance(circle.y, float),
                    not isinstance(circle.z, float),
                    abs(circle.x) > 10, abs(circle.y) > 10, abs(circle.z) > 10  # 防止误差值
                ]):
                    self.get_logger().warn(f"试管口 {idx+1} 坐标非法，跳过显示。")
                    continue

                name = f"T_{idx+1}"
                self.publish_marker(circle, name)
                time.sleep(0.2)

            self.get_cup_coords()
        else:
            self.get_logger().error(f"获取试管口坐标失败: {message}")
            self.fail_current_run(message)
    def get_cup_coords(self):
        # 使用通用的函数获取坐标
        self.current_phase = 'detecting_beaker'
        self.publish_task_status(message='detecting beaker')
        self.get_logger().info("请求获取 'Beaker' 的坐标...")
        self.get_target_coords("Beaker", self.on_get_cup_coords_result)

    def on_get_cup_coords_result(self, success, cup_coords, message):
        if success:
            self.get_logger().info(f"获取待分配容器坐标成功: {message}")
            self.cup_coords = cup_coords
            # 在RViz中显示目标坐标
            self.publish_marker(cup_coords, "Beaker")
            self.start_dispensing_loop()
        else:
            self.get_logger().error(f"获取待分配容器坐标失败: {message}")
            self.fail_current_run(message)

    def publish_marker(self, coords, target_name):
        """
        发布一个目标位置的球体 + 名称文本到 RViz
        """
        marker_id_base = self.marker_id_counter
        self.marker_id_counter += 2  # 预留两个id：一个给球体，一个给文本

        # === 小球 Marker ===
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "base_link"
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "circle_sphere"
        sphere_marker.id = marker_id_base
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position.x = coords.x
        sphere_marker.pose.position.y = coords.y
        sphere_marker.pose.position.z = coords.z
        sphere_marker.scale.x = 0.01
        sphere_marker.scale.y = 0.01
        sphere_marker.scale.z = 0.01
        sphere_marker.color.a = 1.0
        sphere_marker.color.r = 0.0
        sphere_marker.color.g = 0.0
        sphere_marker.color.b = 1.0  # 蓝色
        sphere_marker.lifetime.sec = 0

        self.marker_publisher.publish(sphere_marker)

        # === 文本 Marker ===
        text_marker = Marker()
        text_marker.header.frame_id = "base_link"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "circle_text"
        text_marker.id = marker_id_base + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = coords.x
        text_marker.pose.position.y = coords.y
        text_marker.pose.position.z = coords.z + 0.01  # 抬高一点
        text_marker.scale.z = 0.01
        text_marker.color.a = 1.0
        text_marker.color.r = 0.0
        text_marker.color.g = 1.0
        text_marker.color.b = 0.0
        text_marker.text = target_name
        text_marker.lifetime.sec = 0

        self.marker_publisher.publish(text_marker)

        self.get_logger().info(f"[RViz] 显示: {target_name} | id: {marker_id_base}, {marker_id_base + 1}")

    def move_to_pose(self, pose, callback):
        """
        根据规划方法，选择发送 Action 还是 启动 RL 推理
        """
        if self.planning_method == 'script':
            # 传统方式：发送一次 Goal
            self.set_target_pose_client.send_goal(pose, callback)

        elif self.planning_method == 'quintic':
            self.get_logger().info("启动 IK + 五次样条轨迹规划...")
            self.quintic_pose_client.send_goal(pose, callback)
        
        elif self.planning_method == 'rl':
            # RL 方式：设置目标并启动定时器循环推理
            self.get_logger().info("启动 RL 推理闭环控制...")
            self.rl_target_pose = pose
            self.rl_callback = callback
            self.rl_running = True
            self.rl_timer.reset()

    def rl_control_loop(self):
        """
        RL 闭环控制定时器函数 (10Hz)
        """
        if not self.rl_running:
            return

        self.rl_strategy.predict_and_step(self.rl_target_pose)

        is_arrived, dist = self.rl_strategy.check_success(self.rl_target_pose, threshold=0.01)
        
        if is_arrived:
            self.get_logger().info(f"RL 已到达目标点 (误差: {dist:.4f}m)")
            self.rl_running = False
            self.rl_timer.cancel()
            # 触发回调以进行下一步骤
            if self.rl_callback:
                # 模拟 ActionClient 的返回格式 (success, message)
                self.rl_callback(True, "RL_SUCCESS")

    def start_dispensing_loop(self):
        self.get_logger().info(f"开始分配试剂，试管数量: {self.expected_circle_count}")
        self.current_loop = 0
        self.current_phase = 'dispensing'
        self.current_step = 0
        self.reset_detected_tube_liquid_state()
        self.publish_task_status(message='dispensing started')
        self.dispense_loop()

    def dispense_loop(self):
        if self.current_loop >= self.expected_circle_count:
            self.get_logger().info("分配试剂完成。末端执行器恢复。")
            self.send_final_gpio()
            return

        i = self.current_loop
        self.get_logger().info(f"开始分配试剂到第 {i+1} 个试管。")
        self.highlight_tube_visual(i)
        self.publish_task_status(message=f'processing tube {i+1}')
        # 执行单次移液流程
        self.execute_dispense_steps(i)

    def execute_dispense_steps(self, i):
        # 定义各步骤的回调链
        self.step1_send_pose_above_cup(i)

    # 步骤1: SetTargetPoseActionClient发送cup上方的位姿
    def step1_send_pose_above_cup(self, i):
        self.set_step(1, i, 'moving above beaker')
        self.get_logger().info(f"步骤1: 发送待分配容器上方位姿...")
        pose = Pose()
        pose.position.x = self.cup_coords.x + self.cup_work_correction_x 
        pose.position.y = self.cup_coords.y + self.cup_work_correction_y
        pose.position.z = self.cup_coords.z + self.observe_correction_z
        pose.orientation.x = 0.5
        pose.orientation.y = 0.5
        pose.orientation.z = 0.5
        pose.orientation.w = 0.5
        self.move_to_pose(pose, partial(self.on_step1_result, i))

    def on_step1_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤1成功: {message}")
            self.step2_send_gpio_3(i)
        else:
            self.get_logger().error(f"步骤1失败: {message}")
            self.fail_current_run(message)

    # 步骤2: SetGPIOCombinationActionClient发送“3”
    def step2_send_gpio_3(self, i):
        self.set_step(2, i, 'aspirator stop point 1')
        self.get_logger().info(f"步骤2: 移液器到达第一停止点")
        self.set_gpio_combination_client.send_goal(3, partial(self.on_step2_result, i))

    def on_step2_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤2成功: {message}")
            self.step3_send_pose_cup_working_point(i)
        else:
            self.get_logger().error(f"步骤2失败: {message}")
            self.fail_current_run(message)

    # 步骤3: SetTargetPoseActionClient发送吸液工作点的位姿
    def step3_send_pose_cup_working_point(self, i):
        self.set_step(3, i, 'moving to beaker working point')
        self.get_logger().info(f"步骤3: 发送吸液工作点位姿...")
        pose = Pose()
        pose.position.x = self.cup_coords.x + self.cup_work_correction_x
        pose.position.y = self.cup_coords.y + self.cup_work_correction_y
        pose.position.z = self.cup_coords.z + self.cup_work_correction_z
        pose.orientation.x = 0.5
        pose.orientation.y = 0.5
        pose.orientation.z = 0.5
        pose.orientation.w = 0.5
        self.move_to_pose(pose, partial(self.on_step3_result, i))

    def on_step3_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤3成功: {message}")
            self.step4_send_gpio_2(i)
        else:
            self.get_logger().error(f"步骤3失败: {message}")
            self.fail_current_run(message)

    # 步骤4: SetGPIOCombinationActionClient发送“2”
    def step4_send_gpio_2(self, i):
        self.set_step(4, i, 'aspirating liquid')
        self.get_logger().info(f"步骤4: 完成吸液")
        self.set_gpio_combination_client.send_goal(2, partial(self.on_step4_result, i))

    def on_step4_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤4成功: {message}")
            self.step5_send_pose_above_cup(i)
        else:
            self.get_logger().error(f"步骤4失败: {message}")
            self.fail_current_run(message)

    # 步骤5: SetTargetPoseActionClient再次发送待分配容器上方的位姿
    def step5_send_pose_above_cup(self, i):
        self.set_step(5, i, 'returning above beaker')
        self.get_logger().info(f"步骤5: 返回容器上方...")
        pose = Pose()
        pose.position.x = self.cup_coords.x + self.cup_work_correction_x
        pose.position.y = self.cup_coords.y + self.cup_work_correction_y
        pose.position.z = self.cup_coords.z + self.observe_correction_z
        pose.orientation.x = 0.5
        pose.orientation.y = 0.5
        pose.orientation.z = 0.5
        pose.orientation.w = 0.5
        self.move_to_pose(pose, partial(self.on_step5_result, i))

    def on_step5_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤5成功: {message}")
            self.step6_send_pose_above_circle(i)
        else:
            self.get_logger().error(f"步骤5失败: {message}")
            self.fail_current_run(message)

    # 步骤6: SetTargetPoseActionClient发送试管i上方的位姿
    def step6_send_pose_above_circle(self, i):
        self.set_step(6, i, f'moving above tube {i+1}')
        self.get_logger().info(f"步骤6: 发送第 {i+1} 个试管上方位姿...")
        if i >= len(self.circle_coords):
            self.get_logger().error(f"步骤6: 第 {i+1} 个试管的坐标不存在。")
            self.fail_current_run(f"tube {i+1} coordinates missing")
            return
        circle = self.circle_coords[i]
        pose = Pose()
        pose.position.x = circle.x + self.circle_work_correction_x
        pose.position.y = circle.y + self.circle_work_correction_y
        pose.position.z = circle.z + self.observe_correction_z
        pose.orientation.x = 0.5
        pose.orientation.y = 0.5
        pose.orientation.z = 0.5
        pose.orientation.w = 0.5
        self.move_to_pose(pose, partial(self.on_step6_result, i))

    def on_step6_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤6成功: {message}")
            self.step7_send_pose_circle_working_point(i)
        else:
            self.get_logger().error(f"步骤6失败: {message}")
            self.fail_current_run(message)

    # 步骤7: SetTargetPoseActionClient发送试管i工作点的位姿
    def step7_send_pose_circle_working_point(self, i):
        self.set_step(7, i, f'moving to tube {i+1} dispense point')
        self.get_logger().info(f"步骤7: 发送第 {i+1} 次排液位姿...")
        circle = self.circle_coords[i]
        pose = Pose()
        pose.position.x = circle.x + self.circle_work_correction_x
        pose.position.y = circle.y + self.circle_work_correction_y
        pose.position.z = circle.z + self.circle_work_correction_z
        pose.orientation.x = 0.5
        pose.orientation.y = 0.5
        pose.orientation.z = 0.5
        pose.orientation.w = 0.5
        self.move_to_pose(pose, partial(self.on_step7_result, i))

    def on_step7_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤7成功: {message}")
            self.step8_send_gpio_4(i)
        else:
            self.get_logger().error(f"步骤7失败: {message}")
            self.fail_current_run(message)

    # 步骤8: SetGPIOCombinationActionClient发送“4”
    def step8_send_gpio_4(self, i):
        self.set_step(8, i, f'dispensing into tube {i+1}')
        self.get_logger().info(f"步骤8: 开始排液")
        self.set_gpio_combination_client.send_goal(4, partial(self.on_step8_result, i))

    def on_step8_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤8成功: {message}")
            self.update_tube_visual_state(i, True, 1.0)
            self.step9_send_pose_above_circle(i)
        else:
            self.get_logger().error(f"步骤8失败: {message}")
            self.fail_current_run(message)

    # 步骤9: SetTargetPoseActionClient再次发送试管i上方的位姿
    def step9_send_pose_above_circle(self, i):
        self.set_step(9, i, f'returning above tube {i+1}')
        self.get_logger().info(f"步骤9: 返回第 {i+1} 个试管口上方位姿...")
        circle = self.circle_coords[i]
        pose = Pose()
        pose.position.x = circle.x + self.circle_work_correction_x
        pose.position.y = circle.y + self.circle_work_correction_y
        pose.position.z = circle.z + self.observe_correction_z
        pose.orientation.x = 0.5
        pose.orientation.y = 0.5
        pose.orientation.z = 0.5
        pose.orientation.w = 0.5
        self.move_to_pose(pose, partial(self.on_step9_result, i))

    def on_step9_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤9成功: {message}")
            self.step10_send_gpio_2(i)
        else:
            self.get_logger().error(f"步骤9失败: {message}")
            self.fail_current_run(message)

    # 步骤10: SetGPIOCombinationActionClient发送“2”
    def step10_send_gpio_2(self, i):
        self.set_step(10, i, f'resetting pipette after tube {i+1}')
        self.get_logger().info(f"步骤10: 移液器恢复...")
        self.set_gpio_combination_client.send_goal(2, partial(self.on_step10_result, i))

    def on_step10_result(self, i, success, message):
        if success:
            self.get_logger().info(f"步骤10成功: {message}")
            # 进入下一个循环
            self.current_loop += 1
            self.current_step = 0
            if self.current_loop < self.expected_circle_count:
                self.highlight_tube_visual(self.current_loop)
            else:
                self.clear_active_tube_visual()
            self.publish_task_status(message=f'tube {i+1} complete')
            self.dispense_loop()
        else:
            self.get_logger().error(f"步骤10失败: {message}")
            self.fail_current_run(message)

    def send_final_gpio(self):
        self.get_logger().info("末端执行器恢复...")
        self.set_gpio_combination_client.send_goal(self.gpio_combination_id_final, self.on_final_gpio_result)

    def on_final_gpio_result(self, success, message):
        if success:
            self.get_logger().info(f"末端执行器恢复成功: {message}")
            self.current_phase = 'finished'
            self.publish_task_status(message='pipetting finished')
        else:
            self.get_logger().error(f"末端执行器恢复失败: {message}")
            self.current_phase = 'error'
            self.last_error_message = message
            self.publish_task_status(message='final gpio failed')
        self.finish_current_run()

    def reset_visual_scene(self):
        self.call_trigger_service(self.reset_scene_client, 'reset_scene')

    def highlight_tube_visual(self, tube_index):
        request = SetActiveTube.Request()
        request.tube_index = tube_index
        self.call_service_async(
            self.set_active_tube_client,
            request,
            f'set_active_tube[{tube_index}]',
        )

    def clear_active_tube_visual(self):
        request = SetActiveTube.Request()
        request.tube_index = -1
        self.call_service_async(self.set_active_tube_client, request, 'clear_active_tube')

    def update_tube_visual_state(self, tube_index, filled, fill_ratio):
        request = SetTubeVisualState.Request()
        request.tube_index = tube_index
        request.filled = filled
        request.fill_ratio = float(fill_ratio)
        self.call_service_async(
            self.set_tube_visual_state_client,
            request,
            f'set_tube_visual_state[{tube_index}]',
        )

    def reset_detected_tube_liquid_state(self):
        for index in range(len(self.circle_coords)):
            self.update_tube_visual_state(index, False, 0.0)

    def call_trigger_service(self, client, description):
        request = Trigger.Request()
        self.call_service_async(client, request, description)

    def call_service_async(self, client, request, description):
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=0.2):
                self.get_logger().warn(f'{description}: service unavailable')
                return
        future = client.call_async(request)
        future.add_done_callback(partial(self.on_service_response, description))

    def on_service_response(self, description, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f'{description}: {exc}')
            return

        success = getattr(response, 'success', True)
        message = getattr(response, 'message', '')
        log_fn = self.get_logger().info if success else self.get_logger().warn
        log_fn(f'{description}: {message}')

    def set_step(self, step_index, tube_index, message):
        self.current_step = step_index
        self.current_phase = 'dispensing'
        self.publish_task_status(
            current_tube_index=tube_index,
            message=message,
        )

    def publish_task_status(self, current_tube_index=None, message=''):
        status = PipettingTaskStatus()
        status.is_running = self.dispensing
        status.phase = self.current_phase
        status.current_tube_index = (
            current_tube_index if current_tube_index is not None else self.current_loop
        )
        status.total_tubes = self.expected_circle_count
        status.completed_tubes = min(self.current_loop, self.expected_circle_count)
        status.current_step = self.current_step
        status.total_steps = self.total_steps
        progress = 0.0
        if self.expected_circle_count > 0:
            progress = (
                float(status.completed_tubes)
                + float(self.current_step) / float(self.total_steps)
            ) / float(self.expected_circle_count)
        status.progress = float(max(0.0, min(1.0, progress)))
        status.message = message
        status.last_error = self.last_error_message
        self.task_status_publisher.publish(status)

    def fail_current_run(self, message):
        self.current_phase = 'error'
        self.last_error_message = message
        self.get_logger().error(f'移液流程失败: {message}')
        self.publish_task_status(message=message)
        self.finish_current_run()

    def finish_current_run(self):
        self.get_logger().info("移液控制流程结束，节点保持待机。")
        self.dispensing = False
        self.rl_running = False if hasattr(self, 'rl_running') else False
        if hasattr(self, 'rl_timer'):
            self.rl_timer.cancel()
        self.clear_active_tube_visual()
        self.publish_task_status(message=self.current_phase)

def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

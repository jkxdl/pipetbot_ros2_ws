import os
import yaml  
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, TimerAction

def load_transform_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    transform = data['static_transform_publisher']['transform']
    translation = transform['translation']
    rotation = transform['rotation']
    frame_id = data['static_transform_publisher']['frame_id']
    child_frame_id = data['static_transform_publisher']['child_frame_id']
    
    # 返回命令行参数列表
    args = [
        str(translation['x']),
        str(translation['y']),
        str(translation['z']),
        str(rotation['x']),
        str(rotation['y']),
        str(rotation['z']),
        str(rotation['w']),
        frame_id,
        child_frame_id
    ]
    return args


def generate_launch_description():
    # 定义参数
    realsense_align_depth_enable = LaunchConfiguration('align_depth_enable', default='true')
    aubo_use_fake_hardware = LaunchConfiguration('use_fake_hardware', default='false')
    aubo_type = LaunchConfiguration('aubo_type', default='aubo_C5')
    aubo_robot_ip = LaunchConfiguration('robot_ip', default='192.168.1.200')
    aubo_use_planning = LaunchConfiguration('use_planning', default='true')
    stereo_target_id_path = LaunchConfiguration(
        'stereo_target_id_path', default='pci-0000:00:14.0-usb-0:5.2:1.0'
    )
    stereo_width = LaunchConfiguration('stereo_width', default='1280')
    stereo_height = LaunchConfiguration('stereo_height', default='360')
    stereo_fps = LaunchConfiguration('stereo_fps', default='5.0')
    stereo_fourcc = LaunchConfiguration('stereo_fourcc', default='MJPG')
    circle_pub_start_delay = LaunchConfiguration('circle_pub_start_delay', default='4.0')
    realsense_color_profile = LaunchConfiguration('realsense_color_profile', default='640x480x15')
    realsense_depth_profile = LaunchConfiguration('realsense_depth_profile', default='640x480x15')
    aubo_kinematics_file = LaunchConfiguration(
        'aubo_kinematics_file', default='trac_ik_kinematics.yaml'
    )

    # 获取包的共享目录路径
    pkg_share = get_package_share_directory('pipettingrobot_launch')
    arm_action_share = get_package_share_directory('arm_action')

    # 定义 YAML 文件路径
    eyeinhand_yaml = os.path.join(pkg_share, 'config', 'eyeinhand_transform.yaml')
    eyetohand_yaml = os.path.join(pkg_share, 'config', 'eyetohand_transform.yaml')
    pipetting_targets_yaml = os.path.join(
        arm_action_share, 'config', 'pipetting_targets.yaml'
    )
    
    # 检查 YAML 文件是否存在
    if not os.path.exists(eyeinhand_yaml):
        raise FileNotFoundError(f"YAML 文件未找到: {eyeinhand_yaml}")
    if not os.path.exists(eyetohand_yaml):
        raise FileNotFoundError(f"YAML 文件未找到: {eyetohand_yaml}")
    if not os.path.exists(pipetting_targets_yaml):
        raise FileNotFoundError(f"YAML 文件未找到: {pipetting_targets_yaml}")
    
    # 加载 transforms
    eyeinhand_args = load_transform_from_yaml(eyeinhand_yaml)
    eyetohand_args = load_transform_from_yaml(eyetohand_yaml)
    
    # 1. 启动 Realsense2 Camera Launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': realsense_align_depth_enable,
            'rgb_camera.color_profile': realsense_color_profile,
            'depth_module.depth_profile': realsense_depth_profile,
            'depth_module.infra_profile': realsense_depth_profile,
            'pointcloud.enable': 'true',
        }.items()
    )

    # 2. 启动 target_detection 节点：yolov11_d435i
    yolov11_node = Node(
        package='target_detection',
        executable='yolov11_d435i',
        name='yolov11_d435i',
        output='screen'
    )

    # 3. 启动 circle_pub 节点
    circle_pub_node = Node(
        package='circle_detection',
        executable='circle_pub',
        name='circle_pub',
        output='screen',
        parameters=[
            {
                'target_id_path': stereo_target_id_path,
                'frame_width': stereo_width,
                'frame_height': stereo_height,
                'fps': stereo_fps,
                'fourcc': stereo_fourcc,
            }
        ],
    )

    # 4. 启动 circleAll_detector 节点
    circleAll_detector_node = Node(
        package='circle_detection',
        executable='circleAll_detector',
        name='circleAll_detector',
        output='screen',
        parameters=[{'expected_circle_count': 50}],
        #arguments=['--ros-args', '--log-level', 'debug']
    )

    # 5. 启动 Aubo Bringup Launch
    aubo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('aubo_bringup'),
                'launch',
                'aubo.launch.py'
            )
        ),
        launch_arguments={
            'use_fake_hardware': aubo_use_fake_hardware,
            'aubo_type': aubo_type,
            'robot_ip': aubo_robot_ip,
            'use_planning': aubo_use_planning,
            'launch_rviz': 'false',
            'kinematics_file': aubo_kinematics_file,
        }.items()
    )

    # 6. 启动 arm_action 节点：action_execute
    arm_action_node = Node(
        package='arm_action',
        executable='action_execute',
        name='action_execute',
        output='screen',
        parameters=[{'use_path_constraints': False}]
    )

    quintic_spline_node = Node(
        package='arm_action',
        executable='quintic_spline',
        name='quintic_spline',
        output='screen',
        parameters=[
            {
                'use_ik_constraints': False,
                'ik_avoid_collisions': False,
            }
        ]
    )

    planning_scene_initializer_node = Node(
        package='arm_action',
        executable='planning_scene_initializer',
        name='planning_scene_initializer',
        output='screen',
        parameters=[
            {
                'publish_once_delay': 4.0,
                'add_ground': True,
                'add_wall': True,
            }
        ]
    )

    # 7. 启动 effector_controller 节点：effector_controller
    effector_controller_node = Node(
        package='effector_controller',
        executable='effector_controller',
        name='effector_controller',
        output='screen'
    )

    # 8. 创建 static_transform_publisher 节点
    static_tf_eyeinhand_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_eyeinhand',
        output='screen',
        arguments=eyeinhand_args
    )

    static_tf_eyetohand_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_eyetohand',
        output='screen',
        arguments=eyetohand_args
    )

    obstacle_feature_node = Node(
        package='obstacle_detection',  
        executable='ObsaclePointSeg_yolo11_pub',  
        name='obstacle_feature_extractor',
        output='screen',
        parameters=[{
            'ray_count': 36,
            'max_dist': 2.0,
            'conf_threshold': 0.3,
            'base_frame': 'base_link'
        }]
    )
    delayed_obstacle_node = TimerAction(
        period=15.0,
        actions=[obstacle_feature_node]
    )

    # 定义启动顺序
    ld = LaunchDescription()

    # 1. 启动 Realsense2 Camera Launch
    ld.add_action(realsense_launch)
    

    # 2. 启动 yolov11_d435i 节点
    ld.add_action(yolov11_node)

    # 3. 延后启动 circle_pub，减小与 RealSense 初始化抢占
    ld.add_action(
        TimerAction(
            period=circle_pub_start_delay,
            actions=[circle_pub_node],
        )
    )

    # 4. 启动 circleAll_detector 节点
    ld.add_action(circleAll_detector_node)

    # 5. 启动 Aubo Bringup Launch
    ld.add_action(aubo_launch)

    # 6. 启动 arm_action 节点
    ld.add_action(planning_scene_initializer_node)
    ld.add_action(arm_action_node)
    ld.add_action(quintic_spline_node)

    # 7. 启动 effector_controller 节点
    ld.add_action(effector_controller_node)

    # 8. 启动 static_transform_publisher 节点
    ld.add_action(static_tf_eyeinhand_node)
    ld.add_action(static_tf_eyetohand_node)

    ld.add_action(delayed_obstacle_node)

    return ld

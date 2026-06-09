import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    base_launch_share = get_package_share_directory('pipettingrobot_launch')
    base_launch_path = os.path.join(base_launch_share, 'launch', 'pipettingrobot_launch.launch.py')
    arm_action_share = get_package_share_directory('arm_action')
    pipetting_targets_yaml = os.path.join(
        arm_action_share, 'config', 'pipetting_targets.yaml'
    )

    start_base_system = LaunchConfiguration('start_base_system')
    start_panel = LaunchConfiguration('start_panel')
    planning_method = LaunchConfiguration('planning_method')
    aubo_type = LaunchConfiguration('aubo_type')
    aubo_kinematics_file = LaunchConfiguration('aubo_kinematics_file')
    rack_mesh = LaunchConfiguration('rack_mesh')
    beaker_mesh = LaunchConfiguration('beaker_mesh')
    tube_mesh = LaunchConfiguration('tube_mesh')
    robot_base_frame = LaunchConfiguration('robot_base_frame')
    top_camera_topic = LaunchConfiguration('top_camera_topic')
    stereo_left_topic = LaunchConfiguration('stereo_left_topic')
    stereo_right_topic = LaunchConfiguration('stereo_right_topic')

    return LaunchDescription(
        [
            DeclareLaunchArgument('start_base_system', default_value='true'),
            DeclareLaunchArgument('start_panel', default_value='true'),
            DeclareLaunchArgument('planning_method', default_value='rl'),
            DeclareLaunchArgument('aubo_type', default_value='aubo_C5'),
            DeclareLaunchArgument(
                'aubo_kinematics_file', default_value='trac_ik_kinematics.yaml'
            ),
            DeclareLaunchArgument(
                'rack_mesh',
                default_value='package://pipettingrobot_gui/meshes/TestTubeRack.dae',
            ),
            DeclareLaunchArgument(
                'beaker_mesh',
                default_value='package://pipettingrobot_gui/meshes/Beaker.dae',
            ),
            DeclareLaunchArgument(
                'tube_mesh',
                default_value='package://pipettingrobot_gui/meshes/TestTube.dae',
            ),
            DeclareLaunchArgument('robot_base_frame', default_value='base_link'),
            DeclareLaunchArgument('top_camera_topic', default_value='/camera/camera/color/image_raw'),
            DeclareLaunchArgument('stereo_left_topic', default_value='/stereo/left/image_raw'),
            DeclareLaunchArgument('stereo_right_topic', default_value='/stereo/right/image_raw'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch_path),
                launch_arguments={
                    'aubo_kinematics_file': aubo_kinematics_file,
                }.items(),
                condition=IfCondition(start_base_system),
            ),
            Node(
                package='arm_action',
                executable='pipetting_control',
                name='pipetting_control',
                output='screen',
                parameters=[
                    pipetting_targets_yaml,
                    {
                        'planning_method': planning_method,
                        'enable_terminal_input': False,
                    }
                ],
            ),
            Node(
                package='pipettingrobot_gui',
                executable='scene_bridge',
                name='scene_bridge',
                output='screen',
                parameters=[
                    {
                        'rack_mesh': rack_mesh,
                        'beaker_mesh': beaker_mesh,
                        'tube_mesh': tube_mesh,
                        'aubo_type': aubo_type,
                    }
                ],
            ),
            Node(
                package='pipettingrobot_gui',
                executable='operator_panel',
                name='operator_panel',
                output='screen',
                condition=IfCondition(start_panel),
                parameters=[
                    {
                        'planning_method': planning_method,
                        'rack_mesh': rack_mesh,
                        'beaker_mesh': beaker_mesh,
                        'tube_mesh': tube_mesh,
                        'aubo_type': aubo_type,
                        'robot_base_frame': robot_base_frame,
                        'top_camera_topic': top_camera_topic,
                        'stereo_left_topic': stereo_left_topic,
                        'stereo_right_topic': stereo_right_topic,
                    }
                ],
            ),
        ]
    )

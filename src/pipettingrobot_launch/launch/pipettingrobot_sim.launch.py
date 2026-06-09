from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, ExecuteProcess
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    pkg_share = get_package_share_directory('aubo_description')
    disable_fuel = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    xacro_file = os.path.join(pkg_share, 'urdf', 'pipettingrobot.xacro')
    robot_description = Command([
        FindExecutable(name='xacro'), ' ', xacro_file, ' sim_gazebo:=true'
    ])

    # Gazebo
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # State publisher
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # ros2_control node
    ros2_control = Node(
        package='controller_manager', executable='ros2_control_node',
        parameters=[
            {'use_sim_time': True},
            os.path.join(pkg_share, 'config', 'pipetting_controllers.yaml')
        ],
        output='screen'
    )

    # spawn robot
    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-topic','robot_description','-entity','pipettingrobot'],
        output='screen'
    )

    # spawner: 加载 controllers（只 load，不 start）
    spawn_jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster','--controller-manager','/controller_manager'],
        output='screen'
    )
    spawn_jtc = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_trajectory_controller','--controller-manager','/controller_manager'],
        output='screen'
    )
    spawn_io  = Node(
        package='controller_manager', executable='spawner',
        arguments=['io_and_status_controller','--controller-manager','/controller_manager'],
        output='screen'
    )
    spawn_fwd = Node(
        package='controller_manager', executable='spawner',
        arguments=['forward_command_controller_position','--controller-manager','/controller_manager'],
        output='screen'
    )

    # load_start: 自动切换到 active
    start_jtc = ExecuteProcess(
        cmd=['ros2','control','load_start_controller','joint_trajectory_controller',
             '--controller-manager','/controller_manager'],
        output='screen'
    )
    start_io  = ExecuteProcess(
        cmd=['ros2','control','load_start_controller','io_and_status_controller',
             '--controller-manager','/controller_manager'],
        output='screen'
    )
    start_fwd = ExecuteProcess(
        cmd=['ros2','control','load_start_controller','forward_command_controller_position',
             '--controller-manager','/controller_manager'],
        output='screen'
    )

    return LaunchDescription([
        disable_fuel,
        gazebo,
        rsp,
        ros2_control,
        spawn,
        spawn_jsb,   # joint_state_broadcaster 会自动 active
        spawn_jtc,
        spawn_io,
        spawn_fwd,
        # 三个要手动 start
        start_jtc,
        start_io,
        start_fwd,
    ])

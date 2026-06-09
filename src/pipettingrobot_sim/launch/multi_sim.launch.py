import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # 1. 环境变量设置：仅打印 WARN 及以上级别日志，减少终端冗余
    ld.add_action(SetEnvironmentVariable(
        'RCUTILS_LOGGING_SEVERITY_THRESHOLD', 'WARN'))

    # 2. 包含 Gazebo 仿真环境 Launch
    # 确保 pipettingrobot_sim 包路径正确
    pipet_dir = get_package_share_directory('pipettingrobot_sim')
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pipet_dir, 'launch', 'robot_gazebo.launch.py')
        )
    ))

    # 3. 末端执行器 (EE) 位姿发布节点
    # 将 wrist3_Link 相对于 base_link 的坐标发布到 /ee_pose
    ld.add_action(
        Node(
            package='rl_train',
            executable='ee_pose_publisher',
            name='ee_pose_publisher',
            output='screen',
            parameters=[{
                'target_frame': 'base_link',
                'source_frame': 'wrist3_Link',
                'publish_topic': '/ee_pose',
                'rate': 30.0,
                'timeout_sec': 0.03,
            }],
            arguments=['--ros-args', '--log-level', 'WARN'],
        )
    )

    # 4. 动态障碍物点云生成节点 (仅发布 PLY 模型动态点云)
    ld.add_action(
        Node(
            package='rl_train',
            executable='pointcloud',
            name='pointcloud_generator',
            output='screen',
            parameters=[{
                'pc_dir': '/home/robot/pipetbot_ros2_ws/pc', # PLY文件存放目录
                'dyn_count': 2,                             # 动态障碍物数量
                'dyn_pts_per_obstacle': 200,                # 每个障碍物的采样点数
                'dyn_size_min': 0.10,
                'dyn_size_max': 0.15,

                'interval_range_s': [4.0, 4.0],             # 每 4 秒重置一次障碍物位置

                # 空间范围定义 [min, max]
                'pose_range_x': [0.1, 0.9],
                'pose_range_y': [-0.3, 0.5],
                'pose_range_z': [0.1, 0.5],
                'pose_range_roll': [0.0, 0.0],
                'pose_range_pitch': [0.0, 0.0],
                'pose_range_yaw': [0.0, 0.0],

                # 速度范围定义 [min, max]
                'velocity_range_x': [-0.2, 0.0],
                'velocity_range_y': [-0.2, 0.0],
                'velocity_range_z': [-0.1, 0.0],
                'velocity_range_roll': [-0.05, 0.05],
                'velocity_range_pitch': [-0.05, 0.05],
                'velocity_range_yaw': [-0.05, 0.05],
            }],
            arguments=['--ros-args', '--log-level', 'WARN'],
        )
    )

    # 5. 目标位姿服务节点
    ld.add_action(Node(
        package='rl_train',
        executable='target_pose',
        name='target_pose_service',
        output='screen',
        arguments=['--ros-args', '--log-level', 'WARN']
    ))

    # 6. 障碍物特征提取节点 (PyTorch GPU 加速版)
    ld.add_action(Node(
        package='rl_train',
        executable='obstacle_feature_extractor', 
        name='obstacle_feature_extractor',
        output='screen',
        parameters=[{
            'cloud_topic': '/obstacles',       # 订阅 pointcloud 节点发布的点云
            'output_topic': '/obstacle_features', # 发布给 RL 算法的特征
            'ray_count': 36,                   # 采样射线条数 (特征维度)
            'max_range': 2.0,                  # 最大探测距离
            'ray_frame': 'base_link',          # 投影参考坐标系

            # 采样平面策略: 
            # "x>0": x 正半球
            # "y>0": y 正半球（与 Isaac 配置对齐）
            # "xy>0": 前右 1/4 半球
            'fibonacci_plane': 'y>0', 

            'device': 'cuda',                  # 强制使用 GPU 计算，若无显卡请改为 'cpu'
        }],
        arguments=['--ros-args', '--log-level', 'INFO'], # 初始化阶段开启 INFO 查看设备加载情况
    ))

    return ld

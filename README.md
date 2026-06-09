# PipetBot ROS 2 Workspace

一个面向移液机器人实验流程的 ROS 2 工作区，集成了机械臂控制、末端执行器控制、视觉检测、眼手标定、Gazebo 仿真、操作界面，以及基于强化学习的轨迹训练与避障实验。

虽然当前仓库以移液任务为核心场景，但整体架构并不局限于移液本身。其感知、运动规划、末端控制、GUI、仿真和强化学习训练模块也可以迁移到其他相似的机械臂任务中，例如码垛、抓取、分拣、定位放置、障碍环境下的轨迹执行等。对于这类任务，通常只需要替换目标检测模型、任务流程逻辑、末端执行器定义，以及少量场景参数或接口配置。

需要说明的是，本仓库中的 `rl_train` 并不是整个强化学习体系的唯一来源。作者还维护了一个基于 Isaac Sim 的强化学习项目，用于训练机械臂的动态避障和 reach 能力；而当前仓库中的 `rl_train` 主要用于在 Gazebo / ROS 2 环境中做策略验证、任务适配和微调。

该仓库当前更接近“研究/实验型工作区”而不是单一 Python 包。目录中除了自研包，还包含了第三方 ROS 2 依赖源码、训练日志、模型权重和数据集。因此，在上传 GitHub 前，建议配合 `.gitignore` 进一步筛掉 `build/`、`install/`、`log/`、训练日志和大体积数据文件。

## 项目能力概览

- 基于 Aubo 机械臂的移液任务执行
- RealSense D435i 与双目相机联合感知
- YOLOv11 目标检测与障碍物分割
- 试管圆孔检测与立体定位
- 眼在手上 / 眼在手外标定
- Gazebo + ros2_control 仿真
- Qt / PySide2 操作面板与 3D 场景可视化
- 基于 Gymnasium + skrl 的 ROS 2 强化学习训练环境
- 可扩展到码垛、抓取、分拣等其他类似机械臂任务

## 基于 YOLO11 的改进模型说明

本项目中的目标检测与部分障碍物感知能力，基于原始 `YOLO11` 结构进行了定制化改进，而不是只直接使用官方默认配置。

目前已整理并收回到本仓库中的 YOLO11 改进内容包括：

- 自定义模型结构 YAML
  - 位于 [third_party/ultralytics_yolo11_custom/cfg/models/11](/home/robot/pipetbot_ros2_ws/third_party/ultralytics_yolo11_custom/cfg/models/11)
- 自定义模块源码
  - 位于 [third_party/ultralytics_yolo11_custom/nn/MyModules](/home/robot/pipetbot_ros2_ws/third_party/ultralytics_yolo11_custom/nn/MyModules)

其中包含的改进方向包括：

- `HSFPN`
- `DynamicConv`
- `CBFuse`
- `RFAConv`
- `C3k2_DynamicConv`
- 以及这些模块的组合结构

这部分文件最初位于本机安装的 `ultralytics` 库中，现已同步到当前仓库，便于：

- 保留实验可复现性
- 与训练权重对应
- 避免仓库运行依赖“本机私有修改过的 site-packages”

如果你需要复现实验或继续训练，建议优先参考这些本地归档的配置，而不是仅依赖系统中已安装的 `ultralytics` 默认模型目录。

## 强化学习说明

本仓库中的强化学习相关内容分为两个层次：

- 主训练项目
  - 另有一个基于 Isaac Sim 的项目，用于训练机械臂的动态避障与 reach 能力
- 当前仓库中的 `rl_train`
  - 基于 Gazebo 和 ROS 2
  - 主要用于策略在当前机器人工作流中的验证、微调和任务侧适配

也就是说，如果从训练体系的角色上看：

- Isaac Sim 项目更偏向高效主训练
- 当前仓库的 `rl_train` 更偏向面向任务落地的仿真微调与联调

## 可扩展应用

除了当前的移液流程，这个工作区也适合扩展到其他结构相近的机械臂任务，包括但不限于：

- 码垛与拆垛
- 目标抓取与放置
- 工件分拣与搬运
- 视觉引导定位
- 避障轨迹跟踪

可复用的核心能力包括：

- 基于 ROS 2 的模块化节点通信
- 基于 MoveIt 2 的位姿规划与轨迹执行
- 基于相机和点云的目标/障碍感知
- 基于 Gazebo 的仿真验证
- 基于 Gymnasium + skrl 的强化学习训练框架
- 基于 PySide2 的操作界面与状态可视化

如果要迁移到新的任务类型，通常需要重点调整：

- 任务目标和状态机逻辑
- 目标检测或分割模型
- 末端执行器控制策略
- 场景模型、标定参数和接口配置
- 奖励函数与训练目标（如果使用强化学习）

## 工作区结构

`src/` 下的主要自研包如下：

- `arm_action`
  - 机械臂动作接口、MoveIt 规划、关节轨迹执行、移液任务流程控制
- `effector_controller`
  - 末端执行器 GPIO 组合控制
- `target_detection`
  - 基于改进版 YOLO11 的目标检测，结合 RealSense RGB/Depth 输出目标三维坐标
- `circle_detection`
  - 双目图像中的圆孔/试管位检测、排序与深度估计
- `obstacle_detection`
  - 障碍物分割、点云融合、障碍特征提取与发布
- `eyehand_calibration`
  - 眼手标定与静态外参发布
- `pipettingrobot_interfaces`
  - 自定义 action / msg / srv 接口定义
- `pipettingrobot_launch`
  - 实机系统与仿真系统的 launch 组织
- `pipettingrobot_gui`
  - 操作员面板、相机预览、3D 场景桥接与任务状态可视化
- `pipettingrobot_sim`
  - Gazebo 世界、URDF/Xacro、控制器配置与仿真资源
- `rl_train`
  - 基于 ROS 2 / Gazebo 的 Gymnasium 环境、碰撞检测、点云障碍特征、skrl 训练与回放
  - 主要用于已有策略在 Gazebo 场景中的验证、微调和任务适配

仓库内还包含外部源码依赖：

- `src/realsense-ros`
- `src/arcs_ros2-ros2`

它们用于相机与 Aubo 相关 ROS 2 能力接入，不属于本仓库核心业务代码，但当前工作区运行依赖它们。

## 核心流程

### 1. 感知

- `target_detection/yolov11_d435i.py`
  - 订阅 RealSense 彩色图和对齐深度图
  - 使用 YOLO 模型检测目标
  - 通过相机内参反投影得到目标三维坐标
- `circle_detection/circleAll_detector.py`
  - 同步双目图像
  - 进行圆检测、匹配与期望圆位输出
- `obstacle_detection/ObsaclePointSeg_yolo11_pub.py`
  - 对图像和点云做时间同步
  - 基于改进版 YOLO11 分割障碍物并投影为障碍特征向量

### 2. 控制与任务执行

- `arm_action/action_execute.py`
  - 封装 MoveIt 规划和轨迹执行
- `arm_action/quintic_spline.py`
  - 通过五次样条和平滑轨迹控制机械臂
- `arm_action/pipetting_control.py`
  - 串联检测、定位、机械臂运动和末端执行器动作，执行完整移液流程
- `effector_controller/effector_controller.py`
  - 通过 Aubo IO 服务控制末端执行器状态

### 3. GUI

- `pipettingrobot_gui/operator_panel.py`
  - 提供操作员面板、相机画面预览、任务进度展示
- `pipettingrobot_gui/scene_bridge.py`
  - 将任务状态和视觉对象桥接为场景数据
- `pipettingrobot_gui/qt3d_scene.py`
  - 使用 PySide2 Qt3D 渲染移液场景

### 4. 强化学习

- `rl_train/ros2_pipetbot_env.py`
  - 将 Gazebo + ROS 2 控制接口封装为 Gymnasium 环境
  - 支持障碍特征、目标位姿、碰撞惩罚、动作平滑惩罚等设计
- `rl_train/train.py`
  - 支持 `ppo`、`sac`、`td3`
  - 使用 `skrl` 在 Gazebo 环境中进行策略微调与验证
- `rl_train/play.py`
  - 载入 checkpoint 回放策略

补充说明：

- 动态避障和 reach 能力的主训练工作来自另一套基于 Isaac Sim 的项目
- 本仓库中的 `rl_train` 更适合做 ROS 2 / Gazebo 场景下的策略联调、适配和小规模再训练

## 自定义接口

`pipettingrobot_interfaces` 中定义了项目所需的 ROS 2 接口，包括：

- Actions
  - `SetTargetPose`
  - `SetGPIOCombination`
  - `GetExpectedCirclePosition`
  - `GetCircleCoords`
- Messages
  - `DetectedObject`
  - `CollisionInfo`
  - `PipettingSceneState`
  - `PipettingTaskStatus`
  - `TubeVisualState`
  - `Circle`
  - `CirclePosition`
- Services
  - `GetDetections`
  - `GetExpectedCircleCount`
  - `SetActiveTube`
  - `SetCircleCount`
  - `SetJointAngles`
  - `SetTargetPose`
  - `SetTubeVisualState`
  - `StartPipetting`

## 环境要求

### 系统/ROS 2 层

建议在 Ubuntu 22.04 + ROS 2 Humble 环境下使用，并具备以下能力：

- ROS 2 Humble
- colcon
- gazebo / gazebo_ros / gazebo_ros2_control
- ros2_control / controller_manager
- MoveIt 2
- RealSense ROS 2 驱动
- Aubo 相关 ROS 2 包

### Python 层

项目中实际使用到的主要 Python 依赖包括：

- `numpy`
- `opencv-python`
- `PyYAML`
- `gymnasium`
- `skrl>=1.4.2`
- `torch`
- `ultralytics`
- `trimesh`
- `python-fcl`
- `urdf-parser-py`
- `plyfile`
- `scipy`
- `packaging`
- `PySide2`

此外还需要若干 ROS 2 Python 绑定：

- `rclpy`
- `tf2_ros`
- `cv_bridge`
- `sensor_msgs_py`
- `tf_transformations`
- `tf2_geometry_msgs`
- `ament_index_python`

注意：这些 ROS 2 相关模块通常不是通过 `pip install` 解决，而是通过 ROS 2 deb 包提供。

## 环境安装

下面给出一个按当前仓库组织方式推荐的安装流程。这个流程以根目录的 `setup.py` 为 Python 依赖入口。

### 1. 准备 ROS 2 Humble 基础环境

建议系统：

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble

安装并加载 ROS 2 环境后，先准备工作区常用工具：

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool
```

如果你的系统还没有初始化 `rosdep`：

```bash
sudo rosdep init
rosdep update
```

### 2. 安装 ROS 2 / 系统层依赖

本项目除了 Python 包，还依赖 Gazebo、ros2_control、MoveIt 2、TF、消息桥接等 ROS 2 组件。可先安装一批核心依赖：

```bash
sudo apt install -y \
  ros-humble-gazebo-ros \
  ros-humble-gazebo-ros2-control \
  ros-humble-controller-manager \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-moveit \
  ros-humble-tf-transformations \
  ros-humble-tf2-geometry-msgs \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-sensor-msgs-py \
  ros-humble-xacro \
  ros-humble-urdfdom-py
```

如果你希望让系统自动分析当前工作区缺失的 ROS 依赖，也可以在工作区根目录执行：

```bash
rosdep install --from-paths src --ignore-src -r -y
```

说明：

- `src/realsense-ros` 和 `src/arcs_ros2-ros2` 已经在仓库中带了源码
- 但它们运行时仍可能依赖系统里的额外 deb 包
- 若你使用真实硬件，还需要根据设备实际情况安装 RealSense SDK/Aubo 相关驱动

### 3. 创建 Conda 环境

建议使用 Conda 为该项目创建独立环境，例如：

```bash
conda create -n pipetbot python=3.10 -y
conda activate pipetbot
python -m pip install --upgrade pip setuptools wheel
```

### 4. 通过根目录 `setup.py` 安装 Python 依赖

根目录 `setup.py` 已经汇总了当前项目主要 Python 依赖，可以直接在工作区根目录执行：

```bash
pip install -e .
```

如果你还需要开发相关工具：

```bash
pip install -e ".[dev]"
```

这一步会安装 README 中列出的主要 Python 依赖，包括：

- `numpy`
- `opencv-python`
- `PyYAML`
- `gymnasium`
- `skrl>=1.4.2`
- `torch`
- `ultralytics`
- `trimesh`
- `python-fcl`
- `urdf-parser-py`
- `plyfile`
- `scipy`
- `packaging`
- `PySide2`

注意：

- `python-fcl`、`torch`、`PySide2` 这类包在不同机器上可能对系统环境更敏感
- 如果安装失败，通常需要根据你的 Ubuntu / Python / CUDA 环境单独调整
- 根目录 `setup.py` 负责说明和聚合 Python 依赖，但不替代 ROS 2 的系统依赖安装

## 数据集

仓库中的 `dataset/` 目录默认不随 GitHub 一起上传，以避免仓库体积过大，也方便后续独立管理数据版本。

后续可从 Hugging Face 下载数据集：

- Hugging Face Dataset: `TODO: https://huggingface.co/datasets/<your-dataset-name>`

建议你后续发布数据集时，在这里替换成真实链接，并补充：

- 数据集内容简介
- 标注格式说明
- 训练/验证/测试划分
- 版本号或更新时间

## 权重文件

为了兼顾仓库可用性与体积控制，当前仓库计划只保留少量关键 YOLO 权重与配置示例，而不上传完整训练数据和全部实验产物。

当前建议保留的代表性文件包括：

- 基础或通用权重
  - [src/yolo11n.pt](/home/robot/pipetbot_ros2_ws/src/yolo11n.pt)
  - [src/yolo11n-seg.pt](/home/robot/pipetbot_ros2_ws/src/yolo11n-seg.pt)
- 当前项目中直接使用的检测权重
  - [dataset/runs/train/yolov11n_experiment/weights/best.pt](/home/robot/pipetbot_ros2_ws/dataset/runs/train/yolov11n_experiment/weights/best.pt)
- 对应训练参数记录
  - [dataset/runs/train/yolov11n_experiment/args.yaml](/home/robot/pipetbot_ros2_ws/dataset/runs/train/yolov11n_experiment/args.yaml)

如果后续你把更多权重统一发布到 Hugging Face，也建议在这里补充下载链接。

### 5. 编译 ROS 2 工作区

安装完 Python 依赖后，在工作区根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

如果你只想先编译核心自研包：

```bash
colcon build --symlink-install --packages-select \
  pipettingrobot_interfaces \
  arm_action \
  effector_controller \
  target_detection \
  circle_detection \
  obstacle_detection \
  eyehand_calibration \
  pipettingrobot_launch \
  pipettingrobot_gui \
  pipettingrobot_sim \
  rl_train
```

### 6. 验证安装

可以先验证几个关键入口是否已经可见：

```bash
ros2 pkg list | grep pipettingrobot
ros2 pkg list | grep rl_train
```

再尝试启动仿真或 GUI：

```bash
ros2 launch pipettingrobot_launch pipettingrobot_sim.launch.py
```

或：

```bash
ros2 launch pipettingrobot_gui pipetting_operator.launch.py
```

## 构建

在工作区根目录执行：

```bash
colcon build --symlink-install
source install/setup.bash
```

如果只想编译核心自研包，可以按需选择：

```bash
colcon build --symlink-install --packages-select \
  pipettingrobot_interfaces \
  arm_action \
  effector_controller \
  target_detection \
  circle_detection \
  obstacle_detection \
  eyehand_calibration \
  pipettingrobot_launch \
  pipettingrobot_gui \
  pipettingrobot_sim \
  rl_train
```

## 运行方式

### 1. 启动实机/完整系统

```bash
ros2 launch pipettingrobot_launch pipettingrobot_launch.launch.py
```

该 launch 会组织启动：

- RealSense 相机
- 目标检测
- 双目圆检测
- Aubo bringup
- 机械臂动作节点
- 末端执行器控制
- 静态 TF
- 障碍特征提取

### 2. 启动 Gazebo 仿真

```bash
ros2 launch pipettingrobot_launch pipettingrobot_sim.launch.py
```

### 3. 启动操作面板

```bash
ros2 launch pipettingrobot_gui pipetting_operator.launch.py
```

### 4. 强化学习训练

先启动仿真和相关 ROS 2 节点，再运行：

```bash
python3 src/rl_train/rl_train/train.py --algo td3
```

或根据自己的配置指定：

```bash
python3 src/rl_train/rl_train/train.py \
  --algo ppo \
  --agent_cfg /path/to/skrl_ppo_cfg.yaml \
  --timesteps 100000
```

### 5. 强化学习策略回放

```bash
python3 src/rl_train/rl_train/play.py \
  --algo td3 \
  --checkpoint /path/to/best_agent.pt
```

## 目前仓库中值得整理的内容

为了更适合公开上传 GitHub，建议继续处理以下内容：

- 增加根目录 `.gitignore`
  - 忽略 `build/`、`install/`、`log/`
  - 忽略 `logs_skrl_gazebo/`、`src/pipetbot_ppo/`、`dataset/`
  - 忽略模型权重、临时数据和 `__pycache__/`
- 补充 `LICENSE`
- 将各包中仍为 `TODO` 的 `description`、`license`、`maintainer` 信息补齐
- 将代码中硬编码的模型路径、checkpoint 路径、相机内参改为参数或配置文件
- 视情况移除大体积数据集后再公开仓库

## 适合 GitHub 仓库简介的短描述

可用作 GitHub 仓库 description 的一句话：

`ROS 2 workspace for a pipetting robot with perception, calibration, motion planning, GUI, Gazebo simulation, and reinforcement learning.`

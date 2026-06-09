import rclpy
import torch
import torch.nn as nn
import yaml
import numpy as np
import threading
import tf2_ros
import os 
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# --- 动态构建网络的类 ---
class DynamicPolicyNetwork(nn.Module):
    def __init__(self, yaml_path, input_dim=61, output_dim=6):
        super(DynamicPolicyNetwork, self).__init__()
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML 配置文件未找到: {yaml_path}")

        # 1. 读取 YAML 配置
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        # 2. 提取网络结构参数
        try:
            policy_cfg = cfg['models']['policy']
            net_cfg = policy_cfg['network'][0]
            hidden_layers = net_cfg['layers']
            activation_name = net_cfg['activations']
        except KeyError as e:
            print(f"YAML 解析失败，将使用默认结构: {e}")
            hidden_layers = [512, 256, 128]
            activation_name = "elu"

        # 3. 映射激活函数
        activation_map = {
            "elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "leaky_relu": nn.LeakyReLU()
        }
        act_func = activation_map.get(activation_name.lower(), nn.ELU())

        # 4. 构建层
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_func)
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return self.net(x)

    def act(self, inputs, role="policy"):
        x = inputs["states"]
        mean_actions = self.net(x)
        return list([mean_actions]), list([self.log_std_parameter]), {}

class RLStrategy:
    def __init__(self, node, checkpoint_path, config_path, n_joints=6):
        self.node = node
        self.n_joints = n_joints
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 机器人参数
        self.home_pos = np.array([0.0, 0.7, 2.26, 1.57, 1.57, 0.0], dtype=np.float32)
        self.scale = 0.5
        self.dt_action = 1.0 / 10.0
        self.obs_feat_len = 36
        self.joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", 
                            "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        self.ee_frame = "wrist3_Link"
        self.base_frame = "base_link"
        self.ee_offset = np.array([0.0, 0.237, -0.106], dtype=np.float32)

        # --- 模型加载逻辑 ---
        self.node.get_logger().info(f"Loading Config: {config_path}")
        self.node.get_logger().info(f"Loading Weights: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
             self.node.get_logger().error(f"权重文件不存在: {checkpoint_path}")
             return

        try:
            # 先通过 YAML 创建模型实例
            self.policy = DynamicPolicyNetwork(config_path, input_dim=61, output_dim=n_joints)
            
            # 加载权重字典到临时变量 checkpoint
            self.node.get_logger().info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 解析字典内容 (skrl 的 pt 文件通常有多层嵌套)
            if isinstance(checkpoint, dict):
                # skrl 默认存储位置
                if 'agent' in checkpoint and 'policy' in checkpoint['agent']:
                    state_dict = checkpoint['agent']['policy']
                # 常见的 key
                elif 'policy' in checkpoint:
                    state_dict = checkpoint['policy']
                # 纯权重字典
                else:
                    state_dict = checkpoint
            else:
                # 如果 pt 里存的是整个模型对象（不推荐），则提取其 state_dict
                state_dict = checkpoint.state_dict()

            # 使用 load_state_dict 将权重填入已实例化的 self.policy
            self.policy.load_state_dict(state_dict, strict=False)
            
            # 现在可以安全地调用 .to() 和 .eval()
            self.policy.to(self.device)
            self.policy.eval()
            self.node.get_logger().info("RL model loaded and moved to device successfully.")

        except Exception as e:
            self.node.get_logger().error(f"Failed to load .pt model: {e}")
            raise e

        # --- 状态变量初始化 ---
        self.current_js_pos = np.zeros(self.n_joints, dtype=np.float32)
        self.current_js_vel = np.zeros(self.n_joints, dtype=np.float32)
        self.latest_obs_feat = np.zeros(self.obs_feat_len, dtype=np.float32)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # --- ROS 订阅与 Action ---
        self.js_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.feat_sub = self.node.create_subscription(Float32MultiArray, '/obstacle_features', self._obs_feat_cb, 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        
        self.traj_client = ActionClient(
            self.node, 
            FollowJointTrajectory, 
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
    def _joint_state_cb(self, msg):
        with self.lock:
            pos_dict = dict(zip(msg.name, msg.position))
            vel_dict = dict(zip(msg.name, msg.velocity))
            try:
                self.current_js_pos = np.array([pos_dict[n] for n in self.joint_names], dtype=np.float32)
                self.current_js_vel = np.array([vel_dict[n] for n in self.joint_names], dtype=np.float32)
            except KeyError:
                pass

    def _obs_feat_cb(self, msg):
        with self.lock:
            data = np.array(msg.data, dtype=np.float32)
            #print(f"Received obstacle features: len(data) = {len(data)}")
            if len(data) == self.obs_feat_len:
                self.latest_obs_feat = data
    

    def get_ee_pos_with_offset(self):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())
            p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            return p + self.ee_offset
        except Exception:
            return None

    def build_observation(self, target_pose):
        with self.lock:
            joint_pos_rel = self.current_js_pos - self.home_pos
            js_vel = self.current_js_vel
            pose_cmd = np.array([
                target_pose.position.x, target_pose.position.y, target_pose.position.z,
                target_pose.orientation.x, target_pose.orientation.y, 
                target_pose.orientation.z, target_pose.orientation.w
            ], dtype=np.float32)
            
            obs = np.concatenate([
                joint_pos_rel, js_vel, pose_cmd, self.last_action, self.latest_obs_feat
            ], axis=0)
            obs=torch.from_numpy(obs).view(1, -1).to(self.device).float()
            #print(f"Observation Vector: {obs}")
            return obs

    def predict_and_step(self, target_pose):

        obs_tensor = self.build_observation(target_pose)
        
        with torch.no_grad():
            output = self.policy.act({"states": obs_tensor}, role="policy")
            raw_action = output[0][0].cpu().numpy()[0]

        current_target_q = self.home_pos + (raw_action * self.scale)
        
        with self.lock:
            self.last_action = current_target_q.copy().astype(np.float32)
            
        # 执行控制
        self._apply_action_to_controller(current_target_q)
        print(f"Predicted Joint Targets: {current_target_q}")
        return current_target_q

    def _apply_action_to_controller(self, target_q):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_q.tolist()
        point.time_from_start = RclpyDuration(seconds=float(self.dt_action / 2.0)).to_msg()
        goal.trajectory.points.append(point)
        self.traj_client.send_goal_async(goal)

    def check_success(self, target_pose, threshold=0.01):
        curr_p = self.get_ee_pos_with_offset()
        if curr_p is None: return False, 999.0
        target_p = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])
        dist = np.linalg.norm(curr_p - target_p)
        return dist < threshold, dist
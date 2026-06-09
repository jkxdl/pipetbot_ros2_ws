#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
import threading

import yaml
import rclpy
from rclpy.node import Node

import gymnasium as gym

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import wrap_env  # 等价于 IsaacLab 里的 SkrlVecEnvWrapper(auto)

# Gazebo 环境
from ros2_pipetbot_env import Ros2ArmObstacleEnv


# ============ skrl 版本检查（和 IsaacLab 同步） ============
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)


def start_spin_thread(node: Node):
    """如果你的 Env 内部没有自己 spin，这里可以启一个线程 spin ROS2（可选）"""
    stop_evt = threading.Event()

    def _spin():
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok() and not stop_evt.is_set():
            executor.spin_once(timeout_sec=0.1)
        executor.shutdown()

    th = threading.Thread(target=_spin, daemon=True)
    th.start()
    return stop_evt


# ============ 构造 Gazebo 环境 ============
def make_env(node_name: str) -> gym.Env:
    """创建单个 Gazebo + ROS2 环境"""
    node: Node = rclpy.create_node(node_name)

    contact_topics = [
        "/shoulder_bumper_plugin",
        "/upperarm_bumper_plugin",
        "/forearm_bumper_plugin",
        "/wrist1_bumper_plugin",
        "/wrist2_bumper_plugin",
        "/wrist3_bumper_plugin",
        "/tool0_bumper_plugin",
    ]

    env = Ros2ArmObstacleEnv(
        node=node,
        ee_frame="wrist3_Link",
        ee_offset=(0.0,-0.237 , 0.106),
        contact_topics=contact_topics,

        # 关节 / 观测
        n_joints=6,
        obs_feat_len=36,
        home_pos=(0.0, 0.7, 2.26, 1.57, 1.57, 0.0),

        # 时间相关
        dt_action=1.0 / 10.0,
        episode_time=12.0,
        goal_update_interval=4.0,

        # 奖励权重
        w_pos_linear=-1.5,
        w_pos_exp=1.5,
        w_collision=-0.5,
        collision_sigma=50.0,
        ori_scale=-2.0,
        ori_k=2.0,
        w_action_rate=-0.005,
        w_action=-0.005,

        # --- 避障惩罚权重 ---
        w_obstacle_dist=-2.0,
        d_safe=0.25,
        obs_k=1.0,
        obs_tau=0.05,

        # keypoint 指数奖励参数
        kp_exp_coeffs=[(50.0, 1e-4)],
        kp_use_sum_of_exps=True,
        keypoint_scale=0.45,
        add_cube_center_kp=True,

        # 点云 / 碰撞检测
        pc_max_points=2000,
    )
    return env


def load_skrl_yaml(yaml_path: str) -> dict:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"skrl YAML 配置文件不存在: {yaml_path}")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML 格式不正确（顶层不是 dict）: {yaml_path}")
    return cfg


def resolve_agent_cfg(algo: str, agent_cfg_override: str | None, cfg_dir: str) -> str:
    """
    根据 algo 选择 YAML。若用户传入 --agent_cfg 则优先使用覆盖路径。
    """
    if agent_cfg_override:
        return agent_cfg_override

    # 你可以按自己的实际文件名修改这三个默认名
    algo_to_file = {
        "ppo": "skrl_ppo_cfg.yaml",
        "sac": "skrl_sac_cfg.yaml",
        "td3": "skrl_td3_cfg.yaml",
    }
    if algo not in algo_to_file:
        raise ValueError(f"Unsupported algo: {algo}. Choose from ppo/sac/td3")

    yaml_path = os.path.join(cfg_dir, algo_to_file[algo])
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO/SAC/TD3 in Gazebo using skrl YAML config")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac", "td3"],
        default="td3",
        help="选择训练算法：ppo / sac / td3（将自动选择对应 YAML）",
    )
    parser.add_argument(
        "--cfg_dir",
        type=str,
        default="/home/robot/pipetbot_managerbased_rl/source/pipetbot_managerbased_rl/pipetbot_managerbased_rl/tasks/manager_based/pipetbot_managerbased_rl/agents",
        help="默认 skrl YAML 所在目录（当不传 --agent_cfg 时生效）",
    )
    parser.add_argument(
        "--agent_cfg",
        type=str,
        default=None,
        help="手动指定 YAML 路径（会覆盖 --algo 的自动选择）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/robot/pipetbot_managerbased_rl/logs/skrl/pipetbot_td3/2026-03-15_13-52-45_td3_torch/checkpoints/best_agent.pt",
        help="checkpoint 路径（.pt）。注意：必须与 algo/YAML 匹配",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="../logs_skrl_gazebo",
        help="实验日志根目录",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="覆盖 trainer.timesteps（不传则用 YAML 里的值）",
    )
    args = parser.parse_args()

    yaml_path = resolve_agent_cfg(args.algo, args.agent_cfg, args.cfg_dir)
    print(f"[INFO] Algo: {args.algo}")
    print(f"[INFO] Using YAML: {yaml_path}")

    agent_cfg = load_skrl_yaml(yaml_path)

    # 基本字段检查
    for key in ["models", "memory", "agent", "trainer"]:
        if key not in agent_cfg:
            raise KeyError(f"YAML 中缺少字段 '{key}'，请检查配置: {yaml_path}")

    models_cfg = agent_cfg["models"]
    memory_cfg = agent_cfg["memory"]
    agent_subcfg = agent_cfg["agent"]
    trainer_cfg = agent_cfg["trainer"]

    # seed 同步（和你原先逻辑一致）
    yaml_seed = None
    if isinstance(agent_subcfg, dict):
        yaml_seed = agent_subcfg.get("seed", None)

    agent_cfg["seed"] = yaml_seed
    agent_subcfg["seed"] = yaml_seed
    agent_cfg["agent"] = agent_subcfg

    set_seed(yaml_seed)

    # timesteps：默认读 YAML，也允许 CLI 覆盖
    if args.timesteps is not None:
        trainer_cfg["timesteps"] = int(args.timesteps)
        print(f"[INFO] Overriding trainer.timesteps from CLI: {trainer_cfg['timesteps']}")
    else:
        print(f"[INFO] Using trainer.timesteps from YAML: {trainer_cfg.get('timesteps', 'N/A')}")

    agent_cfg["trainer"] = trainer_cfg

    # 日志路径：按 algo 做目录后缀，避免 PPO/TD3 混在一起
    exp_cfg = agent_subcfg.get("experiment", {})
    base_dir_name = exp_cfg.get("directory", f"pipetbot_{args.algo}")
    log_root_path = os.path.abspath(os.path.join(args.log_root, base_dir_name))
    os.makedirs(log_root_path, exist_ok=True)

    run_suffix = f"_{args.algo}_torch"
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + run_suffix
    exp_name = exp_cfg.get("experiment_name", "")
    if exp_name:
        log_dir_name += f"_{exp_name}"

    agent_subcfg.setdefault("experiment", {})
    agent_subcfg["experiment"]["directory"] = log_root_path
    agent_subcfg["experiment"]["experiment_name"] = log_dir_name
    agent_cfg["agent"] = agent_subcfg

    print(f"[INFO] Logging root: {log_root_path}")
    print(f"[INFO] This run dir: {os.path.join(log_root_path, log_dir_name)}")

    # ROS2 init
    rclpy.init()

    raw_env = make_env("rl_env_node_gazebo")
    print(f"[INFO] Observation space: {raw_env.observation_space}")
    print(f"[INFO] Action space: {raw_env.action_space}")

    # 如果你的环境内部没有 spin，你可以启用这一行：
    # stop_spin = start_spin_thread(raw_env.node if hasattr(raw_env, "node") else None)

    env = wrap_env(raw_env, wrapper="auto")
    runner = Runner(env, agent_cfg)

    # 加载 checkpoint（可选）
    if args.checkpoint is not None:
        ckpt_path = os.path.abspath(args.checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {ckpt_path}")
        print(f"[INFO] Loading checkpoint from: {ckpt_path}")
        runner.agent.load(ckpt_path)

    # 开跑
    runner.run()

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

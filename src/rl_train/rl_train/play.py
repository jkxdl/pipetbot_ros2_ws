#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import rclpy

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import wrap_env

from train import make_env, load_skrl_yaml, resolve_agent_cfg


SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Play a Gazebo checkpoint without training updates")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac", "td3"],
        default="ppo",
        help="选择算法：ppo / sac / td3",
    )
    parser.add_argument(
        "--cfg_dir",
        type=str,
        default="/home/robot/pipetbot_managerbased_rl/source/pipetbot_managerbased_rl/pipetbot_managerbased_rl/tasks/manager_based/pipetbot_managerbased_rl/agents",
        help="默认 skrl YAML 所在目录",
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
        required=True,
        help="checkpoint 路径（.pt）。必须与 algo/YAML 匹配",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="推理步数",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机评估；默认使用 deterministic / mean action",
    )
    args = parser.parse_args()

    yaml_path = resolve_agent_cfg(args.algo, args.agent_cfg, args.cfg_dir)
    print(f"[INFO] Algo: {args.algo}")
    print(f"[INFO] Using YAML: {yaml_path}")

    agent_cfg = load_skrl_yaml(yaml_path)
    for key in ["models", "memory", "agent", "trainer"]:
        if key not in agent_cfg:
            raise KeyError(f"YAML 中缺少字段 '{key}'，请检查配置: {yaml_path}")

    agent_subcfg = agent_cfg["agent"]
    trainer_cfg = agent_cfg["trainer"]

    yaml_seed = None
    if isinstance(agent_subcfg, dict):
        yaml_seed = agent_subcfg.get("seed", None)

    agent_cfg["seed"] = yaml_seed
    agent_subcfg["seed"] = yaml_seed
    agent_cfg["agent"] = agent_subcfg
    set_seed(yaml_seed)

    trainer_cfg["timesteps"] = int(args.timesteps)
    trainer_cfg["headless"] = True
    trainer_cfg["disable_progressbar"] = False
    trainer_cfg["stochastic_evaluation"] = bool(args.stochastic)
    agent_cfg["trainer"] = trainer_cfg

    print(f"[INFO] Evaluation timesteps: {trainer_cfg['timesteps']}")
    print(f"[INFO] Stochastic evaluation: {trainer_cfg['stochastic_evaluation']}")

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"指定的 checkpoint 不存在: {ckpt_path}")

    rclpy.init()

    raw_env = make_env("rl_env_node_gazebo_play")
    print(f"[INFO] Observation space: {raw_env.observation_space}")
    print(f"[INFO] Action space: {raw_env.action_space}")

    env = wrap_env(raw_env, wrapper="auto")
    runner = Runner(env, agent_cfg)

    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    runner.agent.load(ckpt_path)

    try:
        runner.run("eval")
    finally:
        if hasattr(raw_env, "close"):
            raw_env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

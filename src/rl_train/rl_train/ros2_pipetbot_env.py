#!/usr/bin/env python3
import os
import time
import math
import random
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional, Sequence
from rclpy.parameter import Parameter
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import gymnasium as gym

from std_srvs.srv import Empty
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import JointState, PointCloud2
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import ModelStates, LinkStates, ContactsState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from tf_transformations import (
    quaternion_matrix,
    translation_matrix,
    concatenate_matrices,
    quaternion_from_euler,
    quaternion_from_matrix,
)

from urdf_parser_py.urdf import URDF, Box, Sphere, Cylinder, Mesh
from ament_index_python.packages import get_package_share_directory

import fcl
import trimesh
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
import threading

def _resolve_mesh_path(fn: str) -> Optional[str]:
    if fn.startswith("package://"):
        pkg = fn.split("/")[2]
        base = get_package_share_directory(pkg)
        return fn.replace(f"package://{pkg}", base)
    elif os.path.exists(fn):
        return fn
    else:
        return None


def build_collision_models_from_urdf(
    robot: URDF,
) -> Tuple[
    Dict[str, Tuple[fcl.CollisionObject, np.ndarray, np.ndarray, np.ndarray]],
    dict,
]:
    """
    link_name -> (CollisionObject, T_origin(4x4), local_min(3,), local_max(3,))
    adj_graph: child-parent 邻接图
    """
    link_objs: Dict[str, Tuple[fcl.CollisionObject, np.ndarray, np.ndarray, np.ndarray]] = {}
    adj_graph = defaultdict(list)

    # 运动链拓扑
    for child, parent in robot.parent_map.items():
        adj_graph[parent].append(child)
        adj_graph[child].append(parent)

    for link in robot.links:
        if not link.collisions:
            continue

        for coll in link.collisions:
            geom = coll.geometry
            origin = coll.origin

            pos = origin.position if origin and origin.position else [0.0, 0.0, 0.0]
            if origin and origin.rotation:
                rpy_or_quat = origin.rotation
            elif origin and hasattr(origin, "rpy") and origin.rpy:
                rpy_or_quat = origin.rpy
            else:
                rpy_or_quat = None

            if rpy_or_quat:
                if len(rpy_or_quat) == 3:
                    rot = quaternion_from_euler(*rpy_or_quat)
                else:
                    rot = rpy_or_quat
            else:
                rot = [0.0, 0.0, 0.0, 1.0]

            T_origin = concatenate_matrices(
                translation_matrix(pos),
                quaternion_matrix(rot),
            )

            # 碰撞几何体
            if isinstance(geom, Box):
                size = np.array(geom.size, dtype=np.float64)
                half = size / 2.0
                local_min = -half
                local_max = half
                obj = fcl.Box(*geom.size)

            elif isinstance(geom, Sphere):
                r = float(geom.radius)
                local_min = np.array([-r, -r, -r])
                local_max = np.array([r, r, r])
                obj = fcl.Sphere(r)

            elif isinstance(geom, Cylinder):
                r = float(geom.radius)
                l = float(geom.length)
                half_l = l / 2.0
                local_min = np.array([-r, -r, -half_l])
                local_max = np.array([r, r, half_l])
                obj = fcl.Cylinder(r, l)

            elif isinstance(geom, Mesh):
                path = _resolve_mesh_path(geom.filename)
                if not path:
                    continue
                mesh = trimesh.load(path, force="mesh")
                bvh = fcl.BVHModel()
                bvh.beginModel(len(mesh.vertices), len(mesh.faces))
                bvh.addSubModel(mesh.vertices, mesh.faces)
                bvh.endModel()
                obj = bvh
                local_min, local_max = mesh.bounds

            else:
                continue

            co = fcl.CollisionObject(obj, fcl.Transform())
            link_objs[link.name] = (
                co,
                T_origin,
                np.array(local_min, dtype=np.float64),
                np.array(local_max, dtype=np.float64),
            )

    return link_objs, adj_graph


class CollisionManager:
    """
    负责：
    - 订阅 /robot_description, /link_states, /obstacles（点云）
    - 订阅若干 Gazebo ContactsState 话题（每个 link 一个接触传感器）
    - 内部构建 FCL 碰撞模型
    - 提供 compute_collision() -> (collided, force_norm)
    """

    def __init__(
        self,
        node: Node,
        pc_max_points: int = 20000,
        contact_topics: Optional[Sequence[str]] = None,
    ):
        self.node = node
        self.PC_MAX_POINTS = pc_max_points
        self.contact_topics = list(contact_topics) if contact_topics is not None else []

        # FCL 模型
        self.link_objs: Dict[str, Tuple[fcl.CollisionObject, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.adj_graph: dict = {}

        # 当前 link 位姿
        self.link_poses: Dict[str, object] = {}

        # 世界系点云
        self.cloud_world: Optional[np.ndarray] = None

        # 每个 frame_id 的接触合力（从 Gazebo ContactsState 积累而来）
        # key: frame_id (例如 "wrist3_Link" 或 "tool0")
        # val: np.array([Fx, Fy, Fz])
        self.link_contact_forces: Dict[str, np.ndarray] = {}

        self._init_fcl_collision_models()
        self._init_subscribers()

    # --- 初始化 URDF / FCL ---
    def _init_fcl_collision_models(self):
        robot_urdf_xml: Optional[str] = None

        def _cb_urdf(msg: String):
            nonlocal robot_urdf_xml
            if robot_urdf_xml is None:
                robot_urdf_xml = msg.data

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.node.create_subscription(String, "/robot_description", _cb_urdf, qos)

        timeout = time.time() + 5.0
        while robot_urdf_xml is None and time.time() < timeout and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if robot_urdf_xml is None:
            raise RuntimeError("CollisionManager: 未能获取 /robot_description")

        robot = URDF.from_xml_string(robot_urdf_xml)
        self.link_objs, self.adj_graph = build_collision_models_from_urdf(robot)
        self.node.get_logger().info(
            f"CollisionManager: 构建了 {len(self.link_objs)} 个 link 碰撞模型"
        )

    # --- 订阅 link_states / 点云 / contacts ---
    def _init_subscribers(self):
        # Gazebo link_states
        self.node.create_subscription(
            LinkStates,
            "/link_states",
            self._cb_link_states,
            10,
        )

        # 环境点云
        self.node.create_subscription(
            PointCloud2,
            "/obstacles",
            self._cb_cloud,
            10,
        )

        # Gazebo contacts（每个传感器一个 topic）
        for topic in self.contact_topics:
            self.node.get_logger().info(f"CollisionManager: 订阅接触话题 {topic}")
            self.node.create_subscription(
                ContactsState,
                topic,
                lambda msg, t=topic: self._cb_contacts(msg, t),
                10,
            )

    # --- 回调：link_states ---
    def _cb_link_states(self, msg: LinkStates):
        link_poses: Dict[str, object] = {}
        for full_name, pose in zip(msg.name, msg.pose):
            # Gazebo 中通常是 "model::link"
            short = full_name.split("::")[-1]
            if short in self.link_objs:
                link_poses[short] = pose
        self.link_poses = link_poses

    # --- 回调：点云 ---
    def _cb_cloud(self, msg: PointCloud2):
        pts = []
        for p in pc2.read_points(msg, ("x", "y", "z"), skip_nans=True):
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        if not pts:
            self.cloud_world = None
            return
        arr = np.array(pts, dtype=np.float64)
        if arr.shape[0] > self.PC_MAX_POINTS:
            idx = np.random.choice(arr.shape[0], self.PC_MAX_POINTS, replace=False)
            arr = arr[idx]
        self.cloud_world = arr

    # --- 回调：Gazebo ContactsState（多个 topic 共用） ---
    def _cb_contacts(self, msg: ContactsState, topic_name: str):
        """
        将该 contact topic 的所有 total_wrench.force 叠加到对应 frame_id 下。
        优先使用 header.frame_id（通常是插件的 frameName）
        """
        frame = msg.header.frame_id.strip() if msg.header.frame_id else topic_name

        fx = fy = fz = 0.0
        for st in msg.states:
            fx += st.total_wrench.force.x
            fy += st.total_wrench.force.y
            fz += st.total_wrench.force.z

        if abs(fx) + abs(fy) + abs(fz) < 1e-9:
            return

        force_vec = np.array([fx, fy, fz], dtype=np.float64)
        if frame not in self.link_contact_forces:
            self.link_contact_forces[frame] = np.zeros(3, dtype=np.float64)
        self.link_contact_forces[frame] += force_vec

    # --- 小工具：运动链距离 ---
    @staticmethod
    def _kin_chain_distance(adj_graph: dict, a: str, b: str) -> int:
        if a == b:
            return 0
        vis = {a}
        dq = deque([(a, 0)])
        while dq:
            node, d = dq.popleft()
            for nbr in adj_graph[node]:
                if nbr == b:
                    return d + 1
                if nbr not in vis:
                    vis.add(nbr)
                    dq.append((nbr, d + 1))
        return int(1e6)

    # --- 几何碰撞：自碰撞 + 点云 ---
    def _compute_geom_collision(
        self,
        max_skip_dist: int = 1,
        penetration_th: float = 0.0,
        pc_radius: float = 0.01,
    ) -> Tuple[bool, float]:
        """
        只做 FCL 几何检测：
        - link vs link (自碰撞)
        - link vs 点云 (sphere 包围点)
        返回:
            collided_geom: 是否出现几何碰撞
            max_penetration: 所有自碰撞中最大的 penetration_depth
        """
        if not self.link_objs or not self.link_poses:
            return False, 0.0

        link_objs = self.link_objs
        adj_graph = self.adj_graph
        link_poses = self.link_poses
        cloud = self.cloud_world

        # 1) 更新 Transform + AABB
        world_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, (co, T_ori, loc_min, loc_max) in link_objs.items():
            if name not in link_poses:
                continue
            pose = link_poses[name]
            T_link = concatenate_matrices(
                translation_matrix(
                    [pose.position.x, pose.position.y, pose.position.z]
                ),
                quaternion_matrix(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                ),
            )
            T_final = T_link.dot(T_ori)
            co.setTransform(fcl.Transform(T_final[:3, :3], T_final[:3, 3]))

            # 计算世界系 AABB
            corners = np.array(
                [
                    [loc_min[0], loc_min[1], loc_min[2]],
                    [loc_min[0], loc_min[1], loc_max[2]],
                    [loc_min[0], loc_max[1], loc_min[2]],
                    [loc_min[0], loc_max[1], loc_max[2]],
                    [loc_max[0], loc_min[1], loc_min[2]],
                    [loc_max[0], loc_min[1], loc_max[2]],
                    [loc_max[0], loc_max[1], loc_min[2]],
                    [loc_max[0], loc_max[1], loc_max[2]],
                ],
                dtype=np.float64,
            )
            wp = (T_final[:3, :3].dot(corners.T) + T_final[:3, 3].reshape(3, 1)).T
            wmin = wp.min(axis=0)
            wmax = wp.max(axis=0)
            world_aabbs[name] = (wmin, wmax)

        def overlap(a_min, a_max, b_min, b_max):
            return all(a_min[i] <= b_max[i] and a_max[i] >= b_min[i] for i in range(3))

        collided_geom = False
        max_penetration = 0.0

        # 2) 自碰撞：link vs link
        items = list(link_objs.items())
        for i in range(len(items)):
            ni, (co_i, *_i) = items[i]
            if ni not in world_aabbs:
                continue
            mi, Ma = world_aabbs[ni]

            for j in range(i + 1, len(items)):
                nj, (co_j, *_j) = items[j]
                if nj not in world_aabbs:
                    continue

                # 邻接或近邻关节跳过（避免连杆之间必然接触）
                if self._kin_chain_distance(adj_graph, ni, nj) <= max_skip_dist:
                    continue

                mj, Mb = world_aabbs[nj]
                if not overlap(mi, Ma, mj, Mb):
                    continue

                req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
                res = fcl.CollisionResult()
                if fcl.collide(co_i, co_j, req, res) > 0 and res.contacts:
                    for c in res.contacts:
                        d = getattr(c, "penetration_depth", 0.0)
                        if d >= penetration_th:
                            collided_geom = True
                            max_penetration = max(max_penetration, float(d))

        # 3) 环境碰撞：link vs 点云（球近似）
        if cloud is not None and cloud.shape[0] > 0:
            sphere = fcl.Sphere(pc_radius)
            for ni, (co, *_i) in link_objs.items():
                if ni not in world_aabbs:
                    continue
                mi, Ma = world_aabbs[ni]
                # 先用 AABB 过滤点云
                mask = np.all(
                    [
                        cloud[:, 0] >= mi[0],
                        cloud[:, 0] <= Ma[0],
                        cloud[:, 1] >= mi[1],
                        cloud[:, 1] <= Ma[1],
                        cloud[:, 2] >= mi[2],
                        cloud[:, 2] <= Ma[2],
                    ],
                    axis=0,
                )
                pts = cloud[mask]
                if pts.size == 0:
                    continue

                for p in pts[:100]:  # 限制上限，避免太慢
                    co_p = fcl.CollisionObject(
                        sphere, fcl.Transform(np.eye(3), p.astype(np.float64))
                    )
                    req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
                    res = fcl.CollisionResult()
                    if fcl.collide(co, co_p, req, res) > 0 and res.contacts:
                        collided_geom = True
                        # 点云我们没有 penetration 深度，用 pc_radius 代表一次“穿入”
                        max_penetration = max(max_penetration, pc_radius)
                        break

        return collided_geom, max_penetration

    def compute_collision(
        self,
        max_skip_dist: int = 1,
        penetration_th: float = 0.0,
        pc_radius: float = 0.01,
        penetration_scale: float = 100.0,
    ) -> Tuple[bool, float]:
        """
        返回:
        - collided: bool
            是否发生碰撞（Gazebo 接触 + FCL 自碰撞 + FCL 点云 检测综合）
        - force_norm: float
            综合强度 = Gazebo 合力范数 + 几何 penetration 惩罚
        """

        # -------- 1) FCL 几何检测（自碰撞 + 点云） --------
        collided_geom, max_penetration = self._compute_geom_collision(
            max_skip_dist=max_skip_dist,
            penetration_th=penetration_th,
            pc_radius=pc_radius,
        )

        # 将 penetration 映射为一个“等效力”
        penetration_force = max(0.0, max_penetration) * penetration_scale

        # -------- 2) Gazebo 真实接触合力 --------
        gazebo_force = np.zeros(3, dtype=np.float64)
        for ln, f in self.link_contact_forces.items():
            gazebo_force += f

        gazebo_force_norm = float(np.linalg.norm(gazebo_force))

        # -------- 3) 综合强度 --------
        force_norm = gazebo_force_norm + penetration_force

        # -------- 4) 碰撞判定 --------
        collided = (gazebo_force_norm > 1e-6) or collided_geom

        # 读完一次后，可以清空接触缓存（按帧使用）
        self.link_contact_forces = {}

        return collided, force_norm

class ObservationBuilder:
    """
    负责把当前 ROS 状态拼成 obs 向量：
    joint_pos_rel(6) + joint_vel_rel(6) + pose_cmd(7) + last_action(6) + obstacle_features(36)
    """
    def __init__(self, env: "Ros2ArmObstacleEnv"):
        self.env = env
    
    def get_obs(self) -> np.ndarray:
        js = self.env.joint_states
        assert js is not None, "joint_states 还没就绪"

        n = self.env.n_joints
        js_pos = np.array(js.position[:n], dtype=np.float32)

        joint_pos_rel = js_pos - self.env.home_pos
        joint_pos_rel += np.random.uniform(-0.01, 0.01, size=n).astype(np.float32)

        js_vel = np.array(js.velocity, dtype=np.float32)
        if js_vel.shape[0] < n:
            js_vel = np.pad(js_vel, (0, n - js_vel.shape[0]))
        joint_vel_rel = js_vel[:n]
        joint_vel_rel += np.random.uniform(-0.01, 0.01, size=n).astype(np.float32)

        target_pos = self.env.target_pos
        
        p = self.env.target_pos
        target_pos = np.array([p[0], p[1], p[2]], dtype=np.float32)
        
        q = self.env.initial_ori  # xyzw
        target_ori = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)  # wxyz             
        #target_ori = [0.499999, 0.49999, 0.499999, 0.5]  
        pose_cmd = np.concatenate(
            [target_pos, target_ori]
        ).astype(np.float32)
        #print(pose_cmd)
        last_act = self.env.last_action.astype(np.float32)
        #print(f"last action: {last_act}")
        feat = self.env.obs_feat.astype(np.float32)
        #feat = np.ones(36, dtype=np.float32)

        obs = np.concatenate(
            [joint_pos_rel, joint_vel_rel, pose_cmd, last_act, feat], axis=0
        )
        # obs = np.concatenate(
        #     [joint_pos_rel, joint_vel_rel, pose_cmd, last_act], axis=0
        # )
        return obs

class ActionExecutor:
    """
    统一使用 arm_trajectory_controller，避免和其他 arm controller 混用。
    """

    def __init__(self, env: "Ros2ArmObstacleEnv", scale: float = 0.5):
        self.env = env
        self.node = env.node
        self.scale = scale
        self.dt_action = env.dt_action 
        self.last_raw_action = np.zeros(env.n_joints, dtype=np.float32)
        self.last_target_q = np.array(env.home_pos, dtype=np.float32) 

        self.traj_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/arm_trajectory_controller/follow_joint_trajectory",
        )
        self.joint_names = [
            "shoulder_joint",
            "upperArm_joint",
            "foreArm_joint",
            "wrist1_joint",
            "wrist2_joint",
            "wrist3_joint",
        ]

        # reset_world 服务
        self.reset_client = self.node.create_client(Empty, "/reset_world")
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("无法连接到 /reset_world 服务")
        if not self.traj_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("无法连接到 /arm_trajectory_controller/follow_joint_trajectory")
        self.node.get_logger().info(f"[ORDER] action_exec joint_names: {self.joint_names[:self.env.n_joints]}")
        self.node.get_logger().info("ActionExecutor: 仅使用 arm_trajectory_controller")

    def goto_home(self, duration=2.0, wait=True) -> bool:
        """
        使用 arm_trajectory_controller 回到 home 位
        """
        js = self.env.joint_states
        if js is None:
            self.node.get_logger().error("goto_home失败：joint_states 未就绪")
            return False

        n = self.env.n_joints
        target_q = np.asarray(self.env.home_pos[:n], dtype=np.float32)

        if not self.traj_client.wait_for_server(timeout_sec=1.0):
            self.node.get_logger().error(
                "goto_home失败：FollowJointTrajectory action server 不可用"
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names[:n]

        point = JointTrajectoryPoint()
        point.positions = target_q.tolist()
        point.time_from_start = RclpyDuration(seconds=float(duration)).to_msg()
        goal.trajectory.points.append(point)

        send_future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future)
        goal_handle = send_future.result()

        if goal_handle is None:
            self.node.get_logger().error("goto_home失败：goal_handle 为空")
            return False

        if not goal_handle.accepted:
            self.node.get_logger().error("goto_home失败：轨迹目标未被接受")
            return False

        if wait:
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, result_future)
            result = result_future.result()

            if result is None:
                self.node.get_logger().error("goto_home失败：未收到执行结果")
                return False

        self.last_target_q = target_q.copy()
        return True

    def reset_world(self):
        req = Empty.Request()
        fut = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)

    def apply_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        绝对动作控制
        """
        js = self.env.joint_states
        assert js is not None, "joint_states 未就绪"

        n = self.env.n_joints

        raw_action = np.asarray(raw_action, dtype=np.float32)
        self.last_raw_action = raw_action.copy()
        delta_q = raw_action * self.scale
        self.env.last_action = raw_action.copy()

        target_q = self.env.home_pos[:n] + delta_q
        #target_q = delta_q
        self.last_target_q = target_q.copy()
        
        # 使用 FollowJointTrajectory 发送单点轨迹
        if not self.traj_client.wait_for_server(timeout_sec=0.01):
            self.node.get_logger().warn(
                "FollowJointTrajectory action server 不可用，跳过此步命令"
            )
            return delta_q

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names[:n]

        # JointTrajectoryController
        point = JointTrajectoryPoint()
        point.positions = target_q.tolist()
        point.time_from_start = RclpyDuration(
            seconds=float(self.dt_action / 2)
        ).to_msg()

        goal.trajectory.points.append(point)
        prev_stamp = self.env.latest_js_stamp_ns

        self.traj_client.send_goal_async(goal)

        self.env.wait_for_new_joint_state(prev_stamp, timeout=self.dt_action)
        return delta_q
    
class RewardCalculator:

    def __init__(self, env: "Ros2ArmObstacleEnv"):
        self.env = env
        self.node = env.node
        self.collision_mgr = env.collision_mgr

        # 1. 基础位姿与动作奖励权重
        self.w_pos_linear = env.w_pos_linear
        self.w_pos_exp    = env.w_pos_exp
        self.w_collision  = env.w_collision
        self.collision_sigma = env.collision_sigma
        self.ori_scale    = env.ori_scale
        self.ori_k        = env.ori_k
        self.w_action_rate = env.w_action_rate
        self.w_action      = env.w_action

        # 2. 避障惩罚 (Obstacle Distance Penalty) 参数
        self.w_obstacle_dist = getattr(env, "w_obstacle_dist", -1.0)
        self.d_safe          = getattr(env, "d_safe", 0.25)
        self.obs_k           = getattr(env, "obs_k", 1.0)
        self.obs_tau         = getattr(env, "obs_tau", 0.05)
        self.obs_eps         = 1e-3

        # 3. Keypoint 逻辑参数
        self.kp_exp_coeffs      = getattr(env, "kp_exp_coeffs", [(50.0, 1e-4)])
        self.kp_use_sum_of_exps = getattr(env, "kp_use_sum_of_exps", True)
        self.keypoint_scale     = getattr(env, "keypoint_scale", 0.45)
        self.add_cube_center_kp = getattr(env, "add_cube_center_kp", True)

        self.prev_ee_ori: Optional[np.ndarray] = None
        self.prev_action = np.zeros(env.n_joints, dtype=np.float32)

    # --- keypoint helper ---
    @staticmethod
    def get_keypoint_offsets_full_6d(scale: float = 0.45, add_center: bool = True) -> np.ndarray:
        """和 IsaacLab get_keypoint_offsets_full_6d 同构：±xyz，可选中心点。"""
        axes = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )
        if add_center:
            axes = np.vstack([np.zeros((1, 3), dtype=np.float32), axes])
        return axes * float(scale)

    @staticmethod
    def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        x /= n
        y /= n
        z /= n
        w /= n
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        return R
    
    def _compute_obstacle_distance_penalty(self) -> float:
        """
        订阅 /obstacle_centers_ee 获取末端坐标系下的相对位置并计算平方和惩罚。
        注意：此函数不再需要传入 ee_pos，因为订阅的数据已经是相对坐标。
        """
        obs_rel_data = self.env.latest_obs_centers_ee 
        
        if obs_rel_data is None or len(obs_rel_data) == 0:
            return 0.0

        obs_rel_pos = np.asarray(obs_rel_data, dtype=np.float32).reshape(-1, 3)

        dists = np.linalg.norm(obs_rel_pos, axis=-1)
        dists = np.clip(dists, self.obs_eps, None)

        raw = self.obs_k * (1.0 / dists - 1.0 / self.d_safe)

        x_scaled = raw / self.obs_tau
        penalty_vec = self.obs_tau * (np.log1p(np.exp(-np.abs(x_scaled))) + np.maximum(0, x_scaled))

        total_penalty = float(np.sum(np.square(penalty_vec)))
        
        return total_penalty
    
    # --- 获取末端姿态 ---
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        # 读取环境缓存的 /ee_pose
        with self.env._ee_lock:
            pos = None if self.env.latest_ee_pos is None else self.env.latest_ee_pos.copy()
            ori = None if self.env.latest_ee_ori is None else self.env.latest_ee_ori.copy()

        if pos is None or ori is None:
            self.env.node.get_logger().warn("[get_ee_pose] /ee_pose not received yet", throttle_duration_sec=2.0)
            return (
                np.zeros(3, dtype=np.float32),
                np.array([0, 0, 0, 1], dtype=np.float32),
            )

        R = quaternion_matrix([ori[0], ori[1], ori[2], ori[3]])[:3, :3]
        pos_tip = pos + (R @ np.asarray(self.env.ee_offset, dtype=np.float32).reshape(3))

        # 方向：tip 与 wrist 同向
        return pos_tip.astype(np.float32), ori.astype(np.float32)

    # --- 指数 keypoint 奖励（完全对齐 keypoint_command_error_exp） ---
    def _compute_keypoint_exp_reward(self, kp_dist: np.ndarray) -> float:
        """
        kp_dist: (K,) 当前与目标 keypoint 的 L2 距离
        对应 IsaacLab 里 keypoint_dist_sep 对单个 env 的那一行。
        """
        # 防御性转换
        kp_dist = np.asarray(kp_dist, dtype=np.float64)  # 用 float64 做 exp 稳定一点

        keypoint_reward_exp = 0.0

        if self.kp_use_sum_of_exps:
            # 用 sum of exps: 先对每个 keypoint 算指数，再对 K 维求均值
            for (a, b) in self.kp_exp_coeffs:
                denom = np.exp(a * kp_dist) + b + np.exp(-a * kp_dist)
                keypoint_reward_exp += float(np.mean(1.0 / denom))
        else:
            # 单一 exp：先对 keypoint 距离求均值，再套同一个指数
            dist_mean = float(np.mean(kp_dist))
            for (a, b) in self.kp_exp_coeffs:
                denom = np.exp(a * dist_mean) + b + np.exp(-a * dist_mean)
                keypoint_reward_exp += float(1.0 / denom)

        return keypoint_reward_exp

    # --- 主奖励接口 ---
    def compute_reward(self, raw_action: np.ndarray) -> float:
        """
        raw_action: 本步 policy 输出的原始动作（与 IsaacLab 的 last_action / action reward 一致）
        """
        # 1) 末端 / 目标位姿
        ee_pos, ee_ori = self.get_ee_pose()
        target_pos = self.env.target_pos
        target_ori = self.env.target_ori
        #print(f"EE Pos: {ee_pos}, Target Pos: {target_pos}")
        #print(f"EE Ori: {ee_ori}, Target Ori: {target_ori}")
        # 2) 关键点变换
        kp_local = self.get_keypoint_offsets_full_6d(
            scale=self.keypoint_scale,
            add_center=self.add_cube_center_kp,
        )
        R_curr = self.quat_to_rot_matrix(ee_ori)
        R_tgt = self.quat_to_rot_matrix(target_ori)
        kp_curr = (R_curr @ kp_local.T).T + ee_pos[None, :]
        kp_tgt = (R_tgt @ kp_local.T).T + target_pos[None, :]

        kp_diff = kp_tgt - kp_curr           # (K, 3)
        kp_dist = np.linalg.norm(kp_diff, axis=-1)  # (K,)
        mean_kp_dist = float(np.mean(kp_dist))

        reward = 0.0

        # 2.1 线性 keypoint 惩罚（保持原逻辑）
        reward += self.w_pos_linear * mean_kp_dist

        # 2.2 指数 keypoint 奖励（完全对齐 keypoint_command_error_exp）
        kp_reward_exp = self._compute_keypoint_exp_reward(kp_dist)
        reward += self.w_pos_exp * kp_reward_exp

        # 3) 碰撞惩罚
        collided, force_norm = self.collision_mgr.compute_collision(
            max_skip_dist=1,
            penetration_th=0.0,
            pc_radius=0.01,
            penetration_scale=self.env.collision_sigma
        )
        if collided:
            threshold = 0.01
            effective_pen = max(force_norm - threshold, 0.0)
            sigma = self.collision_sigma
            if sigma <= 1e-6:
                penalty_mag = 1.0
            else:
                penalty_mag = 1.0 - math.exp(-0.5 * (effective_pen / sigma) ** 2)
            reward += self.w_collision * penalty_mag
        
        if abs(self.w_obstacle_dist) > 1e-6:
            obs_penalty = self._compute_obstacle_distance_penalty()
            reward += self.w_obstacle_dist * obs_penalty

        # 4) 姿态变化惩罚
        if self.prev_ee_ori is None:
            self.prev_ee_ori = ee_ori.copy()
        else:
            q1 = ee_ori / (np.linalg.norm(ee_ori) + 1e-8)
            q0 = self.prev_ee_ori / (np.linalg.norm(self.prev_ee_ori) + 1e-8)
            dot = abs(float(np.dot(q1, q0)))
            dot = max(-1.0, min(1.0, dot))
            delta_angle = 2.0 * math.acos(dot)

            ori_penalty = self.ori_scale * (math.exp(self.ori_k * delta_angle) - 1.0)
            reward += ori_penalty

            self.prev_ee_ori = ee_ori.copy()

        # 5) action_rate L2 / action L2
        raw_action = np.asarray(raw_action, dtype=np.float32)
        delta_a = raw_action - self.prev_action
        action_rate_cost = float(np.sum(delta_a ** 2))
        reward += self.w_action_rate * action_rate_cost

        action_cost = float(np.sum(raw_action ** 2))
        reward += self.w_action * action_cost

        self.prev_action = raw_action.copy()

        return float(reward)

    def reset(self):
        self.prev_ee_ori = None
        self.prev_action[:] = 0.0


class TerminationManager:
    """
    管理：
    - 步数计数
    - episode 时间截断
    - 目标刷新（每 4s）
    """

    def __init__(self, env: "Ros2ArmObstacleEnv"):
        self.env = env
        self.dt = env.dt_action
        self.episode_time = env.episode_time
        self.goal_update_interval = env.goal_update_interval

        self.max_steps = int(self.episode_time / self.dt)
        self.goal_update_steps = int(self.goal_update_interval / self.dt)

        self.current_step = 0
        self.last_goal_update_step = 0

    def reset(self):
        self.current_step = 0
        self.last_goal_update_step = 0

    def step_and_check_goal(self):
        """
        步数 +1，判断是否需要更新目标
        """
        self.current_step += 1
        if (self.current_step - self.last_goal_update_step) >= self.goal_update_steps:
            self.env.sample_random_target(candidates=self.env.pair)
            self.last_goal_update_step = self.current_step

    def check_truncation(self) -> Tuple[bool, dict]:
        truncated = self.current_step >= self.max_steps
        info = {}
        if truncated:
            info["time_limit"] = True
        return truncated, info

class Ros2ArmObstacleEnv(gym.Env):
    """
    单阶段连续任务：
    - 每 4s 随机选一个 cup_* / tube_*，目标是其上方
    - 控制频率 ~30Hz（dt_action = 1/30）
    - 奖励结构对齐 Isaac Lab:
        * keypoint_command_error + keypoint_command_error_exp
        * collision_penalty (Gaussian)
        * orientation_change_penalty_exp
        * action_rate_l2 + action_l2
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        node: Node,
        ee_frame: str = "tool0",
        ee_offset=(0.0, -0.237, 0.106),
        contact_topics: Optional[Sequence[str]] = None,
        *,
        # 关节 / 观测
        n_joints: int = 6,
        obs_feat_len: int = 36,
        home_pos=(0.0, 0.7, 2.26, 1.57, 1.57, 0.0),
        # 时间相关
        dt_action: float = 1.0 / 15.0,
        episode_time: float = 12.0,
        goal_update_interval: float = 4.0,
        # 奖励权重
        w_pos_linear: float = -0.05,
        w_pos_exp: float = 1.5,
        w_collision: float = -3.0,
        collision_sigma: float = 50.0,
        ori_scale: float = 0.1,
        ori_k: float = 2.0,
        w_action_rate: float = -0.005,
        w_action: float = -0.005,
        w_obstacle_dist: float = -1.0,
        d_safe: float = 0.2,           
        obs_k: float = 1.0,            
        obs_tau: float = 0.01,
        kp_exp_coeffs: Sequence[tuple[float, float]] = ((50.0, 1e-4),),
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 0.45,
        add_cube_center_kp: bool = True,

        # 碰撞管理相关
        pc_max_points: int = 20000,
        # 固定目标姿态
        fixed_target_ori=(0.5, 0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.node = node

        self.node.set_parameters([
            Parameter("use_sim_time", Parameter.Type.BOOL, True)
        ])
        self.n_joints = int(n_joints)   # 提前，避免回调先到时 self.n_joints 还没赋值
        self.desired_joint_names = [
            "shoulder_joint",
            "upperArm_joint",
            "foreArm_joint",
            "wrist1_joint",
            "wrist2_joint",
            "wrist3_joint",
        ][:self.n_joints]
        self._js_reorder_idx = None
        # ======== 基本状态 / 订阅 ========
        self.latest_states: Optional[ModelStates] = None
        self.obs_feat: Optional[np.ndarray] = None
        self.joint_states: Optional[JointState] = None
        
        self.latest_js_stamp_ns = None

        self.js_sub = self.node.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_states,
            qos_profile_sensor_data,
        )
        # model_states
        node.create_subscription(
            ModelStates,
            "/model_states",
            lambda msg: setattr(self, "latest_states", msg),
            10,
        )

        # obstacle_features
        node.create_subscription(
            Float32MultiArray,
            "/obstacle_features",
            lambda msg: setattr(
                self, "obs_feat", np.array(msg.data, dtype=np.float32)
            ),
            10,
        )

        # joint_states
        js_qos = QoSProfile(depth=10)
        js_qos.reliability = ReliabilityPolicy.RELIABLE
        js_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        node.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_states,
            qos_profile=js_qos,
        )
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node, spin_thread=False)

        # Gazebo 目标位姿服务
        self.cli_tgt = node.create_client(GetEntityState, "/get_target_pose")
        if not self.cli_tgt.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("无法连接到 /get_target_pose 服务")

        # ======== 关节 / 观测 / 动作空间 ========
        self.n_joints = int(n_joints)
        self.obs_feat_len = int(obs_feat_len)
        self.obs_dim = self.n_joints + self.n_joints + 7 + self.n_joints + self.obs_feat_len
        #self.obs_dim = self.n_joints + self.n_joints + 7 + self.n_joints 

        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(self.n_joints,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.latest_obs_centers_ee = None
        
        def _obs_ee_cb(msg):
            self.latest_obs_centers_ee = msg.data

        self.obs_sub = self.node.create_subscription(
            Float32MultiArray,
            '/obstacle_centers_ee',
            _obs_ee_cb,
            10
        )
        
        self.home_pos = np.array(home_pos, dtype=np.float32)

        # ======== 时间参数 ========
        self.dt_action = float(dt_action)
        self.episode_time = float(episode_time)
        self.goal_update_interval = float(goal_update_interval)

        # ======== 奖励权重 ========
        self.w_pos_linear = float(w_pos_linear)
        self.w_pos_exp = float(w_pos_exp)
        self.w_collision = float(w_collision)
        self.collision_sigma = float(collision_sigma)
        self.ori_scale = float(ori_scale)
        self.ori_k = float(ori_k)
        self.w_action_rate = float(w_action_rate)
        self.w_action = float(w_action)

        self.kp_exp_coeffs = kp_exp_coeffs
        self.kp_use_sum_of_exps = kp_use_sum_of_exps
        self.keypoint_scale = keypoint_scale
        self.add_cube_center_kp = add_cube_center_kp

        self.w_obstacle_dist = float(w_obstacle_dist)
        self.d_safe = float(d_safe)
        self.obs_k = float(obs_k)
        self.obs_tau = float(obs_tau)
        # ======== 末端与目标姿态 ========
        self.ee_frame = ee_frame
        self.ee_offset = np.array(ee_offset, dtype=np.float32)

        self.fixed_target_ori = np.array(fixed_target_ori, dtype=np.float32)
        self.initial_ori = np.array([1, 0, 0, 0], dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)

        # 动作缓存（joint delta）
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.debug_step_count = 0

        # ======== 组合五个模块 ========
        self.collision_mgr = CollisionManager(
            node,
            pc_max_points=pc_max_points,
            contact_topics=contact_topics,
        )
        self.obs_builder = ObservationBuilder(self)
        self.action_exec = ActionExecutor(self, scale=0.5)
        self.reward_calc = RewardCalculator(self)
        self.termination_mgr = TerminationManager(self)

        # ======== 等待 joint_states 首帧 ========
        while rclpy.ok() and self.joint_states is None:
            rclpy.spin_once(node, timeout_sec=0.01)

        # ---- EE pose cache (from /ee_pose) ----
        self._ee_lock = threading.Lock()
        self.latest_ee_pos = None          # np.ndarray shape (3,)
        self.latest_ee_ori = None          # np.ndarray shape (4,) xyzw
        self.latest_ee_stamp = None        # rclpy.time.Time

        def _ee_pose_cb(msg: PoseStamped):
            pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
            ori = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                            msg.pose.orientation.z, msg.pose.orientation.w], dtype=np.float32)
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)

            with self._ee_lock:
                self.latest_ee_pos = pos
                self.latest_ee_ori = ori
                self.latest_ee_stamp = stamp

        self.node.create_subscription(
            PoseStamped,
            "/ee_pose",          # 如果你发布节点改了 topic，这里同步改
            _ee_pose_cb,
            qos_profile_sensor_data
        )
    
    # ---------- 内部工具 ----------
    def _on_joint_states(self, msg: JointState):
        # 第一次收到 joint_states 时建立 name->index 映射
        if self._js_reorder_idx is None:
            name_to_i = {n: i for i, n in enumerate(msg.name)}
            missing = [j for j in self.desired_joint_names if j not in name_to_i]
            if missing:
                self.node.get_logger().error(
                    f"/joint_states 缺少关节: {missing}. input names={list(msg.name)}"
                )

            self._js_reorder_idx = [name_to_i[j] for j in self.desired_joint_names]
            self.node.get_logger().info(
                f"JointState reorder enabled. desired={self.desired_joint_names}, idx={self._js_reorder_idx}"
            )

        idx = self._js_reorder_idx

        def take(arr, fill=0.0):
            if arr is None or len(arr) == 0:
                return [fill] * len(idx)
            # JointState 的 position/velocity/effort 可能比 name 短，防御一下
            out = []
            for i in idx:
                out.append(arr[i] if i < len(arr) else fill)
            return out

        out = JointState()
        out.header = msg.header
        out.name = list(self.desired_joint_names)
        out.position = take(msg.position, 0.0)
        out.velocity = take(msg.velocity, 0.0)
        out.effort = take(msg.effort, 0.0)

        self.joint_states = out
        self.latest_js_stamp_ns = self.node.get_clock().now().nanoseconds
    
    def wait_for_new_joint_state(self, prev_stamp_ns, timeout=0.1):
        end_t = time.time() + timeout
        while rclpy.ok() and time.time() < end_t:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if self.latest_js_stamp_ns is not None and self.latest_js_stamp_ns != prev_stamp_ns:
                return True
        return False
    
    def _wait_for_multi_obs(self, attrs, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end and rclpy.ok():
            if all(getattr(self, a) is not None for a in attrs):
                return
            rclpy.spin_once(self.node, timeout_sec=0.01)
        missing = [a for a in attrs if getattr(self, a) is None]
        raise RuntimeError(f"等待 {missing} 超时")

    def _call_target_service_world(self, name: str) -> np.ndarray:
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = "world"
        fut = self.cli_tgt.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut)
        res = fut.result()
        pos = np.array(
            [
                res.state.pose.position.x,
                res.state.pose.position.y,
                res.state.pose.position.z,
            ],
            dtype=np.float32,
        )
        return pos

    def _world_to_base(self, p_w: np.ndarray) -> np.ndarray:
        p_w = np.asarray(p_w, dtype=np.float32).reshape(3)

        if not self.tf_buffer.can_transform(
            "base_link", "world", rclpy.time.Time(),
            timeout=RclpyDuration(seconds=1.0)
        ):
            self.node.get_logger().warn("TF not ready: base_link <-> world")
            return p_w  # 或 raise 让上层重试

        trans = self.tf_buffer.lookup_transform(
            "base_link", "world", rclpy.time.Time(),
            timeout=RclpyDuration(seconds=1.0)
        )

        T = translation_matrix([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z,
        ]) @ quaternion_matrix([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w,
        ])

        p_h = np.array([*p_w, 1.0], dtype=np.float32)
        return (T @ p_h)[:3].astype(np.float32)

    def sample_random_target(self, candidates: Optional[Sequence[str]] = None):
        if self.latest_states is None:
            raise RuntimeError("latest_states 尚未就绪")

        if candidates is None:
            # 兼容旧逻辑：从所有 cup* / tube* 中选
            names = self.latest_states.name
            cups = [n for n in names if n.startswith("cup")]
            tubes = [n for n in names if n.startswith("tube")]
            all_objs = cups + tubes
            if not all_objs:
                raise RuntimeError("场景中未找到 cup* 或 tube* 模型")
            candidates = all_objs
        else:
            # 外部传进来的 candidates
            candidates = [n for n in candidates if n]  # 过滤掉 None / 空字符串
            if not candidates:
                raise RuntimeError("传入的候选目标列表为空")

        # 从候选目标中随机选一个
        name = random.choice(candidates)

        # 查询世界坐标
        pos_w = self._call_target_service_world(name)
        pos_above_w = pos_w.copy()

        # 随机高度
        pos_above_w[2] += np.random.uniform(0.05, 0.10)
        #print(f"目标 {name} 世界坐标: {pos_w}, 上方位置: {pos_above_w}")
        # 转到 base_link 坐标系
        pos_above_b = self._world_to_base(pos_above_w)
        
        self.target_pos = pos_above_b.astype(np.float32)
        self.target_ori = self.initial_ori.copy()
        #print(f"目标: {name}, target_pos={self.target_pos}, target_ori={self.target_ori}")

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 清零计数与缓存
        self.termination_mgr.reset()
        self.last_action[:] = 0.0
        self.reward_calc.reset()

        # 等核心观测就绪
        self._wait_for_multi_obs(["latest_states", "obs_feat", "joint_states"])
        
        # reset_world
        self.action_exec.reset_world()
        
        # 回 home
        self.action_exec.goto_home(duration=2.0, wait=True)
        # 记录初始姿态
        ee_pos_start, ee_ori_start = self.reward_calc.get_ee_pose()
        #print(f"初始末端位姿: pos={ee_pos_start}, ori={ee_ori_start}")
        #self.initial_ori = ee_ori_start.copy()
        self.initial_ori = [0.499999, 0.499999, 0.499999, 0.499999]

        # 初始化目标
        if self.latest_states is None:
            raise RuntimeError("latest_states 尚未就绪")

        names = self.latest_states.name
        cups  = [n for n in names if n.startswith("cup")]
        tubes = [n for n in names if n.startswith("tube")]

        if not cups and not tubes:
            raise RuntimeError("场景中未找到 cup* 或 tube* 模型")

        # 每次 reset 随机选出「一个 cup 名」和「一个 tube 名」
        cup_pick  = random.choice(cups)  if cups  else None
        tube_pick = random.choice(tubes) if tubes else None

        # 过滤掉可能为 None 的
        self.pair = [n for n in (cup_pick, tube_pick) if n is not None]

        self.sample_random_target(candidates=self.pair)

        # 等 obstacle_features 更新
        timeout = time.time() + 5.0
        while self.obs_feat is None and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.01)
        if self.obs_feat is None:
            raise RuntimeError("obs_feat 超时")

        obs = self.obs_builder.get_obs()
        info = {}
        
        return obs, info

    def step(self, action):
        # 计步、目标更新
        self.termination_mgr.step_and_check_goal()
        self.debug_step_count += 1

        # 执行动作；执行层内部将 raw action 映射为绝对关节位置目标
        self.action_exec.apply_action(action)

        # 刷新 ROS 状态
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.01)

        # 构造观测
        obs = self.obs_builder.get_obs()
        js = self.joint_states
        if js is not None:
            joint_pos_rel = np.array(js.position[:self.n_joints], dtype=np.float32) - self.home_pos
        else:
            joint_pos_rel = np.zeros(self.n_joints, dtype=np.float32)

        obs_last_action = obs[19:25]
        obs_feat = obs[25:]
        self.node.get_logger().info(
            "[step %d] raw=%s target_q=%s joint_rel=%s obs_last=%s obs_feat[min=%.3f max=%.3f]"
            % (
                self.debug_step_count,
                np.array2string(self.action_exec.last_raw_action, precision=3, suppress_small=True),
                np.array2string(self.action_exec.last_target_q, precision=3, suppress_small=True),
                np.array2string(joint_pos_rel, precision=3, suppress_small=True),
                np.array2string(obs_last_action, precision=3, suppress_small=True),
                float(np.min(obs_feat)) if obs_feat.size else float("nan"),
                float(np.max(obs_feat)) if obs_feat.size else float("nan"),
            )
        )

        # 奖励
        reward = self.reward_calc.compute_reward(action)

        terminated = False
        truncated, info = self.termination_mgr.check_truncation()
        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        return None

import os
import numpy as np
import trimesh
import fcl
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import LinkStates
from tf2_geometry_msgs import do_transform_point
from tf_transformations import (
    translation_matrix,
    quaternion_matrix,
    concatenate_matrices,
    quaternion_from_euler,
)
from urdf_parser_py.urdf import URDF, Box, Sphere, Cylinder, Mesh
from ament_index_python import get_package_share_directory
from collections import deque, defaultdict
from pipettingrobot_interfaces.msg import CollisionInfo


class CollisionChecker(Node):
    def __init__(self):
        super().__init__(
            'collision_checker',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=False,
        )
        # —— 开启仿真时间 —— #
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.get_logger().info(f"use_sim_time = {self.get_parameter('use_sim_time').value}")

        # —— 订阅 robot_description URDF —— #
        self.urdf_xml = None
        self.create_subscription(
            String, '/robot_description', self.cb_urdf,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            )
        )
        while self.urdf_xml is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.robot = URDF.from_xml_string(self.urdf_xml)

        # —— 构建碰撞模型 & 局部 AABB —— #
        # self.link_objs: link_name -> (CollisionObject, T_origin, local_min, local_max)
        self.link_objs = {}
        self._build_collision_models()

        # —— 构建运动链图，用于跳过父子直接碰撞 —— #
        self.adj_graph = defaultdict(list)
        for child, parent in self.robot.parent_map.items():
            self.adj_graph[parent].append(child)
            self.adj_graph[child].append(parent)

        # —— 订阅 Gazebo 发布的 /link_states —— #
        self.link_poses = {}  # link_name -> geometry_msgs/Pose
        self.create_subscription(
            LinkStates, '/link_states', self.cb_link_states, 10
        )

        # —— 订阅环境点云 & 发布碰撞信息 —— #
        self.cloud = None
        self.create_subscription(PointCloud2, '/obstacles', self.cb_cloud, 10)
        self.pub = self.create_publisher(CollisionInfo, '/collision_info', 10)

        # —— 定时器：0.1s 检测一次碰撞 —— #
        self.create_timer(0.1, self.on_timer)

        # —— 参数设定 —— #
        self.MAX_SKIP_DIST = 1      # 跳过直接相连的 link
        self.PENETRATION_TH = 0.03  # 穿透深度阈值
        self.PC_MAX_POINTS = self.declare_parameter('pc_max_points', 19800).value

    def _build_collision_models(self):
        """为每个 link.collision 创建 FCL 对象，并计算局部 AABB"""
        for link in self.robot.links:
            # 仅处理有碰撞体的 link
            for coll in link.collisions:
                geom = coll.geometry
                origin = coll.origin

                # 解析 origin 平移与旋转
                pos = origin.position if origin and origin.position else [0.0,0.0,0.0]
                if origin and origin.rotation:
                    rpy_or_quat = origin.rotation
                elif origin and hasattr(origin, 'rpy') and origin.rpy:
                    rpy_or_quat = origin.rpy
                else:
                    rpy_or_quat = None

                if rpy_or_quat:
                    if len(rpy_or_quat)==3:
                        rot = quaternion_from_euler(*rpy_or_quat)
                    else:
                        rot = rpy_or_quat
                else:
                    rot = [0.0,0.0,0.0,1.0]

                T_origin = concatenate_matrices(
                    translation_matrix(pos),
                    quaternion_matrix(rot)
                )

                # 基本几何体 & Mesh 支持
                if isinstance(geom, Box):
                    size = np.array(geom.size, dtype=np.float64)
                    half = size/2.0
                    local_min = -half; local_max = half
                    obj = fcl.Box(*geom.size)

                elif isinstance(geom, Sphere):
                    r = float(geom.radius)
                    local_min = np.array([-r,-r,-r])
                    local_max = np.array([ r, r, r])
                    obj = fcl.Sphere(r)

                elif isinstance(geom, Cylinder):
                    r=float(geom.radius); l=float(geom.length)
                    half_l=l/2.0
                    local_min = np.array([-r,-r,-half_l])
                    local_max = np.array([ r, r, half_l])
                    obj = fcl.Cylinder(r,l)

                elif isinstance(geom, Mesh):
                    path = self._resolve_mesh_path(geom.filename)
                    mesh = trimesh.load(path, force='mesh')
                    bvh = fcl.BVHModel()
                    bvh.beginModel(len(mesh.vertices), len(mesh.faces))
                    bvh.addSubModel(mesh.vertices, mesh.faces)
                    bvh.endModel()
                    obj = bvh
                    local_min, local_max = mesh.bounds

                else:
                    self.get_logger().warn(f"不支持几何类型: {type(geom)}")
                    continue

                co = fcl.CollisionObject(obj, fcl.Transform())
                self.link_objs[link.name] = (
                    co,
                    T_origin,
                    np.array(local_min, dtype=np.float64),
                    np.array(local_max, dtype=np.float64),
                )
                #self.get_logger().info(f"创建碰撞体: {link.name} [{type(geom).__name__}]")

    def _resolve_mesh_path(self, fn: str) -> str:
        if fn.startswith('package://'):
            pkg = fn.split('/')[2]
            base = get_package_share_directory(pkg)
            return fn.replace(f'package://{pkg}', base)
        elif os.path.exists(fn):
            return fn
        else:
            self.get_logger().error(f"Mesh 文件不存在: {fn}")
            return None

    def cb_urdf(self, msg: String):
        if self.urdf_xml is None:
            self.urdf_xml = msg.data

    def cb_link_states(self, msg: LinkStates):
        """缓存所有 link 的 world pose，并且剥离命名空间"""
        for full_name, pose in zip(msg.name, msg.pose):
            # “包名::link” -> “link”
            short = full_name.split("::")[-1]
            # 仅缓存我们在 link_objs 中真正有碰撞模型的那些 link
            if short in self.link_objs:
                self.link_poses[short] = pose
        # 检查 tool0，如果不存在，则推算
        if "tool0" in self.link_objs and "wrist3_Link" in self.link_poses:
            # 获取 wrist3_Link 的世界位姿（geometry_msgs/Pose -> 4x4矩阵）
            pose = self.link_poses["wrist3_Link"]
            T_wrist3 = concatenate_matrices(
                translation_matrix([pose.position.x, pose.position.y, pose.position.z]),
                quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            )
            # 获取 wrist3_Link 的 flange 位姿（geometry_msgs/Pose -> 4x4矩阵）
            T_flange = self.link_objs.get("flange", (None, np.eye(4)))[1]  # 默认单位阵
            T_tool0  = self.link_objs["tool0"][1]
            # 世界下 tool0 = T_wrist3 * T_flange * T_tool0
            T_world_tool0 = T_wrist3.dot(T_flange).dot(T_tool0)
            # 转 geometry_msgs/Pose 存到 link_poses
            from geometry_msgs.msg import Pose
            from tf_transformations import quaternion_from_matrix
            pos = T_world_tool0[:3, 3]
            quat = quaternion_from_matrix(T_world_tool0)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
            self.link_poses["tool0"] = pose

    def cb_cloud(self, msg: PointCloud2):
        """订阅并更新环境点云（world）"""
        # 无需转换，直接 world frame
        pts = []
        for p in pc2.read_points(msg, ('x','y','z'), skip_nans=True):
            try:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            except:
                continue
        if not pts:
            return
        arr = np.array(pts, dtype=np.float64)
        if arr.shape[0] > self.PC_MAX_POINTS:
            idx = np.random.choice(arr.shape[0], self.PC_MAX_POINTS, replace=False)
            arr = arr[idx]
        self.cloud = arr
        #self.get_logger().info(f"更新点云，共 {len(arr)} 点")

    def _kin_chain_distance(self, a: str, b: str) -> int:
        """计算链路距离，用于滤掉直接相连的 link"""
        if a == b:
            return 0
        vis = {a}
        dq = deque([(a, 0)])
        while dq:
            node, d = dq.popleft()
            for nbr in self.adj_graph[node]:
                if nbr == b:
                    return d+1
                if nbr not in vis:
                    vis.add(nbr)
                    dq.append((nbr, d+1))
        return int(1e6)

    def on_timer(self):
        now = self.get_clock().now().to_msg()

        # —— 1) 从 link_poses 获取世界位姿 & 计算 world AABB —— #
        world_aabbs = {}
        for name, (co, T_ori, loc_min, loc_max) in self.link_objs.items():
            if name not in self.link_poses:
                continue
            pose = self.link_poses[name]
            # 构建 T_link
            T_link = concatenate_matrices(
                translation_matrix([pose.position.x,
                                    pose.position.y,
                                    pose.position.z]),
                quaternion_matrix([pose.orientation.x,
                                   pose.orientation.y,
                                   pose.orientation.z,
                                   pose.orientation.w])
            )
            T_final = T_link.dot(T_ori)
            co.setTransform(fcl.Transform(T_final[:3, :3], T_final[:3, 3]))

            # 日志确认
            #self.get_logger().info(
            #    f"[Pose] {name}: x={pose.position.x:.3f}, "
            #    f"y={pose.position.y:.3f}, z={pose.position.z:.3f}"
            #)

            # 计算 AABB
            corners = np.array([
                [loc_min[0],loc_min[1],loc_min[2]],
                [loc_min[0],loc_min[1],loc_max[2]],
                [loc_min[0],loc_max[1],loc_min[2]],
                [loc_min[0],loc_max[1],loc_max[2]],
                [loc_max[0],loc_min[1],loc_min[2]],
                [loc_max[0],loc_min[1],loc_max[2]],
                [loc_max[0],loc_max[1],loc_min[2]],
                [loc_max[0],loc_max[1],loc_max[2]],
            ], dtype=np.float64)
            wp = (T_final[:3, :3].dot(corners.T) +
                  T_final[:3, 3].reshape(3,1)).T
            wmin = wp.min(axis=0) 
            wmax = wp.max(axis=0)
            world_aabbs[name] = (wmin, wmax)

        # —— 2) 自碰撞检测 —— #
        def overlap(a_min,a_max,b_min,b_max):
            return all(a_min[i] <= b_max[i] and a_max[i]>= b_min[i]
                       for i in range(3))

        self_pairs = []
        items = list(self.link_objs.items())
        for i in range(len(items)):
            ni, (co_i, *_ ) = items[i]
            if ni not in world_aabbs:
                continue
            mi, Ma = world_aabbs[ni]

            for j in range(i+1, len(items)):
                nj, (co_j, *_ ) = items[j]
                if nj not in world_aabbs:
                    continue

                # 跳过直接相连
                if self._kin_chain_distance(ni, nj) <= self.MAX_SKIP_DIST:
                    continue
                # 跳过 wrist1_Link 和 tool0 之间的检测
                if ( (ni == 'wrist1_Link' and nj == 'tool0') or 
                    (ni == 'tool0' and nj == 'wrist1_Link') ):
                    continue

                mj, Mb = world_aabbs[nj]
                if not overlap(mi, Ma, mj, Mb):
                    continue

                # FCL 精细碰撞
                req = fcl.CollisionRequest(num_max_contacts=100,
                                           enable_contact=True)
                res = fcl.CollisionResult()
                if fcl.collide(co_i, co_j, req, res) > 0 and res.contacts:
                    for c in res.contacts:
                        d = getattr(c, 'penetration_depth', 0.0)
                        if d >= self.PENETRATION_TH:
                            self_pairs.append((ni, nj))
                            break

        if self_pairs:
            msg = CollisionInfo()
            msg.collision_type = 'self'
            for a,b in self_pairs:
                msg.involved_links += [a,b]
            #self.get_logger().warn(f"自碰撞: {self_pairs}")
            self.pub.publish(msg)
            return

        # —— 3) 环境碰撞检测 —— #
        env = []
        if self.cloud is not None:
            for ni, (co, *_ ) in self.link_objs.items():
                if ni not in world_aabbs:
                    continue
                mi, Ma = world_aabbs[ni]
                mask = np.all([
                    self.cloud[:,0] >= mi[0], self.cloud[:,0] <= Ma[0],
                    self.cloud[:,1] >= mi[1], self.cloud[:,1] <= Ma[1],
                    self.cloud[:,2] >= mi[2], self.cloud[:,2] <= Ma[2],
                ], axis=0)
                pts = self.cloud[mask]
                self.get_logger().debug(f"[Env AABB 筛点] {ni}: {len(pts)} pts")
                for p in pts[:100]:
                    sphere = fcl.Sphere(0.01)
                    co_p = fcl.CollisionObject(sphere,
                                               fcl.Transform(np.eye(3), p))
                    req = fcl.CollisionRequest(num_max_contacts=1,
                                               enable_contact=True)
                    res = fcl.CollisionResult()
                    if fcl.collide(co, co_p, req, res)>0 and res.contacts:
                        env.append(ni)
                        break

        msg = CollisionInfo()
        if env:
            msg.collision_type = 'environment'
            msg.involved_links = env
            #self.get_logger().warn(f"环境碰撞: {env}")
        else:
            msg.collision_type = 'none'
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CollisionChecker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import math
import random
import numpy as np
import os
from plyfile import PlyData

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.duration import Duration
from std_msgs.msg import Header, Float32MultiArray
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from tf_transformations import (
    quaternion_matrix,
    quaternion_about_axis,
    quaternion_multiply,
    quaternion_from_euler,
    translation_matrix
)

# ------------------ Dynamic Obstacle with Orientation ------------------
class Obstacle:
    def __init__(self, pc_dir, pose_range, vel_range, size_min, size_max):
        self.pc_dir = pc_dir
        self.pose_range = pose_range
        self.vel_range = vel_range
        self.size_min = size_min
        self.size_max = size_max

        self.P = np.zeros(3, dtype=np.float32) # 中心位置 (base_link)
        self.vel = np.zeros(3, dtype=np.float32)
        self.base_q = [0.0, 0.0, 0.0, 1.0]
        self.dyn_q = [0.0, 0.0, 0.0, 1.0]
        self.rot_axis_dyn = np.array([0, 0, 1], dtype=np.float32)
        self.ang_speed_dyn = 0.0

        self.reset()

    def _rand_from_range(self, key):
        r = self.pose_range[key]
        return random.uniform(float(r[0]), float(r[1]))

    def _rand_vel_from_range(self, key):
        r = self.vel_range[key]
        return random.uniform(float(r[0]), float(r[1]))

    def reset(self):
        self.P = np.array([
            self._rand_from_range("x"),
            self._rand_from_range("y"),
            self._rand_from_range("z"),
        ], dtype=np.float32)

        roll  = self._rand_from_range("roll")
        pitch = self._rand_from_range("pitch")
        yaw   = self._rand_from_range("yaw")
        self.base_q = quaternion_from_euler(roll, pitch, yaw)
        self.dyn_q = [0.0, 0.0, 0.0, 1.0]

        self.vel = np.array([
            self._rand_vel_from_range("x"),
            self._rand_vel_from_range("y"),
            self._rand_vel_from_range("z"),
        ], dtype=np.float32)

        wx, wy, wz = self._rand_vel_from_range("roll"), self._rand_vel_from_range("pitch"), self._rand_vel_from_range("yaw")
        w = np.array([wx, wy, wz], dtype=np.float32)
        wn = np.linalg.norm(w)
        if wn < 1e-6:
            self.ang_speed_dyn = 0.0
            self.rot_axis_dyn = np.array([0, 0, 1], dtype=np.float32)
        else:
            self.ang_speed_dyn = wn
            self.rot_axis_dyn = w / wn

        ply_files = [f for f in os.listdir(self.pc_dir) if f.endswith(".ply")]
        ply = PlyData.read(os.path.join(self.pc_dir, random.choice(ply_files)))
        pts = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_d = np.max(np.linalg.norm(pts, axis=1))
        size = random.uniform(self.size_min, self.size_max)
        self.points_local = (pts / max_d * size).astype(np.float32)

    def update(self, dt):
        self.P += self.vel * dt
        if self.ang_speed_dyn > 0:
            dq = quaternion_about_axis(self.ang_speed_dyn * dt, self.rot_axis_dyn)
            self.dyn_q = quaternion_multiply(dq, self.dyn_q)

    def sample_points(self, n):
        R = quaternion_matrix(self.base_q)[:3, :3] @ quaternion_matrix(self.dyn_q)[:3, :3]
        idx = np.random.choice(len(self.points_local), size=n, replace=True)
        pts = (R @ self.points_local[idx].T).T + self.P
        return pts.astype(np.float32)

class ObstacleCloudPublisher(Node):
    def __init__(self):
        super().__init__("obstacle_cloud_publisher")

        # 参数声明
        self.declare_parameter("pc_dir", "/home/robot/pipetbot_ros2_ws/pc")
        self.declare_parameter("dyn_count", 2)
        self.declare_parameter("dyn_pts_per_obstacle", 600)
        self.declare_parameter("dyn_size_min", 0.05)
        self.declare_parameter("dyn_size_max", 0.15)
        self.declare_parameter("interval_range_s", [1.0, 2.0])
        self.declare_parameter("ee_frame", "wrist3_Link")  # 末端坐标系
        self.declare_parameter("base_frame", "base_link") # 基础坐标系

        # 读取参数
        pc_dir = self.get_parameter("pc_dir").value
        dyn_count = int(self.get_parameter("dyn_count").value)
        self.pts_per_dyn = int(self.get_parameter("dyn_pts_per_obstacle").value)
        dyn_size_min = float(self.get_parameter("dyn_size_min").value)
        dyn_size_max = float(self.get_parameter("dyn_size_max").value)
        self.interval_range_s = [float(v) for v in self.get_parameter("interval_range_s").value]
        self.ee_frame = self.get_parameter("ee_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        pose_range = {k: self.declare_parameter(f"pose_range_{k}", [0.0, 0.0]).value for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
        vel_range = {k: self.declare_parameter(f"velocity_range_{k}", [0.0, 0.0]).value for k in ["x", "y", "z", "roll", "pitch", "yaw"]}

        # 发布者
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE)
        self.pub_cloud = self.create_publisher(PointCloud2, "/obstacles", qos)
        self.pub_centers_ee = self.create_publisher(Float32MultiArray, "/obstacle_centers_ee", 10)

        # TF 监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 动态障碍物
        self.dyn_obs = [Obstacle(pc_dir, pose_range, vel_range, dyn_size_min, dyn_size_max) for _ in range(dyn_count)]
        now = self.get_clock().now()
        self.next_reset_times = [self._next_reset_time(now) for _ in range(dyn_count)]

        self.timer = self.create_timer(1.0 / 33.0, self.cb_timer)
        self.last_time = now
        self.last_print_ns = 0

    def _next_reset_time(self, now):
        delay = random.uniform(self.interval_range_s[0], self.interval_range_s[1])
        return now + Duration(seconds=delay)

    def cb_timer(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        pts_list = []
        centers_base = []

        for i, obs in enumerate(self.dyn_obs):
            if now >= self.next_reset_times[i]:
                obs.reset()
                self.next_reset_times[i] = self._next_reset_time(now)
            obs.update(dt)
            pts_list.append(obs.sample_points(self.pts_per_dyn))
            centers_base.append(obs.P) # 记录中心点 (x,y,z)

        # 1. 发布点云
        self._publish_cloud(pts_list, now)

        # 2. 计算并发布末端坐标系下的中心位置
        self._publish_ee_centers(centers_base)

    def _publish_cloud(self, pts_list, now):
        pts_all = np.vstack(pts_list) if pts_list else np.zeros((0, 3), dtype=np.float32)
        header = Header(stamp=now.to_msg(), frame_id=self.base_frame)
        fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        cloud = pc2.create_cloud(header, fields, pts_all)
        self.pub_cloud.publish(cloud)

    def _publish_ee_centers(self, centers_base):
        try:
            # 获取从 base 到 ee 的变换 T_base_ee
            # 注意：我们需要的是点在 EE 坐标系下的表示，所以变换应该是 lookup_transform(target=ee, source=base)
            trans = self.tf_buffer.lookup_transform(self.ee_frame, self.base_frame, rclpy.time.Time())
            
            # 构建变换矩阵
            r_mat = quaternion_matrix([trans.transform.rotation.x, trans.transform.rotation.y, 
                                       trans.transform.rotation.z, trans.transform.rotation.w])
            t_mat = translation_matrix([trans.transform.translation.x, trans.transform.translation.y, 
                                        trans.transform.translation.z])
            T_ee_base = t_mat @ r_mat # 这是一个 4x4 矩阵

            centers_ee_flat = []
            for p_base in centers_base:
                p_h = np.append(p_base, 1.0) # 齐次坐标
                p_ee = (T_ee_base @ p_h)[:3]
                centers_ee_flat.extend(p_ee.tolist())

            msg = Float32MultiArray()
            msg.data = centers_ee_flat
            self.pub_centers_ee.publish(msg)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
import tf2_ros

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped

class EEPosePublisher(Node):
    def __init__(self):
        super().__init__("ee_pose_publisher")

        # ---- params ----
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("source_frame", "wrist3_Link")
        self.declare_parameter("publish_topic", "/ee_pose")
        self.declare_parameter("rate", 30.0)          # Hz
        self.declare_parameter("timeout_sec", 0.03)   # can_transform 等待时间

        self.target_frame = self.get_parameter("target_frame").value
        self.source_frame = self.get_parameter("source_frame").value
        self.topic = self.get_parameter("publish_topic").value
        self.rate = float(self.get_parameter("rate").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        self.pub = self.create_publisher(PoseStamped, self.topic, 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        period = 1.0 / max(self.rate, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"Publishing {self.target_frame}->{self.source_frame} as PoseStamped on {self.topic} @ {self.rate}Hz"
        )

    
    def _tick(self):
        timeout = Duration(seconds=self.timeout_sec)
        try:
            # ✅ 用最新一帧（Time=0），不要用 now
            if not self.tf_buffer.can_transform(
                self.target_frame, self.source_frame, rclpy.time.Time(), timeout=timeout
            ):
                # 给你一个节流提示：否则你不知道它一直在失败
                self.get_logger().warn(
                    f"TF not ready: {self.target_frame}->{self.source_frame}",
                    throttle_duration_sec=2.0
                )
                return

            t = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, rclpy.time.Time()
            )

            msg = PoseStamped()
            msg.header.stamp = t.header.stamp
            msg.header.frame_id = self.target_frame
            msg.pose.position.x = t.transform.translation.x
            msg.pose.position.y = t.transform.translation.y
            msg.pose.position.z = t.transform.translation.z
            msg.pose.orientation = t.transform.rotation

            self.pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"publish failed: {e}", throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = EEPosePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

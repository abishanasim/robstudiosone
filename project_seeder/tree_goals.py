#!/usr/bin/env python3
import asyncio
import math
import threading
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from rcl_interfaces.msg import Log  # <- /rosout mirroring

# ---- CONFIG ----
WORLD_FRAME = "map"

# SDF poses copied from your world file: (x, y, z, roll, pitch, yaw)
SDF_POSES = {
    "pinea": (-2.5,  9.7,  0.0, 0.0, 0.0, 0.9),
    "pineb": (-2.5,  7.55, 0.0, 0.0, 0.0, 0.9),
    "pinec": (-2.5,  5.4,  0.0, 0.0, 0.0, 0.9),
    "pined": (-2.5,  3.25, 0.0, 0.0, 0.0, 0.9),
    "pine1": (-2.5,  1.10, 0.0, 0.0, 0.0, 0.1),
    "pine2": (-2.5, -1.05, 0.0, 0.0, 0.0, 0.3),
    "pine3": (-2.5, -3.25, 0.0, 0.0, 0.0, 0.2),
    "pine4": (-2.5, -5.40, 0.0, 0.0, 0.0, 0.0),
    "pine5": (-2.5, -7.60, 0.0, 0.0, 0.0, 0.5),
    "pine6": (-2.5, -9.70, 0.0, 0.0, 0.0, 0.9),
}

TREE_ORDER = [
    "pinea","pineb","pinec","pined","pine1","pine2","pine3","pine4","pine5","pine6"
]

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

def quat_to_yaw(q) -> float:
    if (q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0):
        return 0.0
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

class TreeGoalRunner(Node):
    def __init__(self):
        super().__init__("tree_goal_runner")

        # Tunables
        self.declare_parameter("inspection_offset", 4.0)
        self.declare_parameter("hold_seconds", 3.0)
        self.declare_parameter("max_trees", 9)
        self.declare_parameter("extra_dx", 0.0)
        self.declare_parameter("extra_dy", 0.0)
        self.declare_parameter("use_dummy_if_missing", True)
        self.declare_parameter("map_half", 12.5)
        self.declare_parameter("goal_clearance", 3.0)
        self.declare_parameter("mirror_rosout", True)
        self.declare_parameter(
            "rosout_nodes_csv",
            "waypoint_follower,bt_navigator,controller_server,planner_server"
        )

        self.dx = float(self.get_parameter("extra_dx").value)
        self.dy = float(self.get_parameter("extra_dy").value)
        self.inspect_offset = float(self.get_parameter("inspection_offset").value)
        self.hold_s = float(self.get_parameter("hold_seconds").value)
        self.max_trees = int(self.get_parameter("max_trees").value)
        self.use_dummy = bool(self.get_parameter("use_dummy_if_missing").value)
        self.map_half = float(self.get_parameter("map_half").value)
        self.goal_clr = float(self.get_parameter("goal_clearance").value)

        # Action client
        self.follow = ActionClient(self, FollowWaypoints, "follow_waypoints")

        # ---- Marker publishers with TRANSIENT_LOCAL QoS (RViz-friendly latch) ----
        qos_vis = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.marker_pub = self.create_publisher(MarkerArray, "tree_waypoints", qos_vis)
        self.highlight_pub = self.create_publisher(Marker, "marker", qos_vis)

        self._cached_markers: Optional[MarkerArray] = None
        self._marker_timer = self.create_timer(0.5, self._republish_markers)

        # /rosout mirroring
        self.mirror_rosout = bool(self.get_parameter("mirror_rosout").value)
        self.rosout_nodes = set(
            n.strip() for n in str(self.get_parameter("rosout_nodes_csv").value).split(",")
            if n.strip()
        )
        if self.mirror_rosout:
            self.create_subscription(Log, "/rosout", self._on_rosout, 100)
            self.get_logger().info(f"Mirroring /rosout for: {sorted(self.rosout_nodes)}")

        self.get_logger().info(
            f"SEQUENTIAL INSPECTION — outward offset={self.inspect_offset:.2f} m, "
            f"hold={self.hold_s:.1f} s, min goal clearance={self.goal_clr:.2f} m"
        )

    # ----------- /rosout mirroring -----------
    def _on_rosout(self, msg: Log):
        name = msg.name.lstrip('/')
        if name not in self.rosout_nodes:
            return
        text = msg.msg
        lvl = msg.level  # 10=DEBUG,20=INFO,30=WARN,40=ERROR,50=FATAL
        if lvl >= 50:
            self.get_logger().fatal(f"[{name}] {text}")
        elif lvl >= 40:
            self.get_logger().error(f"[{name}] {text}")
        elif lvl >= 30:
            self.get_logger().warn(f"[{name}] {text}")
        elif lvl >= 20:
            self.get_logger().info(f"[{name}] {text}")
        else:
            self.get_logger().debug(f"[{name}] {text}")

    # ---------------- Pose sources ----------------
    def get_pose_from_sdf(self, model_name: str) -> Optional[Pose]:
        if model_name not in SDF_POSES:
            return None
        x, y, z, r, p, yaw = SDF_POSES[model_name]
        pose = Pose()
        pose.position.x = x + self.dx
        pose.position.y = y + self.dy
        pose.position.z = z
        pose.orientation = quat_from_yaw(yaw)
        return pose

    # ---------- Helpers ----------
    def _clamp_goal(self, x: float, y: float) -> Tuple[float, float]:
        min_x, max_x = -self.map_half + self.goal_clr, self.map_half - self.goal_clr
        min_y, max_y = -self.map_half + self.goal_clr, self.map_half - self.goal_clr
        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))
        return x, y

    def inspection_waypoint(self, tree_pose: Pose) -> Pose:
        sign = 1.0 if tree_pose.position.x <= 0.0 else -1.0
        xw = tree_pose.position.x + sign * self.inspect_offset
        yw = tree_pose.position.y
        xw, yw = self._clamp_goal(xw, yw)

        wp = Pose()
        wp.position.x = xw
        wp.position.y = yw
        wp.position.z = max(0.2, tree_pose.position.z)
        face_yaw = math.atan2(tree_pose.position.y - yw, tree_pose.position.x - xw)
        wp.orientation = quat_from_yaw(face_yaw)
        return wp

    # ---------------- Nav2 FollowWaypoints ----------------
    async def send_waypoints(self, poses: List[Pose]) -> bool:
        if not self.follow.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("follow_waypoints not available (Nav2 not active)")
            return False

        goal = FollowWaypoints.Goal()
        for p in poses:
            ps = PoseStamped()
            ps.header.frame_id = WORLD_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose = p
            if (ps.pose.orientation.x == ps.pose.orientation.y ==
                ps.pose.orientation.z == ps.pose.orientation.w == 0.0):
                ps.pose.orientation.w = 1.0
            goal.poses.append(ps)

        self.get_logger().info(f"Sending {len(goal.poses)} waypoint(s)…")
        gh = await self.follow.send_goal_async(goal)
        if not gh.accepted:
            self.get_logger().warn("FollowWaypoints goal rejected")
            return False

        result = await gh.get_result_async()
        if result is None or result.result is None:
            self.get_logger().warn("No result returned (Nav2 may not be active)")
            return False

        if result.result.missed_waypoints:
            self.get_logger().warn(f"Missed indices: {result.result.missed_waypoints}")
            return False
        return True

    # ---------------- Markers ----------------
    def _make_markers(self, named_poses: List[Tuple[str, Pose]]) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        life = Duration()  # forever

        for i, (name, p) in enumerate(named_poses):
            s = Marker()
            s.header.frame_id = WORLD_FRAME
            s.header.stamp = now
            s.ns = "tree_wp"
            s.id = i * 2
            s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose = p
            if s.pose.position.z < 0.2:
                s.pose.position.z = 0.2
            s.scale.x = s.scale.y = s.scale.z = 0.6
            s.color.r, s.color.g, s.color.b, s.color.a = (0.1, 0.8, 0.2, 0.98)
            s.lifetime = life
            ma.markers.append(s)

            t = Marker()
            t.header.frame_id = WORLD_FRAME
            t.header.stamp = now
            t.ns = "tree_label"
            t.id = i * 2 + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = p.position.x
            t.pose.position.y = p.position.y
            t.pose.position.z = max(0.2, p.position.z) + 0.8
            t.scale.z = 0.35
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = name
            t.lifetime = life
            ma.markers.append(t)

        return ma

    def publish_waypoint_markers(self, named_poses: List[Tuple[str, Pose]]):
        self._cached_markers = self._make_markers(named_poses)
        self.marker_pub.publish(self._cached_markers)

        # Highlight current waypoint on /marker
        if named_poses:
            _, p = named_poses[0]
            m = Marker()
            m.header.frame_id = WORLD_FRAME
            m.ns = "tree_wp_big"
            m.id = 999
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose = p
            if m.pose.position.z < 0.2:
                m.pose.position.z = 0.2
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = (0.2, 0.5, 1.0, 0.98)
            self.highlight_pub.publish(m)

        self.get_logger().info(
            f"Published {len(self._cached_markers.markers)} marker objects "
            f"({len(named_poses)} waypoints + labels) on /tree_waypoints"
        )

    def _republish_markers(self):
        if self._cached_markers is not None:
            self.marker_pub.publish(self._cached_markers)

# ---------------- Main async flow with a background executor ----------------
async def main_async():
    rclpy.init()
    node = TreeGoalRunner()

    # Start a background executor so timers, actions, and subscriptions work
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    node.get_logger().info("Executor spinning in background (markers will refresh @0.5s).")

    try:
        # Collect trees in order (up to max_trees)
        tree_named_poses: List[Tuple[str, Pose]] = []
        for name in TREE_ORDER:
            if len(tree_named_poses) >= node.max_trees:
                break
            p = node.get_pose_from_sdf(name)
            if p is not None:
                tree_named_poses.append((name, p))
            else:
                node.get_logger().warn(f"Skipping '{name}' (no pose)")

        if not tree_named_poses and node.use_dummy:
            node.get_logger().warn("No tree poses; using DUMMY waypoints to test.")
            dummy_pts = [(5.0, 0.0), (5.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
            for i, (x, y) in enumerate(dummy_pts):
                p = Pose()
                p.position.x = x
                p.position.y = y
                p.orientation.w = 1.0
                tree_named_poses.append((f"dummy_{i+1}", p))

        # Sequential: one tree at a time
        for idx, (name, tree_pose) in enumerate(tree_named_poses, start=1):
            wp = node.inspection_waypoint(tree_pose)
            node.get_logger().info(
                f"[{idx}/{len(tree_named_poses)}] {name}: waypoint "
                f"({wp.position.x:.2f}, {wp.position.y:.2f}, yaw={quat_to_yaw(wp.orientation):.2f})"
            )
            node.publish_waypoint_markers([(name, wp)])

            ok = await node.send_waypoints([wp])
            if ok:
                node.get_logger().info(f"{name}: reached; holding {node.hold_s:.1f}s facing the tree")
                if node.hold_s > 0.0:
                    await asyncio.sleep(node.hold_s)
            else:
                node.get_logger().warn(f"{name}: could not reach waypoint (continuing)")

        node.get_logger().info("Inspection complete for all trees.")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

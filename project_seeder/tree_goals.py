#!/usr/bin/env python3
import asyncio
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

# ---- CONFIG ----
WORLD_FRAME = "map"

# SDF poses copied from your world file: (x, y, z, roll, pitch, yaw)
SDF_POSES = {
    # row A (north)
    "pinea": (-2.5,  9.7,  0.0, 0.0, 0.0, 0.9),
    "pineb": (-2.5,  7.55, 0.0, 0.0, 0.0, 0.9),
    "pinec": (-2.5,  5.4,  0.0, 0.0, 0.0, 0.9),
    "pined": (-2.5,  3.25, 0.0, 0.0, 0.0, 0.9),

    # central strip
    "pine1": (-2.5,  1.10, 0.0, 0.0, 0.0, 0.1),
    "pine2": (-2.5, -1.05, 0.0, 0.0, 0.0, 0.3),
    "pine3": (-2.5, -3.25, 0.0, 0.0, 0.0, 0.2),
    "pine4": (-2.5, -5.40, 0.0, 0.0, 0.0, 0.0),
    "pine5": (-2.5, -7.60, 0.0, 0.0, 0.0, 0.5),
    "pine6": (-2.5, -9.70, 0.0, 0.0, 0.0, 0.9),
}
TREE_NAMES = list(SDF_POSES.keys())

# Try Gazebo Classic service first (fast + simple); fall back to /model/<name>/pose
try:
    from gazebo_msgs.srv import GetModelState
    HAVE_CLASSIC = True
except Exception:
    HAVE_CLASSIC = False


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

        # Parameters (tune at runtime with --ros-args -p name:=value)
        self.declare_parameter("approach_distance", 2.0)   # distance in front of each tree
        self.declare_parameter("extra_dx", 0.0)            # global nudges (map frame)
        self.declare_parameter("extra_dy", 0.0)
        self.declare_parameter("use_dummy_if_missing", True)

        # Map bounds (your walls are at ±12.5)
        self.declare_parameter("map_half", 12.5)

        # For visuals/markers clamp
        self.declare_parameter("safety_margin", 1.0)

        # NEW: minimum clearance from walls for actual GOALS (stronger than safety_margin)
        self.declare_parameter("goal_clearance", 2.5)      # <-- keeps goals well inside

        # Inward bias (“purple-dot”)
        self.declare_parameter("blend_to_center", 0.4)     # 0→pure front, 1→pure inward
        self.declare_parameter("face_along_path", True)    # face along approach vector

        self.approach_dist = float(self.get_parameter("approach_distance").value)
        self.dx = float(self.get_parameter("extra_dx").value)
        self.dy = float(self.get_parameter("extra_dy").value)
        self.use_dummy = bool(self.get_parameter("use_dummy_if_missing").value)
        self.map_half = float(self.get_parameter("map_half").value)
        self.margin = float(self.get_parameter("safety_margin").value)
        self.goal_clr = float(self.get_parameter("goal_clearance").value)
        self.blend = float(self.get_parameter("blend_to_center").value)
        self.face_along = bool(self.get_parameter("face_along_path").value)

        # Nav2 action
        self.follow = ActionClient(self, FollowWaypoints, "follow_waypoints")

        # Gazebo Classic client (optional fallback)
        self.classic_cli = None
        if HAVE_CLASSIC:
            self.classic_cli = self.create_client(GetModelState, "/gazebo/get_model_state")

        # Markers
        self.marker_pub = self.create_publisher(MarkerArray, "tree_waypoints", 10)
        self.highlight_pub = self.create_publisher(Marker, "marker", 10)
        self._cached_markers: Optional[MarkerArray] = None
        self._marker_timer = self.create_timer(0.5, self._republish_markers)

        self.get_logger().info(
            f"BATCH MODE (one per tree) — approach={self.approach_dist:.2f} m, "
            f"blend={self.blend:.2f}, goal_clearance={self.goal_clr:.2f} m"
        )

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

    def get_pose_via_classic(self, model_name: str) -> Optional[Pose]:
        if not self.classic_cli:
            return None
        if not self.classic_cli.wait_for_service(timeout_sec=0.5):
            return None
        req = GetModelState.Request()
        req.model_name = model_name
        resp = self.classic_cli.call(req)
        if not resp.success:
            self.get_logger().warn(f"[classic] no model '{model_name}'")
            return None
        p = resp.pose
        if (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) == (0, 0, 0, 0):
            p.orientation.w = 1.0
        p.position.x += self.dx
        p.position.y += self.dy
        return p

    async def get_pose_via_topic(self, model_name: str, timeout: float = 2.0) -> Optional[Pose]:
        topic = f"/model/{model_name}/pose"
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        def cb(msg: Pose):
            if not fut.done():
                fut.set_result(msg)

        sub = self.create_subscription(Pose, topic, cb, 10)
        self.get_logger().info(f"[gz] waiting for pose on {topic}")
        try:
            p = await asyncio.wait_for(fut, timeout=timeout)
            p.position.x += self.dx
            p.position.y += self.dy
            return p
        except asyncio.TimeoutError:
            self.get_logger().warn(f"[gz] timeout waiting for {topic}")
            return None
        finally:
            self.destroy_subscription(sub)

    async def get_tree_pose(self, name: str) -> Optional[Pose]:
        p = self.get_pose_from_sdf(name)
        if p is not None:
            return p
        p = self.get_pose_via_classic(name)
        if p is not None:
            return p
        return await self.get_pose_via_topic(name)

    # ---------- Helpers ----------
    def _clamp(self, x: float, y: float, margin: float) -> Tuple[float, float]:
        min_x, max_x = -self.map_half + margin, self.map_half - margin
        min_y, max_y = -self.map_half + margin, self.map_half - margin
        x = max(min_x, min(max_x, x))
        y = max(min_y, min(max_y, y))
        return x, y

    def _inside(self, x: float, y: float, margin: float) -> bool:
        min_x, max_x = -self.map_half + margin, self.map_half - margin
        min_y, max_y = -self.map_half + margin, self.map_half - margin
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    # ---------- Safe waypoint placement (inward bias + guaranteed clearance) ----------
    def safe_waypoint_in_front(self, p: Pose, distance: float) -> Pose:
        """
        Place waypoint at 'distance' using a blend of front-of-tree and inward-to-center.
        Always keep the result at least goal_clearance meters away from all walls.
        If the tree is itself close to a wall, auto-bias fully inward.
        """
        yaw = quat_to_yaw(p.orientation)

        # How close is the TREE to walls relative to the strong goal box?
        min_x, max_x = -self.map_half + self.goal_clr, self.map_half - self.goal_clr
        min_y, max_y = -self.map_half + self.goal_clr, self.map_half - self.goal_clr
        d_edge_tree = min(
            max_x - p.position.x,
            p.position.x - min_x,
            max_y - p.position.y,
            p.position.y - min_y,
        )

        # Unit vectors
        v_front = (math.cos(yaw), math.sin(yaw))
        to_center = (-p.position.x, -p.position.y)
        n = math.hypot(to_center[0], to_center[1]) or 1.0
        v_in = (to_center[0] / n, to_center[1] / n)

        # Start with user blend; if tree is near the wall, force stronger inward bias
        alpha = max(0.0, min(1.0, self.blend))
        if d_edge_tree < (self.goal_clr + 0.5):
            alpha = max(alpha, 0.9)      # close to wall → bias harder
        if d_edge_tree < (self.goal_clr + 0.2):
            alpha = 1.0                  # very close → pure inward

        vx = (1.0 - alpha) * v_front[0] + alpha * v_in[0]
        vy = (1.0 - alpha) * v_front[1] + alpha * v_in[1]
        vn = math.hypot(vx, vy) or 1.0
        vx, vy = vx / vn, vy / vn

        # Candidate waypoint
        xw = p.position.x + distance * vx
        yw = p.position.y + distance * vy

        # Keep the GOAL inside the stronger goal box
        if not self._inside(xw, yw, self.goal_clr):
            xw = p.position.x + distance * v_in[0]
            yw = p.position.y + distance * v_in[1]
        xw, yw = self._clamp(xw, yw, self.goal_clr)

        wp = Pose()
        wp.position.x = xw
        wp.position.y = yw
        wp.position.z = max(0.2, p.position.z)

        # Orientation
        if self.face_along:
            face_yaw = math.atan2(vy, vx)             # along approach vector
        else:
            face_yaw = math.atan2(p.position.y - yw, p.position.x - xw)  # toward tree
        wp.orientation = quat_from_yaw(face_yaw)
        return wp

    # ---------------- Nav2 FollowWaypoints ----------------
    async def send_waypoints(self, poses: List[Pose]) -> bool:
        if not self.follow.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("follow_waypoints not available (Nav2 not active)")
            return False
        goal = FollowWaypoints.Goal()
        now = self.get_clock().now().to_msg()
        for p in poses:
            ps = PoseStamped()
            ps.header.frame_id = WORLD_FRAME
            ps.header.stamp = now
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
            self.get_logger().warn(f"Missed waypoint indices: {result.result.missed_waypoints}")
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
        # Highlight first
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
        self.get_logger().info(f"Published {len(self._cached_markers.markers)} waypoint markers")

    def _republish_markers(self):
        if self._cached_markers is not None:
            self.marker_pub.publish(self._cached_markers)


# ---------------- Main async flow ----------------
async def main_async():
    rclpy.init()
    node = TreeGoalRunner()

    # Collect tree poses
    tree_named_poses: List[Tuple[str, Pose]] = []
    for name in TREE_NAMES:
        p = await node.get_tree_pose(name)
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

    # ---------- Build ONE waypoint per tree with guaranteed clearance ----------
    named_wps: List[Tuple[str, Pose]] = []
    for name, tree_pose in tree_named_poses:
        final_wp = node.safe_waypoint_in_front(tree_pose, node.approach_dist)
        named_wps.append((name, final_wp))

    node.publish_waypoint_markers(named_wps)

    all_wps = [wp for _, wp in named_wps]
    node.get_logger().info(
        f"Sending {len(all_wps)} waypoint(s) for {len(tree_named_poses)} tree(s) (1 per tree)…"
    )
    ok = await node.send_waypoints(all_wps)
    if ok:
        node.get_logger().info(f"Reached all {len(all_wps)} waypoints.")
    else:
        node.get_logger().warn("Some waypoints were missed or the goal was rejected.")

    node.get_logger().info("All waypoints processed.")
    node.destroy_node()
    rclpy.shutdown()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

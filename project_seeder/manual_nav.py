#!/usr/bin/env python3
import re
import asyncio
import math
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.duration import Duration as RclpyDuration

from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints, NavigateToPose, Spin
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as MsgDuration
from tf2_ros import Buffer, TransformListener

DEFAULT_WORLD_FRAME = "map"

# SDF poses for your five pines (x, y, z, r, p, yaw)
SDF_POSES = {
    "pine1": (-2.5,  9.0, 0.0, 0.0, 0.0, 0.0),
    "pine2": (-2.5,  4.5, 0.0, 0.0, 0.0, 0.0),
    "pine3": (-2.5,  0.0, 0.0, 0.0, 0.0, 0.0),
    "pine4": (-2.5, -4.5, 0.0, 0.0, 0.0, 0.0),
    "pine5": (-2.5, -9.0, 0.0, 0.0, 0.0, 0.0),
}
TREE_NAMES = list(SDF_POSES.keys())


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def waypoint_beside_tree(tree_pose: Pose, x_offset: float, face_tree: bool = False) -> Pose:
    wp = Pose()
    wp.position.x = tree_pose.position.x + x_offset
    wp.position.y = tree_pose.position.y
    wp.position.z = max(0.2, tree_pose.position.z)
    yaw = math.atan2(tree_pose.position.y - wp.position.y,
                     tree_pose.position.x - wp.position.x) if face_tree else 0.0
    wp.orientation = quat_from_yaw(yaw)
    return wp


class ManualNav(Node):
    """
    Single-shot manual runner:
      - Waits for /seeder/manual_mode=True
      - Waits for /seeder/target_tree and a /seeder/manual_send pulse
      - Runs the full per-tree sequence ONCE
      - Returns to idle and waits for another /seeder/manual_send
    """
    def __init__(self):
        super().__init__("manual_nav")

        # Params (kept in sync with tree_goals semantics)
        self.declare_parameter("world_frame", DEFAULT_WORLD_FRAME)
        self.declare_parameter("x_offset_world", 4.5)
        self.declare_parameter("per_goal_timeout_sec", 120.0)
        self.declare_parameter("base_frame", "base_link")

        # LiDAR / detection
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("front_check_window_deg", 12.0)
        self.declare_parameter("goal_match_tolerance_m", 1.0)
        self.declare_parameter("gap_check_timeout_sec", 1.5)
        self.declare_parameter("tree_expected_dist_min", 0.5)
        self.declare_parameter("tree_expected_dist_max", 4.0)
        self.declare_parameter("gap_report_topic", "/gap_report")

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.x_offset = float(self.get_parameter("x_offset_world").value)
        self.per_goal_timeout = float(self.get_parameter("per_goal_timeout_sec").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.front_window = float(self.get_parameter("front_check_window_deg").value)
        self.goal_match_tol = float(self.get_parameter("goal_match_tolerance_m").value)
        self.gap_check_timeout = float(self.get_parameter("gap_check_timeout_sec").value)
        self.tree_dmin = float(self.get_parameter("tree_expected_dist_min").value)
        self.tree_dmax = float(self.get_parameter("tree_expected_dist_max").value)
        self.gap_report_topic = str(self.get_parameter("gap_report_topic").value)

        # ---------- Treatment tanks (same as tree_goals) ----------
        self.tank_ml = {
            "Copper oxychloride": 10.0,
            "Phosphite injections": 10.0,
            "Mancozeb": 10.0,
        }
        self.default_dose_ml = 5.0
        self.min_dose_ml = 0.5
        self.max_dose_ml = 10.0

        # State
        self.manual_enabled = False
        self.current_target_name: Optional[str] = None
        self._busy = False  # block re-entry while a single run is active

        self._min_front = float("inf")
        self._last_scan_time: Optional[Time] = None

        # TF + Actions
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.follow = ActionClient(self, FollowWaypoints, "follow_waypoints")
        self.nav2 = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_ac = ActionClient(self, Spin, "spin")

        # Pubs/subs
        self.status_pub = self.create_publisher(String, "/seeder/status", 10)
        self.progress_pub = self.create_publisher(Float32, "mission_progress", 10)
        self.gap_pub = self.create_publisher(String, self.gap_report_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "tree_waypoints", 10)
        self.highlight_pub = self.create_publisher(Marker, "marker", 10)
        self.scan_done_pub = self.create_publisher(Marker, "tree_scan_done", 10)
        self.scan_req_pub = self.create_publisher(String, "/scan_request", 10)

        # QoS: latch manual_mode from GUI
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Bool,   "/seeder/manual_mode", self._on_manual, qos_transient)
        self.create_subscription(String, "/seeder/target_tree", self._on_target, 10)
        self.create_subscription(Bool,   "/seeder/manual_send", self._on_send,   10)
        self.create_subscription(LaserScan, self.scan_topic,    self._on_scan,   10)

        self._say("manual_nav ready. Turn Manual ON in GUI, pick a tree, then click 'Send Manual Control'.")

    # ---------- basic callbacks ----------
    def _say(self, text: str):
        self.get_logger().info(text)
        try:
            self.status_pub.publish(String(data=text))
        except Exception:
            pass

    def _on_manual(self, msg: Bool):
        self.manual_enabled = bool(msg.data)
        self._say(f"manual_mode = {self.manual_enabled}")

    def _on_target(self, msg: String):
        name = (msg.data or "").strip().lower()
        if name in SDF_POSES:
            self.current_target_name = name
            self._say(f"Target set to: {name}")
        else:
            self.get_logger().warn(f"Unknown target '{name}'")

    def _on_scan(self, msg: LaserScan):
        a_min, a_inc = msg.angle_min, msg.angle_increment
        n = len(msg.ranges)
        window = math.radians(self.front_window)
        mval = float("inf")
        for i in range(n):
            a = a_min + i*a_inc
            if -window <= a <= window:
                r = msg.ranges[i]
                if math.isfinite(r) and r > 0.0:
                    mval = min(mval, r)
        self._min_front = mval
        self._last_scan_time = self.get_clock().now()

    def _on_send(self, _msg: Bool):
        """Only act when manual is enabled AND a valid target is selected."""
        if not self.manual_enabled:
            self.get_logger().warn("Manual send ignored (manual_mode is False).")
            return
        if not self.current_target_name:
            self.get_logger().warn("Manual send ignored (no target).")
            return
        if self._busy:
            self.get_logger().warn("Manual run already in progress; wait for it to finish.")
            return
        asyncio.create_task(self._run_single_tree(self.current_target_name))

    # ---------- pose/TF helpers ----------
    def _pose_from_sdf(self, model: str) -> Optional[Pose]:
        if model not in SDF_POSES:
            return None
        x, y, z, r, p, yaw = SDF_POSES[model]
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = quat_from_yaw(yaw)
        return pose

    async def _current_base_pose_yaw(self) -> Tuple[float, float, float]:
        tf = await self.tf_buffer.lookup_transform_async(self.world_frame, self.base_frame, Time())
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        return x, y, yaw

    def _dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    # ---------- spin helpers ----------
    def _normalize_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0*math.pi
        while a < -math.pi:
            a += 2.0*math.pi
        return a

    async def _wait_for_server(self, client: ActionClient, label: str, tries: int = 50, dt: float = 0.1) -> bool:
        self.get_logger().info(f"Waiting for {label} server...")
        for _ in range(tries):
            if client.wait_for_server(timeout_sec=dt):
                return True
            await asyncio.sleep(dt)
        self.get_logger().error(f"{label} server not available")
        return False

    async def _spin_once(self, delta: float, time_allowance_s: float) -> Optional[int]:
        if not await self._wait_for_server(self.spin_ac, "spin"):
            self.get_logger().warn("Spin action server not available")
            return None
        goal = Spin.Goal()
        goal.target_yaw = delta
        goal.time_allowance = RclpyDuration(seconds=time_allowance_s).to_msg()
        gh = await self.spin_ac.send_goal_async(goal)
        if not gh.accepted:
            self.get_logger().warn("Spin goal not accepted")
            return None
        res = await gh.get_result_async()
        return getattr(res, "status", None)

    async def _spin_to_face(self, target_x: float, target_y: float, timeout_sec: float = 12.0) -> bool:
        angle_tolerance = math.radians(20.0)
        nudge_window   = math.radians(35.0)

        x_now, y_now, yaw_now = await self._current_base_pose_yaw()
        desired = math.atan2(target_y - y_now, target_x - x_now)
        delta = self._normalize_angle(desired - yaw_now)
        if abs(delta) <= angle_tolerance:
            return True

        _ = await self._spin_once(delta, timeout_sec)

        x2, y2, yaw2 = await self._current_base_pose_yaw()
        residual = self._normalize_angle(math.atan2(target_y - y2, target_x - x2) - yaw2)
        if abs(residual) <= angle_tolerance:
            return True
        if abs(residual) <= nudge_window:
            _ = await self._spin_once(residual, 4.0)
            x3, y3, yaw3 = await self._current_base_pose_yaw()
            residual2 = self._normalize_angle(math.atan2(target_y - y3, target_x - x3) - yaw3)
            return abs(residual2) <= angle_tolerance
        return False

    # ---------- stationary helper (ported from tree_goals) ----------
    async def _wait_until_stationary(self, max_wait_s: float = 5.0, check_dt: float = 0.3, thresh: float = 0.02) -> bool:
        """
        Returns True if robot becomes stationary (motion < thresh over check_dt) within max_wait_s.
        """
        self.get_logger().info("Step 6: Checking if Husky is stationary...")
        deadline = self.get_clock().now() + RclpyDuration(seconds=max_wait_s)
        while self.get_clock().now() < deadline:
            try:
                x1, y1, _ = await self._current_base_pose_yaw()
                await asyncio.sleep(check_dt)
                x2, y2, _ = await self._current_base_pose_yaw()
                if self._dist(x1, y1, x2, y2) < thresh:
                    self.get_logger().info("Husky stationary confirmed.")
                    return True
            except Exception as e:
                self.get_logger().warn(f"Stationary check error: {e}")
                break
        self.get_logger().warn("Husky not stationary within timeout; proceeding anyway.")
        return False

    # ---------- lidar gates ----------
    async def _pregoal_occluded(self, goal_x: float, goal_y: float) -> Tuple[bool, float, float]:
        # wait briefly for a fresh scan
        deadline = self.get_clock().now() + RclpyDuration(seconds=0.5)
        while self.get_clock().now() < deadline:
            if self._last_scan_time is not None:
                age = (self.get_clock().now() - self._last_scan_time).nanoseconds * 1e-9
                if age < 0.5:
                    break
            await asyncio.sleep(0.05)

        try:
            x, y, _ = await self._current_base_pose_yaw()
        except Exception:
            return (False, float("inf"), float("inf"))

        goal_d = self._dist(x, y, goal_x, goal_y)
        lidar_d = self._min_front
        if math.isfinite(lidar_d) and abs(lidar_d - goal_d) <= self.goal_match_tol:
            return (True, lidar_d, goal_d)
        return (False, lidar_d, goal_d)

    async def _check_tree_gap(self, tree_name: str) -> bool:
        deadline = self.get_clock().now() + RclpyDuration(seconds=self.gap_check_timeout)
        while self.get_clock().now() < deadline:
            if self._last_scan_time is not None:
                age = (self.get_clock().now() - self._last_scan_time).nanoseconds * 1e-9
                if age < 0.5:
                    break
            await asyncio.sleep(0.05)

        if self._last_scan_time is None:
            self.get_logger().warn("No LiDAR yet; proceeding.")
            return True
        if (self.get_clock().now() - self._last_scan_time).nanoseconds * 1e-9 >= 0.5:
            self.get_logger().warn("Stale LiDAR; proceeding.")
            return True

        d = self._min_front
        if math.isfinite(d) and self.tree_dmin <= d <= self.tree_dmax:
            return True

        msg = f"{tree_name}: GAP DETECTED (front_min={d:.2f} m not in [{self.tree_dmin:.2f}, {self.tree_dmax:.2f}]). Replant needed."
        self.get_logger().warn(msg)
        self.gap_pub.publish(String(data=msg))

        # red cylinder at expected location
        expected = self._pose_from_sdf(tree_name)
        if expected:
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "gap_detected"
            m.id = 5000 + (TREE_NAMES.index(tree_name) if tree_name in TREE_NAMES else 999)
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.4
            m.pose.position.x = expected.position.x
            m.pose.position.y = expected.position.y
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.9
            self.highlight_pub.publish(m)
        return False

    # ---------- action senders ----------
    async def _send_follow_waypoints(self, poses: List[Pose]) -> Tuple[bool, str]:
        if not await self._wait_for_server(self.follow, "follow_waypoints"):
            return False, "server_unavailable"
        goal = FollowWaypoints.Goal()
        now = self.get_clock().now().to_msg()
        for p in poses:
            ps = PoseStamped()
            ps.header.frame_id = self.world_frame
            ps.header.stamp = now
            ps.pose = p
            if (ps.pose.orientation.x == ps.pose.orientation.y ==
                ps.pose.orientation.z == ps.pose.orientation.w == 0.0):
                ps.pose.orientation.w = 1.0
            goal.poses.append(ps)
        gh = await self.follow.send_goal_async(goal)
        if not gh.accepted:
            return False, "rejected"
        try:
            res = await asyncio.wait_for(gh.get_result_async(), timeout=self.per_goal_timeout)
        except asyncio.TimeoutError:
            try:
                await gh.cancel_goal_async()
            except Exception:
                pass
            return False, "timeout"
        if res is None or res.result is None:
            return False, "no_result"
        missed = getattr(res.result, "missed_waypoints", [])
        if missed:
            return False, f"missed_indices={list(missed)}"
        return True, "succeeded"

    async def _send_nav_to_pose(self, pose: Pose) -> Tuple[bool, str]:
        if not await self._wait_for_server(self.nav2, "navigate_to_pose"):
            return False, "server_unavailable"
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.world_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose = pose
        if (goal.pose.pose.orientation.x == goal.pose.pose.orientation.y ==
            goal.pose.pose.orientation.z == goal.pose.pose.orientation.w == 0.0):
            goal.pose.pose.orientation.w = 1.0
        gh = await self.nav2.send_goal_async(goal)
        if not gh.accepted:
            return False, "rejected"
        try:
            res = await asyncio.wait_for(gh.get_result_async(), timeout=self.per_goal_timeout)
        except asyncio.TimeoutError:
            try:
                await gh.cancel_goal_async()
            except Exception:
                pass
            return False, "timeout"
        status = getattr(res, "status", None)
        return (status == 4, f"status={status}")

    async def _go_to_waypoint(self, pose: Pose) -> bool:
        ok, why = await self._send_follow_waypoints([pose])
        if ok:
            return True
        self.get_logger().warn(f"FollowWaypoints failed: {why}; trying NavigateToPose")
        ok2, _ = await self._send_nav_to_pose(pose)
        return ok2

    # ---------- markers ----------
    def _infection_color_rgba(self, infection_text: str):
        s = (infection_text or "").lower()
        if "healthy" in s:
            return (0.0, 1.0, 0.0, 1.0)
        if "myrtle rust" in s or "rust" in s:
            return (1.0, 1.0, 0.0, 1.0)
        if "phytophthora" in s or "cinnamon fungus" in s:
            return (1.0, 0.0, 0.0, 1.0)
        if "eucalyptus leaf blister" in s or "blister" in s:
            return (0.0, 0.4, 1.0, 1.0)
        if "dead" in s:
            return (0.0, 0.0, 0.0, 1.0)
        return (0.0, 0.0, 0.0, 1.0)

    def publish_scan_complete_cylinder(self, tree_name: str, tree_pose: Pose, infection_text: str):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "scan_done"
        try:
            m.id = 1000 + TREE_NAMES.index(tree_name)
        except ValueError:
            m.id = 1999
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale.x = 0.35
        m.scale.y = 0.35
        m.scale.z = 0.60
        m.pose.position.x = float(tree_pose.position.x)
        m.pose.position.y = float(tree_pose.position.y)
        m.pose.position.z = max(0.0, float(tree_pose.position.z)) + m.scale.z * 0.5
        m.pose.orientation.w = 1.0
        r, g, b, a = self._infection_color_rgba(infection_text)
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a
        m.lifetime = MsgDuration()
        self.scan_done_pub.publish(m)

    def publish_waypoint_marker(self, name: str, wp: Pose):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        s = Marker()
        s.header.frame_id = self.world_frame
        s.header.stamp = stamp
        s.ns = "tree_waypoints_spheres"
        s.id = 1
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position.x = wp.position.x
        s.pose.position.y = wp.position.y
        s.pose.position.z = max(0.2, wp.position.z)
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.25
        s.color.r = 0.2
        s.color.g = 0.8
        s.color.b = 1.0
        s.color.a = 0.9
        s.lifetime = MsgDuration()
        ma.markers.append(s)

        t = Marker()
        t.header.frame_id = self.world_frame
        t.header.stamp = stamp
        t.ns = "tree_waypoints_labels"
        t.id = 10001
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = wp.position.x
        t.pose.position.y = wp.position.y
        t.pose.position.z = max(0.2, wp.position.z) + 0.35
        t.pose.orientation.w = 1.0
        t.scale.z = 0.30
        t.color.r = t.color.g = t.color.b = 1.0
        t.color.a = 0.95
        t.text = name.capitalize()
        t.lifetime = MsgDuration()
        ma.markers.append(t)

        self.marker_pub.publish(ma)

        big = Marker()
        big.header.frame_id = self.world_frame
        big.ns = "tree_wp_big"
        big.id = 999
        big.type = Marker.SPHERE
        big.action = Marker.ADD
        big.pose = wp
        if big.pose.position.z < 0.2:
            big.pose.position.z = 0.2
        big.scale.x = big.scale.y = big.scale.z = 1.0
        big.color.r, big.color.g, big.color.b, big.color.a = (0.2, 0.5, 1.0, 0.98)
        self.highlight_pub.publish(big)

    # ---------- dosing helpers (ported) ----------
    def _parse_dose_ml(self, text: str) -> Optional[float]:
        """
        Accepts either 'Dose: X ml' or 'dose_ml=X' inside the scan report text.
        Returns float or None.
        """
        if not text:
            return None
        m = re.search(r"[Dd]ose:\s*([0-9]*\.?[0-9]+)\s*ml", text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        m2 = re.search(r"dose_ml\s*=\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
        if m2:
            try:
                return float(m2.group(1))
            except Exception:
                pass
        return None

    def _print_treatments_block(self, infection_text: str, dose_ml: Optional[float] = None, chosen: Optional[str] = None) -> float:
        """
        Prints a treatment block identical in style to tree_goals/color_3d_detector:
          - Available treatments
          - Chosen solution
          - Dose: X.X ml
        Deducts from tank with safety clamps, returns actual ml dispensed.
        """
        tlc = (infection_text or "").lower()

        # Healthy / Dead -> no dosing
        if "healthy" in tlc or "dead" in tlc:
            return 0.0

        if not chosen:
            if "myrtle rust" in tlc:
                chosen = "Copper oxychloride"
            elif "cinnamon fungus" in tlc or "phytophthora" in tlc:
                chosen = "Phosphite injections"
            elif "eucalyptus leaf blister" in tlc:
                chosen = "Mancozeb"
            else:
                chosen = "N/A"

        title_fix = {
            "phosphite injections": "Phosphite Injections",
            "copper oxychloride":   "Copper oxychloride",
            "mancozeb":             "Mancozeb",
            "n/a":                  "N/A",
        }
        chosen_pest = title_fix.get(chosen.lower(), chosen)

        if chosen not in self.tank_ml:
            print("- Available treatments:")
            print("  Copper oxychloride")
            print("  Phosphite injections")
            print("  Mancozeb")
            print(f"Chosen solution: {chosen_pest}")
            return 0.0

        if dose_ml is None:
            dose_ml = self.default_dose_ml
        try:
            dose_ml = float(dose_ml)
        except Exception:
            dose_ml = self.default_dose_ml
        dose_ml = max(self.min_dose_ml, min(self.max_dose_ml, dose_ml))

        remaining = self.tank_ml.get(chosen, 0.0)
        actual = dose_ml if dose_ml <= remaining else remaining
        self.tank_ml[chosen] = max(0.0, remaining - actual)

        print("- Available treatments:")
        print("  Copper oxychloride")
        print("  Phosphite injections")
        print("  Mancozeb")
        print(f"Chosen solution: {chosen_pest}")
        print(f"Dose: {actual:.1f} ml")

        return float(actual)

    # ---------- SINGLE RUN ----------
    async def _run_single_tree(self, tree_name: str):
        self._busy = True
        try:
            # start fresh progress for this one goal
            self.progress_pub.publish(Float32(data=0.0))

            tree_pose = self._pose_from_sdf(tree_name)
            if not tree_pose:
                self.get_logger().error(f"No SDF pose for {tree_name}")
                return

            normal_wp = waypoint_beside_tree(tree_pose, self.x_offset, face_tree=False)
            self.publish_waypoint_marker(tree_name, normal_wp)

            # Step 2: face the (normal) goal
            self._say(f"[Manual] Facing goal for {tree_name} (x={normal_wp.position.x:.2f}, y={normal_wp.position.y:.2f})")
            _ = await self._spin_to_face(normal_wp.position.x, normal_wp.position.y, timeout_sec=8.0)

            # Step 3: occlusion test -> mirror if needed
            blocked, lidar_d, goal_d = await self._pregoal_occluded(normal_wp.position.x, normal_wp.position.y)
            if blocked:
                self.get_logger().warn(
                    f"Obstacle on goal line (LiDAR {lidar_d:.2f} m vs goal {goal_d:.2f} m, tol �{self.goal_match_tol:.1f}). Mirroring."
                )
                wp = waypoint_beside_tree(tree_pose, -self.x_offset, face_tree=False)
                _ = await self._spin_to_face(wp.position.x, wp.position.y, timeout_sec=6.0)
            else:
                wp = normal_wp

            # Step 4: travel
            ok = await self._go_to_waypoint(wp)

            # proximity acceptance (0.6 m)
            reached = ok
            if not ok:
                try:
                    x_now, y_now, _ = await self._current_base_pose_yaw()
                    if self._dist(x_now, y_now, wp.position.x, wp.position.y) <= 0.6:
                        self.get_logger().warn("Nav2 failed but within 0.6 m; accepting.")
                        reached = True
                except Exception as e:
                    self.get_logger().warn(f"Proximity check failed: {e}")

            if not reached:
                self.get_logger().warn(f"FAILED to reach waypoint for {tree_name}.")
                self.progress_pub.publish(Float32(data=100.0))
                return

            self._say(f"Goal reached for {tree_name}.")

            # Step 6: stationary gate
            await self._wait_until_stationary(5.0, 0.3, 0.02)

            # Step 7/8: turn to tree
            tx, ty = tree_pose.position.x, tree_pose.position.y
            _ = await self._spin_to_face(tx, ty, timeout_sec=12.0)

            # Step 9: tree present?
            present = await self._check_tree_gap(tree_name)
            if not present:
                self.progress_pub.publish(Float32(data=100.0))
                return

            # Step 9(cont): request scan + wait
            self._say("Tree present. Requesting scan report...")
            report = await self._request_and_wait_scan(tree_name, 10.0)

            # Full report handling (matches tree_goals formatting/logic)
            marker_status = "unknown"
            if report:
                clean = re.sub(r"\s*\(.*?\)\s*", "", report).strip()
                pretty = re.sub(r"(\D+)(\d+)", lambda m: f"{m[1].capitalize()} {m[2]}", tree_name)

                if "Tree Infected:" in clean:
                    m = re.search(
                        r":\s*Tree Infected:\s*(.*?)(?:\s+dose_ml\s*=\s*[0-9]*\.?[0-9]+)?\s*$",
                        clean, flags=re.IGNORECASE
                    )
                    infection_raw = (m.group(1).strip() if m else "Unknown")
                    infection = re.sub(
                        r"\s*dose_ml\s*=\s*[0-9]*\.?[0-9]+", "", infection_raw, flags=re.IGNORECASE
                    ).strip()

                    print(f"Detected: {infection}")
                    print("Health:")

                    # choose treatment like tree_goals
                    if "myrtle rust" in infection.lower():
                        treatment = "Copper oxychloride"
                    elif "cinnamon fungus" in infection.lower() or "phytophthora" in infection.lower():
                        treatment = "Phosphite injections"
                    elif "eucalyptus leaf blister" in infection.lower():
                        treatment = "Mancozeb"
                    else:
                        treatment = "N/A"

                    # parse intensity-based dose (if provided)
                    dose_ml = self._parse_dose_ml(report)
                    _ = self._print_treatments_block(
                        infection_text=infection,
                        dose_ml=dose_ml,
                        chosen=treatment
                    )

                    marker_status = infection

                elif "Healthy" in clean:
                    self._say(f"{pretty} -> Health: Healthy")
                    marker_status = "healthy"

                elif "Dead" in clean:
                    self._say(f"{pretty} -> Health: Dead")
                    marker_status = "dead"

                else:
                    self._say(f"{pretty} -> {clean}")
                    marker_status = "unknown"
            else:
                self.get_logger().warn("Scan report timeout.")
                marker_status = "unknown"

            # Step 11: RViz cylinder
            try:
                self.publish_scan_complete_cylinder(tree_name, tree_pose, marker_status)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish scan marker: {e}")

            print("\n---------------------------------------------\n")
            self.progress_pub.publish(Float32(data=100.0))

        finally:
            # end of single run  return to idle and wait for the next /seeder/manual_send
            self._busy = False

    # ---------- scan req/await ----------
    async def _request_and_wait_scan(self, pine_name: str, timeout: float = 10.0) -> Optional[str]:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        def _on_report(msg: String):
            if pine_name.lower() in (msg.data or "").lower() and not fut.done():
                fut.set_result(msg.data)

        sub = self.create_subscription(String, "/scan_report", _on_report, 10)
        self.scan_req_pub.publish(String(data=pine_name))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.destroy_subscription(sub)


# -------- spin helper for asyncio+rclpy --------
async def _spin(node: Node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        await asyncio.sleep(0.01)


async def main_async():
    rclpy.init()
    node = ManualNav()
    spin_task = asyncio.create_task(_spin(node))
    try:
        node.get_logger().info("manual_nav running (single-shot mode).")
        while rclpy.ok():
            await asyncio.sleep(0.25)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if not spin_task.done():
            spin_task.cancel()
            try:
                await spin_task
            except asyncio.CancelledError:
                pass


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
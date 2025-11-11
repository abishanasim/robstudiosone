#!/usr/bin/env python3
# =============================================================================
# TREE_GOALS.PY  NODE OVERVIEW
#
# Purpose:
#   Autonomous control node for a Husky robot performing tree scanning
#   and health classification in a plantation environment.
#   It navigates between known tree positions, faces each tree, validates its
#   presence, triggers a scan, applies logic-based decisions, and publishes
#   visual markers for feedback in RViz.
#
# CTRL-F tags you can jump to:
#   ### PERCEPTION (LiDAR, TF)
#   ### MAPPING (SDF, Gazebo, Markers)
#   ### NAVIGATION (Nav2 actions, spin)
#   ### DECISION MAKING (occlusion, mirroring, dosing)
#   ### CONTROL & RECOVERY (timeouts, proximity, stationary)
#   ### OUTPUT & LOGGING (progress, CSV, RViz)
#   ### MAIN (mission loop)
# =============================================================================

import re
import asyncio
import math
import csv
import os
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import FollowWaypoints, NavigateToPose, Spin
from visualization_msgs.msg import Marker, MarkerArray

# Alias the two Durations so they dont collide
from builtin_interfaces.msg import Duration as MsgDuration
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time

from tf2_ros import Buffer, TransformListener
from std_msgs.msg import String, Float32   # progress topic
from sensor_msgs.msg import LaserScan      # LiDAR

# ---- CONFIG ----
# If you are not using AMCL/map_server, set default_world_frame to "odom"
DEFAULT_WORLD_FRAME = "map"

# =============================================================================
# ### MAPPING (SDF, Gazebo, Markers)
# - SDF_POSES: Static world coordinates for the five pines (simulation truth).
# - TREE_NAMES: Convenience list of the keys to iterate consistently.
#   These provide a known, structured map without running SLAM.
# =============================================================================
SDF_POSES = {
    "pine1": (-2.5,  9.0, 0.0, 0.0, 0.0, 0.0),
    "pine2": (-2.5,  4.5, 0.0, 0.0, 0.0, 0.0),
    "pine3": (-2.5,  0.0, 0.0, 0.0, 0.0, 0.0),
    "pine4": (-2.5, -4.5, 0.0, 0.0, 0.0, 0.0),
    "pine5": (-2.5, -9.0, 0.0, 0.0, 0.0, 0.0),
}
TREE_NAMES = list(SDF_POSES.keys())

# Try Gazebo Classic service first (optional)
try:
    from gazebo_msgs.srv import GetModelState
    HAVE_CLASSIC = True
except Exception:
    HAVE_CLASSIC = False

# =============================================================================
# ### MAPPING (SDF, Gazebo, Markers)  math helpers
# - quat_from_yaw / quat_to_yaw: convert between yaw (heading) and quaternion.
# - waypoint_beside_tree: place a goal beside a tree and optionally face it.
# ============================================================================
def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

# --------------------------------------------------------------------------------------
# UTILS (quaternions, helpers)
# -------------------------------------------------------------------------------------
def quat_to_yaw(q) -> float:
    if (q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0):
        return 0.0
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def waypoint_beside_tree(tree_pose: Pose, x_offset: float, face_tree: bool = True) -> Pose:
    """
    Place a waypoint beside the tree by shifting +x in the WORLD frame:
      wp = (tree_x + x_offset, tree_y)
    If face_tree=True, orient the robot to look at the tree.
    """
    wp = Pose()
    wp.position.x = tree_pose.position.x + x_offset
    wp.position.y = tree_pose.position.y
    wp.position.z = max(0.2, tree_pose.position.z)

    if face_tree:
        # Yaw that faces from waypoint toward the tree
        yaw = math.atan2(tree_pose.position.y - wp.position.y,
                         tree_pose.position.x - wp.position.x)
    else:
        yaw = quat_to_yaw(tree_pose.orientation)
    wp.orientation = quat_from_yaw(yaw)
    return wp

# =============================================================================
# Node class encapsulating perception, mapping, navigation, decision-making,
# control/recovery, and logging. Each section below has a CTRL-F header.
# =============================================================================
class TreeGoalRunner(Node):
    def __init__(self):
        super().__init__("tree_goal_runner")

        # ---------------------------------------------------------------------
        # ### CONFIG / PARAMS (shared across blocks)
        # - world_frame/base_frame: TF frames used for navigation.
        # - x_offset_world: lateral distance from tree to approach.
        # - pause_seconds: dwell time at a tree before switching context.
        # - per_goal_timeout_sec: action timeouts for robust control.
        # ---------------------------------------------------------------------

        self.declare_parameter("world_frame", DEFAULT_WORLD_FRAME)
        # IMPORTANT: 2.5 => normal goals at x H 0, mirrored at x H -5 (trees at x = -2.5)
        self.declare_parameter("x_offset_world", 4.5)
        self.declare_parameter("extra_dx", 0.0)
        self.declare_parameter("extra_dy", 0.0)
        self.declare_parameter("pause_seconds", 1.0)
        self.declare_parameter("use_dummy_if_missing", True)
        self.declare_parameter("per_goal_timeout_sec", 120.0)
        self.declare_parameter("base_frame", "base_link")  # matches Nav2

        # ---------------------------------------------------------------------
        # ### PERCEPTION (LiDAR, TF)  parameters
        # - scan_topic: LiDAR source.
        # - front_check_window_deg: angular window around forward direction.
        # - goal_match_tolerance_m: distance agreement to call occlusion.
        # - gap_check_timeout_sec: wait-for-fresh-scan gate.
        # - tree_expected_dist_min/max: band to assert "tree present".
        # ---------------------------------------------------------------------

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("front_check_window_deg", 12.0)     # +/- degrees about forward
        self.declare_parameter("goal_match_tolerance_m", 1.0)       # step 3 threshold (�1 m)
        self.declare_parameter("gap_check_timeout_sec", 1.5)        # wait for fresh scan before tree check
        self.declare_parameter("tree_expected_dist_min", 0.5)       # expected tree distance band
        self.declare_parameter("tree_expected_dist_max", 4.0)
        self.declare_parameter("gap_report_topic", "/gap_report")

        # Pull parameter value
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.x_offset = float(self.get_parameter("x_offset_world").value)
        self.dx = float(self.get_parameter("extra_dx").value)
        self.dy = float(self.get_parameter("extra_dy").value)
        self.pause_s = float(self.get_parameter("pause_seconds").value)
        self.use_dummy = bool(self.get_parameter("use_dummy_if_missing").value)
        self.per_goal_timeout = float(self.get_parameter("per_goal_timeout_sec").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        # LiDAR config
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.front_window = float(self.get_parameter("front_check_window_deg").value)
        self.goal_match_tol = float(self.get_parameter("goal_match_tolerance_m").value)
        self.gap_check_timeout = float(self.get_parameter("gap_check_timeout_sec").value)
        self.tree_dmin = float(self.get_parameter("tree_expected_dist_min").value)
        self.tree_dmax = float(self.get_parameter("tree_expected_dist_max").value)
        self.gap_report_topic = str(self.get_parameter("gap_report_topic").value)

        # ---------------------------------------------------------------------
        # ### PERCEPTION (LiDAR, TF)  runtime state
        # - _min_front: min valid range within the front angle window.
        # - _last_scan_time/msg: recency tracking to avoid stale decisions.
        # ---------------------------------------------------------------------
        self._min_front = float("inf")
        self._last_scan_time: Optional[Time] = None
        self._last_scan_msg: Optional[LaserScan] = None

        # ---------------------------------------------------------------------
        # ### PERCEPTION (LiDAR, TF)  TF stack
        # - Buffer/TransformListener: async TF lookups for pose/yaw.
        # ---------------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------------------
        # ### NAVIGATION (Nav2 actions, spin)  action clients
        # - follow: FollowWaypoints (primary navigator).
        # - nav2: NavigateToPose (fallback).
        # - spin_ac: Spin (orientation control).
        # ---------------------------------------------------------------------
        self.follow = ActionClient(self, FollowWaypoints, "follow_waypoints")
        self.nav2 = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_ac = ActionClient(self, Spin, "spin")

        # ---------------------------------------------------------------------
        # ### MAPPING (SDF, Gazebo, Markers)  optional classic Gazebo client
        # - classic_cli: service client for /gazebo/get_model_state.
        # --------------------------------------------------------------------
        self.classic_cli = None
        if HAVE_CLASSIC:
            self.classic_cli = self.create_client(GetModelState, "/gazebo/get_model_state")

        # ---------------------------------------------------------------------
        # ### OUTPUT & LOGGING (progress, CSV, RViz)  publishers/markers
        # - marker_pub/highlight_pub/scan_done_pub: RViz markers.
        # - progress_pub: mission completion percent.
        # - results/csv_path: batch summary persisted to CSV.
        # ---------------------------------------------------------------------
        self.marker_pub = self.create_publisher(MarkerArray, "tree_waypoints", 10)
        self.highlight_pub = self.create_publisher(Marker, "marker", 10)
        self.scan_done_pub = self.create_publisher(Marker, "tree_scan_done", 10)  # for per-tree scan result cylinders
        self._cached_markers: Optional[MarkerArray] = None
        self._marker_timer = self.create_timer(0.5, self._republish_markers)

        # Proximity acceptance (meters) should match your Nav2 xy_goal_tolerance
        self.accept_radius = 0.6

        # ---------- Mission progress ----------
        self.progress_pub = self.create_publisher(Float32, "mission_progress", 10)
        self.total_goals = 0
        self.completed_goals = 0

        # ---------- Reporting ----------
        self.results = []
        self.csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tree_health_report.csv"
        )

        # ---------------------------------------------------------------------
        # ### DECISION MAKING (occlusion, mirroring, dosing)  treatments store
        # - tank_ml: remaining ml per chemical.
        # - default/min/max dose: clamped dosing safety.
        # ---------------------------------------------------------------------

        self.tank_ml = {
            "Copper oxychloride": 10.0,
            "Phosphite injections": 10.0,
            "Mancozeb": 10.0,
        }
        # Default dose if no intensity provided
        self.default_dose_ml = 5.0
        # Hard safety limits
        self.min_dose_ml = 0.5
        self.max_dose_ml = 10.0

        # ---------------------------------------------------------------------
        # ### PERCEPTION (LiDAR) + OUTPUT  subscriptions for scan + gap report
        # - _on_scan: LiDAR callback.
        # - gap_pub: textual "gap detected" reports.
        # ---------------------------------------------------------------------

        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, 10)
        self.gap_pub = self.create_publisher(String, self.gap_report_topic, 10)

        # Visual mirror reference (not a param; we compute per-goal)
        self.center_x = -2.5  # line between rows

    
    # -------------------------------------------------------------------------
    # ### OUTPUT & LOGGING (progress, CSV, RViz) color mapping helper
    # - Maps infection strings to RGBA used by the scan-complete cylinder.
    # -------------------------------------------------------------------------

    def _infection_color_rgba(self, infection_text: str):
        s = (infection_text or "").lower()
        # green, red, blue, yellow, black (as requested)
        if "healthy" in s:
            return (0.0, 1.0, 0.0, 1.0)          # green
        if "myrtle rust" in s or "rust" in s:
            return (1.0, 1.0, 0.0, 1.0)          # yellow
        if "phytophthora" in s or "cinnamon fungus" in s:
            return (1.0, 0.0, 0.0, 1.0)          # red
        if "eucalyptus leaf blister" in s or "blister" in s:
            return (0.0, 0.4, 1.0, 1.0)          # blue
        if "dead" in s:
            return (0.0, 0.0, 0.0, 1.0)          # black
        return (0.0, 0.0, 0.0, 1.0)              # default: black

    # -------------------------------------------------------------------------
    # ### MAPPING (SDF, Gazebo, Markers)  tree pose sources
    # - get_pose_from_sdf: primary, from static dict + optional dx/dy offsets.
    # - get_pose_via_classic: Gazebo Classic service, if available.
    # - get_pose_via_topic: topic fallback (/model/<tree>/pose).
    # - get_tree_pose: unified "try SDF  Classic  Topic".
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # ### PERCEPTION (LiDAR, TF) pose, distance, and spin math
    # - _normalize_angle: wrap to [-pi, pi].
    # - _current_base_pose_yaw(_sync): TF lookups for (x,y,yaw).
    # - _dist: Euclidean distance helper.
    # -------------------------------------------------------------------------

    def _normalize_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    async def _current_base_pose_yaw(self) -> Tuple[float, float, float]:
        tf = await self.tf_buffer.lookup_transform_async(
            self.world_frame, self.base_frame, Time()
        )
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )
        return x, y, yaw

    def _current_base_pose_yaw_sync(self) -> Tuple[float, float, float]:
        tf = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, Time())
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        return x, y, yaw

    def _dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)
    
    # -------------------------------------------------------------------------
    # ### DECISION MAKING (occlusion, mirroring, dosing) dosing parser
    # - _parse_dose_ml: extract "Dose: X ml" or "dose_ml=X" from scan text.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # ### CONTROL & RECOVERY (timeouts, proximity, stationary)
    # - wait_until_stationary: ensures robot stops before face-the-tree step.
    #   Samples base pose twice; movement < thresh => stationary confirmed.
    # -------------------------------------------------------------------------

    async def wait_until_stationary(self, max_wait_s: float = 5.0, check_dt: float = 0.3, thresh: float = 0.02) -> bool:
        """
        Returns True if robot becomes stationary (motion < thresh over check_dt) within max_wait_s.
        """
        self.get_logger().info("Checking if Husky is stationary...")
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

    async def _spin_once(self, delta: float, time_allowance_s: float) -> Optional[int]:
        if not await self._wait_for_server(self.spin_ac, "spin"):
            self.get_logger().warn("Spin action server not available")
            return None
        goal = Spin.Goal()
        goal.target_yaw = delta
        goal.time_allowance = RclpyDuration(seconds=time_allowance_s).to_msg()
        self.get_logger().info(f"Spin delta: {math.degrees(delta):.1f} deg")
        gh = await self.spin_ac.send_goal_async(goal)
        if not gh.accepted:
            self.get_logger().warn("Spin goal not accepted")
            return None
        res = await gh.get_result_async()
        return getattr(res, "status", None)

    async def spin_to_face(self, target_x: float, target_y: float, timeout_sec: float = 15.0) -> bool:
        angle_tolerance = math.radians(20.0)
        nudge_window   = math.radians(35.0)

        x_now, y_now, yaw_now = await self._current_base_pose_yaw()
        desired = math.atan2(target_y - y_now, target_x - x_now)
        delta = self._normalize_angle(desired - yaw_now)

        if abs(delta) <= angle_tolerance:
            self.get_logger().info(
                f"Already facing target within 20 deg tolerance (delta={math.degrees(delta):.1f} deg)."
            )
            return True

        _ = await self._spin_once(delta, timeout_sec)

        x_now2, y_now2, yaw_now2 = await self._current_base_pose_yaw()
        desired2 = math.atan2(target_y - y_now2, target_x - x_now2)
        residual = self._normalize_angle(desired2 - yaw_now2)

        if abs(residual) <= angle_tolerance:
            self.get_logger().info(
                f"Within 20 deg tolerance after spin (residual={math.degrees(residual):.1f} deg)."
            )
            return True

        if abs(residual) <= nudge_window:
            self.get_logger().info(
                f"Not within tolerance, nudging (residual={math.degrees(residual):.1f} deg)."
            )
            _ = await self._spin_once(residual, 4.0)
            x_now3, y_now3, yaw_now3 = await self._current_base_pose_yaw()
            desired3 = math.atan2(target_y - y_now3, target_x - x_now3)
            residual2 = self._normalize_angle(desired3 - yaw_now3)

            if abs(residual2) <= angle_tolerance:
                self.get_logger().info(
                    f"Nudge succeeded, now within 20 deg tolerance (residual={math.degrees(residual2):.1f} deg)."
                )
                return True
            else:
                self.get_logger().warn(
                    f"Nudge ended outside tolerance (residual={math.degrees(residual2):.1f} deg)."
                )
                return False

        self.get_logger().warn(
            f"Spin ended outside tolerance (residual={math.degrees(residual):.1f} deg >35 deg window)."
        )
        return False

    # -------------------------------------------------------------------------
    # ### PERCEPTION (LiDAR, TF) SCAN TRIGGER
    # - request_and_wait_scan: publish /scan_request and await /scan_report
    #   matching this tree name. Used after presence check.
    # -------------------------------------------------------------------------

    async def request_and_wait_scan(self, pine_name: str, timeout: float = 10.0) -> Optional[str]:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        def _on_report(msg: String):
            if pine_name.lower() in msg.data.lower() and not fut.done():
                fut.set_result(msg.data)

        sub = self.create_subscription(String, "/scan_report", _on_report, 10)
        pub = self.create_publisher(String, "/scan_request", 10)
        pub.publish(String(data=pine_name))

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.destroy_subscription(sub)
            self.destroy_publisher(pub)

    # -------------------------------------------------------------------------
    # ### NAVIGATION (Nav2 actions, spin)  action helpers
    # - _wait_for_server: robustly wait for action servers.
    # - _send_follow_waypoints / _send_nav_to_pose: travel commands with
    #   timeouts and fallback.
    # - send_waypoint_one_by_one: FW first, then Nav2 fallback.
    # -------------------------------------------------------------------------

    async def _wait_for_server(self, client: ActionClient, label: str, tries: int = 50, dt: float = 0.1) -> bool:
        self.get_logger().info(f"Waiting for {label} server...")
        for _ in range(tries):
            if client.wait_for_server(timeout_sec=dt):
                return True
            await asyncio.sleep(dt)
        self.get_logger().error(f"{label} server not available")
        return False

    async def _send_follow_waypoints(self, poses: List[Pose]) -> Tuple[bool, str]:
        if not await self._wait_for_server(self.follow, "follow_waypoints"):
            return False, "server_unavailable"

        goal = FollowWaypoints.Goal()
        for p in poses:
            ps = PoseStamped()
            ps.header.frame_id = self.world_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose = p
            if (ps.pose.orientation.x == ps.pose.orientation.y ==
                ps.pose.orientation.z == ps.pose.orientation.w == 0.0):
                ps.pose.orientation.w = 1.0
            goal.poses.append(ps)

        self.get_logger().info("Travelling to goal (FollowWaypoints)...")
        gh = await self.follow.send_goal_async(goal)
        if not gh.accepted:
            return False, "rejected"

        try:
            res = await asyncio.wait_for(gh.get_result_async(), timeout=self.per_goal_timeout)
        except asyncio.TimeoutError:
            self.get_logger().warn("FollowWaypoints timed out")
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

        self.get_logger().info("Travelling to goal (NavigateToPose fallback)...")
        gh = await self.nav2.send_goal_async(goal)
        if not gh.accepted:
            return False, "rejected"

        try:
            res = await asyncio.wait_for(gh.get_result_async(), timeout=self.per_goal_timeout)
        except asyncio.TimeoutError:
            self.get_logger().warn("NavigateToPose timed out")
            try:
                await gh.cancel_goal_async()
            except Exception:
                pass
            return False, "timeout"

        status = getattr(res, "status", None)
        if status == 4:
            return True, "succeeded"
        return False, f"status={status}"

    async def send_waypoint_one_by_one(self, pose: Pose) -> bool:
        ok, why = await self._send_follow_waypoints([pose])
        if ok:
            return True
        self.get_logger().warn(f"FollowWaypoints failed: {why}; trying NavigateToPose fallback")
        ok2, why2 = await self._send_nav_to_pose(pose)
        if not ok2:
            self.get_logger().warn(f"NavigateToPose also failed: {why2}")
        return ok2

    # -------------------------------------------------------------------------
    # ### MAPPING (SDF, Gazebo, Markers) RViz marker builders
    # - _make_markers: small spheres + labels for waypoints.
    # - publish_waypoint_markers: publish both small spheres + one large sphere
    #   to highlight current target. Cached for periodic republish.
    # - publish_scan_complete_cylinder: color-coded cylinder per tree after scan.
    # -------------------------------------------------------------------------

    def _make_markers(self, named_poses: List[Tuple[str, Pose]]) -> MarkerArray:
        """
        Build a MarkerArray for the provided named poses:
          - Small spheres at each waypoint
          - Text labels above each sphere
        """
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, (name, pose) in enumerate(named_poses):
            # Sphere marker
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = "tree_waypoints_spheres"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose = Pose()
            m.pose.position.x = pose.position.x
            m.pose.position.y = pose.position.y
            m.pose.position.z = max(0.2, pose.position.z)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 0.25
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 1.0
            m.color.a = 0.9
            m.lifetime = MsgDuration()
            ma.markers.append(m)

            # Text marker
            t = Marker()
            t.header.frame_id = self.world_frame
            t.header.stamp = stamp
            t.ns = "tree_waypoints_labels"
            t.id = 10000 + i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = pose.position.x
            t.pose.position.y = pose.position.y
            t.pose.position.z = max(0.2, pose.position.z) + 0.35
            t.pose.orientation.w = 1.0
            t.scale.z = 0.30  # text height
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 0.95
            t.text = self._pretty_tree_name(name)
            t.lifetime = MsgDuration()
            ma.markers.append(t)

        return ma

    def publish_scan_complete_cylinder(self, tree_name: str, tree_pose: Pose, infection_text: str):
        """
        Publish a CYLINDER marker in RViz at the tree's pose once its scan is complete.
        One cylinder per tree (stable id), tinted by infection status.
        """
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "scan_done"

        # Stable ID per tree so updating replaces the same marker
        try:
            m.id = 1000 + TREE_NAMES.index(tree_name)
        except ValueError:
            m.id = 1999  # fallback

        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        # Cylinder geometry (meters)
        diameter = 0.35
        height   = 0.60
        m.scale.x = diameter     # x = diameter
        m.scale.y = diameter     # y = diameter
        m.scale.z = height       # z = height

        # Put cylinder on the ground at tree's x,y (pose is the center of the cylinder)
        m.pose.position.x = float(tree_pose.position.x)
        m.pose.position.y = float(tree_pose.position.y)
        ground_z = float(tree_pose.position.z)
        m.pose.position.z = max(0.0, ground_z) + height * 0.5

        # Upright
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # Color by infection
        r, g, b, a = self._infection_color_rgba(infection_text)
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)

        # Persist in RViz (0 duration)
        m.lifetime = MsgDuration()

        self.scan_done_pub.publish(m)
        self.get_logger().info(f"[RViz] Scan-done cylinder published for {tree_name} ({infection_text}).")

    def publish_waypoint_markers(self, named_poses: List[Tuple[str, Pose]]):
        self._cached_markers = self._make_markers(named_poses)
        self.marker_pub.publish(self._cached_markers)

        if named_poses:
            _, p = named_poses[0]
            m = Marker()
            m.header.frame_id = self.world_frame
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

    # # -------------------------------------------------------------------------
    # ### OUTPUT & LOGGING (progress, CSV, RViz)  progress helper
    # - Publishes mission coverage percentage after each tree.
    # -------------------------------------------------------------------------

    def _publish_progress(self):
        pct = 0.0 if self.total_goals <= 0 else (100.0 * float(self.completed_goals) / float(self.total_goals))
        self.progress_pub.publish(Float32(data=float(pct)))
        self.get_logger().info(f"Mission coverage: {pct:.1f}% ({self.completed_goals}/{self.total_goals})")

    # -------------------------------------------------------------------------
    # ### OUTPUT & LOGGING  small formatter for pretty labels
    # -------------------------------------------------------------------------

    def _pretty_tree_name(self, raw: str) -> str:
        m = re.match(r"([A-Za-z]+)(\d+)$", raw)
        if m:
            return f"{m.group(1).capitalize()} {int(m.group(2))}"
        return raw.capitalize()

    # -------------------------------------------------------------------------
    # ### DECISION MAKING (occlusion, mirroring, dosing) treatment printer
    # - Chooses default chemical based on infection text (if not provided),
    #   clamps dose, deducts from tank, prints a console block like detector.
    # - Returns the actual ml dispensed.
    # -------------------------------------------------------------------------
    def _print_treatments_block(self, infection_text: str, dose_ml: Optional[float] = None, chosen: Optional[str] = None) -> float:
        """
        Prints a treatment block identical in style to color_3d_detector:
            - Available treatments (with " bullets)
            - Chosen solution: <Name>
            - Dose: X.X ml
        Still deducts from tank and clamps to safety limits. Returns actual ml dispensed.
        """
        tlc = (infection_text or "").lower()

        # Healthy / Dead -> no dispersion text nor dose
        if "healthy" in tlc or "dead" in tlc:
            return 0.0

        # Pick treatment unless caller provided one
        if not chosen:
            if "myrtle rust" in tlc:
                chosen = "Copper oxychloride"
            elif "cinnamon fungus" in tlc or "phytophthora" in tlc:
                chosen = "Phosphite injections"
            elif "eucalyptus leaf blister" in tlc:
                chosen = "Mancozeb"
            else:
                chosen = "N/A"

        # Match the exact casing you show in color_3d_detector
        title_fix = {
            "phosphite injections": "Phosphite Injections",
            "copper oxychloride":   "Copper oxychloride",
            "mancozeb":             "Mancozeb",
            "n/a":                  "N/A",
        }
        chosen_pest = title_fix.get(chosen.lower(), chosen)

        # If not tracked, just print the block without dosing
        if chosen not in self.tank_ml:
            print("- Available treatments:")
            print("  Copper oxychloride")
            print("  Phosphite injections")
            print("  Mancozeb")
            print(f"Chosen solution: {chosen_pest}")
            return 0.0

        # Determine dose (parse result or default), clamp, and deduct from tank
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

        # Print in the *same* style as color_3d_detector
        print("- Available treatments:")
        print("  Copper oxychloride")
        print("  Phosphite injections")
        print("  Mancozeb")
        print(f"Chosen solution: {chosen_pest}")
        print(f"Dose: {actual:.1f} ml")

        return float(actual)

    # -------------------------------------------------------------------------
    # ### OUTPUT & LOGGING (RViz) alternate marker helper (not core path)
    # -------------------------------------------------------------------------

    def publish_tree_marker(self, tree_name: str, tree_pose: Pose, color: Tuple[float, float, float] = (0.0, 0.8, 0.0)):
        """
        Publishes a cylinder marker at the scanned tree's position (alternate helper).
        """
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "tree_scan_results"
        m.id = hash(tree_name) % 10000  # ensure unique id per tree
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose = Pose()
        m.pose.position.x = tree_pose.position.x
        m.pose.position.y = tree_pose.position.y
        m.pose.position.z = 0.5  # raise slightly above ground
        m.pose.orientation.w = 1.0
        m.scale.x = 0.4  # cylinder diameter
        m.scale.y = 0.4
        m.scale.z = 1.0  # height
        m.color.r, m.color.g, m.color.b = color
        m.color.a = 1.0
        m.lifetime = MsgDuration()  # forever

        self.highlight_pub.publish(m)
        self.get_logger().info(f"Placed cylinder marker for {tree_name}")

    # ---------- LiDAR callbacks ----------
    def _on_scan(self, msg: LaserScan):
        """
        Track min range in a small window about forward (0 rad).
        Assumes the sensor frame is aligned with +x forward (typical base_link mounting).
        """
        if not msg.ranges:
            self._min_front = float("inf")
            self._last_scan_time = self.get_clock().now()
            self._last_scan_msg = msg
            return

        a_min = msg.angle_min
        a_inc = msg.angle_increment
        n = len(msg.ranges)
        window = math.radians(self.front_window)

        min_val = float("inf")
        for i in range(n):
            a = a_min + i * a_inc
            if -window <= a <= window:
                r = msg.ranges[i]
                if math.isfinite(r) and r > 0.0:
                    if r < min_val:
                        min_val = r

        self._min_front = min_val
        self._last_scan_time = self.get_clock().now()
        self._last_scan_msg = msg

    async def _check_tree_gap(self, tree_name: str) -> bool:
        """
        Returns True if something is in front within [tree_dmin, tree_dmax] (assume it's the tree),
        otherwise publishes a 'gap detected' and returns False. Waits briefly for a fresh scan.
        Also publishes a red cylinder marker for the gap location in RViz.
        """
        deadline = self.get_clock().now() + RclpyDuration(seconds=self.gap_check_timeout)

        # wait up to gap_check_timeout for a scan newer than 0.5s
        while self.get_clock().now() < deadline:
            if self._last_scan_time is not None:
                age = (self.get_clock().now() - self._last_scan_time).nanoseconds * 1e-9
                if age < 0.5:
                    break
            await asyncio.sleep(0.05)

        if self._last_scan_time is None:
            self.get_logger().warn("No LiDAR data yet; cannot check for tree presence. Proceeding to scan.")
            return True

        age = (self.get_clock().now() - self._last_scan_time).nanoseconds * 1e-9
        if age >= 0.5:
            self.get_logger().warn(f"Stale LiDAR ({age:.2f}s); proceeding without gap check.")
            return True

        d = self._min_front
        if math.isfinite(d) and self.tree_dmin <= d <= self.tree_dmax:
            self.get_logger().info(f"Tree detected at ~{d:.2f} m (within [{self.tree_dmin:.2f}, {self.tree_dmax:.2f}]).")
            return True

        # --- GAP DETECTED ---
        msg = f"{tree_name}: GAP DETECTED (front_min={d:.2f} m not in [{self.tree_dmin:.2f}, {self.tree_dmax:.2f}]). Replant needed."
        self.get_logger().warn(msg)
        self.gap_pub.publish(String(data=msg))

        # publish a red cylinder at the expected tree position
        try:
            expected_pose = self.get_pose_from_sdf(tree_name)
            if expected_pose:
                m = Marker()
                m.header.frame_id = self.world_frame
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "gap_detected"
                m.id = 5000 + TREE_NAMES.index(tree_name) if tree_name in TREE_NAMES else 5999
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.scale.x = 0.5
                m.scale.y = 0.5
                m.scale.z = 0.4
                m.pose.position.x = expected_pose.position.x
                m.pose.position.y = expected_pose.position.y
                m.pose.position.z = 0.2
                m.pose.orientation.w = 1.0
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 0.9
                self.highlight_pub.publish(m)
                self.get_logger().info(f"[RViz] Red gap marker published for {tree_name}.")
        except Exception as e:
            self.get_logger().warn(f"Failed to publish gap marker: {e}")

        return False

    # -------------------------------------------------------------------------
    # ### DECISION MAKING (occlusion, mirroring)  pre-goal occlusion test
    # - pregoal_occluded: compares LiDAR front distance vs geometric distance
    #   to the candidate approach waypoint; if within tolerance => blocked.
    # -------------------------------------------------------------------------

    # ---------- Build waypoints (normal or mirrored) ----------
    def build_wp_normal(self, tree_pose: Pose) -> Pose:
        # Normal: x = -2.5 + x_offset -> nominally 0.0 with x_offset=2.5
        return waypoint_beside_tree(tree_pose, self.x_offset, face_tree=False)

    def build_wp_mirrored(self, tree_pose: Pose) -> Pose:
        # Mirrored across center -2.5: x = -2.5 - x_offset -> nominally -5.0 with x_offset=2.5
        return waypoint_beside_tree(tree_pose, -self.x_offset, face_tree=False)

    # ---------- Step 3: Pre-goal occlusion test ----------
    async def pregoal_occluded(self, goal_x: float, goal_y: float) -> Tuple[bool, float, float]:
        """
        Returns (is_occluded, lidar_d, goal_d).
        is_occluded iff a valid front LiDAR distance exists and is within goal_match_tol of distance to goal.
        """
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
        if math.isfinite(lidar_d):
            if abs(lidar_d - goal_d) <= self.goal_match_tol:
                return (True, lidar_d, goal_d)
        return (False, lidar_d, goal_d)

# -----------------------------------------------------------------------------
# Spin helper keeps ROS callbacks alive while we await coroutines in main().
# -----------------------------------------------------------------------------

async def spin_node(node: Node):
    """Keep callbacks pumping while awaiting actions."""
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        await asyncio.sleep(0.01)


# =============================================================================
# ### MAIN (mission loop)
# Orchestrates the per-tree sequence:
#   1) collect tree poses; 2) publish markers; 3) for each tree:
#   - face normal wp; occlusion check (mirror if needed); travel; proximity
#     acceptance; stationary gate; face tree; presence band; scan; interpret;
#     RViz cylinder; pre-orient next; progress update; CSV write on finish.
# =============================================================================

async def main_async():
    rclpy.init()
    node = TreeGoalRunner()
    spin_task = asyncio.create_task(spin_node(node))

    try:
        node.get_logger().info("Node Running.")

        # Collect tree poses (expecting 5)
        tree_named_poses: List[Tuple[str, Pose]] = []
        for name in TREE_NAMES:
            p = await node.get_tree_pose(name)
            if p is not None:
                tree_named_poses.append((name, p))
            else:
                node.get_logger().warn(f"Skipping '{name}' (no pose)")
            if len(tree_named_poses) >= 5:
                break

    except Exception as e:
        node.get_logger().error(f"Error while collecting tree poses: {e}")

        if not tree_named_poses and bool(node.get_parameter("use_dummy_if_missing").value):
            node.get_logger().warn("No tree poses; using DUMMY waypoints.")
            for i, (x, y) in enumerate([(5.0, 0.0), (5.0, 3.0), (0.0, 3.0), (0.0, 0.0)]):
                p = Pose()
                p.position.x = x + node.dx
                p.position.y = y + node.dy
                p.orientation.w = 1.0
                tree_named_poses.append((f"dummy_{i+1}", p))

        # Build initial normal waypoints for visualization
        beside_named_poses = []
        for name, tree_pose in tree_named_poses:
            wp = node.build_wp_normal(tree_pose)
            beside_named_poses.append((name, wp))

        node.total_goals = len(beside_named_poses)
        node.completed_goals = 0
        node.publish_waypoint_markers(beside_named_poses)
        node._publish_progress()

        # Drive to each tree waypoint one-by-one
        for idx, (name, tree_pose) in enumerate(tree_named_poses, start=1):

            # ---- Step 2: Spin to face the (normal) goal ----
            normal_wp = node.build_wp_normal(tree_pose)
            node.get_logger().info(
                f"[{idx}/{len(tree_named_poses)}] {name} | Facing goal at (x={normal_wp.position.x:.2f}, y={normal_wp.position.y:.2f})"
            )
            _ = await node.spin_to_face(normal_wp.position.x, normal_wp.position.y, timeout_sec=8.0)

            # ---- Step 3: LiDAR vs distance-to-goal pre-check ----
            node.get_logger().info("Checking LiDAR vs distance-to-goal for occlusion")
            blocked, lidar_d, goal_d = await node.pregoal_occluded(normal_wp.position.x, normal_wp.position.y)
            if blocked:
                node.get_logger().warn(
                    f"Obstacle detected near goal line (LiDAR {lidar_d:.2f} m vs goal {goal_d:.2f} m, tol �{node.goal_match_tol:.1f} m)."
                )
                node.get_logger().warn("Cancelling this goal and MIRRORING to other side of center line x=-2.5 (goal at xH-5).")
                wp = node.build_wp_mirrored(tree_pose)
                node.get_logger().info(
                    f"Mirrored goal: (x={wp.position.x:.2f}, y={wp.position.y:.2f}). Spinning to face mirrored goal..."
                )
                _ = await node.spin_to_face(wp.position.x, wp.position.y, timeout_sec=6.0)
            else:
                node.get_logger().info(
                    f"No obstacle near goal line (LiDAR {lidar_d if math.isfinite(lidar_d) else float('inf'):.2f} m vs goal {goal_d:.2f} m). Proceeding."
                )
                wp = normal_wp

            # ---- Step 4: Travel to goal ----
            ok = await node.send_waypoint_one_by_one(wp)

            # Proximity acceptance
            reached = ok
            if not ok:
                try:
                    x_now, y_now, _ = await node._current_base_pose_yaw()
                    dist = node._dist(x_now, y_now, wp.position.x, wp.position.y)
                    if dist <= node.accept_radius:
                        node.get_logger().warn(
                            f"Nav2 didn't report success but we're {dist:.2f} m from goal (<= {node.accept_radius}); accepting."
                        )
                        reached = True
                except Exception as e:
                    node.get_logger().warn(f"Proximity check failed: {e}")

            if not reached:
                node.get_logger().warn(f"FAILED to reach approach waypoint for {name}; skipping to next.")
                # Still count as... (8 KB left)
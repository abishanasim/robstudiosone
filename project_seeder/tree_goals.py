#tree goals yo 
import asyncio
import math
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
from std_msgs.msg import String
from std_srvs.srv import Empty  # <-- for costmap clearing


# ---- CONFIG ----
# If you are not using AMCL/map_server, set default_world_frame to "odom"
DEFAULT_WORLD_FRAME = "map"

# SDF poses copied from your world file: (x, y, z, roll, pitch, yaw)
SDF_POSES = {
    "pine1": (-2.5,  7.5,  0.0, 0.0, 0.0, 0.0),
    "pine2": (-2.5,  3.75, 0.0, 0.0, 0.0, 0.0),
    "pine3": (-2.5,  0.0,  0.0, 0.0, 0.0, 0.0),
    "pine4": (-2.5, -3.75, 0.0, 0.0, 0.0, 0.0),
    "pine5": (-2.5, -7.5,  0.0, 0.0, 0.0, 0.0),
}
TREE_NAMES = list(SDF_POSES.keys())

# Try Gazebo Classic service first (optional)
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


def waypoint_beside_tree(tree_pose: Pose, x_offset: float, face_tree: bool = True) -> Pose:
    """
    Place a waypoint beside the tree by shifting +x in the WORLD frame:
      wp = (tree_x + x_offset, tree_y)
    If face_tree=True, orient the robot to look at the tree.
    """
    wp = Pose()
    wp.position.x = tree_pose.position.x + x_offset
    wp.position.y = tree_pose.position.y
    wp.position.z = 0.0  # <-- Ground the goal; Nav2 expects 2D goals

    if face_tree:
        # Yaw that faces from waypoint toward the tree
        yaw = math.atan2(tree_pose.position.y - wp.position.y,
                         tree_pose.position.x - wp.position.x)
    else:
        yaw = quat_to_yaw(tree_pose.orientation)
    wp.orientation = quat_from_yaw(yaw)
    return wp


class TreeGoalRunner(Node):
    def __init__(self):
        super().__init__("tree_goal_runner")

        # Parameters
        self.declare_parameter("world_frame", DEFAULT_WORLD_FRAME)
        self.declare_parameter("x_offset_world", 3.0)   # offset from tree in +X of map
        self.declare_parameter("extra_dx", 0.0)
        self.declare_parameter("extra_dy", 0.3)         # <-- small lateral nudge by default
        self.declare_parameter("pause_seconds", 1.0)
        self.declare_parameter("use_dummy_if_missing", True)
        self.declare_parameter("per_goal_timeout_sec", 120.0)
        self.declare_parameter("base_frame", "base_link")  # so this matches Nav2

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.x_offset = float(self.get_parameter("x_offset_world").value)
        self.dx = float(self.get_parameter("extra_dx").value)
        self.dy = float(self.get_parameter("extra_dy").value)
        self.pause_s = float(self.get_parameter("pause_seconds").value)
        self.use_dummy = bool(self.get_parameter("use_dummy_if_missing").value)
        self.per_goal_timeout = float(self.get_parameter("per_goal_timeout_sec").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        # TF listener
        self.tf_buffer = Buffer()
               # noqa
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Actions
        self.follow = ActionClient(self, FollowWaypoints, "follow_waypoints")
        self.nav2 = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_ac = ActionClient(self, Spin, "spin")

        # Gazebo Classic client (optional)
        self.classic_cli = None
        if HAVE_CLASSIC:
            self.classic_cli = self.create_client(GetModelState, "/gazebo/get_model_state")

        # Costmap clear service clients (created lazily)
        self._clear_global_cli = None
        self._clear_local_cli = None

        # Markers
        self.marker_pub = self.create_publisher(MarkerArray, "tree_waypoints", 10)
        self.highlight_pub = self.create_publisher(Marker, "marker", 10)
        self._cached_markers: Optional[MarkerArray] = None
        self._marker_timer = self.create_timer(0.5, self._republish_markers)

        # Proximity acceptance (meters) should match your Nav2 xy_goal_tolerance
        self.accept_radius = 0.8  # <-- more forgiving than 0.6

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

    # ---------------- Helpers for spin-to-face & proximity ----------------
    def _normalize_angle(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    async def _current_base_pose_yaw(self) -> Tuple[float, float, float]:
        # Latest transform base -> world_frame
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

    def _dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    async def _spin_once(self, delta: float, time_allowance_s: float) -> Optional[int]:
        """Send one Spin goal and return its action status code (or None if goal rejected)."""
        if not await self._wait_for_server(self.spin_ac, "spin"):
            self.get_logger().warn("Spin action server not available")
            return None
        goal = Spin.Goal()
        goal.target_yaw = delta
        goal.time_allowance = RclpyDuration(seconds=time_allowance_s).to_msg()
        self.get_logger().info(f"Spin delta: {math.degrees(delta):.1f}�")
        gh = await self.spin_ac.send_goal_async(goal)
        if not gh.accepted:
            self.get_logger().warn("Spin goal not accepted")
            return None
        res = await gh.get_result_async()
        return getattr(res, "status", None)

    async def spin_to_face(self, target_x: float, target_y: float, timeout_sec: float = 15.0) -> bool:
        """Face (target_x, target_y) using Spin action only, with tolerance and a single nudge."""
        # Tolerances
        angle_tolerance = math.radians(20.0)   # accept if within �20�
        nudge_window   = math.radians(35.0)    # allow one extra nudge if within �35�

        # Current error
        x_now, y_now, yaw_now = await self._current_base_pose_yaw()
        desired = math.atan2(target_y - y_now, target_x - x_now)
        delta = self._normalize_angle(desired - yaw_now)

        if abs(delta) <= angle_tolerance:
            self.get_logger().info(
                f"Already facing target within {math.degrees(angle_tolerance):.1f}� "
                f"(delta={math.degrees(delta):.1f}�)."
            )
            return True

        # First spin attempt
        _ = await self._spin_once(delta, timeout_sec)
        # Recompute residual
        x_now2, y_now2, yaw_now2 = await self._current_base_pose_yaw()
        desired2 = math.atan2(target_y - y_now2, target_x - x_now2)
        residual = self._normalize_angle(desired2 - yaw_now2)

        if abs(residual) <= angle_tolerance:
            self.get_logger().info(
                f"Facing target within tolerance after spin "
                f"(residual={math.degrees(residual):.1f}�)."
            )
            return True

        # If we're close but not there, try ONE short nudge
        if abs(residual) <= nudge_window:
            self.get_logger().info(
                f"Residual {math.degrees(residual):.1f}� within nudge window "
                f"({math.degrees(nudge_window):.1f}�). Nudging once&"
            )
            _ = await self._spin_once(residual, 4.0)
            # Check again
            x_now3, y_now3, yaw_now3 = await self._current_base_pose_yaw()
            desired3 = math.atan2(target_y - y_now3, target_x - x_now3)
            residual2 = self._normalize_angle(desired3 - yaw_now3)

            if abs(residual2) <= angle_tolerance:
                self.get_logger().info(
                    f"Nudge succeeded. Facing target within tolerance "
                    f"(residual={math.degrees(residual2):.1f}�)."
                )
                return True
            else:
                self.get_logger().warn(
                    f"Nudge ended with residual {math.degrees(residual2):.1f}�. "
                    f"Proceeding anyway."
                )
                return False

        self.get_logger().warn(
            f"Spin ended with residual {math.degrees(residual):.1f}� "
            f"(exceeds nudge window)."
        )
        return False

    # --------- SCAN TRIGGER (no __init__ edits; temp pub/sub per scan) ---------
    async def request_and_wait_scan(self, pine_name: str, timeout: float = 10.0) -> Optional[str]:
        """
        Ask the color detector to scan this pine via /scan_request and wait for a
        matching one-line report on /scan_report. Creates temporary pub/sub and
        cleans them up afterwards. Returns the report text or None on timeout.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        # Callback resolves when the line mentions our pine (case-insensitive)
        def _on_report(msg: String):
            if pine_name.lower() in msg.data.lower() and not fut.done():
                fut.set_result(msg.data)

        sub = self.create_subscription(String, "/scan_report", _on_report, 10)
        pub = self.create_publisher(String, "/scan_request", 10)

        # Send request
        pub.publish(String(data=pine_name))

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.destroy_subscription(sub)
            self.destroy_publisher(pub)

    # ---------------- Costmap helpers ----------------
    async def _clear_costmaps(self, wait_timeout: float = 1.0) -> None:
        """Call Nav2 clear costmap services (global + local)."""
        if self._clear_global_cli is None:
            self._clear_global_cli = self.create_client(
                Empty, "/global_costmap/clear_entirely_global_costmap"
            )
        if self._clear_local_cli is None:
            self._clear_local_cli = self.create_client(
                Empty, "/local_costmap/clear_entirely_local_costmap"
            )

        for name, cli in (
            ("global", self._clear_global_cli),
            ("local", self._clear_local_cli),
        ):
            if not cli.wait_for_service(timeout_sec=wait_timeout):
                self.get_logger().warn(f"Clear {name} costmap service unavailable")
                continue
            try:
                self.get_logger().info(f"Clearing {name} costmap&")
                await cli.call_async(Empty.Request())
            except Exception as e:
                self.get_logger().warn(f"Clear {name} costmap failed: {e}")

    # ---------------- Action helpers ----------------
    async def _wait_for_server(self, client: ActionClient, label: str, tries: int = 50, dt: float = 0.1) -> bool:
        self.get_logger().info(f"Waiting for {label} server&")
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

        self.get_logger().info(f"Sending {len(goal.poses)} waypoint(s) to FollowWaypoints&")
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

        self.get_logger().info("Sending goal to NavigateToPose (fallback)&")
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
        """Try FollowWaypoints with a single pose; on failure, try NavigateToPose.
           If NavigateToPose fails, clear costmaps and retry once.
        """
        ok, why = await self._send_follow_waypoints([pose])
        if ok:
            return True
        self.get_logger().warn(f"FollowWaypoints failed: {why}; trying NavigateToPose fallback")

        ok2, why2 = await self._send_nav_to_pose(pose)
        if ok2:
            return True

        self.get_logger().warn(f"NavigateToPose failed: {why2}. Clearing costmaps and retrying once&")
        await self._clear_costmaps()
        ok3, why3 = await self._send_nav_to_pose(pose)
        if not ok3:
            self.get_logger().warn(f"NavigateToPose retry also failed: {why3}")
        return ok3

    # ---------------- Markers ----------------
    def _make_markers(self, named_poses: List[Tuple[str, Pose]]) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        life = MsgDuration()  # 0 => forever

        for i, (name, p) in enumerate(named_poses):
            s = Marker()
            s.header.frame_id = self.world_frame
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
            t.header.frame_id = self.world_frame
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


async def spin_node(node: Node):
    """Keep callbacks pumping while awaiting actions."""
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        await asyncio.sleep(0.01)


# ---------------- Main async flow ----------------
async def main_async():
    rclpy.init()
    node = TreeGoalRunner()
    spin_task = asyncio.create_task(spin_node(node))

    try:
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

        if not tree_named_poses and bool(node.get_parameter("use_dummy_if_missing").value):
            node.get_logger().warn("No tree poses; using DUMMY waypoints.")
            for i, (x, y) in enumerate([(5.0, 0.0), (5.0, 3.0), (0.0, 3.0), (0.0, 0.0)]):
                p = Pose()
                p.position.x = x + node.dx
                p.position.y = y + node.dy
                p.orientation.w = 1.0
                tree_named_poses.append((f"dummy_{i+1}", p))

        # Build "beside-tree" waypoints at +x offset; let Nav2 approach first, then we spin
        beside_named_poses = []
        for name, tree_pose in tree_named_poses:
            wp = waypoint_beside_tree(tree_pose, node.x_offset, face_tree=False)  # <- FALSE
            beside_named_poses.append((name, wp))

        # Publish all waypoint markers so you can verify offsets in RViz
        node.publish_waypoint_markers(beside_named_poses)

        # Drive to each beside-tree waypoint one-by-one
        for idx, (name, wp) in enumerate(beside_named_poses, start=1):
            node.get_logger().info(
                f"[{idx}/{len(beside_named_poses)}] -> {name}: "
                f"goal (x={wp.position.x:.2f}, y={wp.position.y:.2f}) in '{node.world_frame}'"
            )
            ok = await node.send_waypoint_one_by_one(wp)

            # --- Proximity acceptance so we don't hang waiting for a perfect success ---
            reached = ok
            if not ok:
                try:
                    x_now, y_now, _ = await node._current_base_pose_yaw()
                    dist = node._dist(x_now, y_now, wp.position.x, wp.position.y)
                    if dist <= node.accept_radius:
                        node.get_logger().warn(
                            f"Nav2 didn't report success but we're {dist:.2f} m away (<= {node.accept_radius}); accepting."
                        )
                        reached = True
                except Exception as e:
                    node.get_logger().warn(f"Proximity check failed: {e}")

            if not reached:
                node.get_logger().warn(f"FAILED to reach: {name} (skipping to next)")
                continue

            node.get_logger().info(f"goal reached (accepted): {name}")

            # Face the actual tree location
            tree_pose_by_name = dict(tree_named_poses)
            tx = tree_pose_by_name[name].position.x
            ty = tree_pose_by_name[name].position.y
            faced = await node.spin_to_face(tx, ty, timeout_sec=12.0)
            if not faced:
                node.get_logger().warn(f"Spin-to-face failed at {name}, continuing")

            # === Trigger a scan now that we're facing the tree ===
            report = await node.request_and_wait_scan(name, timeout=10.0)
            if report:
                node.get_logger().info(report)
            else:
                node.get_logger().warn(f"No scan report for {name} within timeout")

            # Pause 3 seconds at the tree
            await asyncio.sleep(3.0)

            # Optional extra pause you already supported
            if node.pause_s > 0.0:
                await asyncio.sleep(node.pause_s)

        node.get_logger().info("All waypoints processed.")
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

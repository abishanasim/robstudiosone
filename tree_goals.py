#!/usr/bin/env python3
import asyncio
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose
from nav2_msgs.action import FollowWaypoints
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

TREE_NAMES = ["pine_tree", "pine1", "pine1_1", "pine_tree_1"]
WORLD_FRAME = "map"

try:
    from gazebo_msgs.srv import GetModelState
    HAVE_CLASSIC = True
except Exception:
    HAVE_CLASSIC = False


class TreeGoalRunner(Node):
    def __init__(self):
        super().__init__('tree_goal_runner')
        self.declare_parameter('approach_dx', 0.0)
        self.declare_parameter('approach_dy', 0.0)
        self.declare_parameter('use_dummy_if_missing', True)
        self.dx = float(self.get_parameter('approach_dx').value)
        self.dy = float(self.get_parameter('approach_dy').value)
        self.use_dummy = bool(self.get_parameter('use_dummy_if_missing').value)

        self.follow = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        self.classic_cli = None
        if HAVE_CLASSIC:
            self.classic_cli = self.create_client(GetModelState, '/gazebo/get_model_state')

        # --- Visualisation
        self.marker_pub = self.create_publisher(MarkerArray, 'tree_waypoints', 10)
        self.single_marker_pub = self.create_publisher(Marker, 'marker', 10)   # NEW: RViz default topic
        self._cached_markers = None
        self._marker_timer = self.create_timer(0.5, self._republish_markers)

    # --- pose sources (unchanged) ---
    def get_pose_via_classic(self, model_name: str) -> Pose | None:
        if not self.classic_cli: return None
        if not self.classic_cli.wait_for_service(timeout_sec=0.5): return None
        req = GetModelState.Request(); req.model_name = model_name
        resp = self.classic_cli.call(req)
        if not resp.success:
            self.get_logger().warn(f"[classic] no model '{model_name}'")
            return None
        p = resp.pose
        p.position.x += self.dx; p.position.y += self.dy
        if (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) == (0,0,0,0):
            p.orientation.w = 1.0
        return p

    async def get_pose_via_topic(self, model_name: str, timeout: float = 2.0) -> Pose | None:
        topic = f"/model/{model_name}/pose"
        fut = asyncio.get_event_loop().create_future()
        def cb(msg: Pose):
            if not fut.done(): fut.set_result(msg)
        sub = self.create_subscription(Pose, topic, cb, 10)
        self.get_logger().info(f"[gz] waiting for pose on {topic}")
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self.get_logger().warn(f"[gz] timeout waiting for {topic}")
            return None
        finally:
            self.destroy_subscription(sub)

    async def get_tree_pose(self, name: str) -> Pose | None:
        p = self.get_pose_via_classic(name)
        return p if p else await self.get_pose_via_topic(name)

    # --- FollowWaypoints (unchanged) ---
    async def send_waypoints(self, poses: list[Pose]) -> bool:
        if not self.follow.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("follow_waypoints not available (start waypoint_follower + activate Nav2)")
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
        self.get_logger().info(f"Sending {len(goal.poses)} waypoints to waypoint_follower…")
        gh = await self.follow.send_goal_async(goal)
        if not gh.accepted:
            self.get_logger().warn("FollowWaypoints goal rejected")
            return False
        result = await gh.get_result_async()
        if result is None or result.result is None:
            self.get_logger().warn("No result returned (Nav2 may not be active)")
            return False
        missed = list(result.result.missed_waypoints)
        if missed:
            self.get_logger().warn(f"Missed waypoint indices: {missed}")
            return False
        self.get_logger().info("Completed all waypoints.")
        return True

    # --- markers ---
    def _make_markers(self, named_poses: list[tuple[str, Pose]]) -> MarkerArray:
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        life = Duration()  # 0 => forever
        for i, (name, p) in enumerate(named_poses):
            # Spheres: BIG + above ground so you can’t miss them
            s = Marker()
            s.header.frame_id = WORLD_FRAME; s.header.stamp = now
            s.ns = "tree_wp"; s.id = i*2; s.type = Marker.SPHERE; s.action = Marker.ADD
            s.pose = p
            if s.pose.position.z < 0.2: s.pose.position.z = 0.2   # NEW: lift up a bit
            s.scale.x = s.scale.y = s.scale.z = 0.6               # NEW: larger
            s.color.r, s.color.g, s.color.b, s.color.a = (0.1, 0.8, 0.2, 0.98)
            s.lifetime = life
            ma.markers.append(s)

            t = Marker()
            t.header.frame_id = WORLD_FRAME; t.header.stamp = now
            t.ns = "tree_label"; t.id = i*2+1; t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
            t.pose.position.x = p.position.x; t.pose.position.y = p.position.y
            t.pose.position.z = max(0.2, p.position.z) + 0.8       # NEW: higher label
            t.scale.z = 0.35
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = name
            t.lifetime = life
            ma.markers.append(t)
        return ma

    def publish_waypoint_markers(self, named_poses: list[tuple[str, Pose]]):
        self._cached_markers = self._make_markers(named_poses)
        self.marker_pub.publish(self._cached_markers)
        # Also publish one big marker on /marker so you see *something* immediately
        if named_poses:
            first_name, first_pose = named_poses[0]
            m = Marker()
            m.header.frame_id = WORLD_FRAME
            m.ns = "tree_wp_big"
            m.id = 999
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose = first_pose
            if m.pose.position.z < 0.2: m.pose.position.z = 0.2
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = (0.2, 0.5, 1.0, 0.98)
            self.single_marker_pub.publish(m)   # NEW
        self.get_logger().info(f"Published {len(self._cached_markers.markers)} waypoint markers")

    def _republish_markers(self):
        if self._cached_markers is not None:
            self.marker_pub.publish(self._cached_markers)

async def main_async():
    rclpy.init()
    node = TreeGoalRunner()

    named_poses: list[tuple[str, Pose]] = []
    for name in TREE_NAMES:
        p = await node.get_tree_pose(name)
        if p is not None:
            p.position.x += node.dx; p.position.y += node.dy
            named_poses.append((name, p))
        else:
            node.get_logger().warn(f"Skipping '{name}' (no pose)")

    if not named_poses and node.use_dummy:
        node.get_logger().warn("No tree poses; using DUMMY waypoints to test.")
        # Larger square (about 4m sides) so Husky visibly moves
        pts = [
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0)
        ]
        for i, (x, y) in enumerate(pts):
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = 0.0
            p.orientation.w = 1.0
            named_poses.append((f"dummy_{i+1}", p))


    if named_poses:
        node.publish_waypoint_markers(named_poses)
        poses_only = [p for _, p in named_poses]
        await node.send_waypoints(poses_only)
    else:
        node.get_logger().error("No waypoints to send.")

    node.destroy_node()
    rclpy.shutdown()

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import math, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# --- tree poses from your SDF (x, y only used here) ---
SDF_POSES = {
    "pinea": (-2.5,  9.7),
    "pineb": (-2.5,  7.55),
    "pinec": (-2.5,  5.4),
    "pined": (-2.5,  3.25),
    "pine1": (-2.5,  1.10),
    "pine2": (-2.5, -1.05),
    "pine3": (-2.5, -3.25),
    "pine4": (-2.5, -5.40),
    "pine5": (-2.5, -7.60),
    "pine6": (-2.5, -9.70),
}

def wrap_to_pi(a): 
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class HuskySeeder(Node):
    # --- simple states ---
    CLEAR, CAUTION, AVOID = 0, 1, 2

    def __init__(self):
        super().__init__('husky_seeder')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        # Params (you can tweak at launch)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odometry')
        self.declare_parameter('imu_topic',  '/imu/data')
        self.declare_parameter('cmd_topic',  '/cmd_vel')
        self.declare_parameter('turn_dist_enter', 2.0)   # enter CAUTION below this
        self.declare_parameter('turn_dist_exit',  2.3)   # exit CAUTION above this
        self.declare_parameter('stop_dist_enter', 1.2)   # enter AVOID below this
        self.declare_parameter('stop_dist_exit',  1.4)   # exit AVOID above this

        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        imu_topic  = self.get_parameter('imu_topic').value
        cmd_topic  = self.get_parameter('cmd_topic').value

        self.turn_enter = float(self.get_parameter('turn_dist_enter').value)
        self.turn_exit  = float(self.get_parameter('turn_dist_exit').value)
        self.stop_enter = float(self.get_parameter('stop_dist_enter').value)
        self.stop_exit  = float(self.get_parameter('stop_dist_exit').value)

        # Subs/Pub
        self.create_subscription(LaserScan, scan_topic, self.on_scan, qos)
        self.create_subscription(Odometry,  odom_topic, self.on_odom, qos)
        self.create_subscription(Imu,       imu_topic,  self.on_imu,  qos)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        # State
        self.min_front = float('inf')
        self._last_print = 0.0
        self.state = self.CLEAR
        self.last_yaw_rate = 0.0

        # Odometry state
        self.x = self.y = self.yaw = 0.0

        # Goal management
        self.goals = list(SDF_POSES.values())
        self.goal_index = 0
        self.goal_tolerance = 0.5

        self.timer = self.create_timer(0.05, self.control_step)

        self.get_logger().info(
            f'Started HuskySeeder; listening on {scan_topic}, {odom_topic}, {imu_topic}; publishing {cmd_topic}'
        )

    # --- callbacks ---
    def on_scan(self, msg: LaserScan):
        if not msg.ranges:
            self.min_front = float('inf'); return
        a0, da = msg.angle_min, msg.angle_increment
        window = math.radians(20)
        m = float('inf')
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r): continue
            a = wrap_to_pi(a0 + i*da)
            if -window <= a <= window and r < m:
                m = r
        self.min_front = m
        now = time.time()
        if now - self._last_print > 1.0:
            self._last_print = now
            print(f"[husky_seeder] min_front: {self.min_front:.2f} m")

    def on_odom(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def on_imu(self, msg): 
        pass

    # --- state transitions ---
    def _set_state(self, new_state):
        if new_state == self.state: return
        if new_state == self.AVOID:
            self.get_logger().warn("Obstacle detected!")
            print("[husky_seeder] Obstacle detected!")
        elif new_state == self.CAUTION:
            self.get_logger().info("Redirection around obstacle...")
            print("[husky_seeder] Redirection around obstacle...")
        elif new_state == self.CLEAR:
            self.get_logger().info("Path clear, resuming forward motion.")
            print("[husky_seeder] Path clear, resuming forward motion.")
        self.state = new_state

    # --- main control loop ---
    def control_step(self):
        # hysteresis state machine
        d = self.min_front
        if d <= self.stop_enter:
            self._set_state(self.AVOID)
        elif self.state == self.AVOID and d < self.stop_exit:
            pass
        elif d <= self.turn_enter:
            self._set_state(self.CAUTION)
        elif self.state == self.CAUTION and d < self.turn_exit:
            pass
        else:
            self._set_state(self.CLEAR)

        # control
        if self.state == self.CLEAR:
            v, w = self._goal_tracking()
        elif self.state == self.CAUTION:
            v, w = 0.3, 0.3
        else:  # AVOID
            v, w = 0.0, 0.6

        # yaw-rate smoothing
        alpha = 0.3
        w = alpha*w + (1-alpha)*self.last_yaw_rate
        self.last_yaw_rate = w

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

    # --- goal seeking ---
    def _goal_tracking(self):
        if self.goal_index >= len(self.goals):
            return 0.0, 0.0  # all done
        gx, gy = self.goals[self.goal_index]
        dx, dy = gx - self.x, gy - self.y
        dist = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        yaw_error = wrap_to_pi(heading - self.yaw)

        if dist < self.goal_tolerance:
            print(f"[husky_seeder] Reached goal {self.goal_index} at ({gx:.2f},{gy:.2f})")
            self.goal_index += 1
            return 0.0, 0.0

        # proportional controller
        v = min(0.8, 0.3 + 0.5*dist)
        w = 1.0 * yaw_error
        return v, w

def main():
    rclpy.init()
    node = HuskySeeder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

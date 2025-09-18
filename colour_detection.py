#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class Color3DDetector(Node):
    def __init__(self):
        super().__init__('color_3d_detector')

        # --------- Parameters ----------
        self.declare_parameter('rgb_topic', '/camera/image')
        self.declare_parameter('depth_topic', '/camera/depth/image')
        self.declare_parameter('info_topic', '/camera/camera_info')  # optional

        # Colors to evaluate every frame (winner is published on /detected_color)
        self.declare_parameter('colors', 'green,yellow,red')
        self.declare_parameter('min_area', 20)                        # px
        self.declare_parameter('show_debug', True)
        self.declare_parameter('min_coverage', 0.005)                 # fraction of image for "winner" color

        # Independent rules:
        self.declare_parameter('yellow_infected_threshold', 0.001)    # 0.1% of image -> "infected"
        self.declare_parameter('red_remove_threshold',      0.001)    # 0.1% of image -> "remove"

        # Print throttling
        self.declare_parameter('print_every_n', 10)                   # log every N frames
        self.declare_parameter('log_period_sec', 0.0)                 # min secs between logs (0 = off)

        self.rgb_topic    = self.get_parameter('rgb_topic').value
        self.depth_topic  = self.get_parameter('depth_topic').value
        self.info_topic   = self.get_parameter('info_topic').value
        self.colors       = [c.strip().lower()
                             for c in self.get_parameter('colors').value.split(',')
                             if c.strip()]
        self.min_area     = int(self.get_parameter('min_area').value)
        self.show_debug   = bool(self.get_parameter('show_debug').value)
        self.min_coverage = float(self.get_parameter('min_coverage').value)
        self.yellow_thresh= float(self.get_parameter('yellow_infected_threshold').value)
        self.red_thresh   = float(self.get_parameter('red_remove_threshold').value)

        # Throttle settings
        self.print_every_n = max(1, int(self.get_parameter('print_every_n').value))
        self.log_period_sec = float(self.get_parameter('log_period_sec').value)
        self._frame_idx = 0
        self._last_log_time = None  # rclpy.time.Time

        self.bridge = CvBridge()
        self.last_depth = None
        self.K = None  # fx, fy, cx, cy

        # --------- Sim-friendly HSV ranges ----------
        self.hsv_ranges = {
            'red':    [(np.array([  0,  70,  35]), np.array([ 10, 255, 255])),
                       (np.array([170,  70,  35]), np.array([180, 255, 255]))],
            'green':  [(np.array([ 30,  25,  25]), np.array([ 95, 255, 255]))],
            'blue':   [(np.array([ 90,  25,  25]), np.array([130, 255, 255]))],
            'brown':  [(np.array([ 10,  20,  20]), np.array([ 30, 255, 255]))],
            'yellow': [(np.array([ 18,  20,  20]), np.array([ 40, 255, 255]))],
        }

        self.kernel = np.ones((3, 3), np.uint8)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.sub_rgb   = self.create_subscription(Image, self.rgb_topic,   self.cb_rgb,   qos_sensor)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_sensor)
        self.sub_info  = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 10)

        self.pub_color   = self.create_publisher(String,  '/detected_color', 10)
        self.pub_point   = self.create_publisher(PointStamped, '/detected_color_point', 10)
        self.pub_debug   = self.create_publisher(Image,  '/debug/color_mask', 10)

        # Independent status publishers
        self.pub_y_flag  = self.create_publisher(Bool,    '/tree_yellow_flag', 10)
        self.pub_y_stat  = self.create_publisher(String,  '/tree_yellow_status', 10)   # "infected"/"healthy"
        self.pub_y_conf  = self.create_publisher(Float32, '/tree_yellow_conf', 10)

        self.pub_r_flag  = self.create_publisher(Bool,    '/tree_red_flag', 10)
        self.pub_r_stat  = self.create_publisher(String,  '/tree_red_status', 10)      # "remove"/"ok"
        self.pub_r_conf  = self.create_publisher(Float32, '/tree_red_conf', 10)

        self.get_logger().info(
            f"Listening: {self.rgb_topic} + {self.depth_topic} (info: {self.info_topic})"
        )

    # ---------------- Callbacks ----------------
    def cb_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy

    def cb_depth(self, msg: Image):
        try:
            if msg.encoding == '32FC1':
                self.last_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            elif msg.encoding == '16UC1':
                d16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                self.last_depth = d16.astype(np.float32) / 1000.0  # mm -> m
            else:
                self.last_depth = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)
        except Exception as e:
            if self._should_log():
                self.get_logger().warn(f"Depth conversion failed: {e}")

    def cb_rgb(self, msg: Image):
        self._frame_idx += 1
        bgr = self._to_bgr(msg)
        if bgr is None:
            return
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # ---- Independent decisions ----
        yellow_cov, yellow_mask = self._coverage_for_color(hsv, 'yellow')
        red_cov,    red_mask    = self._coverage_for_color(hsv, 'red')

        # Yellow status
        y_flag = yellow_cov >= self.yellow_thresh
        y_status = 'infected' if y_flag else 'healthy'
        self.pub_y_flag.publish(Bool(data=y_flag))
        self.pub_y_stat.publish(String(data=y_status))
        self.pub_y_conf.publish(Float32(data=float(yellow_cov)))
        if self._should_log():
            self.get_logger().info(f"[YELLOW {y_status}] coverage={yellow_cov:.4f} (thr={self.yellow_thresh:.4f})")

        # Red status
        r_flag = red_cov >= self.red_thresh
        r_status = 'remove' if r_flag else 'ok'
        self.pub_r_flag.publish(Bool(data=r_flag))
        self.pub_r_stat.publish(String(data=r_status))
        self.pub_r_conf.publish(Float32(data=float(red_cov)))
        if self._should_log():
            self.get_logger().info(f"[RED {r_status}] coverage={red_cov:.4f} (thr={self.red_thresh:.4f})")

        # >>> ADD THIS BLOCK <<<
        # Decide overall health based on majority coverage between GREEN and YELLOW
        green_cov, _ = self._coverage_for_color(hsv, 'green')
        health = 'healthy' if green_cov >= yellow_cov else 'unhealthy'
        if self._should_log():
            self.get_logger().info(f"Tree health is {health}")
        # <<< END ADD >>>

        # ---- Multi-color detection (for /detected_color) ----
        best = None  # (score, name, coverage, area, cnt, cx, cy, mask)
        for name in self.colors:
            if name not in self.hsv_ranges:
                if self._should_log():
                    self.get_logger().warn(f"Unknown color '{name}' in 'colors' param; skipping")
                continue
            coverage, mask, cnt_main, area, cx, cy = self._analyze_color(hsv, name)
            score = 1e5 * coverage + (area if area >= self.min_area else 0.0)
            if best is None or score > best[0]:
                best = (score, name, coverage, area, cnt_main, cx, cy, mask)


        if best is None or best[2] < self.min_coverage:
            self.pub_color.publish(String(data='none'))
            # Show yellow mask if yellow flagged, else red if red flagged, else empty
            dbg = yellow_mask if y_flag else (red_mask if r_flag else np.zeros(hsv.shape[:2], np.uint8))
            self._maybe_publish_debug(dbg)
            return

        _, name, coverage, area, cnt_main, cx, cy, mask = best

        if self._should_log():
            msg_extra = f" @{cx},{cy}" if (cx is not None and cy is not None) else ""
            self.get_logger().info(f"[{name.upper()}] coverage={coverage:.3f} area={int(area)}{msg_extra}")
            if cx is not None and cy is not None:
                b, g, r = bgr[cy, cx].tolist()
                H, S, V = hsv[cy, cx].tolist()
                self.get_logger().info(
                    f"{name.upper()} pixel BGR=({b},{g},{r}) RGB=({r},{g},{b}) HSV=({H},{S},{V})"
                )

        if cx is not None and cy is not None:
            self._publish_point(msg, cx, cy)

        self.pub_color.publish(String(data=name))

        # Debug priority: yellow if flagged, else red if flagged, else winner
        dbg_mask = yellow_mask if y_flag else (red_mask if r_flag else mask)
        self._maybe_publish_debug(dbg_mask, cnt_main, (cx, cy) if (cx is not None and cy is not None) else None)

    # ---------------- Helpers ----------------
    def _should_log(self) -> bool:
        if (self._frame_idx % self.print_every_n) != 0:
            return False
        if self.log_period_sec > 0.0:
            now = self.get_clock().now()
            if self._last_log_time is None:
                self._last_log_time = now
                return True
            elapsed = (now - self._last_log_time).nanoseconds * 1e-9
            if elapsed < self.log_period_sec:
                return False
            self._last_log_time = now
        return True

    def _coverage_for_color(self, hsv, name):
        if name not in self.hsv_ranges:
            return 0.0, np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.hsv_ranges[name]:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        return coverage, mask

    def _analyze_color(self, hsv, name):
        coverage, mask = self._coverage_for_color(hsv, name)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0.0
        cnt_main = None
        cx = cy = None
        if cnts:
            cnt_main = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt_main)
            if area >= self.min_area:
                M = cv2.moments(cnt_main)
                if M['m00'] > 0:
                    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        return coverage, mask, cnt_main, area, cx, cy

    def _publish_point(self, rgb_msg: Image, cx: int, cy: int):
        if self.last_depth is None:
            return
        if not (0 <= cy < self.last_depth.shape[0] and 0 <= cx < self.last_depth.shape[1]):
            return
        Z = float(self.last_depth[cy, cx])
        if not (np.isfinite(Z) and Z > 0.0):
            return
        pt = PointStamped()
        pt.header = rgb_msg.header
        if self.K is not None:
            fx, fy, cx0, cy0 = self.K
            X = (cx - cx0) * Z / fx
            Y = (cy - cy0) * Z / fy
            pt.point.x, pt.point.y, pt.point.z = float(X), float(Y), float(Z)
        else:
            pt.point.x, pt.point.y, pt.point.z = float(cx), float(cy), Z
        self.pub_point.publish(pt)

    def _to_bgr(self, msg: Image):
        try:
            enc = msg.encoding.lower()
            if enc == 'bgr8':
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if enc == 'rgb8':
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if enc in ('rgba8', 'bgra8'):
                rgba = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgba8')
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if enc == 'mono8':
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            if self._should_log():
                self.get_logger().warn(f"RGB conversion failed: {e}")
            return None

    def _maybe_publish_debug(self, mask, cnt=None, centroid=None):
        if not (self.show_debug or self.pub_debug.get_subscription_count() > 0):
            return
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if cnt is not None:
            cv2.drawContours(dbg, [cnt], -1, (0, 255, 0), 2)
        if centroid is not None:
            cx, cy = centroid
            cv2.circle(dbg, (cx, cy), 4, (0, 0, 255), -1)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))

def main():
    rclpy.init()
    node = Color3DDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

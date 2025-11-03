#colour detection yo
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2, numpy as np, os, time

cv2.setNumThreads(1)  # keep CPU use reasonable

class Color3DDetector(Node):
    def __init__(self):
        super().__init__('color_3d_detector')

        # ---------- Parameters ----------
        self.declare_parameter('rgb_topic', '/camera/image')
        self.declare_parameter('depth_topic', '/camera/depth/image')
        self.declare_parameter('info_topic', '/camera/camera_info')
        self.declare_parameter('colors', 'green,yellow,red,blue,black')
        self.declare_parameter('min_area', 20)
        self.declare_parameter('show_debug', False)
        self.declare_parameter('min_coverage', 0.005)
        self.declare_parameter('yellow_infected_threshold', 0.001)
        self.declare_parameter('red_remove_threshold', 0.001)
        self.declare_parameter('print_every_n', 10)
        self.declare_parameter('log_period_sec', 0.0)
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('max_hz', 10.0)

        # request/response topics
        self.declare_parameter('scan_request_topic', '/scan_request')
        self.declare_parameter('scan_report_topic',  '/scan_report')

        # NEW: auto publish on startup
        self.declare_parameter('auto_scan_on_start', True)
        self.declare_parameter('auto_pub_hz', 0.2)
        self.declare_parameter('auto_scan_name', 'auto')

        self.rgb_topic   = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic  = self.get_parameter('info_topic').value
        self.req_topic   = self.get_parameter('scan_request_topic').value
        self.rep_topic   = self.get_parameter('scan_report_topic').value

        self.auto_on     = bool(self.get_parameter('auto_scan_on_start').value)
        self.auto_hz     = float(self.get_parameter('auto_pub_hz').value)
        self.auto_name   = str(self.get_parameter('auto_scan_name').value)

        requested = [c.strip().lower() for c in self.get_parameter('colors').value.split(',') if c.strip()]
        self.show_debug     = bool(self.get_parameter('show_debug').value)
        self.min_coverage   = float(self.get_parameter('min_coverage').value)
        self.process_every_n= int(self.get_parameter('process_every_n').value)
        self.max_hz         = float(self.get_parameter('max_hz').value)
        self._last_proc_time = None

        self.bridge = CvBridge()
        self._frame_idx = 0
        self._last_log_time = None
        self.last_depth = None
        self.K = None

        # --- cached frames for one-shot scans ---
        self._latest_bgr = None
        self._t_bgr = 0.0
        self._t_depth = 0.0

        # Inspection gating (for continuous on-gate prints)
        self.inspect_enabled = False
        self.inspect_label = ""
        self._printed_this_window = False

        self.sub_gate  = self.create_subscription(Bool, '/inspect_enable', self.cb_gate, 10)
        self.sub_label = self.create_subscription(String, '/inspect_label', self.cb_label, 10)

        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5)
        self.sub_rgb   = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, qos_sensor)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_sensor)
        self.sub_info  = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 10)

        # scan request/response bridge
        self.req_sub = self.create_subscription(String, self.req_topic, self._on_scan_request, 10)
        self.rep_pub = self.create_publisher(String, self.rep_topic, 10)

        self.get_logger().info(f"Listening on {self.rgb_topic} + {self.depth_topic}")
        self.get_logger().info(f"Scan bridge: req={self.req_topic} rep={self.rep_topic}")
        self.get_logger().info(f"Auto publish: on={self.auto_on} hz={self.auto_hz:.2f} name='{self.auto_name}'")

        # ===== Calibrated HSV for SDF pure colors =====
        # OpenCV HSV: H∈[0..180], S,V∈[0..255]
        # Global vivid floors to reject bark/ground & lighting washout
        self.declare_parameter('s_min', 175)
        self.declare_parameter('v_min', 170)
        S_MIN = int(self.get_parameter('s_min').value)
        V_MIN = int(self.get_parameter('v_min').value)

        def band(name, h_lo, h_hi, s_lo=S_MIN, v_lo=V_MIN):
            """Create a tunable band with per-color params (tweak via ROS params at launch)."""
            self.declare_parameter(f'{name}_h_lo', h_lo)
            self.declare_parameter(f'{name}_h_hi', h_hi)
            self.declare_parameter(f'{name}_s_min', s_lo)
            self.declare_parameter(f'{name}_v_min', v_lo)
            H_LO = int(self.get_parameter(f'{name}_h_lo').value)
            H_HI = int(self.get_parameter(f'{name}_h_hi').value)
            S_LO = int(self.get_parameter(f'{name}_s_min').value)
            V_LO = int(self.get_parameter(f'{name}_v_min').value)
            return (np.array([H_LO, S_LO, V_LO]), np.array([H_HI, 255, 255]))

        # Canonical hues from SDF RGBA:
        # Yellow: H≈30 → 26..34 (tight)
        Y_BANDS = [band('yellow', 26, 34, s_lo=180, v_lo=180)]

        # Green: H≈60 → 56..66 (tight, moved away from yellow)
        G_BANDS = [band('green', 56, 66, s_lo=185, v_lo=180)]

        # Blue: H≈120 → 114..126 (tight)
        B_BANDS = [band('blue', 114, 126, s_lo=175, v_lo=170)]

        # Red: H≈0 and wrap near 180 → two narrow bands
        R_BANDS = [
            band('red1', 0,   6,   s_lo=190, v_lo=180),  # near 0°
            band('red2', 174, 180, s_lo=190, v_lo=180),  # wrap near 180°
        ]

        # Black: hue-agnostic, very low value
        self.declare_parameter('black_s_max', 60)
        self.declare_parameter('black_v_max', 55)
        B_S_MAX = int(self.get_parameter('black_s_max').value)
        B_V_MAX = int(self.get_parameter('black_v_max').value)
        BLACK_BANDS = [(np.array([0, 0, 0]), np.array([180, B_S_MAX, B_V_MAX]))]

        self.hsv_ranges = {
            'red':    R_BANDS,
            'yellow': Y_BANDS,
            'green':  G_BANDS,
            'blue':   B_BANDS,
            'black':  BLACK_BANDS,
        }

        # Keep only requested colors (or all if none specified)
        self.colors = [c for c in requested if c in self.hsv_ranges] or list(self.hsv_ranges.keys())

        self.status_map = {
            'green':  'healthy',
            'red':    'infection 1',
            'blue':   'infection 2',
            'yellow': 'infection 3',
            'black':  'remove',
        }

        self.kernel = np.ones((3,3), np.uint8)

        # NEW: start auto publish timer
        self._auto_timer = None
        if self.auto_on and self.auto_hz > 0:
            period = max(0.02, 1.0 / self.auto_hz)
            self._auto_timer = self.create_timer(period, self._auto_publish_tick)

    # --- gate callbacks ---
    def cb_gate(self, msg: Bool):
        if msg.data and not self.inspect_enabled:
            self._printed_this_window = False
        self.inspect_enabled = bool(msg.data)

    def cb_label(self, msg: String):
        self.inspect_label = msg.data or ""

    # --- Camera callbacks ---
    def cb_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def cb_depth(self, msg: Image):
        try:
            if msg.encoding == '32FC1':
                self.last_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                d16 = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.last_depth = d16.astype(np.float32)/1000.0
            self._t_depth = time.time()
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def cb_rgb(self, msg: Image):
        bgr_latest = self._to_bgr(msg)
        if bgr_latest is not None:
            self._latest_bgr = bgr_latest
            self._t_bgr = time.time()

        if not self.inspect_enabled:
            return

        self._frame_idx += 1
        if (self._frame_idx % self.process_every_n) != 0:
            return
        if self.max_hz > 0:
            now = self.get_clock().now()
            if self._last_proc_time:
                dt = (now - self._last_proc_time).nanoseconds * 1e-9
                if dt < 1.0/self.max_hz:
                    return
            self._last_proc_time = now

        if self._latest_bgr is None:
            return

        label, status, covs, dist_m = self._classify_dominant(self._latest_bgr, self.last_depth)
        if not self._printed_this_window:
            self._printed_this_window = True
            cov_txt = " ".join([f"{k}={covs.get(k,0.0):.4f}" for k in self.colors])
            self.get_logger().info(
                f"[INSPECTION {self.inspect_label}] status={status} label={label} {cov_txt} dist_m={dist_m:.2f}"
            )

    # --- AUTO publish tick (runs even without /scan_request) ---
    def _auto_publish_tick(self):
        # Require fresh-ish frames
        now = time.time()
        if (self._latest_bgr is None or self.last_depth is None or
            (now - self._t_bgr) > 1.0 or (now - self._t_depth) > 1.0):
            return

        label, status, covs, dist_m = self._classify_dominant(self._latest_bgr, self.last_depth)
        cov_txt = " ".join([f"{k}={covs.get(k,0.0):.4f}" for k in self.colors])
        name = self.inspect_label if self.inspect_label else self.auto_name
        line = f"{name}: {status} (label={label} {cov_txt} dist_m={dist_m:.2f})"
        self._publish_report(line)

    # --- request/response scan bridge (kept) ---
    def _on_scan_request(self, msg: String):
        name = (msg.data or "").strip()
        self.get_logger().info(f"[color_3d_detector] scan request: '{name}'")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if self._latest_bgr is not None and self.last_depth is not None:
                if (time.time() - self._t_bgr) < 1.0 and (time.time() - self._t_depth) < 1.0:
                    break
            rclpy.spin_once(self, timeout_sec=0.05)

        if self._latest_bgr is None or self.last_depth is None:
            self._publish_report(f"{name}: no_data")
            return

        label, status, covs, dist_m = self._classify_dominant(self._latest_bgr, self.last_depth)
        cov_txt = " ".join([f"{k}={covs.get(k,0.0):.4f}" for k in self.colors])
        line = f"{name}: {status} (label={label} {cov_txt} dist_m={dist_m:.2f})"
        self._publish_report(line)

    def _publish_report(self, text: str):
        self.rep_pub.publish(String(data=text))
        self.get_logger().info(f"[color_3d_detector] report -> {self.rep_topic}: {text}")

    # --- core classification ---
    def _classify_dominant(self, bgr: np.ndarray, depth_m: np.ndarray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        covs, masks = {}, {}
        for name in self.colors:
            cov, mask = self._coverage_for_color(hsv, name)
            covs[name] = float(cov); masks[name] = mask

        label = max(covs, key=covs.get) if covs else 'unknown'
        if covs.get(label, 0.0) < self.min_coverage:
            return label, 'unknown', covs, float('nan')

        status = self.status_map.get(label, 'unknown')

        dist_m = float('nan')
        pick = masks.get(label, None)
        if pick is not None and depth_m is not None:
            ys, xs = np.where(pick > 0)
            if xs.size > 50:
                z = depth_m[ys, xs].astype(np.float32)
                z = z[(z > 0.05) & (z < 20.0)]
                if z.size > 0:
                    dist_m = float(np.median(z))

        return label, status, covs, dist_m

    # --- helpers ---
    def _coverage_for_color(self,hsv,name):
        if name not in self.hsv_ranges:
            return 0.0,np.zeros(hsv.shape[:2],dtype=np.uint8)
        mask=np.zeros(hsv.shape[:2],dtype=np.uint8)
        for lo,hi in self.hsv_ranges[name]:
            mask|=cv2.inRange(hsv,lo,hi)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernel)
        return float(np.count_nonzero(mask))/mask.size,mask

    def _to_bgr(self,msg:Image):
        try:
            enc=msg.encoding.lower()
            if enc=='bgr8':
                return self.bridge.imgmsg_to_cv2(msg,'bgr8')
            if enc=='rgb8':
                rgb=self.bridge.imgmsg_to_cv2(msg,'rgb8')
                return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            if enc in('rgba8','bgra8'):
                rgba=self.bridge.imgmsg_to_cv2(msg,'rgba8')
                return cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
            if enc=='mono8':
                gray=self.bridge.imgmsg_to_cv2(msg,'mono8')
                return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
        return None


def main():
    rclpy.init()
    node=Color3DDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()

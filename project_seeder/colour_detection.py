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
        self.declare_parameter('show_debug', False)

        # Loosen coverage threshold so distant trees classify
        self.declare_parameter('min_coverage', 0.002)  # was 0.005

        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('max_hz', 10.0)

        # request/response topics
        self.declare_parameter('scan_request_topic', '/scan_request')
        self.declare_parameter('scan_report_topic',  '/scan_report')

        # auto publish on startup
        self.declare_parameter('auto_scan_on_start', True)
        self.declare_parameter('auto_pub_hz', 0.4)
        self.declare_parameter('auto_scan_name', 'Health')

        # Optional: central ROI to focus on the middle (helps small/distant trees)
        self.declare_parameter('use_central_roi', True)
        self.declare_parameter('roi_margin_frac', 0.20)  # keep central 60%

        # --- Read params
        self.rgb_topic   = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic  = self.get_parameter('info_topic').value
        self.req_topic   = self.get_parameter('scan_request_topic').value
        self.rep_topic   = self.get_parameter('scan_report_topic').value #communicates with tree goals

        self.auto_on     = bool(self.get_parameter('auto_scan_on_start').value)
        self.auto_hz     = float(self.get_parameter('auto_pub_hz').value)
        self.auto_name   = str(self.get_parameter('auto_scan_name').value)

        requested = [c.strip().lower() for c in self.get_parameter('colors').value.split(',') if c.strip()]
        self.show_debug      = bool(self.get_parameter('show_debug').value)
        self.min_coverage    = float(self.get_parameter('min_coverage').value)
        self.process_every_n = int(self.get_parameter('process_every_n').value)
        self.max_hz          = float(self.get_parameter('max_hz').value)
        self.use_central_roi = bool(self.get_parameter('use_central_roi').value)
        self.roi_margin_frac = float(self.get_parameter('roi_margin_frac').value)

        self._last_proc_time = None

        self.bridge = CvBridge()
        self._frame_idx = 0
        self._last_log_time = None
        self.last_depth = None
        self.K = None

        # updating the Latest frames
        self._latest_bgr = None
        self._t_bgr = 0.0
        self._t_depth = 0.0

        # Inspect gthe ate
        self.inspect_enabled = False
        self.inspect_label = ""
        self._printed_this_window = False

        # Gate subs
        self.sub_gate  = self.create_subscription(Bool, '/inspect_enable', self.cb_gate, 10)
        self.sub_label = self.create_subscription(String, '/inspect_label', self.cb_label, 10)

        #hor ros 2 communicates messages
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5)
        self.sub_rgb   = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, qos_sensor)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_sensor)
        self.sub_info  = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 10)

        # scan request/response bridge
        self.req_sub = self.create_subscription(String, self.req_topic, self._on_scan_request, 10)
        self.rep_pub = self.create_publisher(String, self.rep_topic, 10)

        # ===== HSV ranges =====
        # Looser global saturation and value given to determine the given value (open cv considers from values 0-180)
        self.declare_parameter('s_min', 130)  # was 175
        self.declare_parameter('v_min', 100)  # was 170
        S_MIN = int(self.get_parameter('s_min').value)
        V_MIN = int(self.get_parameter('v_min').value)

        def band(name, h_lo, h_hi, s_lo=S_MIN, v_lo=V_MIN): #set for rach colour
            self.declare_parameter(f'{name}_h_lo', h_lo) #max hue
            self.declare_parameter(f'{name}_h_hi', h_hi) #min hue
            self.declare_parameter(f'{name}_s_min', s_lo) #min saturation
            self.declare_parameter(f'{name}_v_min', v_lo) #min value
            H_LO = int(self.get_parameter(f'{name}_h_lo').value) #lower hue bound
            H_HI = int(self.get_parameter(f'{name}_h_hi').value) #upper hue bound 
            S_LO = int(self.get_parameter(f'{name}_s_min').value) #saturation lower bound
            V_LO = int(self.get_parameter(f'{name}_v_min').value) #min value #brighthtness
            return (np.array([H_LO, S_LO, V_LO]), np.array([H_HI, 255, 255]))

        # Yellow 30 degrees
        Y_range = [band('yellow', 26, 34, s_lo=max(170, S_MIN), v_lo=max(170, V_MIN))]
        # Green (wider H, softer S/V) 
        G_range = [band('green', 50, 80, s_lo=max(140, S_MIN), v_lo=max(90, V_MIN))]
        # Blue round 120 degrees
        B_range = [band('blue', 114, 126, s_lo=max(160, S_MIN), v_lo=max(120, V_MIN))]
        # Red (two bands)
        R_range = [
            band('red1', 0,   6,   s_lo=max(170, S_MIN), v_lo=max(140, V_MIN)), #0 degrees begining of sclae
            band('red2', 174, 180, s_lo=max(170, S_MIN), v_lo=max(140, V_MIN)), #180 end of hue scale 
        ]
        # Black
        self.declare_parameter('black_s_max', 60) #max saturation for black
        self.declare_parameter('black_v_max', 55) #max valuye for black
        B_smax = int(self.get_parameter('black_s_max').value) #max saturation value 
        B_vmax = int(self.get_parameter('black_v_max').value) #get max value 
        Black_range = [(np.array([0, 0, 0]), np.array([180, B_smax, B_vmax]))] #any hue, low saturation, low value 

        self.hsv_ranges = {
            'red':    R_range,
            'yellow': Y_range,
            'green':  G_range,
            'blue':   B_range,
            'black':  Black_range,
        }

        self.colors = [c for c in requested if c in self.hsv_ranges] or list(self.hsv_ranges.keys()) #keep requested colours, select them withing set ranges

        self.status_map = {
            'green':  'Healthy',
            'red':    'Tree Infected: Cinnamon Fungus',
            'blue':   'Tree Infected: Eucalyptus Leaf Blister',
            'yellow': 'Tree Infected: Myrtle Rust',
            'black':  'Dead: Remove Tree',
        }

        # treatment selection 
        self.declare_parameter('pesticide_info_delay_sec', 2.0)
        self.pesticide_delay = float(self.get_parameter('pesticide_info_delay_sec').value)
        self.solution_map = {
            'yellow': 'Copper Oxychloride',
            'red':    'Phosphite Injections',
            'blue':   'Mancozeb',
            'green':  'None required',
            'black':  'Removal / no pesticide',
        }

        #  dose scaling params
        self.declare_parameter('dose_min_ml', 2.0)
        self.declare_parameter('dose_max_ml', 10.0)
        self.declare_parameter('dose_gamma', 1.0)  # >1 makes high-intensity doses ramp faster
        self.dose_min_ml = float(self.get_parameter('dose_min_ml').value)
        self.dose_max_ml = float(self.get_parameter('dose_max_ml').value)
        self.dose_gamma  = float(self.get_parameter('dose_gamma').value)

        self.kernel = np.ones((3,3), np.uint8)

        # Auto publish timer
        self._auto_timer = None
        if self.auto_on and self.auto_hz > 0:
            period = max(0.02, 1.0 / self.auto_hz)
            self._auto_timer = self.create_timer(period, self._auto_publish_tick)

        self._pending_pesticide_timers = []

    # --- gate callbacks ---
    def cb_gate(self, msg: Bool):
        if msg.data and not self.inspect_enabled: #if the gate is enabled
            self._printed_this_window = False #resetting the flag 
        self.inspect_enabled = bool(msg.data)

    def cb_label(self, msg: String):
        self.inspect_label = msg.data or "" #receive the inspection message 

    # --- Camera callbacks ---
    def cb_info(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5]) #receive camera calibration information
    #focal elgnths and principal points form camera calibration

    def cb_depth(self, msg: Image): #depth image 
        try:
            if msg.encoding == '32FC1': #open cv format
                self.last_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1') 
            elif msg.encoding == '16UC1':
                d16 = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self.last_depth = d16.astype(np.float32)/1000.0
            self._t_depth = time.time() #record the depth image 
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def cb_rgb(self, msg: Image): #receving colour image 
        bgr_latest = self._to_bgr(msg) #ros to bgr
        if bgr_latest is not None:
            self._latest_bgr = bgr_latest #store it 
            self._t_bgr = time.time()

        if not self.inspect_enabled:
            return

        self._frame_idx += 1 #incrmetnt eh frame counnters
        if (self._frame_idx % self.process_every_n) != 0: #detemriens if frame shoudl eb kepth
            return #otherwise skip 
        if self.max_hz > 0:
            now = self.get_clock().now()
            if self._last_proc_time: #if frame has been rpocessed 
                dt = (now - self._last_proc_time).nanoseconds * 1e-9
                if dt < 1.0/self.max_hz: #skip this frame to stay in limit 
                    return
            self._last_proc_time = now

        if self._latest_bgr is None:
            return

        label, status, covs, dist_m, intensity = self._classify_dominant(self._latest_bgr, self.last_depth)
        if not self._printed_this_window:
            self._printed_this_window = True
            cov_txt = " ".join([f"{k}={covs.get(k,0.0):.4f}" for k in self.colors])
            self.get_logger().info(
                f"[INSPECTION {self.inspect_label}] status={status} label={label} {cov_txt} dist_m={dist_m:.2f} intensity={intensity:.2f}"
            )
            follow_name = self.inspect_label if self.inspect_label else self.auto_name
            self._schedule_pesticide_append(follow_name, label, intensity)

    # --- AUTO publish tick ---
    def _auto_publish_tick(self):
        now = time.time()
        if (self._latest_bgr is None or self.last_depth is None or
            (now - self._t_bgr) > 1.0 or (now - self._t_depth) > 1.0):
            return

         # Classify what color/disease is dominant in the image
        label, status, covs, dist_m, intensity = self._classify_dominant(self._latest_bgr, self.last_depth)
        name = self.inspect_label if self.inspect_label else self.auto_name
        chosen, ml = self._dose_from_intensity(label, intensity)
        # include dose so tree_goals (supports your _parse_dose_ml)
        line = f"{name}: {status} dose_ml={ml:.1f}"
        self._publish_report(line)

        # still print the nice block after a short delay
        self._schedule_pesticide_append(name, label, intensity)

    # --- dose helper (same math as printing block) ---
    def _dose_from_intensity(self, label: str, intensity: float) -> tuple[str, float]:
        """
        Returns (chosen_solution, dose_ml) based on given colour 
        For healthy/black, returns ('None required' or 'Removal / no pesticide', 0.0).
        """
        chosen = self.solution_map.get(label, 'N/A')

        # healthy/dead -> 0 ml
        if label in ('green', 'black'):
            return chosen, 0.0

        i = max(0.0, min(1.0, float(intensity)))
        ml = self.dose_min_ml + (self.dose_max_ml - self.dose_min_ml) * (i ** self.dose_gamma)
        ml = float(np.clip(ml, self.dose_min_ml, self.dose_max_ml))
        return chosen, ml

    # --- request/response scan bridge ---
    def _on_scan_request(self, msg: String): #manual scan 
        name = (msg.data or "").strip()
        self.get_logger().info(f"scan request: '{name}'")
        t0 = time.time() 
        while time.time() - t0 < 2.0: #
            if self._latest_bgr is not None and self.last_depth is not None: #cheks if the images are recent 
                if (time.time() - self._t_bgr) < 1.0 and (time.time() - self._t_depth) < 1.0:
                    break
            rclpy.spin_once(self, timeout_sec=0.05)

        if self._latest_bgr is None or self.last_depth is None:
            self._publish_report(f"{name}: no_data")
            return

        label, status, covs, dist_m, intensity = self._classify_dominant(self._latest_bgr, self.last_depth)
        # wire format (no brackets, no covs) so tree_goal_runner nicely formatted easily
        chosen, ml = self._dose_from_intensity(label, intensity)
        line = f"{name}: {status} dose_ml={ml:.1f}"
        self._publish_report(line)
        self._schedule_pesticide_append(name, label, intensity)


    def _publish_report(self, text: str):  # will communicate with the tree_goals 
        self.rep_pub.publish(String(data=text))
        self.get_logger().info(f" {self.rep_topic}: {text}")

    # --- pesticide follow-up helper (dose scales with color intensity) ---
    def _schedule_pesticide_append(self, name: str, label: str, intensity: float):
        # Skip for healthy/black (no treatment)
        if label in ('green', 'black'):
            return

        chosen = self.solution_map.get(label, 'N/A') #appropirate treatment for teh diesease

        # Compute dose in ml from intensity (0..1)
        i = max(0.0, min(1.0, float(intensity)))
        ml = self.dose_min_ml + (self.dose_max_ml - self.dose_min_ml) * (i ** self.dose_gamma)
        ml = float(np.clip(ml, self.dose_min_ml, self.dose_max_ml))
        ml_str = f"{ml:.1f}"

        text = "\n".join([
            f"{name}:",
            "- Available treatments:",
            "  Copper oxychloride",
            "  Phosphite injections",
            "  Mancozeb",
            f"Chosen solution: {chosen}",
            f"Dose: {ml_str} ml",
        ])

        holder = {}
        def fire_once():
            print(text)
            t = holder.get('t')
            if t is not None:
                try: t.cancel()
                except Exception: pass
                try: self._pending_pesticide_timers.remove(t)
                except Exception: pass

        t = self.create_timer(self.pesticide_delay, fire_once)
        holder['t'] = t
        self._pending_pesticide_timers.append(t)

    # --- core classification ---
    def _classify_dominant(self, bgr: np.ndarray, depth_m: np.ndarray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) #bgr to hsv

        # Optional central ROI (keeps central 60% if margin=0.2)
        if self.use_central_roi:
            h, w = hsv.shape[:2]
            m = self.roi_margin_frac
            x0, x1 = int(w*m), int(w*(1.0-m))
            y0, y1 = int(h*m), int(h*(1.0-m))
            hsv = hsv[y0:y1, x0:x1]
            if depth_m is not None:
                depth_m = depth_m[y0:y1, x0:x1]

        covs, masks = {}, {}
        for name in self.colors:
            cov, mask = self._coverage_for_color(hsv, name)
            covs[name] = float(cov); masks[name] = mask

        label = max(cvs := covs, key=cvs.get) if covs else 'unknown'
        if covs.get(label, 0.0) < self.min_coverage:
            return label, 'unknown', covs, float('nan'), 0.0

        status = self.status_map.get(label, 'unknown')

        # Distance estimate
        dist_m = float('nan')
        pick = masks.get(label, None)
        if pick is not None and depth_m is not None:
            ys, xs = np.where(pick > 0)
            if xs.size > 50:
                z = depth_m[ys, xs].astype(np.float32)
                z = z[(z > 0.05) & (z < 20.0)]
                if z.size > 0:
                    dist_m = float(np.median(z))

        # --- Color intensity (0..1) from S and V within mask ---
        intensity = 0.0
        if pick is not None:
            S = hsv[..., 1]
            V = hsv[..., 2]
            idx = np.where(pick > 0)
            if idx[0].size > 0:
                mean_s = float(np.mean(S[idx])) / 255.0
                mean_v = float(np.mean(V[idx])) / 255.0
                # Average of normalized S and V (simple, robust)
                intensity = 0.5 * (mean_s + mean_v)
                intensity = max(0.0, min(1.0, intensity))

        return label, status, covs, dist_m, intensity

    # --- helpers ---
    def _coverage_for_color(self, hsv, name): #how muchnof the image consists of that colour
        if name not in self.hsv_ranges: #if colour is wihin the ranged we set 
            return 0.0, np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8) #empty binary amsk 
        for lo, hi in self.hsv_ranges[name]:#hsc range this colour 
            mask |= cv2.inRange(hsv, lo, hi)
        # Clean speckle, then bridge gaps & thicken blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return float(np.count_nonzero(mask)) / mask.size, mask

    def _to_bgr(self, msg: Image): #convert ros image message ot open vcv bgr format 
        try:
            enc = msg.encoding.lower() #store picels 
            if enc == 'bgr8': #alreayd in bgr format 
                return self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if enc == 'rgb8': #open cv rgb 
                rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) #swap red and blue chnnels to get BGR
            if enc in ('rgba8','bgra8'):
                rgba = self.bridge.imgmsg_to_cv2(msg, 'rgba8') #opencv rgb
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if enc == 'mono8':
                gray = self.bridge.imgmsg_to_cv2(msg, 'mono8') #greyscale to bgr 
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
        return None


def main():
    rclpy.init()
    node = Color3DDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()

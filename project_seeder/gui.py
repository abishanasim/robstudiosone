#!/usr/bin/env python3
import os, sys, time, threading, xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import signal
import subprocess
import tkinter as tk
from tkinter import ttk

# --- Camera deps (optional) ---
try:
    from sensor_msgs.msg import Image, CompressedImage
except Exception:
    Image = CompressedImage = None
try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None
try:
    from PIL import Image as PILImage
    from PIL import ImageTk
except Exception:
    PILImage = ImageTk = None
try:
    import numpy as np
    import cv2
except Exception:
    np = cv2 = None

# ---------- Colors (Dark-Green theme) ----------
BG        = "#0b120e"
PANEL     = "#0f1a14"
FG        = "#e7f2ea"
SUBFG     = "#a9c4b2"
BTN_BG    = "#17261f"
ACCENT    = "#2b604a"
EDGE      = "#1e3a2f"
RED       = "#cc2e2e"
RED_DARK  = "#9e1f1f"
POSE_OUTLINE          = "#2a4136"
POSE_SELECTED_OUTLINE = "#ffffff"

# ---------- Camera config ----------
IMAGE_TOPIC       = "/camera/image"
COMPRESSED_TOPIC  = "/camera/image/compressed"
CAM_W, CAM_H      = 240, 180

# ---------- World/Grid mapping ----------
METERS_PER_CELL = 0.5
DOT_RADIUS_PX   = 5
PINE_KEYWORDS   = ["pine","pinetree","pine_tree","pine-tree"]

# Exactly 5 pines in your GUI mini-map
SDF_POSES = {
    "pine1": (-2.5,  1.10, 0,0,0,0),
    "pine2": (-2.5, -1.05,0,0,0,0),
    "pine3": (-2.5, -3.25,0,0,0,0),
    "pine4": (-2.5, -5.40,0,0,0,0),
    "pine5": (-2.5, -7.60,0,0,0,0),
}
ROW1_NAMES = list(SDF_POSES.keys())

# ---------- ROS Bridge ----------
try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
    from std_msgs.msg import Bool, String
    HAVE_ROS = True
except Exception as e:
    HAVE_ROS = False
    print("[GUI] ROS unavailable:", e)

SCRIPT_DIR = Path(__file__).resolve().parent
SDF_PATH = (SCRIPT_DIR / "seeder.sdf") if (SCRIPT_DIR / "seeder.sdf").exists() else None

def looks_like_pine(name: str) -> bool:
    return any(k in (name or "").lower() for k in PINE_KEYWORDS)

def rgba_to_hex(r, g, b):
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def extract_tree_colors_from_sdf(path):
    colors = {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return colors
    for model in root.iter("model"):
        mname = model.attrib.get("name", "")
        if not looks_like_pine(mname):
            continue
        diff = model.find(".//diffuse")
        if diff is None or not diff.text:
            continue
        vals = diff.text.split()
        try:
            r, g, b = float(vals[0]), float(vals[1]), float(vals[2])
            colors[mname] = rgba_to_hex(r, g, b)
        except Exception:
            pass
    return colors

# --------- Terminal helpers ---------
_TERMINALS = ["gnome-terminal", "konsole", "xfce4-terminal", "xterm"]

_autonomous_procs = []
_autonomous_headless = []
_manual_procs = []
_manual_headless = []

def _pick_terminal():
    for t in _TERMINALS:
        if shutil.which(t):
            return t
    return None

def _launch_in_terminal(title: str, bash_cmd: str):
    term = _pick_terminal()
    if not term:
        return None
    if term == "gnome-terminal":
        return subprocess.Popen([term, "--title", title, "--", "bash", "-lc", bash_cmd],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if term == "konsole":
        return subprocess.Popen([term, "-p", f"tabtitle={title}", "-e", "bash", "-lc", bash_cmd],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if term == "xfce4-terminal":
        return subprocess.Popen([term, "--title", title, "--command", f"bash -lc '{bash_cmd}'"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if term == "xterm":
        return subprocess.Popen([term, "-T", title, "-e", "bash", "-lc", bash_cmd],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None

def _launch_headless(bash_cmd: str):
    return subprocess.Popen(["bash", "-lc", bash_cmd],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)

def _pkill(cmd_substr):
    try:
        subprocess.run(["pkill", "-f", cmd_substr],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

# --------- AUTONOMOUS (tree_goals + colour_detection) ----------
def _start_autonomous_terms():
    global _autonomous_procs, _autonomous_headless
    _autonomous_procs.clear(); _autonomous_headless.clear()
    cmd_goals = "source ~/41068_ws/install/setup.bash; ros2 run project_seeder tree_goals"
    cmd_color = "source ~/41068_ws/install/setup.bash; ros2 run project_seeder colour_detection"

    p1 = _launch_in_terminal("tree_goals", cmd_goals)
    p2 = _launch_in_terminal("colour_detection", cmd_color)
    if p1 and p2:
        _autonomous_procs.extend([p1, p2])
        return True

    for p in (p1, p2):
        try:
            if p: p.terminate()
        except Exception: pass
    h1 = _launch_headless(cmd_goals)
    h2 = _launch_headless(cmd_color)
    _autonomous_headless.extend([h1, h2])
    return True

def _stop_autonomous_terms():
    global _autonomous_procs, _autonomous_headless
    for p in list(_autonomous_procs):
        try: p.terminate()
        except Exception: pass
    time.sleep(0.4)
    for p in list(_autonomous_procs):
        try:
            if p.poll() is None: p.kill()
        except Exception: pass
    _autonomous_procs.clear()

    for p in list(_autonomous_headless):
        try:
            if p.poll() is None: os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception: pass
    time.sleep(0.4)
    for p in list(_autonomous_headless):
        try:
            if p.poll() is None: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
    _autonomous_headless.clear()

    _pkill("ros2 run project_seeder tree_goals")
    _pkill("ros2 run project_seeder colour_detection")

# --------- MANUAL (manual_nav + colour_detection) ----------
def _start_manual_terms():
    global _manual_procs, _manual_headless
    _manual_procs.clear(); _manual_headless.clear()
    cmd_manual = "source ~/41068_ws/install/setup.bash; ros2 run project_seeder manual_nav"
    cmd_color  = "source ~/41068_ws/install/setup.bash; ros2 run project_seeder colour_detection"

    p1 = _launch_in_terminal("manual_nav", cmd_manual)
    p2 = _launch_in_terminal("colour_detection", cmd_color)
    if p1 and p2:
        _manual_procs.extend([p1, p2])
        return True

    for p in (p1, p2):
        try:
            if p: p.terminate()
        except Exception: pass
    h1 = _launch_headless(cmd_manual)
    h2 = _launch_headless(cmd_color)
    _manual_headless.extend([h1, h2])
    return True

def _stop_manual_terms():
    global _manual_procs, _manual_headless
    for p in list(_manual_procs):
        try: p.terminate()
        except Exception: pass
    time.sleep(0.4)
    for p in list(_manual_procs):
        try:
            if p.poll() is None: p.kill()
        except Exception: pass
    _manual_procs.clear()

    for p in list(_manual_headless):
        try:
            if p.poll() is None: os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception: pass
    time.sleep(0.4)
    for p in list(_manual_headless):
        try:
            if p.poll() is None: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
    _manual_headless.clear()

    _pkill("ros2 run project_seeder manual_nav")
    _pkill("ros2 run project_seeder colour_detection")

# --------- ROS bridge ---------
class RosBridge:
    """GUI<->ROS bridge, plus camera subscription & echoed state."""
    def __init__(self):
        self.node = None
        self.estop_pub = None
        self.resume_pub = None
        self.target_pub = None
        self.operate_pub = None
        self.status_sub = None
        self.executor = None
        self._status_cb = None

        # UI state callback + flags
        self._state_cb = None
        self.estopped_flag = False
        self.operating_flag = False
        self.manual_flag = False
        self.move_pub = None
        self.manual_mode_pub = None
        self.manual_send_pub = None

        # Camera bits
        self.bridge = CvBridge() if CvBridge else None
        self.latest_bgr = None

        if not HAVE_ROS:
            return

        rclpy.init(args=None)
        self.node = rclpy.create_node("seeder_gui")

        qos_durable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # pubs/subs
        self.estop_pub    = self.node.create_publisher(Bool,   "/seeder/estop",       qos_durable)
        self.resume_pub   = self.node.create_publisher(Bool,   "/seeder/resume",      qos_fast)
        self.target_pub   = self.node.create_publisher(String, "/seeder/target_tree", qos_fast)
        self.operate_pub  = self.node.create_publisher(Bool,   "/seeder/operate",     qos_durable)
        self.status_sub   = self.node.create_subscription(String, "/seeder/status", self._on_status, 10)
        self.move_pub     = self.node.create_publisher(Bool, "/seeder/move", 10)
        self.manual_mode_pub = self.node.create_publisher(Bool, "/seeder/manual_mode", qos_durable)
        self.manual_send_pub = self.node.create_publisher(Bool, "/seeder/manual_send", 10)

        # Echoed states
        self.node.create_subscription(Bool, "/seeder/estop",        self._on_estop_state,   10)
        self.node.create_subscription(Bool, "/seeder/operate",      self._on_operate_state, 10)
        self.node.create_subscription(Bool, "/seeder/manual_mode",  self._on_manual_state,  10)

        # Camera
        if Image and CompressedImage:
            cam_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            img_topic, cmp_topic = self._pick_camera_topics()
            img_topic = img_topic or IMAGE_TOPIC
            cmp_topic = cmp_topic or COMPRESSED_TOPIC
            try:
                self.node.create_subscription(Image,           img_topic, self._on_image,      cam_qos)
                self.node.create_subscription(CompressedImage, cmp_topic, self._on_compressed, cam_qos)
                self.node.get_logger().info(f"[GUI] Camera subscribe -> raw='{img_topic}', compressed='{cmp_topic}'")
            except Exception as e:
                self.node.get_logger().warn(f"[GUI] Camera subscription failed: {e}")

        from rclpy.executors import SingleThreadedExecutor
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        print("[GUI] Topics: /seeder/estop, /seeder/resume, /seeder/target_tree, /seeder/operate, /seeder/status, /seeder/move, /seeder/manual_mode, /seeder/manual_send (+ camera)")

    # --- publish API ---
    def publish_estop(self, val: bool):
        if self.estop_pub:
            self.estop_pub.publish(Bool(data=val))
    def publish_resume(self):
        if self.resume_pub:
            self.resume_pub.publish(Bool(data=True))
    def publish_target(self, name: str):
        if self.target_pub:
            self.target_pub.publish(String(data=name))
    def publish_operate(self, val: bool):
        if self.operate_pub:
            self.operate_pub.publish(Bool(data=val))
    def publish_move(self):
        if self.move_pub:
            self.move_pub.publish(Bool(data=True))
    def publish_manual_mode(self, val: bool):
        if self.manual_mode_pub:
            self.manual_mode_pub.publish(Bool(data=val))
    def publish_manual_send(self):
        if self.manual_send_pub:
            self.manual_send_pub.publish(Bool(data=True))

    # --- callbacks wiring ---
    def set_status_callback(self, cb): self._status_cb = cb
    def set_state_callback(self, cb):  self._state_cb  = cb
    def _on_status(self, msg):
        if self._status_cb: self._status_cb(msg.data)
    def _on_estop_state(self, msg):
        self.estopped_flag = bool(msg.data)
        if self._state_cb: self._state_cb(self.estopped_flag, self.operating_flag, self.manual_flag)
    def _on_operate_state(self, msg):
        self.operating_flag = bool(msg.data)
        if self._state_cb: self._state_cb(self.estopped_flag, self.operating_flag, self.manual_flag)
    def _on_manual_state(self, msg):
        self.manual_flag = bool(msg.data)
        if self._state_cb: self._state_cb(self.estopped_flag, self.operating_flag, self.manual_flag)

    # --- camera internals ---
    def _pick_camera_topics(self):
        img_topic = None
        cmp_topic = None
        try:
            for name, types in self.node.get_topic_names_and_types():
                if "sensor_msgs/msg/Image" in types and ("/camera/" in name or name.endswith("/image")):
                    img_topic = name
                if "sensor_msgs/msg/CompressedImage" in types and ("compressed" in name):
                    cmp_topic = name
        except Exception:
            pass
        return img_topic, cmp_topic

    def _on_image(self, msg):
        if not (self.bridge and cv2 and np): return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_bgr = cv_img
        except Exception: pass

    def _on_compressed(self, msg):
        if not (cv2 and np): return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                self.latest_bgr = cv_img
        except Exception: pass

    def shutdown(self):
        if not HAVE_ROS: return
        try:
            if self.executor: self.executor.shutdown()
            if self.node: self.node.destroy_node()
            rclpy.shutdown()
        except Exception: pass

# ---------------- Splash + Boot ----------------
class Splash:
    def __init__(self, root, title="Seeder", version="", img_left=None, img_right=None):
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.configure(bg=BG)
        self.top.attributes("-topmost", True)
        self._start_ms = int(time.time() * 1000)
        self._min_show_ms = 2400

        frame = tk.Frame(self.top, bg=PANEL, bd=0, highlightthickness=0)
        frame.pack(padx=48, pady=48)

        header = tk.Frame(frame, bg=PANEL)
        header.grid(row=0, column=0, pady=(16, 20))

        def _load_img(path):
            if path and os.path.exists(path):
                try:
                    pil = PILImage.open(path).convert("RGBA")
                    pil.thumbnail((180, 180), PILImage.LANCZOS)
                    return ImageTk.PhotoImage(pil)
                except Exception:
                    return None
            return None

        self._img_left = _load_img(img_left)
        self._img_right = _load_img(img_right)

        left_box = tk.Label(header, bg=PANEL)
        left_box.grid(row=0, column=0, padx=(0, 20))
        if self._img_left:
            left_box.config(image=self._img_left, width=180, height=180)
        else:
            left_box.config(text="(no image)", fg=SUBFG, width=20, height=10)

        logo = tk.Canvas(header, width=180, height=180, bg=PANEL, highlightthickness=0)
        logo.grid(row=0, column=1)
        cx = 90
        trunk_w = 26
        logo.create_rectangle(cx - trunk_w/2, 130, cx + trunk_w/2, 170, fill="#5c3b22", outline="#3d2818", width=2)
        logo.create_polygon(30, 130, 150, 130, cx, 70, fill=ACCENT, outline=EDGE, width=2)
        logo.create_polygon(40, 108, 140, 108, cx, 60, fill=ACCENT, outline=EDGE, width=2)
        logo.create_polygon(50, 88, 130, 88, cx, 46, fill=ACCENT, outline=EDGE, width=2)

        right_box = tk.Label(header, bg=PANEL)
        right_box.grid(row=0, column=2, padx=(20, 0))
        if self._img_right:
            right_box.config(image=self._img_right, width=180, height=180)
        else:
            right_box.config(text="(no image)", fg=SUBFG, width=20, height=10)

        self.title_lbl = tk.Label(frame, text=title, fg=FG, bg=PANEL, font=("Segoe UI", 20, "bold"))
        self.title_lbl.grid(row=1, column=0, pady=(6, 4))
        self.ver_lbl = tk.Label(frame, text=version, fg=SUBFG, bg=PANEL, font=("Segoe UI", 12))
        self.ver_lbl.grid(row=2, column=0, pady=(0, 18))

        self.status = tk.Label(frame, text="Starting...", fg=FG, bg=PANEL, font=("Segoe UI", 12))
        self.status.grid(row=3, column=0, sticky="w", padx=8, pady=(0,10))
        self.pbar = ttk.Progressbar(frame, mode="determinate", length=500, maximum=100)
        self.pbar.grid(row=4, column=0, padx=8, pady=(0,20))

        self._center()

    def _center(self):
        self.top.update_idletasks()
        w = self.top.winfo_width(); h = self.top.winfo_height()
        sw = self.top.winfo_screenwidth(); sh = self.top.winfo_screenheight()
        x = int((sw - w) / 2); y = int((sh - h) / 2.5)
        self.top.geometry(f"{w}x{h}+{x}+{y}")

    def update(self, pct, msg):
        self.status.config(text=msg)
        self.pbar['value'] = pct
        self.top.update_idletasks()

    def close(self, on_closed=None):
        elapsed = int(time.time() * 1000) - self._start_ms
        wait_ms = max(0, self._min_show_ms - elapsed)
        def _finish():
            try:
                self.top.destroy()
            finally:
                if on_closed:
                    on_closed()
        self.top.after(wait_ms, _finish)

# ------------ Globals for UI ------------
root = None
style = None
left = None
grid = None
origin_px = origin_py = 0
ppm = 12 / METERS_PER_CELL
ROW_MAP = {1: []}
PREV_ID = None
selected = None
combo = None
feedback = None
send_manual_btn = None
reload_btn = None
estop = None
circle = None
resume_btn = None
autonomous_btn = None
manual_btn = None
cam_label = None
_cam_imgtk = None
ROS = None
CACHED_COLORS = {}
UI_READY = False
estopped = False

# echoed state
estopped_flag = False
operating_flag = False
manual_flag = False

# local toggles
autonomous_on = False
manual_on = False

# ------------ Drawing helpers ------------
def draw_checkerboard():
    grid.delete("bg")
    w = max(1, grid.winfo_width())
    h = max(1, grid.winfo_height())
    cell = 12
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            color = "#122018" if ((i//cell + j//cell) % 2 == 0) else "#16271f"
            grid.create_rectangle(j, i, j+cell, i+cell, fill=color, outline="#1c2f26", tags=("bg",))

def world_to_canvas(x, y):
    return origin_px + x*ppm, origin_py - y*ppm

def _on_canvas_resize(event=None):
    global origin_px, origin_py
    origin_px = grid.winfo_width()/2.0
    origin_py = grid.winfo_height()/2.0
    draw_checkerboard()
    if UI_READY:
        draw_trees()

def highlight(row, idx):
    global PREV_ID
    if PREV_ID:
        grid.itemconfig(PREV_ID, width=1, outline=POSE_OUTLINE)
    ids = ROW_MAP.get(row, [])
    if 0 <= idx-1 < len(ids):
        cid = ids[idx-1]
        grid.itemconfig(cid, width=3, outline=POSE_SELECTED_OUTLINE)
        PREV_ID = cid

def draw_trees():
    grid.delete("pose")
    ROW_MAP[1].clear()
    for name in ROW1_NAMES:
        x, y, *_ = SDF_POSES[name]
        cx, cy = world_to_canvas(x, y)
        fill = CACHED_COLORS.get(name, "#4caf50")
        cid = grid.create_oval(cx-DOT_RADIUS_PX, cy-DOT_RADIUS_PX,
                               cx+DOT_RADIUS_PX, cy+DOT_RADIUS_PX,
                               fill=fill, outline=POSE_OUTLINE, width=1, tags=("pose",))
        ROW_MAP[1].append(cid)

    values = [f"Row 1 tree {i+1}" for i in range(len(ROW_MAP[1]))]
    if combo and combo.winfo_exists():
        combo["values"] = values
        if not selected.get() or selected.get() not in values:
            selected.set(values[0])
    refresh_controls()

# ------------ Manual-selection helpers ------------
def _selection_allowed():
    estop_ok  = not estopped and not estopped_flag
    manual_ok = (manual_on or manual_flag)
    return estop_ok and manual_ok

def _on_combo_selected(event=None):
    if _selection_allowed():
        send_manual_control()
    else:
        feedback.config(text="Manual is off or E-STOP is active  cant move yet.")

def _on_canvas_click(event):
    if not _selection_allowed():
        feedback.config(text="Manual is off or E-STOP is active  cant move yet.")
        return
    x, y = event.x, event.y
    best_i, best_d2 = None, 10_000_000
    for i, cid in enumerate(ROW_MAP.get(1, [])):
        x1, y1, x2, y2 = grid.bbox(cid)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        if d2 < best_d2:
            best_d2, best_i = d2, i
    if best_i is not None and best_d2 <= (20 ** 2):
        selected.set(f"Row 1 tree {best_i + 1}")
        highlight(1, best_i + 1)
        send_manual_control()

# ------------ Camera tick ------------
def _tick_camera():
    global _cam_imgtk
    try:
        if ROS and ROS.latest_bgr is not None and cam_label is not None and PILImage and ImageTk and cv2:
            frame = ROS.latest_bgr
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(CAM_W / float(w), CAM_H / float(h))
            new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
            rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pil = PILImage.fromarray(rgb_resized)
            _cam_imgtk = ImageTk.PhotoImage(image=pil)
            cam_label.config(image=_cam_imgtk, text="", width=CAM_W, height=CAM_H)
            cam_label.place(x=(CAM_W-new_w)//2, y=(CAM_H-new_h)//2)
        else:
            cam_label.config(image="", text="No camera", fg=SUBFG, bg=PANEL, width=CAM_W, height=CAM_H)
            cam_label.place(x=0, y=0)
    except Exception:
        pass
    if root:
        root.after(50, _tick_camera)

# ------------ UI gating ------------
def refresh_controls():
    """Enable dropdown + Send Manual when (E-STOP cleared) AND manual mode on (local or echoed)."""
    global estopped, autonomous_on, manual_on
    global estopped_flag, operating_flag, manual_flag

    estop_ok   = not estopped and not estopped_flag
    manual_ok  = (manual_on or manual_flag)

    try:
        combo.configure(state=("readonly" if (estop_ok and manual_ok) else "disabled"))
        if estop_ok and manual_ok:
            send_manual_btn.state(["!disabled"])
        else:
            send_manual_btn.state(["disabled"])

        autonomous_btn.config(text=("Stop Autonomous" if (autonomous_on or operating_flag) else "Autonomous Control"))
        manual_btn.config(text=("Disable Manual" if (manual_on or manual_flag) else "Manual Control"))
    except Exception:
        pass

# ------------ Button callbacks ------------
def send_manual_control():
    choice = combo.get().strip()
    if not choice:
        feedback.config(text="Select a tree first")
        return
    try:
        idx = int(choice.split()[3])
        highlight(1, idx)
        name = f"pine{idx}"
        feedback.config(text=f"Manual: sending target {name}")
        ROS.publish_target(name)     # /seeder/target_tree
        ROS.publish_manual_send()    # /seeder/manual_send (edge-trigger)
    except Exception:
        feedback.config(text="Invalid selection")

def emergency_stop(event=None):
    """
    E-STOP:
    1) Publish /seeder/estop True so running nodes can self-halt.
    2) Also hard-stop any spawned processes (autonomous & manual).
    """
    global estopped, autonomous_on, manual_on
    if estopped: return
    estopped = True
    feedback.config(text="E-STOP engaged. Robot halted.")
    try:
        ROS.publish_estop(True)
    except Exception:
        pass

    _stop_autonomous_terms()
    _stop_manual_terms()
    autonomous_on = False
    manual_on = False

    estop.itemconfig(circle, fill=RED_DARK)
    resume_btn.state(["!disabled"])
    refresh_controls()

def resume():
    global estopped
    # Clearing E-STOP alone does NOT resume  we also pulse /seeder/resume.
    estopped = False
    feedback.config(text="Resume pressed. Sending resume pulse.")
    try:
        ROS.publish_estop(False)   # clear E-STOP
        ROS.publish_resume()       # pulse resume
        root.after(200, ROS.publish_resume)  # race-safe double pulse
    except Exception:
        pass
    estop.itemconfig(circle, fill=RED)
    resume_btn.state(["disabled"])
    refresh_controls()

def toggle_autonomous():
    """Start/stop BOTH autonomous nodes: tree_goals + colour_detection; mirror /seeder/operate."""
    global autonomous_on
    new_state = not (autonomous_on or operating_flag)

    if new_state:
        ok = _start_autonomous_terms()
        if ok:
            autonomous_on = True
            ROS.publish_operate(True)
            feedback.config(text="Autonomous: started (tree_goals + colour_detection)")
        else:
            autonomous_on = False
            feedback.config(text="Failed to launch autonomous terminals")
    else:
        _stop_autonomous_terms()
        autonomous_on = False
        ROS.publish_operate(False)
        feedback.config(text="Autonomous: stopped")

    refresh_controls()

def toggle_manual():
    """
    Manual Control toggle:
      - start/stop (manual_nav + colour_detection)
      - publish /seeder/manual_mode (True twice with a short delay to avoid startup races)
    """
    global manual_on
    new_state = not (manual_on or manual_flag)

    if new_state:
        ok = _start_manual_terms()
        if ok:
            manual_on = True
            feedback.config(text="Manual: started (manual_nav + colour_detection)")
            ROS.publish_manual_mode(True)
            root.after(250, lambda: ROS.publish_manual_mode(True))
        else:
            manual_on = False
            feedback.config(text="Failed to launch manual terminals")
    else:
        _stop_manual_terms()
        manual_on = False
        ROS.publish_manual_mode(False)
        feedback.config(text="Manual: stopped")

    refresh_controls()

def close_all():
    _stop_autonomous_terms()
    _stop_manual_terms()
    if ROS: ROS.shutdown()
    root.destroy()

# ------------ Build UI ------------
def build_main_ui():
    global style, left, grid, selected, combo, feedback
    global send_manual_btn, reload_btn, estop, circle, resume_btn, autonomous_btn, manual_btn
    global cam_label, UI_READY, origin_px, origin_py

    root.title("Seeder Control")
    root.configure(bg=BG)

    style = ttk.Style()
    try: style.theme_use("clam")
    except Exception: pass
    style.configure("Dark.TFrame", background=PANEL)
    style.configure("Dark.TLabel", background=PANEL, foreground=FG)
    style.configure("Dark.TButton", background=BTN_BG, foreground=FG, padding=8)
    style.map("Dark.TButton", background=[("active", ACCENT)])

    left = ttk.Frame(root, style="Dark.TFrame", padding=12)
    left.pack(side="left", fill="y")

    grid = tk.Canvas(root, bg=BG, highlightthickness=0)
    grid.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    grid.bind("<Configure>", _on_canvas_resize)
    grid.bind("<Button-1>", _on_canvas_click)

    selected = tk.StringVar(value="Row 1 tree 1")
    combo = ttk.Combobox(left, textvariable=selected, state="disabled", width=22, values=[selected.get()])
    combo.grid(row=0, column=0, columnspan=2, sticky="w")
    combo.bind("<<ComboboxSelected>>", _on_combo_selected)

    feedback = ttk.Label(left, text="Toggle manual or autonomous. E-STOP to halt.", style="Dark.TLabel")
    feedback.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 10))

    # Camera panel
    cam_frame = tk.Frame(left, bg=PANEL, width=CAM_W, height=CAM_H)
    cam_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0,8))
    cam_frame.grid_propagate(False)
    cam_bg = tk.Canvas(cam_frame, width=CAM_W, height=CAM_H, bg=BG, highlightthickness=0)
    cam_bg.pack()
    cam_label = tk.Label(cam_frame, bg=PANEL)
    cam_label.place(x=0, y=0)

    # Buttons
    autonomous_btn = ttk.Button(left, text="Autonomous Control", style="Dark.TButton", command=toggle_autonomous)
    manual_btn      = ttk.Button(left, text="Manual Control", style="Dark.TButton", command=toggle_manual)
    send_manual_btn = ttk.Button(left, text="Send Manual Control", style="Dark.TButton", command=send_manual_control)
    autonomous_btn.grid(row=3, column=0, sticky="ew", pady=(0,8))
    manual_btn.grid(row=3, column=1, sticky="ew", pady=(0,8))
    send_manual_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0,8))
    send_manual_btn.state(["disabled"])

    reload_btn = ttk.Button(left, text="Reload Trees", style="Dark.TButton", command=draw_trees)
    reload_btn.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0,8))

    # E-stop
    global estop, circle
    estop = tk.Canvas(left, width=110, height=110, bg=PANEL, highlightthickness=0)
    estop.grid(row=6, column=0, columnspan=2, pady=(6,0))
    circle = estop.create_oval(8,8,102,102, fill=RED, outline=RED_DARK, width=4)
    label  = estop.create_text(55,55, text="E-STOP", fill="white", font=("Segoe UI", 14, "bold"))
    for tag in (circle, label):
        estop.tag_bind(tag, "<Button-1>", emergency_stop)

    global resume_btn
    resume_btn = ttk.Button(left, text="Resume", style="Dark.TButton", command=resume)
    resume_btn.grid(row=7, column=0, sticky="w", pady=(10,0))
    resume_btn.state(["disabled"])

    close_btn = ttk.Button(left, text="Close", style="Dark.TButton", command=close_all)
    close_btn.grid(row=7, column=1, sticky="ew", pady=(10,0))

    # Finalize + first draw
    global UI_READY, origin_px, origin_py
    UI_READY = True
    root.update_idletasks()
    origin_px = grid.winfo_width()/2.0
    origin_py = grid.winfo_height()/2.0
    _on_canvas_resize()

    if ROS:
        ROS.set_status_callback(lambda m: feedback.config(text=m))
        def _on_state(e, o, m):
            global estopped_flag, operating_flag, manual_flag
            estopped_flag, operating_flag, manual_flag = bool(e), bool(o), bool(m)
            root.after(0, refresh_controls)
        ROS.set_state_callback(_on_state)

    refresh_controls()

# ------------ Boot sequence with splash ------------
def boot_sequence(splash: Splash):
    steps = []
    def step(percent, msg, fn=None):
        def _do():
            splash.update(percent, msg)
            if fn:
                try: fn()
                except Exception as e: print(f"[Startup] {msg} -> {e}")
            if steps:
                root.after(400, steps.pop(0))
            else:
                def show_main():
                    root.deiconify()
                    root.after(100, _tick_camera)
                splash.update(100, "Ready")
                splash.close(on_closed=show_main)
        return _do

    def init_ros():
        global ROS
        ROS = RosBridge()

    def parse_sdf():
        global CACHED_COLORS
        if SDF_PATH and SDF_PATH.exists():
            CACHED_COLORS = extract_tree_colors_from_sdf(str(SDF_PATH))
        else:
            CACHED_COLORS = {}

    def build_ui():
        build_main_ui()

    steps.extend([
        step(5,   "Starting Seeder"),
        step(30,  "Initialising ROS 2 bridge" if HAVE_ROS else "Skipping ROS 2 (not available)", init_ros if HAVE_ROS else None),
        step(60,  "Parsing SDF for tree colours" if SDF_PATH else "No seeder.sdf found, using defaults", parse_sdf if SDF_PATH else None),
        step(85,  "Building widgets", build_ui),
        step(95,  "Drawing scene"),
    ])
    root.after(220, steps.pop(0))

# ---------------- Entry point ----------------
def main():
    global root
    root = tk.Tk()
    root.withdraw()  # hide while loading

    # Progressbar styling
    s = ttk.Style()
    try: s.theme_use("clam")
    except Exception: pass
    s.configure("TProgressbar",
                troughcolor=PANEL,
                background=ACCENT,
                bordercolor=PANEL,
                lightcolor=ACCENT,
                darkcolor=ACCENT)

    splash = Splash(
        root,
        title="Seeder",
        version="By Abisha, Arya, Jess, William",
        img_left=str(Path(__file__).resolve().parent / "Anh.png"),
        img_right=str(Path(__file__).resolve().parent / "louis.png"),
    )
    boot_sequence(splash)
    root.mainloop()

if __name__ == "__main__":
    main()
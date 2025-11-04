#!/usr/bin/env python3
import os, sys, time, threading, xml.etree.ElementTree as ET
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# --- Camera deps ---
from sensor_msgs.msg import Image, CompressedImage
try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None
from PIL import Image as PILImage
from PIL import ImageTk
import numpy as np
import cv2

# ---------- Colors (Dark-Green theme) ----------
BG        = "#0b120e"   # near-black green
PANEL     = "#0f1a14"   # deep panel green
FG        = "#e7f2ea"   # soft off-white
SUBFG     = "#a9c4b2"   # muted mint
BTN_BG    = "#17261f"   # button base
ACCENT    = "#2b604a"   # progress/active green
EDGE      = "#1e3a2f"   # outlines
RED       = "#cc2e2e"
RED_DARK  = "#9e1f1f"
POSE_OUTLINE          = "#2a4136"
POSE_SELECTED_OUTLINE = "#ffffff"

# ---------- Camera config (UPDATED) ----------
IMAGE_TOPIC       = "/camera/image"                # matches your RViz panel
COMPRESSED_TOPIC  = "/camera/image/compressed"
CAM_W, CAM_H      = 240, 180

# ---------- World/Grid mapping ----------
METERS_PER_CELL = 0.5
DOT_RADIUS_PX   = 5
PINE_KEYWORDS   = ["pine", "pinetree", "pine_tree", "pine-tree"]

# Exactly 5 pines in your world (x, y, z, r, p, yaw)
SDF_POSES = {
    "pine1": (-2.5,  1.10, 0.0, 0.0, 0.0, 0.1),
    "pine2": (-2.5, -1.05, 0.0, 0.0, 0.0, 0.3),
    "pine3": (-2.5, -3.25, 0.0, 0.0, 0.0, 0.2),
    "pine4": (-2.5, -5.40, 0.0, 0.0, 0.0, 0.0),
    "pine5": (-2.5, -7.60, 0.0, 0.0, 0.0, 0.5),
}
ROW1_NAMES = ["pine1", "pine2", "pine3", "pine4", "pine5"]

# ---------- Optional ROS bridge (durable /seeder/estop) ----------
try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
    from std_msgs.msg import Bool
    HAVE_ROS = True
except Exception as e:
    HAVE_ROS = False
    print("[GUI] ROS 2 not available:", e)

class RosEstopBridge:
    def __init__(self):
        self.node = None
        self.pub  = None
        self.executor = None
        self.bridge = CvBridge() if CvBridge else None
        self.latest_bgr = None  # last camera frame (BGR numpy)

        if not HAVE_ROS:
            return

        rclpy.init(args=None)
        self.node = rclpy.create_node("seeder_gui_estop")

        # durable E-STOP publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.node.create_publisher(Bool, "/seeder/estop", qos)

        # ---- Camera subs (UPDATED QoS + auto-topic pick) ----
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # sim cameras are usually best-effort
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        img_topic, cmp_topic = self._pick_camera_topics()
        img_topic = img_topic or IMAGE_TOPIC
        cmp_topic = cmp_topic or COMPRESSED_TOPIC

        self.node.create_subscription(Image,            img_topic, self._on_image,      cam_qos)
        self.node.create_subscription(CompressedImage,  cmp_topic, self._on_compressed, cam_qos)
        self.node.get_logger().info(f"[GUI] Camera subscribe -> raw='{img_topic}', compressed='{cmp_topic}'")

        from rclpy.executors import SingleThreadedExecutor
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        print("[GUI] /seeder/estop publisher + camera subscribers running")

    def _pick_camera_topics(self):
        """Try to find Image/CompressedImage topics; fall back to constants."""
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

    def _on_image(self, msg: Image):
        try:
            if self.bridge:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                if msg.encoding.lower() != "rgb8":
                    return
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.latest_bgr = cv_img
        except Exception:
            pass

    def _on_compressed(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                self.latest_bgr = cv_img
        except Exception:
            pass

    def publish(self, val: bool):
        if HAVE_ROS and self.pub:
            self.pub.publish(Bool(data=val))

    def shutdown(self):
        if not HAVE_ROS:
            return
        try:
            if self.executor:
                self.executor.shutdown()
            if self.node:
                self.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

# ---------- SDF color helpers ----------
def rgba_to_hex(r, g, b):
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def looks_like_pine(name: str) -> bool:
    return any(k in (name or "").lower() for k in PINE_KEYWORDS)

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

# ---------- Splash / Loading ----------
class Splash:
    def __init__(self, root, title="Seeder", version=""):
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.configure(bg=BG)
        self.top.attributes("-topmost", True)
        self._start_ms = int(time.time() * 1000)
        self._min_show_ms = 2400  # keep splash visible a bit longer

        # Bigger card
        frame = tk.Frame(self.top, bg=PANEL, bd=0, highlightthickness=0)
        frame.pack(padx=28, pady=28)

        # Bigger tree icon (triangles + trunk)
        logo = tk.Canvas(frame, width=130, height=130, bg=PANEL, highlightthickness=0)
        logo.grid(row=0, column=0, pady=(12, 10))
        cx = 65  # center x
        trunk_w = 18
        logo.create_rectangle(cx - trunk_w/2, 94, cx + trunk_w/2, 120,
                              fill="#5c3b22", outline="#3d2818", width=2)
        logo.create_polygon(20, 94, 110, 94, cx, 50, fill=ACCENT, outline=EDGE, width=2)
        logo.create_polygon(28, 78, 102, 78, cx, 42, fill=ACCENT, outline=EDGE, width=2)
        logo.create_polygon(36, 64,  94, 64, cx, 28, fill=ACCENT, outline=EDGE, width=2)

        # Title / version
        self.title_lbl = tk.Label(frame, text=title, fg=FG, bg=PANEL, font=("Segoe UI", 16, "bold"))
        self.title_lbl.grid(row=1, column=0, pady=(2,2))
        self.ver_lbl = tk.Label(frame, text=version, fg=SUBFG, bg=PANEL, font=("Segoe UI", 10))
        self.ver_lbl.grid(row=2, column=0, pady=(0,12))

        # Status + progress (cleaned text)
        self.status = tk.Label(frame, text="Starting&", fg=FG, bg=PANEL, font=("Segoe UI", 11))
        self.status.grid(row=3, column=0, sticky="w", padx=6, pady=(0,8))
        self.pbar = ttk.Progressbar(frame, mode="determinate", length=360, maximum=100)
        self.pbar.grid(row=4, column=0, padx=6, pady=(0,14))

        self._center()

    def _center(self):
        self.top.update_idletasks()
        w = self.top.winfo_width()
        h = self.top.winfo_height()
        sw = self.top.winfo_screenwidth()
        sh = self.top.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2.5)
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

# ---------- Globals initialised during build ----------
root = None
style = None
left = None
grid_canvas = None
origin_px = origin_py = None
pixels_per_meter = None
SCRIPT_DIR = Path(__file__).resolve().parent
SDF_PATH = (SCRIPT_DIR / "seeder.sdf") if (SCRIPT_DIR / "seeder.sdf").exists() else None
ALL_DOTS, ROW_MAP, PREV_SELECTED_ID = [], {1: []}, None
selected = None
combo = None
feedback = None
move_btn = None
load_btn = None
estop_canvas = None
circle = None
label = None
resume_btn = None
ROS = None
estopped = False
CACHED_COLORS = {}  # filled during splash

# Camera widget globals
cam_label = None
_cam_imgtk = None  # keep a reference to avoid GC

# ---------- Coordinate helper ----------
def world_to_canvas_xy(x, y):
    return origin_px + x*pixels_per_meter, origin_py - y*pixels_per_meter

# ---------- UI construction split into functions ----------
def draw_checkerboard():
    grid_canvas.delete("all")
    h = int(grid_canvas['height']); w = int(grid_canvas['width'])
    for i in range(0, h, 12):
        for j in range(0, w, 12):
            color = "#122018" if ((i//12 + j//12) % 2 == 0) else "#16271f"
            grid_canvas.create_rectangle(j, i, j+12, i+12, fill=color, outline="#1c2f26")

def highlight_one(row, idx1):
    global PREV_SELECTED_ID
    if PREV_SELECTED_ID:
        grid_canvas.itemconfig(PREV_SELECTED_ID, width=1, outline=POSE_OUTLINE)
    ids = ROW_MAP.get(row, [])
    if 0 <= idx1-1 < len(ids):
        cid = ids[idx1-1]
        grid_canvas.itemconfig(cid, width=3, outline=POSE_SELECTED_OUTLINE)
        PREV_SELECTED_ID = cid

def draw_poses_on_grid():
    global ALL_DOTS, ROW_MAP
    grid_canvas.delete("pose_dot")
    ALL_DOTS, ROW_MAP = [], {1: []}
    colors = CACHED_COLORS or (extract_tree_colors_from_sdf(str(SDF_PATH)) if SDF_PATH else {})
    for name in ROW1_NAMES:
        (x, y, *_)= SDF_POSES[name]
        cx, cy = world_to_canvas_xy(x, y)
        fill = colors.get(name, "#4caf50")
        cid = grid_canvas.create_oval(cx-DOT_RADIUS_PX, cy-DOT_RADIUS_PX,
                                      cx+DOT_RADIUS_PX, cy+DOT_RADIUS_PX,
                                      fill=fill, outline=POSE_OUTLINE, width=1, tags=("pose_dot",))
        ROW_MAP[1].append(cid)
    combo.configure(values=[f"Row 1 tree {i+1}" for i in range(len(ROW_MAP[1]))])
    selected.set("Row 1 tree 1")
    feedback.config(text=f"Loaded {len(ROW_MAP[1])} trees")

def move_seeder():
    feedback.config(text=f"Moving to {combo.get()}")
    try:
        r = int(combo.get().split()[1]); t = int(combo.get().split()[3])
        highlight_one(r, t)
    except Exception:
        pass

def emergency_stop(event=None):
    global estopped
    if estopped: return
    estopped = True
    feedback.config(text="EMERGENCY STOP  Husky halted")
    move_btn.state(["disabled"]); load_btn.state(["disabled"]); combo.configure(state="disabled")
    estop_canvas.itemconfig(circle, fill=RED_DARK)
    if ROS: ROS.publish(True)
    resume_btn.state(["!disabled"])

def resume_from_estop():
    global estopped
    if not estopped: return
    estopped = False
    feedback.config(text="RESUME  Husky allowed to move")
    move_btn.state(["!disabled"]); load_btn.state(["!disabled"]); combo.configure(state=["readonly"])
    estop_canvas.itemconfig(circle, fill=RED)
    if ROS: ROS.publish(False)
    resume_btn.state(["disabled"])

def _on_close():
    if ROS: ROS.shutdown()
    root.destroy()

def _tick_camera():
    """Pull latest frame from ROS bridge and draw into cam_label."""
    global _cam_imgtk
    try:
        if ROS and ROS.latest_bgr is not None and cam_label is not None:
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
            # placeholder (UPDATED)
            cam_label.config(image="", text="No camera", fg=SUBFG, bg=PANEL, width=CAM_W, height=CAM_H)
            cam_label.place(x=0, y=0)
    except Exception:
        pass
    if root:
        root.after(50, _tick_camera)  # ~20 FPS

def build_main_ui():
    global style, left, grid_canvas, origin_px, origin_py, pixels_per_meter
    global selected, combo, feedback, move_btn, load_btn, estop_canvas, circle, label, resume_btn
    global cam_label

    root.title("Seeder")
    root.configure(bg=BG)

    style = ttk.Style()
    try: style.theme_use("clam")
    except Exception: pass
    style.configure("Dark.TFrame", background=PANEL)
    style.configure("Dark.TLabel", background=PANEL, foreground=FG)
    style.configure("Dark.TButton", background=BTN_BG, foreground=FG, padding=8)
    style.map("Dark.TButton", background=[("active", ACCENT)])

    left = ttk.Frame(root, style="Dark.TFrame", padding=12); left.pack(side="left", fill="y")

    canvas_w = 500; canvas_h = 500
    grid_canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg=BG, highlightthickness=0)
    grid_canvas.pack(side="right", padx=10, pady=10)

    draw_checkerboard()

    origin_px = canvas_w/2; origin_py = canvas_h/2
    global pixels_per_meter
    pixels_per_meter = 12 / METERS_PER_CELL

    # Controls
    selected = tk.StringVar(value="Row 1 tree 1")
    combo = ttk.Combobox(left, textvariable=selected, values=[selected.get()], state="readonly", width=22)
    combo.grid(row=0, column=0, columnspan=2, sticky="w")

    feedback = ttk.Label(left, text="Select a tree or press E-STOP", style="Dark.TLabel")
    feedback.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 10))

    # --- Camera panel (left corner) ---
    cam_frame = tk.Frame(left, bg=PANEL, width=CAM_W, height=CAM_H)
    cam_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0,8))
    cam_frame.grid_propagate(False)
    cam_bg = tk.Canvas(cam_frame, width=CAM_W, height=CAM_H, bg=BG, highlightthickness=0)
    cam_bg.pack()
    cam_label = tk.Label(cam_frame, bg=PANEL)
    cam_label.place(x=0, y=0)

    # Buttons (shifted down one row)
    move_btn = ttk.Button(left, text="Move", style="Dark.TButton", command=move_seeder)
    load_btn = ttk.Button(left, text="Load Poses", style="Dark.TButton", command=draw_poses_on_grid)
    move_btn.grid(row=3, column=0, sticky="w", pady=(0,8))
    load_btn.grid(row=3, column=1, sticky="w", pady=(0,8))

    # E-STOP & Resume (shifted)
    estop_canvas = tk.Canvas(left, width=110, height=110, bg=PANEL, highlightthickness=0)
    estop_canvas.grid(row=4, column=0, columnspan=2, pady=(6,0))
    circle = estop_canvas.create_oval(8,8,102,102, fill=RED, outline=RED_DARK, width=4)
    label  = estop_canvas.create_text(55,55, text="E-STOP", fill="white", font=("Segoe UI", 14, "bold"))
    for tag in (circle, label):
        estop_canvas.tag_bind(tag, "<Button-1>", emergency_stop)

    resume_btn = ttk.Button(left, text="Resume", style="Dark.TButton", command=resume_from_estop)
    resume_btn.grid(row=5, column=0, sticky="w", pady=(10,0))
    resume_btn.state(["disabled"])

    close_btn = ttk.Button(left, text="Close", style="Dark.TButton", command=_on_close)
    close_btn.grid(row=5, column=1, sticky="e", pady=(10,0))

    # start camera pump
    root.after(100, _tick_camera)

def boot_sequence(splash: Splash):
    """Non-blocking startup with small delays; splash stays visible longer."""
    steps = []

    def step(percent, msg, fn=None):
        def _do():
            splash.update(percent, msg)
            if fn:
                try:
                    fn()
                except Exception as e:
                    print(f"[Startup] {msg} -> {e}")
            if steps:
                root.after(450, steps.pop(0))  # slower cadence for a beefier splash
            else:
                def show_main():
                    root.deiconify()
                    root.after(0, draw_poses_on_grid)
                splash.update(100, "Ready")
                splash.close(on_closed=show_main)
        return _do

    def init_ros():
        global ROS
        ROS = RosEstopBridge()

    def parse_sdf():
        global CACHED_COLORS
        if SDF_PATH and SDF_PATH.exists():
            CACHED_COLORS = extract_tree_colors_from_sdf(str(SDF_PATH))
        else:
            CACHED_COLORS = {}

    def build_ui():
        build_main_ui()

    steps.extend([
        step(5,   "Starting Seeder&"),
        step(30,  "Initialising ROS 2 bridge&" if HAVE_ROS else "Skipping ROS 2 (not available)&", init_ros if HAVE_ROS else None),
        step(60,  "Parsing SDF for tree colours&" if SDF_PATH else "No seeder.sdf found, using defaults&", parse_sdf if SDF_PATH else None),
        step(85,  "Building widgets&", build_ui),
        step(95,  "Drawing scene&"),
    ])
    root.after(220, steps.pop(0))

# ---------- Main entry ----------
def main():
    global root
    root = tk.Tk()
    root.withdraw()  # hide while loading

    # Progressbar styling for dark-green theme
    s = ttk.Style()
    try: s.theme_use("clam")
    except Exception: pass
    s.configure("TProgressbar",
                troughcolor=PANEL,
                background=ACCENT,
                bordercolor=PANEL,
                lightcolor=ACCENT,
                darkcolor=ACCENT)

    splash = Splash(root, title="Seeder", version="By Abisha, Arya, Jess, William")
    boot_sequence(splash)
    root.mainloop()

if __name__ == "__main__":
    main()

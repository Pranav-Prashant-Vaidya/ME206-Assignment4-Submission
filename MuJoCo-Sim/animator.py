"""
animator.py
Pose the robot yourself with the viewer's sliders, save keyframes, and play them
back as a smooth physics-based animation.

    python main.py animate                 (or: python animator.py)

Scene: the G1 with five-finger hands (pelvis fixed), a table in front of it and
an apple on the table. Nothing moves unless you move it.

How an animation is formed here
  * A KEYFRAME is a snapshot of all 65 actuator targets (data.ctrl): the arm, leg,
    waist and finger joint angles you set with the sliders.
  * Each keyframe has a DURATION: the time the robot takes to move from the previous
    keyframe into this one.
  * PLAY interpolates the targets between consecutive keyframes with a minimum-jerk
    curve (smooth start and stop) and feeds them to the position servos every 2 ms.
    MuJoCo then simulates the motion, so the fingers really touch, push and lift
    the apple, and gravity really acts on it.

You type commands in the VS Code terminal while the viewer window is open
(see HELP below).
"""

import json
import queue
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

import config
from controllers import HAND_PRESETS, min_jerk

ANIM_XML = config.ROBOT_DIR / "animation_scene.xml"
ANIM_DIR = config.PROJECT_DIR / "animations"

TABLE_CENTER = (0.40, 0.0)      # x, y of the table centre (robot faces +x)
TABLE_TOP_Z = 0.75              # m
TABLE_HALF = (0.18, 0.35)       # half length (x), half width (y)
APPLE_ON_TABLE = (0.30, -0.12)  # in front of the right hand

HELP = """
COMMANDS (type in this terminal, press Enter):
  k [sec]          add keyframe from the current pose (sec = time to reach it, default 1.0)
  ins N [sec]      insert keyframe at position N
  del [N]          delete keyframe N (default: last)
  dur N sec        change duration of keyframe N
  list             list keyframes
  goto N           move the robot to keyframe N (sliders jump there too)
  play             play the animation once from the start (scene is reset first)
  loop             play it forever          stop   stop playing
  reset            put the robot and the apple back to the start
  hand left|right open|pregrasp|cup|grasp      set all 18 finger sliders at once
  set NAME VALUE   set one actuator by (part of) its name, e.g. set right_elbow 1.2
  pose             print the current arm targets
  throw [h] [s]    the hand holding the apple turns palm-up and tosses it:
                   h = height above the hand in m (default 0.25),
                   s = sideways distance towards the other hand in m (default 0.25).
                   A green marker shows where it comes down. Example: throw 0.3 0.2
  rewind           go back to the moment before the last throw (apple in hand)
  ready left|right move that hand palm-up under the green marker
  speed X          slow motion, e.g. speed 0.25 (quarter speed); speed 1 = normal
  save [name]      save to animations/<name>.json (default: my_animation)
  load [name]      load animations/<name>.json
  help             show this list           quit   close everything
SLIDERS: right-hand panel of the viewer -> "Control". One slider per actuator.
"""


# --------------------------------------------------------------- scene
def build_animation_scene():
    """Copy the generated toss scene, swap the apple stand for a table."""
    root = ET.parse(str(config.SCENE_XML)).getroot() if config.SCENE_XML.exists() else None
    size = None
    if root is not None:
        g = next((g for g in root.iter("geom") if g.get("name") == "apple_collision"), None)
        size = float(g.get("size")) if g is not None else None
    if size is None or abs(size - config.APPLE_RADIUS) > 1e-6:
        print("apple size changed in config.py -> rebuilding the model")
        import build_model
        build_model.build()
        root = ET.parse(str(config.SCENE_XML)).getroot()
    wb = root.find("worldbody")
    for body in list(wb.findall("body")):
        if body.get("name") == "apple_stand":
            wb.remove(body)
    asset = root.find("asset")
    asset.append(ET.Element("material", name="table_wood", rgba="0.55 0.38 0.24 1",
                            specular="0.2", shininess="0.3"))
    cx, cy = TABLE_CENTER
    hx, hy = TABLE_HALF
    table = ET.SubElement(wb, "body", name="table", pos=f"{cx} {cy} 0")
    ET.SubElement(table, "geom", name="table_top", type="box",
                  size=f"{hx} {hy} 0.02", pos=f"0 0 {TABLE_TOP_Z - 0.02}",
                  material="table_wood", friction="1 0.01 0.002")
    leg_h = (TABLE_TOP_Z - 0.04) / 2
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ET.SubElement(table, "geom", name=f"table_leg{i}", type="box",
                      size=f"0.02 0.02 {leg_h:.4f}",
                      pos=f"{sx * (hx - 0.03)} {sy * (hy - 0.03)} {leg_h:.4f}",
                      material="table_wood")
    marker = ET.SubElement(wb, "body", name="landing_marker", mocap="true", pos="0 0 -1")
    ET.SubElement(marker, "geom", type="cylinder", size="0.035 0.0015",
                  rgba="0.1 0.9 0.2 0.55", contype="0", conaffinity="0")
    ET.SubElement(marker, "geom", type="sphere", size="0.006",
                  rgba="0.1 0.9 0.2 0.9", contype="0", conaffinity="0")
    for body in wb.iter("body"):
        if body.get("name") == "apple":
            ax, ay = APPLE_ON_TABLE
            body.set("pos", f"{ax} {ay} {TABLE_TOP_Z + config.APPLE_RADIUS + 0.0005:.4f}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(ANIM_XML), encoding="utf-8")
    return ANIM_XML


# --------------------------------------------------------------- animator
class Animator:
    def __init__(self):
        xml = build_animation_scene()
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.pose = json.loads(config.POSE_JSON.read_text(encoding="utf-8"))
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(self.m.nu)]
        self.lo = self.m.actuator_ctrlrange[:, 0]
        self.hi = self.m.actuator_ctrlrange[:, 1]
        self.keyframes = []          # list of {"ctrl": [...], "duration": s}
        self.playing = False
        self.looping = False
        self.play_t0 = 0.0
        self.play_start_ctrl = None
        self.commands = queue.Queue()
        self.running = True
        from apple import AppleTracker
        self.apple = AppleTracker(self.m, self.d)
        mid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "landing_marker")
        self.marker_id = int(self.m.body_mocapid[mid])
        self.marker_pos = None
        self.action = None            # running ThrowAction / ReadyAction
        self.pre_throw = None         # snapshot for  rewind
        self.last_apple_local = None
        self.speed = 1.0
        self.resync = False
        self.reset()

    # ---------------------------------------------------- state
    def stand_ctrl(self):
        ctrl = np.zeros(self.m.nu)
        for i in range(self.m.nu):
            if self.m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                j = int(self.m.actuator_trnid[i, 0])
                jn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
                ctrl[i] = self.pose["joints"].get(jn, 0.0)
        return np.clip(ctrl, self.lo, self.hi)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for jn, v in self.pose["joints"].items():
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self.d.qpos[self.m.jnt_qposadr[jid]] = v
        self.d.ctrl[:] = self.stand_ctrl()
        if self.marker_pos is not None:
            self.d.mocap_pos[self.marker_id] = self.marker_pos
        mujoco.mj_forward(self.m, self.d)

    def set_marker(self, pos):
        self.marker_pos = np.array(pos, float)
        self.d.mocap_pos[self.marker_id] = self.marker_pos

    # ---------------------------------------------------- playback
    def total_time(self):
        return sum(k["duration"] for k in self.keyframes)

    def ctrl_at(self, t):
        """Actuator targets at time t into the animation."""
        prev = self.play_start_ctrl
        for k in self.keyframes:
            target = np.array(k["ctrl"])
            if t <= k["duration"]:
                return prev + min_jerk(t / max(k["duration"], 1e-3)) * (target - prev)
            t -= k["duration"]
            prev = target
        return prev

    def start_play(self, loop):
        if not self.keyframes:
            print("no keyframes yet: pose the robot and type  k")
            return
        self.reset()
        self.play_start_ctrl = self.d.ctrl.copy()
        self.play_t0 = self.d.time
        self.playing, self.looping = True, loop
        print(f"playing {len(self.keyframes)} keyframes, {self.total_time():.1f} s"
              + (" (looping)" if loop else ""))

    def step_playback(self):
        t = self.d.time - self.play_t0
        if t > self.total_time() + 0.5:
            if self.looping:
                self.start_play(True)
                return
            self.playing = False
            print("animation finished (sliders are live again)")
            return
        self.d.ctrl[:] = np.clip(self.ctrl_at(t), self.lo, self.hi)

    # ---------------------------------------------------- commands
    def find_actuators(self, text):
        return [i for i, n in enumerate(self.names) if text.lower() in n.lower()]

    def handle(self, line):
        parts = line.strip().split()
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd == "k":
                dur = float(args[0]) if args else 1.0
                self.keyframes.append({"ctrl": self.d.ctrl.tolist(), "duration": dur})
                print(f"keyframe {len(self.keyframes) - 1} added ({dur:.2f} s)")
            elif cmd == "ins":
                n, dur = int(args[0]), float(args[1]) if len(args) > 1 else 1.0
                self.keyframes.insert(n, {"ctrl": self.d.ctrl.tolist(), "duration": dur})
                print(f"keyframe inserted at {n}")
            elif cmd == "del":
                n = int(args[0]) if args else len(self.keyframes) - 1
                self.keyframes.pop(n)
                print(f"keyframe {n} deleted, {len(self.keyframes)} left")
            elif cmd == "dur":
                self.keyframes[int(args[0])]["duration"] = float(args[1])
                print("duration changed")
            elif cmd == "list":
                t = 0.0
                for i, k in enumerate(self.keyframes):
                    t += k["duration"]
                    print(f"  {i:2d}: reach in {k['duration']:.2f} s  (at {t:.2f} s)")
                if not self.keyframes:
                    print("  (empty)")
            elif cmd == "goto":
                self.playing = False
                self.d.ctrl[:] = self.keyframes[int(args[0])]["ctrl"]
                print(f"moving to keyframe {args[0]}")
            elif cmd in ("play", "loop"):
                self.start_play(cmd == "loop")
            elif cmd == "stop":
                self.playing, self.action = False, None
                print("stopped")
            elif cmd == "reset":
                self.playing, self.action = False, None
                self.reset()
                print("scene reset")
            elif cmd == "hand":
                side, preset = args[0].lower(), args[1].lower()
                prefix = "lh_A_" if side.startswith("l") else "rh_A_"
                for key, val in HAND_PRESETS[preset].items():
                    i = self.names.index(prefix + key) if prefix + key in self.names else -1
                    if i >= 0:
                        self.d.ctrl[i] = np.clip(val, self.lo[i], self.hi[i])
                print(f"{side} hand -> {preset}")
            elif cmd == "set":
                ids = self.find_actuators(args[0])
                if len(ids) != 1:
                    print("matches:", [self.names[i] for i in ids] or "none")
                    return
                i = ids[0]
                self.d.ctrl[i] = np.clip(float(args[1]), self.lo[i], self.hi[i])
                print(f"{self.names[i]} = {self.d.ctrl[i]:.3f}  "
                      f"(range {self.lo[i]:.2f} .. {self.hi[i]:.2f})")
            elif cmd == "pose":
                for i, n in enumerate(self.names):
                    if "shoulder" in n or "elbow" in n or "wrist" in n:
                        print(f"  {n:28s} {self.d.ctrl[i]:+.3f}")
            elif cmd == "throw":
                height = float(args[0]) if args else 0.25
                sideways = float(args[1]) if len(args) > 1 else 0.25
                c = self.apple.contacts()
                if c["left"] == 0 and c["right"] == 0:
                    print("neither hand is touching the apple: grab it first")
                    return
                side = "right" if c["right"] >= c["left"] else "left"
                self.playing = False
                self.pre_throw = (self.d.qpos.copy(), self.d.qvel.copy(), self.d.ctrl.copy())
                from throw_action import ThrowAction
                self.action = ThrowAction(self, side, height, sideways)
            elif cmd == "rewind":
                if self.pre_throw is None:
                    print("nothing to rewind yet")
                    return
                self.action, self.playing = None, False
                self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = self.pre_throw
                mujoco.mj_forward(self.m, self.d)
                print("rewound to just before the throw (marker kept). Position the other "
                      "hand, then  throw  again with the same numbers.")
            elif cmd == "ready":
                if self.marker_pos is None or self.last_apple_local is None:
                    print("throw once first so there is a landing marker")
                    return
                side = "left" if args and args[0].lower().startswith("l") else "right"
                loc = self.last_apple_local
                mirrored = np.array([-loc[0], loc[1], loc[2]])
                from throw_action import ReadyAction
                self.action = ReadyAction(self, side, self.marker_pos, mirrored)
            elif cmd == "speed":
                self.speed = min(max(float(args[0]), 0.05), 2.0)
                print(f"playback speed x{self.speed}")
            elif cmd == "save":
                name = args[0] if args else "my_animation"
                ANIM_DIR.mkdir(exist_ok=True)
                path = ANIM_DIR / f"{name}.json"
                path.write_text(json.dumps({"actuators": self.names,
                                            "keyframes": self.keyframes}, indent=1),
                                encoding="utf-8")
                print(f"saved {len(self.keyframes)} keyframes to {path}")
            elif cmd == "load":
                name = args[0] if args else "my_animation"
                data = json.loads((ANIM_DIR / f"{name}.json").read_text(encoding="utf-8"))
                if data["actuators"] != self.names:
                    print("!! actuator list differs from this model; loading by name")
                    idx = {n: i for i, n in enumerate(data["actuators"])}
                    for k in data["keyframes"]:
                        k["ctrl"] = [k["ctrl"][idx[n]] if n in idx else 0.0 for n in self.names]
                self.keyframes = data["keyframes"]
                print(f"loaded {len(self.keyframes)} keyframes")
            elif cmd == "help":
                print(HELP)
            elif cmd == "quit":
                self.running = False
            else:
                print("unknown command; type  help")
        except (IndexError, ValueError, KeyError, OSError) as exc:
            print(f"could not do that: {exc}  (type  help)")

    def input_thread(self):
        while self.running:
            try:
                line = input("anim> ")
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            self.commands.put(line)

    # ---------------------------------------------------- main loop
    def run(self):
        print(HELP)
        threading.Thread(target=self.input_thread, daemon=True).start()
        config.apply_fast_graphics(self.m)
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            config.apply_fast_graphics(self.m, viewer)
            viewer.cam.lookat[:] = (0.25, 0.0, 0.9)
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 160
            viewer.cam.elevation = -18
            wall0, sim0 = time.perf_counter(), self.d.time
            while viewer.is_running() and self.running:
                frame_start = time.perf_counter()
                with viewer.lock():
                    while not self.commands.empty():
                        self.handle(self.commands.get())
                        wall0, sim0 = time.perf_counter(), self.d.time
                    target = (frame_start - wall0) * self.speed
                    steps = 0
                    while self.d.time - sim0 < target and steps < 40:
                        if self.playing:
                            self.step_playback()
                        if self.action is not None:
                            self.action.step(self.d.time)
                            if self.action.done:
                                self.action = None
                        mujoco.mj_step(self.m, self.d)
                        steps += 1
                        if self.resync:
                            break
                    if steps == 40 or self.resync:
                        wall0, sim0 = time.perf_counter(), self.d.time
                        self.resync = False
                viewer.sync()
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - frame_start)))
        self.running = False
        print("viewer closed. Remember to  save  next time before closing!")


if __name__ == "__main__":
    Animator().run()

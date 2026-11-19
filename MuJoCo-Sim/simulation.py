"""
simulation.py
Loads the generated scene, puts the robot in its standing pose, and runs the
physics loop either in the MuJoCo viewer (real time) or headless (for checks).
Later steps plug controllers into Simulation.run_viewer(controller=...).
"""

import json
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

import config

OBJ = mujoco.mjtObj


class Simulation:
    def __init__(self, xml_path=config.SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.pose = json.loads(config.POSE_JSON.read_text(encoding="utf-8"))
        m = self.model
        self.actuator_names = [mujoco.mj_id2name(m, OBJ.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.stand_ctrl = self._stand_ctrl()
        from apple import AppleTracker
        self.apple = AppleTracker(self.model, self.data)

    # ------------------------------------------------------------ lookups
    def joint_qpos_adr(self, name):
        jid = mujoco.mj_name2id(self.model, OBJ.mjOBJ_JOINT, name)
        return -1 if jid < 0 else int(self.model.jnt_qposadr[jid])

    def actuator_id(self, name):
        return mujoco.mj_name2id(self.model, OBJ.mjOBJ_ACTUATOR, name)

    def hand_actuators(self, side):
        """Indices of the Shadow Hand actuators for side 'left' or 'right'."""
        prefix = "lh_" if side == "left" else "rh_"
        return [i for i, n in enumerate(self.actuator_names) if n and n.startswith(prefix)]

    def clip_ctrl(self, i, value):
        if self.model.actuator_ctrllimited[i]:
            lo, hi = self.model.actuator_ctrlrange[i]
            return float(np.clip(value, lo, hi))
        return float(value)

    def _stand_ctrl(self):
        m = self.model
        ctrl = np.zeros(m.nu)
        for i in range(m.nu):
            target = 0.0
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                jname = mujoco.mj_id2name(m, OBJ.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
                target = self.pose["joints"].get(jname, 0.0)
            ctrl[i] = self.clip_ctrl(i, target)
        return ctrl

    # ------------------------------------------------------------ state
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for name, value in self.pose["joints"].items():
            adr = self.joint_qpos_adr(name)
            if adr >= 0:
                self.data.qpos[adr] = value
        self.data.ctrl[:] = self.stand_ctrl
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ viewer
    def _setup_camera(self, viewer):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = config.CAM_LOOKAT
        viewer.cam.distance = config.CAM_DISTANCE
        viewer.cam.azimuth = config.CAM_AZIMUTH
        viewer.cam.elevation = config.CAM_ELEVATION

    def run_viewer(self, controller=None, max_steps_per_frame=40):
        """Real-time loop. controller(sim, t) is called before every physics step."""
        self.reset()
        m, d = self.model, self.data
        frame_dt = 1.0 / config.VIEW_FPS
        config.apply_fast_graphics(m)
        with mujoco.viewer.launch_passive(m, d) as viewer:
            config.apply_fast_graphics(m, viewer)
            self._setup_camera(viewer)
            wall0, sim0 = time.perf_counter(), d.time
            while viewer.is_running():
                frame_start = time.perf_counter()
                target = frame_start - wall0
                steps = 0
                while d.time - sim0 < target and steps < max_steps_per_frame:
                    if controller is not None:
                        controller(self, d.time)
                    mujoco.mj_step(m, d)
                    steps += 1
                if steps == max_steps_per_frame:      # PC too slow: drop time, never jump
                    wall0 = time.perf_counter() - (d.time - sim0)
                viewer.sync()
                time.sleep(max(0.0, frame_dt - (time.perf_counter() - frame_start)))

    # ------------------------------------------------------------ headless run
    def run_headless(self, controller, seconds):
        """Run a controller without a window (fast); returns wall-clock seconds."""
        self.reset()
        m, d = self.model, self.data
        t0 = time.perf_counter()
        while d.time < seconds:
            controller(self, d.time)
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                print(f"!! simulation became unstable (NaN) at t={d.time:.3f}s")
                break
        return time.perf_counter() - t0

    # ------------------------------------------------------------ apple test
    def run_apple_test(self, drop_height=0.20, seconds=2.0):
        """Drop the apple onto its stand and report bounce, drift and settling."""
        from apple import apple_rest_position
        self.reset()
        m, d, a = self.model, self.data, self.apple
        rest = np.array(apple_rest_position())
        r = config.APPLE_RADIUS

        # 1) resting check: does it stay put from the start?
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        drift = np.linalg.norm(a.pos - rest) * 1000
        sink = (config.STAND_TOP_Z + r - a.pos[2]) * 1000
        print(f"resting: drift {drift:.2f} mm, sinks into stand {sink:.2f} mm, "
              f"contacts {a.contacts()}")

        # 2) drop test
        a.set_free_state(rest + (0, 0, drop_height))
        mujoco.mj_forward(m, d)
        landed, peak_after, settle_t, t0 = False, 0.0, None, d.time
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
            z, speed = a.pos[2], np.linalg.norm(a.vel)
            if not landed and a.contacts()["stand"] > 0:
                landed = True
                land_t = d.time - t0
            elif landed:
                peak_after = max(peak_after, z - rest[2])
                if settle_t is None and speed < 0.005 and a.contacts()["stand"] > 0:
                    settle_t = d.time - t0
        fall_expected = math.sqrt(2 * drop_height / 9.81)
        if landed:
            print(f"drop {drop_height * 100:.0f} cm: landed after {land_t:.3f} s "
                  f"(free-fall theory {fall_expected:.3f} s)")
        else:
            print("!! apple never touched the stand")
        print(f"bounce height: {peak_after * 1000:.1f} mm, settled: "
              f"{'%.2f s' % settle_t if settle_t else 'no'}, final offset from start "
              f"{np.linalg.norm(a.pos[:2] - rest[:2]) * 1000:.1f} mm, on stand: "
              f"{a.contacts()['stand'] > 0}")

    # ------------------------------------------------------------ headless check
    def run_check(self, seconds=3.0, controller=None):
        self.reset()
        m, d = self.model, self.data
        n = int(seconds / m.opt.timestep)
        t0 = time.perf_counter()
        for _ in range(n):
            if controller is not None:
                controller(self, d.time)
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                print(f"!! simulation became unstable (NaN) at t={d.time:.3f}s")
                return False
        wall = time.perf_counter() - t0
        print(f"simulated {seconds:.1f}s in {wall:.2f}s wall  ->  {seconds / wall:.1f}x real time")

        # joint tracking of the arms
        worst = (0.0, "")
        for i in range(m.nu):
            if m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = int(m.actuator_trnid[i, 0])
            err = abs(d.qpos[m.jnt_qposadr[jid]] - d.ctrl[i])
            if err > worst[0]:
                worst = (err, mujoco.mj_id2name(m, OBJ.mjOBJ_JOINT, jid))
        print(f"largest joint tracking error: {worst[0]:.4f} rad ({worst[1]})")

        # contacts / penetration
        deepest = (0.0, "")
        for c in d.contact[:d.ncon]:
            if c.dist < deepest[0]:
                g1n = mujoco.mj_id2name(m, OBJ.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
                g2n = mujoco.mj_id2name(m, OBJ.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
                b1 = mujoco.mj_id2name(m, OBJ.mjOBJ_BODY, int(m.geom_bodyid[c.geom1]))
                b2 = mujoco.mj_id2name(m, OBJ.mjOBJ_BODY, int(m.geom_bodyid[c.geom2]))
                deepest = (c.dist, f"{g1n}[{b1}] <-> {g2n}[{b2}]")
        print(f"contacts: {d.ncon}, deepest penetration: {-deepest[0] * 1000:.2f} mm {deepest[1]}")

        for i, w in enumerate(d.warning):
            if w.number:
                print(f"MuJoCo warning {i}: {w.number} times")

        # hand orientation sanity check
        for side, prefix in (("left", "lh_"), ("right", "rh_")):
            bid = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, prefix + "palm")
            if bid < 0:
                print(f"!! {prefix}palm missing")
                continue
            R = d.xmat[bid].reshape(3, 3)
            normal = R @ np.array(config.SHADOW_PALM_NORMAL)
            fingers = R @ np.array(config.SHADOW_FINGER_AXIS)
            print(f"{side:5s} palm at {np.round(d.xpos[bid], 3)}, fingers point "
                  f"{np.round(fingers, 2)}, palm faces {np.round(normal, 2)}")
        return True

"""
toss.py
The full behaviour as a state machine, used as a controller by Simulation:

  settle -> reach (right hand above apple, palm down) -> descend -> close (5-finger grasp)
  -> lift -> carry (rotate palm up to hold pose; left hand gets ready, palm up)
  -> cup (fingers relax into a bowl, apple rests in the palm)
  -> [throw -> flight -> catch -> stabilise -> recentre -> swap hands] repeated forever

Physics, not animation:
  * The apple is never moved by code. It leaves the throwing palm on its own when the
    hand decelerates, flies ballistically under MuJoCo gravity, and is stopped only
    by contact with the other hand's fingers and palm.
  * Throw planning: before the first throw in each direction, candidate throws are
    simulated in a copy of the MjData and the one whose real flight lands closest
    to the catching hand is kept (cached in robot_model/throw_calibration.json).
  * Catch: after release the catching hand predicts the landing point from the
    apple's measured position/velocity and re-solves its IK every 10 ms to meet it.
"""

import json
import math

import mujoco
import numpy as np

import config
from controllers import ArmController, ArmIK, HandController, min_jerk, palm_prefix

G = 9.81
CAL_FILE = config.ROBOT_DIR / "throw_calibration.json"


def mirror(v):
    return np.array([v[0], -v[1], v[2]])


def other(side):
    return "left" if side == "right" else "right"


class TossSequence:
    def __init__(self, sim, verbose=True):
        self.sim = sim
        self.m, self.d = sim.model, sim.data
        self.dt = self.m.opt.timestep
        self.apple = sim.apple
        self.verbose = verbose
        self.ik = {s: ArmIK(self.m, s) for s in ("left", "right")}
        self.arm = {s: ArmController(self.m, s) for s in ("left", "right")}
        self.hand = {s: HandController(self.m, s) for s in ("left", "right")}
        tilt = math.radians(config.CUP_TILT_DEG)
        self.palm_up = (np.array([math.cos(tilt), 0, -math.sin(tilt)]),
                        np.array([math.sin(tilt), 0, math.cos(tilt)]))
        self.palm_down = (np.array([1.0, 0, 0]), np.array([0, 0, -1.0]))
        self.hold = {"right": np.array(config.HOLD_POS_RIGHT, float),
                     "left": mirror(config.HOLD_POS_RIGHT)}
        self.stats = {"grasps": 0, "throws": 0, "catches": 0, "drops": 0, "resets": 0}
        self.cal = self._load_cal()
        self.started = False

    # ------------------------------------------------------------ utilities
    def log(self, t, msg):
        if self.verbose:
            print(f"[t={t:6.2f}s] {msg}")

    def _fingerprint(self):
        return [config.HAND_SCALE, list(config.HOLD_POS_RIGHT), config.APPLE_RADIUS,
                config.APPLE_MASS, config.CUP_TILT_DEG, list(config.THROW_DURATIONS),
                list(config.THROW_LATERAL), list(config.THROW_UP), config.THROW_FORWARD]

    def _load_cal(self):
        try:
            data = json.loads(CAL_FILE.read_text(encoding="utf-8"))
            if data.get("fingerprint") == self._fingerprint():
                return data
        except (OSError, ValueError):
            pass
        return {"fingerprint": self._fingerprint()}

    def _save_cal(self):
        try:
            CAL_FILE.write_text(json.dumps(self.cal, indent=2), encoding="utf-8")
        except OSError:
            pass

    def palm_local(self, side, world_point):
        b = self.ik[side].palm
        R = self.d.xmat[b].reshape(3, 3)
        return R.T @ (np.asarray(world_point) - self.d.xpos[b])

    def solve_fast(self, side, point_local, pos, orient):
        """Single warm-started IK, cheap enough to run every 10 ms while catching."""
        q, _, _ = self.ik[side].solve(self.d.qpos.copy(), point_local, pos, orient[0],
                                      orient[1], self.arm[side].q_cmd, iters=40)
        return q

    def solve(self, side, point_local, pos, orient, extra_seeds=()):
        seeds = [self.arm[side].q_cmd, *self.ik[side].seed_postures, *extra_seeds]
        q, ep, er = self.ik[side].solve_best(self.d.qpos.copy(), point_local, pos,
                                             orient[0], orient[1], seeds)
        return q, ep, er

    def report_ik(self, name, ep, er):
        flag = "" if ep < 0.01 and er < 0.15 else "   <-- not fully reachable"
        self.log(self.d.time, f"IK {name:22s} pos err {ep * 1000:6.1f} mm, "
                              f"rot err {math.degrees(er):5.1f} deg{flag}")

    def ballistic_intercept(self, p, v, z_catch):
        """Time and point where a free projectile from (p, v) descends through z_catch."""
        a, b, c = -0.5 * G, v[2], p[2] - z_catch
        disc = b * b - 4 * a * c
        if disc < 0:
            return None, None
        t = (-b - math.sqrt(disc)) / (2 * a)       # later (descending) root
        if t <= 0:
            return None, None
        return t, np.array([p[0] + v[0] * t, p[1] + v[1] * t, z_catch])

    # ------------------------------------------------------------ start / reset
    def start(self, t):
        for s in ("left", "right"):
            self.arm[s].sync_from(self.d)
            self.hand[s].sync_from(self.d)
        mujoco.mj_forward(self.m, self.d)
        # reference point in each palm: the Shadow model's own grasp_site
        self.grasp_local = {}
        for s in ("left", "right"):
            sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, palm_prefix(s) + "grasp_site")
            if sid >= 0:
                self.grasp_local[s] = self.palm_local(s, self.d.site_xpos[sid])
            else:
                self.grasp_local[s] = np.array([0.0, -0.03, 0.06]) * config.HAND_SCALE
        self.apple_local = dict(self.grasp_local)   # refined after the apple is cupped
        a0 = self.apple.pos
        self.log(t, f"apple at {np.round(a0, 3)}; planning pick with right hand")
        above = a0 + (0, 0, config.PICK_ABOVE)
        grasp = a0 + (0, 0, config.GRASP_OFFSET_Z)
        gl = self.grasp_local["right"]
        self.q_above, ep, er = self.solve("right", gl, above, self.palm_down)
        self.report_ik("right above apple", ep, er)
        self.q_grasp, ep, er = self.solve("right", gl, grasp, self.palm_down, [self.q_above])
        self.report_ik("right grasp", ep, er)
        self.phase, self.phase_end = "settle", t + 0.5
        self.started = True

    def reset_scene(self, t, why):  # noqa: D401  (returns None; phase_failed does the reset)
        self.stats["drops"] += 1
        self.stats["resets"] += 1
        self.log(t, f"DROP ({why}); resetting the scene in 1.5 s")
        self.phase, self.phase_end = "failed", t + 1.5

    # ------------------------------------------------------------ throw planning
    def calibrate(self, thrower, t):
        catcher = other(thrower)
        key = f"{thrower}_to_{catcher}"
        if key in self.cal:
            return self.cal[key]
        self.log(t, f"planning {key} throw: simulating "
                    f"{len(config.THROW_DURATIONS) * len(config.THROW_LATERAL) * len(config.THROW_UP)}"
                    " candidates (one-time, cached) ...")
        m = self.m
        base = mujoco.MjData(m)
        mujoco.mj_copyData(base, m, self.d)
        roll = mujoco.MjData(m)
        q0 = self.arm[thrower].q_cmd.copy()
        aids = self.arm[thrower].aids
        inward = 1.0 if thrower == "right" else -1.0
        target = self.hold[catcher]
        best = None
        for T in config.THROW_DURATIONS:
            for lat in config.THROW_LATERAL:
                for up in config.THROW_UP:
                    end = self.hold[thrower] + (config.THROW_FORWARD, inward * lat, up)
                    q1, ep, _ = self.solve(thrower, self.apple_local[thrower], end, self.palm_up)
                    if ep > 0.015:
                        continue
                    mujoco.mj_copyData(roll, m, base)
                    free_steps, result = 0, None
                    for k in range(int((T + 0.25) / self.dt)):
                        roll.ctrl[aids] = q0 + min_jerk(k * self.dt / T) * (q1 - q0)
                        mujoco.mj_step(m, roll)
                        if self.apple.contacts(roll)[thrower] == 0 and k * self.dt > 0.03:
                            free_steps += 1
                            if free_steps >= 5:
                                p, v = self.apple.pos_in(roll), self.apple.vel_in(roll)
                                tf, hit = self.ballistic_intercept(p, v, target[2])
                                if hit is not None:
                                    err = np.linalg.norm(hit[:2] - target[:2])
                                    apex = p[2] + v[2] ** 2 / (2 * G) - target[2]
                                    result = (err, T, lat, up, apex, tf + k * self.dt)
                                break
                        else:
                            free_steps = 0
                    if result and (best is None or result[0] < best[0]):
                        best = result
        if best is None:
            self.log(t, "!! no candidate throw released the apple; using middle values")
            best = (1.0, config.THROW_DURATIONS[1], config.THROW_LATERAL[0], config.THROW_UP[0], 0, 0)
        cal = {"error": best[0], "T": best[1], "lat": best[2], "up": best[3],
               "apex": best[4], "flight": best[5]}
        self.log(t, f"best {key}: stroke {cal['T']:.2f}s, lateral {cal['lat']:.2f} m, up "
                    f"{cal['up']:.2f} m -> predicted miss {cal['error'] * 1000:.0f} mm, "
                    f"apex {cal['apex'] * 100:.0f} cm above hand")
        self.cal[key] = cal
        self._save_cal()
        return cal

    # ------------------------------------------------------------ main controller
    def __call__(self, sim, t):
        if not self.started or t < 1e-9:
            self.start(t)
        handler = getattr(self, "phase_" + self.phase)
        if handler(t) == "reset":
            return                      # scene was reset: leave the fresh stand ctrl alone
        for s in ("left", "right"):
            self.arm[s].apply(self.d, t, self.dt)
            self.hand[s].apply(self.d, t)

    # ---- pick ------------------------------------------------------------
    def phase_settle(self, t):
        if t >= self.phase_end:
            self.arm["right"].move_to(self.q_above, t, 1.3)
            self.hand["right"].set("pregrasp", t, 0.8)
            self.phase, self.phase_end = "reach", t + 1.4

    def phase_reach(self, t):
        if t >= self.phase_end:
            self.arm["right"].move_to(self.q_grasp, t, 0.9)
            self.phase, self.phase_end = "descend", t + 1.0

    def phase_descend(self, t):
        if t >= self.phase_end:
            self.hand["right"].set("grasp", t, 0.7)
            self.phase, self.phase_end = "close", t + 0.9

    def phase_close(self, t):
        if t >= self.phase_end:
            c = self.apple.contacts()
            self.log(t, f"grasp: {c['right']} hand contacts on the apple")
            self.stats["grasps"] += 1
            self.arm["right"].move_to(self.q_above, t, 0.9)
            self.phase, self.phase_end = "lift", t + 1.0

    def phase_lift(self, t):
        if t >= self.phase_end:
            if self.apple.contacts()["right"] == 0:
                return self.reset_scene(t, "apple not lifted")
            q, ep, er = self.solve("right", self.grasp_local["right"], self.hold["right"],
                                   self.palm_up)
            self.report_ik("right hold (palm up)", ep, er)
            self.arm["right"].move_to(q, t, 1.5)
            ql, ep, er = self.solve("left", self.grasp_local["left"], self.hold["left"],
                                    self.palm_up)
            self.report_ik("left hold (palm up)", ep, er)
            self.arm["left"].move_to(ql, t, 1.5)
            self.hand["left"].set("cup", t, 1.0)
            self.phase, self.phase_end = "carry", t + 1.6

    def phase_carry(self, t):
        if t >= self.phase_end:
            self.hand["right"].set("cup", t, 0.6)
            self.phase, self.phase_end = "cup", t + 1.0

    def phase_cup(self, t):
        if t >= self.phase_end:
            self.begin_throw_cycle("right", t)

    # ---- toss cycle ---------------------------------------------------------
    def begin_throw_cycle(self, thrower, t):
        c = self.apple.contacts()
        if c[thrower] == 0:
            return self.reset_scene(t, f"apple not resting in {thrower} hand")
        # measure where the apple actually sits in this palm; mirror it for the other hand
        loc = self.palm_local(thrower, self.apple.pos)
        self.apple_local[thrower] = loc
        self.apple_local[other(thrower)] = np.array([-loc[0], loc[1], loc[2]])
        # put the catching hand exactly at its hold pose for the measured apple offset
        catcher = other(thrower)
        q, ep, er = self.solve(catcher, self.apple_local[catcher], self.hold[catcher], self.palm_up)
        self.arm[catcher].move_to(q, t, 0.4)
        self.hand[catcher].set("cup", t, 0.3)
        self.thrower, self.catcher = thrower, catcher
        self.phase, self.phase_end = "aim", t + 0.45

    def phase_aim(self, t):
        if t < self.phase_end:
            return
        cal = self.calibrate(self.thrower, t)
        inward = 1.0 if self.thrower == "right" else -1.0
        end = self.hold[self.thrower] + (config.THROW_FORWARD, inward * cal["lat"], cal["up"])
        q1, _, _ = self.solve(self.thrower, self.apple_local[self.thrower], end, self.palm_up)
        self.arm[self.thrower].move_to(q1, t, cal["T"])
        self.stats["throws"] += 1
        self.free_steps = 0
        self.throw_t0 = t
        self.log(t, f"THROW {self.thrower} -> {self.catcher}")
        self.phase, self.phase_end = "throw", t + cal["T"] + 0.35

    def phase_throw(self, t):
        if self.apple.contacts()[self.thrower] == 0 and t - self.throw_t0 > 0.03:
            self.free_steps += 1
        else:
            self.free_steps = 0
        if self.free_steps >= 5:
            v = self.apple.vel
            self.log(t, f"release: speed {np.linalg.norm(v):.2f} m/s, "
                        f"vel {np.round(v, 2)}")
            q_back, _, _ = self.solve(self.thrower, self.apple_local[self.thrower],
                                      self.hold[self.thrower], self.palm_up)
            self.arm[self.thrower].move_to(q_back, t, 0.7)
            self.flight_count, self.first_prediction = 0, None
            self.catch_goal = self.hold[self.catcher].copy()
            self.phase, self.phase_end = "flight", t + 2.0
        elif t >= self.phase_end:
            self.reset_scene(t, "apple never left the throwing hand")

    def phase_flight(self, t):
        p, v = self.apple.pos, self.apple.vel
        if self.apple.contacts()[self.catcher] > 0:
            self.on_catch_contact(t)
            return
        if p[2] < config.DROP_HEIGHT or t >= self.phase_end:
            return self.reset_scene(t, "missed the catch")
        self.flight_count += 1
        if self.flight_count % 5 != 1:                 # re-plan every 10 ms
            return
        tf, hit = self.ballistic_intercept(p, v, self.hold[self.catcher][2])
        if hit is None:
            return
        offset = hit - self.hold[self.catcher]
        n = np.linalg.norm(offset[:2])
        if n > config.CATCH_MAX_CORRECTION:
            offset[:2] *= config.CATCH_MAX_CORRECTION / n
        goal = self.hold[self.catcher] + offset
        if self.first_prediction is None:
            self.first_prediction = goal
            self.log(t, f"predicted landing {np.round(hit, 3)} "
                        f"({n * 1000:.0f} mm from the waiting hand), {tf:.2f} s to go")
        q = self.solve_fast(self.catcher, self.apple_local[self.catcher], goal, self.palm_up)
        self.arm[self.catcher].track(q, config.TRACK_TAU)
        self.catch_goal = goal

    def on_catch_contact(self, t):
        self.log(t, f"contact with {self.catcher} hand at speed {np.linalg.norm(self.apple.vel):.2f} m/s")
        self.hand[self.catcher].set("grasp", t, 0.15)
        goal = getattr(self, "catch_goal", self.hold[self.catcher]) - (0, 0, config.CATCH_CUSHION)
        q = self.solve_fast(self.catcher, self.apple_local[self.catcher], goal, self.palm_up)
        self.arm[self.catcher].move_to(q, t, 0.2)
        self.phase, self.phase_end = "stabilise", t + 0.7

    def phase_stabilise(self, t):
        if self.apple.pos[2] < config.DROP_HEIGHT:
            return self.reset_scene(t, "apple fell out after contact")
        if t >= self.phase_end:
            c = self.apple.contacts()[self.catcher]
            speed = np.linalg.norm(self.apple.vel)
            if c == 0:
                return self.reset_scene(t, "apple bounced out of the hand")
            self.stats["catches"] += 1
            self.log(t, f"CATCH OK by {self.catcher} hand ({c} contacts, apple speed "
                        f"{speed:.3f} m/s)   [throws {self.stats['throws']}, "
                        f"catches {self.stats['catches']}]")
            q, _, _ = self.solve(self.catcher, self.apple_local[self.catcher],
                                 self.hold[self.catcher], self.palm_up)
            self.arm[self.catcher].move_to(q, t, 0.6)
            self.hand[self.catcher].set("cup", t, 0.5)
            self.phase, self.phase_end = "recentre", t + 0.9

    def phase_recentre(self, t):
        if t >= self.phase_end:
            self.begin_throw_cycle(self.catcher, t)

    def phase_failed(self, t):
        if t >= self.phase_end:
            self.sim.reset()             # full scene reset: apple back on its stand
            self.started = False
            self.log(0.0, "scene reset, starting again")
            return "reset"

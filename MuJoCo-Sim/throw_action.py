"""
throw_action.py
Robot-driven actions used by the animator (animator.py):

  ThrowAction  the hand that is holding the apple turns palm-up, relaxes its fingers
               into a cup, and makes a quick upward stroke. The apple leaves the palm
               on its own when the hand decelerates (nothing pushes the apple except
               the hand). The stroke is chosen by simulating candidate strokes in a
               copy of the physics, and the predicted landing point is shown by a
               green marker.
  ReadyAction  moves a hand palm-up so its palm sits under the green marker.
"""

import math

import mujoco
import numpy as np

from controllers import ArmController, ArmIK, HandController, min_jerk

G = 9.81
TILT = math.radians(8)
PALM_UP = (np.array([math.cos(TILT), 0.0, -math.sin(TILT)]),     # fingers forward
           np.array([math.sin(TILT), 0.0, math.cos(TILT)]))      # palm facing up

# candidate strokes the planner tries (duration s, upward travel m, sideways travel m)
STROKE_TIMES = (0.15, 0.20, 0.25)
STROKE_UP = (0.08, 0.12, 0.17, 0.23)
STROKE_SIDE = (0.0, 0.04, 0.08, 0.12)


def other(side):
    return "left" if side == "right" else "right"


def ballistic_landing(p, v, z):
    """Time and point where a projectile from (p, v) comes down through height z."""
    a, b, c = -0.5 * G, v[2], p[2] - z
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    t = (-b - math.sqrt(disc)) / (2 * a)
    if t <= 0:
        return None, None
    return t, np.array([p[0] + v[0] * t, p[1] + v[1] * t, z])


class ThrowAction:
    def __init__(self, anim, side, height, sideways):
        self.a, self.m, self.d = anim, anim.m, anim.d
        self.side, self.height, self.sideways = side, height, sideways
        self.inward = 1.0 if side == "right" else -1.0      # towards the other hand
        self.ik = ArmIK(self.m, side)
        self.arm = ArmController(self.m, side)
        self.hand = HandController(self.m, side)
        self.arm.sync_from(self.d)
        self.hand.sync_from(self.d)
        self.done = False
        t = self.d.time

        # reference point = where the apple is, expressed in the palm frame
        self.apple_local = self.palm_local(anim.apple.pos)
        target = anim.apple.pos + np.array([0.0, 0.0, 0.10])
        q, ep, er = self.ik.solve_best(self.d.qpos.copy(), self.apple_local, target,
                                       *PALM_UP, [self.arm.q_cmd, *self.ik.seed_postures])
        print(f"[throw] {side} hand: turning palm up "
              f"(IK error {ep * 1000:.0f} mm, {math.degrees(er):.0f} deg)")
        self.arm.move_to(q, t, 1.2)
        self.phase, self.phase_end = "turn", t + 1.4

    # ------------------------------------------------------------ helpers
    def palm_local(self, world_point):
        R = self.d.xmat[self.ik.palm].reshape(3, 3)
        return R.T @ (np.asarray(world_point) - self.d.xpos[self.ik.palm])

    def finish(self, msg):
        print(f"[throw] {msg}")
        self.done = True

    # ------------------------------------------------------------ planning
    def plan(self, t):
        m, apple = self.m, self.a.apple
        print(f"[throw] planning: simulating up to "
              f"{len(STROKE_TIMES) * len(STROKE_UP) * len(STROKE_SIDE)} candidate strokes ...")
        base = mujoco.MjData(m)
        mujoco.mj_copyData(base, m, self.d)
        roll = mujoco.MjData(m)
        q0 = self.arm.q_cmd.copy()
        hold = apple.pos
        best = None
        for T in STROKE_TIMES:
            for up in STROKE_UP:
                for lat in STROKE_SIDE:
                    end = hold + np.array([0.0, self.inward * lat, up])
                    q1, ep, _ = self.ik.solve(base.qpos.copy(), self.apple_local, end,
                                              *PALM_UP, q0, iters=80)
                    if ep > 0.015:
                        continue
                    mujoco.mj_copyData(roll, m, base)
                    free = 0
                    for k in range(int((T + 0.2) / m.opt.timestep)):
                        roll.ctrl[self.arm.aids] = q0 + min_jerk(k * m.opt.timestep / T) * (q1 - q0)
                        mujoco.mj_step(m, roll)
                        if apple.contacts(roll)[self.side] == 0 and k * m.opt.timestep > 0.03:
                            free += 1
                            if free >= 5:
                                p, v = apple.pos_in(roll), apple.vel_in(roll)
                                apex = p[2] + max(v[2], 0) ** 2 / (2 * G) - hold[2]
                                _, hit = ballistic_landing(p, v, hold[2])
                                if hit is not None:
                                    side_dist = self.inward * (hit[1] - hold[1])
                                    score = abs(apex - self.height) + abs(side_dist - self.sideways)
                                    if best is None or score < best[0]:
                                        best = (score, T, q1, apex, side_dist)
                                break
                        else:
                            free = 0
        self.a.resync = True                      # planning took wall time; re-sync the clock
        if best is None:
            return self.finish("no stroke made the apple leave the palm. Try a lower height, "
                               "or check the apple is resting in the palm.")
        _, T, q1, apex, side_dist = best
        print(f"[throw] chosen stroke {T:.2f} s -> apex {apex * 100:.0f} cm "
              f"(asked {self.height * 100:.0f}), lands {side_dist * 100:.0f} cm "
              f"sideways (asked {self.sideways * 100:.0f})")
        self.q_hold, self.hold = q0, hold
        self.arm.move_to(q1, t, T)
        self.stroke_t0, self.free = t, 0
        self.phase, self.phase_end = "stroke", t + T + 0.4

    # ------------------------------------------------------------ per physics step
    def step(self, t):
        apple = self.a.apple
        if self.phase == "turn" and t >= self.phase_end:
            self.hand.set("cup", t, 0.5)
            self.phase, self.phase_end = "cup", t + 0.9
        elif self.phase == "cup" and t >= self.phase_end:
            if apple.contacts()[self.side] == 0:
                return self.finish("the apple fell out while the palm turned up. Grip it "
                                   "more firmly (hand right grasp) and try again.")
            self.apple_local = self.palm_local(apple.pos)   # where it really rests now
            self.a.last_apple_local = self.apple_local
            self.plan(t)
            if self.done:
                return
        elif self.phase == "stroke":
            if apple.contacts()[self.side] == 0 and t - self.stroke_t0 > 0.03:
                self.free += 1
            else:
                self.free = 0
            if self.free >= 5:
                p, v = apple.pos, apple.vel
                tf, hit = ballistic_landing(p, v, self.hold[2])
                print(f"[throw] released at {np.linalg.norm(v):.2f} m/s")
                if hit is not None:
                    self.a.set_marker(hit)
                    print(f"[throw] it will come down through hand height at "
                          f"x={hit[0]:.3f} y={hit[1]:.3f} z={hit[2]:.3f} in {tf:.2f} s "
                          f"(green marker)")
                self.arm.move_to(self.q_hold, t, 0.6)
                self.max_z = p[2]
                self.phase, self.phase_end = "flight", t + 3.0
            elif t >= self.phase_end:
                self.arm.move_to(self.q_hold, t, 0.5)
                return self.finish("the apple never left the palm.")
        elif self.phase == "flight":
            p = apple.pos
            self.max_z = max(self.max_z, p[2])
            c = apple.contacts()
            if apple.vel[2] < 0 and sum(c.values()) > 0:
                where = max(c, key=c.get)
                names = {other(self.side): f"{other(self.side)} hand (caught?)",
                         self.side: f"{self.side} hand (it came back)",
                         "floor": "floor", "other": "table", "stand": "stand"}
                return self.finish(f"apex was {(self.max_z - self.hold[2]) * 100:.0f} cm "
                                   f"above the hand; it touched the {names[where]}")
            if t >= self.phase_end:
                return self.finish("flight finished")
        self.arm.apply(self.d, t, self.m.opt.timestep)
        self.hand.apply(self.d, t)


class ReadyAction:
    """Move a hand palm-up so the apple would sit on it at the marker."""

    def __init__(self, anim, side, target, apple_local):
        self.m, self.d = anim.m, anim.d
        ik = ArmIK(self.m, side)
        self.arm = ArmController(self.m, side)
        self.hand = HandController(self.m, side)
        self.arm.sync_from(self.d)
        self.hand.sync_from(self.d)
        q, ep, er = ik.solve_best(self.d.qpos.copy(), apple_local, target, *PALM_UP,
                                  [self.arm.q_cmd, *ik.seed_postures])
        flag = "" if ep < 0.01 else "  <-- cannot fully reach; use the sliders to finish"
        print(f"[ready] {side} hand to marker: IK error {ep * 1000:.0f} mm{flag}")
        t = self.d.time
        self.arm.move_to(q, t, 1.0)
        self.hand.set("cup", t, 0.6)
        self.end = t + 1.1
        self.done = False

    def step(self, t):
        self.arm.apply(self.d, t, self.m.opt.timestep)
        self.hand.apply(self.d, t)
        if t >= self.end:
            print("[ready] done")
            self.done = True

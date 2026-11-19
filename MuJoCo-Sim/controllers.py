"""
controllers.py
Low-level control building blocks.

  min_jerk()        smooth 0->1 profile (zero velocity and acceleration at both ends)
  ArmIK             inverse kinematics for one 7-DoF G1 arm using MuJoCo Jacobians
                    (damped least squares + posture regularisation + joint limits)
  ArmController     sends smooth joint-space trajectories to the G1 position servos,
                    or tracks a moving goal (used while catching)
  HandController    blends the 18 Shadow Hand actuators between grasp presets
"""

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")

# Shadow Hand frame (confirmed by `python main.py check`): fingers +z, palm faces -y
PALM_FINGER_AXIS = np.array([0.0, 0.0, 1.0])
PALM_NORMAL = np.array([0.0, -1.0, 0.0])

OBJ = mujoco.mjtObj


def min_jerk(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def target_rotation(finger_dir, palm_normal):
    """World rotation of the palm body so its fingers/normal point as requested."""
    def frame(a, n):
        a = np.asarray(a, float) / np.linalg.norm(a)
        n = np.asarray(n, float) - np.dot(n, a) * a
        n /= np.linalg.norm(n)
        return np.column_stack([a, n, np.cross(a, n)])
    return frame(finger_dir, palm_normal) @ frame(PALM_FINGER_AXIS, PALM_NORMAL).T


def palm_prefix(side):
    return "lh_" if side == "left" else "rh_"


# ============================================================== inverse kinematics
class ArmIK:
    def __init__(self, model, side):
        self.m = model
        self.d = mujoco.MjData(model)               # private scratch data
        self.side = side
        self.jids = []
        for j in ARM_JOINTS:
            jid = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT, f"{side}_{j}_joint")
            if jid < 0:
                raise ValueError(f"joint {side}_{j}_joint not found")
            self.jids.append(jid)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids]) + 0.02
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids]) - 0.02
        self.palm = mujoco.mj_name2id(model, OBJ.mjOBJ_BODY, palm_prefix(side) + "palm")
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        # neutral "elbow bent, hand in front" posture used to resolve redundancy
        self.nominal = np.clip(np.array([-0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), self.lo, self.hi)
        # extra starting guesses: forearm rotated both ways (palm-up / palm-down branches)
        self.seed_postures = [self.nominal]
        for roll in (-1.5, 1.5):
            s = self.nominal.copy()
            s[4] = roll
            self.seed_postures.append(np.clip(s, self.lo, self.hi))

    def point_world(self, d, point_local):
        R = d.xmat[self.palm].reshape(3, 3)
        return d.xpos[self.palm] + R @ point_local

    def solve(self, base_qpos, point_local, target_pos, finger_dir, palm_normal,
              q_init, iters=120, rot_weight=0.35, tol_pos=5e-4, tol_rot=0.01):
        """Return (q, position_error_m, rotation_error_rad)."""
        m, d = self.m, self.d
        d.qpos[:] = base_qpos
        q = np.clip(np.asarray(q_init, float), self.lo, self.hi)
        d.qpos[self.qadr] = q
        R_t = target_rotation(finger_dir, palm_normal)
        target_pos = np.asarray(target_pos, float)
        lam2, mu2 = 1e-4, 2e-3
        ep = er = 1.0
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.palm].reshape(3, 3)
            p = d.xpos[self.palm] + R @ point_local
            e_pos = target_pos - p
            e_rot = 0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))
            ep, er = np.linalg.norm(e_pos), np.linalg.norm(e_rot)
            if ep < tol_pos and er < tol_rot:
                break
            mujoco.mj_jac(m, d, self.jacp, self.jacr, p, self.palm)
            J = np.vstack([self.jacp[:, self.dadr], rot_weight * self.jacr[:, self.dadr]])
            e = np.concatenate([e_pos, rot_weight * e_rot])
            A = J.T @ J + (lam2 + mu2) * np.eye(7)
            b = J.T @ e + mu2 * (self.nominal - q)
            dq = np.linalg.solve(A, b)
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            q = np.clip(q + dq, self.lo, self.hi)
            d.qpos[self.qadr] = q
        return q, ep, er

    def solve_best(self, base_qpos, point_local, target_pos, finger_dir, palm_normal, seeds):
        best = None
        for s in seeds:
            res = self.solve(base_qpos, point_local, target_pos, finger_dir, palm_normal, s)
            score = res[1] + 0.05 * res[2]
            if best is None or score < best[0]:
                best = (score, res)
        return best[1]


# ============================================================== arm controller
class ArmController:
    """Writes joint targets to the G1 arm position actuators (actuator name == joint name)."""

    def __init__(self, model, side):
        self.m = model
        self.aids = []
        for j in ARM_JOINTS:
            aid = mujoco.mj_name2id(model, OBJ.mjOBJ_ACTUATOR, f"{side}_{j}_joint")
            if aid < 0:
                raise ValueError(f"actuator {side}_{j}_joint not found")
            self.aids.append(aid)
        self.aids = np.array(self.aids)
        self.lo = model.actuator_ctrlrange[self.aids, 0]
        self.hi = model.actuator_ctrlrange[self.aids, 1]
        self.q_cmd = np.zeros(7)
        self.seg = None          # (q_start, q_goal, t0, duration)
        self.track_goal = None
        self.track_tau = 0.05

    def sync_from(self, data):
        self.q_cmd = data.ctrl[self.aids].copy()
        self.seg, self.track_goal = None, None

    def move_to(self, q_goal, t, duration):
        self.seg = (self.q_cmd.copy(), np.asarray(q_goal, float).copy(), t, max(duration, 1e-3))
        self.track_goal = None

    def track(self, q_goal, tau=0.05):
        self.seg = None
        self.track_goal = np.asarray(q_goal, float).copy()
        self.track_tau = tau

    def busy(self, t):
        return self.seg is not None and t < self.seg[2] + self.seg[3]

    def apply(self, data, t, dt):
        if self.seg is not None:
            q0, q1, t0, T = self.seg
            self.q_cmd = q0 + min_jerk((t - t0) / T) * (q1 - q0)
        elif self.track_goal is not None:
            alpha = min(1.0, dt / self.track_tau)
            step = np.clip(alpha * (self.track_goal - self.q_cmd), -0.03, 0.03)  # <= 15 rad/s
            self.q_cmd = self.q_cmd + step
        data.ctrl[self.aids] = np.clip(self.q_cmd, self.lo, self.hi)


# ============================================================== hand controller
# Values are joint targets (rad; *J0 = summed PIP+DIP tendon). Same values work for
# both hands because Menagerie's left/right Shadow Hands share actuator ranges.
HAND_PRESETS = {
    "open":   {"THJ5": 0.0, "THJ4": 0.3, "THJ3": 0.0, "THJ2": 0.0, "THJ1": 0.0,
               "FFJ4": 0.0, "FFJ3": 0.0, "FFJ0": 0.2, "MFJ4": 0.0, "MFJ3": 0.0, "MFJ0": 0.2,
               "RFJ4": 0.0, "RFJ3": 0.0, "RFJ0": 0.2, "LFJ5": 0.0, "LFJ4": 0.0, "LFJ3": 0.0,
               "LFJ0": 0.2},
    # thumb swung under the palm, fingers straight: ready to close over the apple
    "pregrasp": {"THJ5": 0.25, "THJ4": 1.05, "THJ3": 0.0, "THJ2": 0.0, "THJ1": 0.1,
                 "FFJ4": -0.1, "FFJ3": 0.05, "FFJ0": 0.3, "MFJ4": 0.0, "MFJ3": 0.05, "MFJ0": 0.3,
                 "RFJ4": 0.1, "RFJ3": 0.05, "RFJ0": 0.3, "LFJ5": 0.2, "LFJ4": 0.2,
                 "LFJ3": 0.05, "LFJ0": 0.3},
    # bowl shape: palm up, fingers and thumb raised a little so the apple cannot roll off
    "cup":    {"THJ5": 0.35, "THJ4": 0.85, "THJ3": 0.0, "THJ2": 0.15, "THJ1": 0.35,
               "FFJ4": 0.0, "FFJ3": 0.40, "FFJ0": 0.70, "MFJ4": 0.0, "MFJ3": 0.40, "MFJ0": 0.70,
               "RFJ4": 0.0, "RFJ3": 0.45, "RFJ0": 0.75, "LFJ5": 0.40, "LFJ4": 0.0,
               "LFJ3": 0.50, "LFJ0": 0.80},
    # wrap all five fingers around the apple
    "grasp":  {"THJ5": 0.55, "THJ4": 1.15, "THJ3": 0.0, "THJ2": 0.40, "THJ1": 0.90,
               "FFJ4": 0.0, "FFJ3": 1.05, "FFJ0": 1.60, "MFJ4": 0.0, "MFJ3": 1.05, "MFJ0": 1.60,
               "RFJ4": 0.0, "RFJ3": 1.10, "RFJ0": 1.65, "LFJ5": 0.55, "LFJ4": 0.0,
               "LFJ3": 1.15, "LFJ0": 1.70},
}


class HandController:
    def __init__(self, model, side):
        self.m = model
        p = palm_prefix(side)
        self.keys, aids = [], []
        for key in HAND_PRESETS["open"]:
            aid = mujoco.mj_name2id(model, OBJ.mjOBJ_ACTUATOR, f"{p}A_{key}")
            if aid >= 0:
                self.keys.append(key)
                aids.append(aid)
        if len(aids) < 15:
            raise ValueError(f"only {len(aids)} Shadow actuators found for {side} hand")
        self.aids = np.array(aids)
        self.lo = model.actuator_ctrlrange[self.aids, 0]
        self.hi = model.actuator_ctrlrange[self.aids, 1]
        self.cmd = np.zeros(len(aids))
        self.seg = None
        self.preset = "open"

    def vector(self, name):
        return np.clip(np.array([HAND_PRESETS[name][k] for k in self.keys]), self.lo, self.hi)

    def sync_from(self, data):
        self.cmd = data.ctrl[self.aids].copy()
        self.seg = None

    def set(self, name, t, duration):
        self.seg = (self.cmd.copy(), self.vector(name), t, max(duration, 1e-3))
        self.preset = name

    def apply(self, data, t):
        if self.seg is not None:
            c0, c1, t0, T = self.seg
            self.cmd = c0 + min_jerk((t - t0) / T) * (c1 - c0)
        data.ctrl[self.aids] = self.cmd

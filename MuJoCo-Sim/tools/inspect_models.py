"""
Step 1 + 2 inspection script (v2, hardened).

Finds MuJoCo Menagerie on this machine, lists the Unitree humanoid models,
and prints the real joint / actuator / hand details we will build on.
Also checks for five-finger hand models and MjSpec support.

Run from ANY folder:
    python tools/inspect_models.py
Optional: point it at Menagerie directly
    python tools/inspect_models.py "C:/path/to/mujoco_menagerie"

A copy of the output is always saved to tools/inspect_report.txt,
even if something fails part-way.
"""

import importlib.util
import os
import site
import sys
import traceback
from pathlib import Path

REPORT_LINES = []
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def out(text=""):
    text = str(text)
    try:
        print(text)
    except UnicodeEncodeError:  # Windows consoles with a legacy code page
        print(text.encode("ascii", "replace").decode("ascii"))
    REPORT_LINES.append(text)


def save_report():
    path = SCRIPT_DIR / "inspect_report.txt"
    try:
        path.write_text("\n".join(REPORT_LINES), encoding="utf-8")
        print(f"\nReport saved to {path}")
    except OSError as exc:
        print(f"\nCould not save report ({exc}). Copy the terminal output instead.")


# --------------------------------------------------------------------------
# 0. Import MuJoCo with a helpful message if the wrong interpreter is used
# --------------------------------------------------------------------------
try:
    import mujoco
except ImportError as exc:
    out("!! Could not import mujoco: " + str(exc))
    out("   This script is running with: " + sys.executable)
    out("   In VS Code: Ctrl+Shift+P > 'Python: Select Interpreter' and pick the")
    out("   environment where you ran 'pip install mujoco', then run:")
    out("       python -m pip install mujoco")
    save_report()
    sys.exit(1)


JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
FINGER_WORDS = ("thumb", "index", "middle", "ring", "little", "pinky", "finger")
HAND_WORDS = ("hand", "palm", "wrist", "rubber")
OBJ = mujoco.mjtObj


def name_of(model, objtype, idx):
    try:
        n = mujoco.mj_id2name(model, objtype, int(idx))
    except Exception:  # noqa: BLE001
        n = None
    return n if n else f"<unnamed {idx}>"


# --------------------------------------------------------------------------
# 1. Locate Menagerie
# --------------------------------------------------------------------------
def candidate_roots():
    roots = []
    if len(sys.argv) > 1:
        roots.append(Path(sys.argv[1]).expanduser())
    if os.environ.get("MENAGERIE_PATH"):
        roots.append(Path(os.environ["MENAGERIE_PATH"]).expanduser())
    roots += [
        PROJECT_DIR / "mujoco_menagerie",
        PROJECT_DIR / "mujoco_menagerie-main",     # name used by GitHub ZIP downloads
        PROJECT_DIR.parent / "mujoco_menagerie",
        PROJECT_DIR.parent / "mujoco_menagerie-main",
        Path.cwd() / "mujoco_menagerie",
        Path.home() / "mujoco_menagerie",
        Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie",
    ]
    for name in ("mujoco_menagerie", "menagerie", "mujoco_menagerie_models"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        if spec is not None and spec.submodule_search_locations:
            roots += [Path(p) for p in spec.submodule_search_locations]
    sp_dirs = []
    for getter in ("getsitepackages", "getusersitepackages"):
        try:
            res = getattr(site, getter)()
            sp_dirs += res if isinstance(res, list) else [res]
        except Exception:  # noqa: BLE001  (missing in some venvs)
            pass
    for sp in sp_dirs:
        p = Path(sp)
        if not p.is_dir():
            continue
        for pattern in ("*/unitree_g1", "*/*/unitree_g1"):
            try:
                roots += [hit.parent for hit in p.glob(pattern)]
            except OSError:
                pass
    return roots


def find_menagerie():
    for root in candidate_roots():
        try:
            if (root / "unitree_g1").is_dir():
                return root.resolve()
            # handle a ZIP extracted one level too deep: mujoco_menagerie/mujoco_menagerie-main/
            for child in root.glob("*/unitree_g1"):
                return child.parent.resolve()
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------
# 2. Describe a model
# --------------------------------------------------------------------------
def actuator_target(m, a):
    trntype = int(m.actuator_trntype[a])
    trnid = int(m.actuator_trnid[a, 0])
    if trntype in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
        return "joint:" + name_of(m, OBJ.mjOBJ_JOINT, trnid)
    if trntype == int(mujoco.mjtTrn.mjTRN_TENDON):
        return "tendon:" + name_of(m, OBJ.mjOBJ_TENDON, trnid)
    if trntype == int(mujoco.mjtTrn.mjTRN_SITE):
        return "site:" + name_of(m, OBJ.mjOBJ_SITE, trnid)
    return f"trntype{trntype}:{trnid}"


def side_of(name):
    low = name.lower()
    if low.startswith(("left", "l_", "lh_")):
        return "left"
    if low.startswith(("right", "r_", "rh_")):
        return "right"
    return "other"


def describe(xml_path, full=True):
    out("=" * 78)
    out(f"MODEL: {xml_path}")
    try:
        m = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # noqa: BLE001
        out(f"  !! failed to load: {exc}")
        return
    try:
        out(f"  nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody} "
            f"ngeom={m.ngeom} nmesh={m.nmesh} nkey={m.nkey} ntendon={m.ntendon}")
        out(f"  timestep={m.opt.timestep} integrator={int(m.opt.integrator)} "
            f"cone={int(m.opt.cone)} iterations={m.opt.iterations} "
            f"total_mass={mujoco.mj_getTotalmass(m):.2f} kg")
        if m.nkey:
            out("  keyframes: " + ", ".join(
                name_of(m, OBJ.mjOBJ_KEY, k) for k in range(m.nkey)))

        hand_joints = {"left": [], "right": [], "other": []}
        if full:
            out("  JOINTS (index | name | type | range):")
        for j in range(m.njnt):
            jn = name_of(m, OBJ.mjOBJ_JOINT, j)
            jt = JOINT_TYPES.get(int(m.jnt_type[j]), "?")
            if bool(m.jnt_limited[j]):
                r = m.jnt_range[j]
                rng_s = f"[{r[0]:+.3f}, {r[1]:+.3f}]"
            else:
                rng_s = "unlimited"
            if full:
                out(f"    {j:3d} {jn:40s} {jt:6s} {rng_s}")
            low = jn.lower()
            if "hand" in low or any(w in low for w in FINGER_WORDS):
                hand_joints[side_of(jn)].append(jn)

        out("  ACTUATORS (index | name | drives | ctrlrange | kp | bias):")
        for a in range(m.nu):
            an = name_of(m, OBJ.mjOBJ_ACTUATOR, a)
            target = actuator_target(m, a)
            if bool(m.actuator_ctrllimited[a]):
                cr = m.actuator_ctrlrange[a]
                cr_s = f"[{cr[0]:+.3f}, {cr[1]:+.3f}]"
            else:
                cr_s = "unlimited"
            kp = float(m.actuator_gainprm[a, 0])
            b = m.actuator_biasprm[a]
            out(f"    {a:3d} {an:36s} -> {target:42s} {cr_s} kp={kp:.1f} "
                f"bias=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f})")

        out("  HAND / FINGER JOINTS FOUND:")
        for side, js in hand_joints.items():
            if js:
                out(f"    {side:5s} ({len(js)}): {', '.join(js)}")
        bodies = [name_of(m, OBJ.mjOBJ_BODY, b) for b in range(m.nbody)]
        hand_bodies = [b for b in bodies if any(w in b.lower() for w in HAND_WORDS)]
        out("  HAND-RELATED BODIES: " + (", ".join(hand_bodies) or "none"))
    except Exception:  # noqa: BLE001
        out("  !! error while describing this model:")
        out(traceback.format_exc())


# --------------------------------------------------------------------------
def list_dir_xml(folder):
    try:
        return sorted(x.name for x in folder.glob("*.xml"))
    except OSError:
        return []


def main():
    out(f"python  : {sys.executable} ({sys.version.split()[0]})")
    out(f"platform: {sys.platform}")
    out(f"mujoco  : {getattr(mujoco, '__version__', 'unknown')}")
    has_spec = hasattr(mujoco, "MjSpec")
    out(f"MjSpec  : {'yes' if has_spec else 'NO'}"
        f" | attach: {'yes' if has_spec and hasattr(mujoco.MjSpec, 'attach') else 'no'}")
    try:
        # importlib avoids rebinding the name 'mujoco' as a local variable
        importlib.import_module("mujoco.viewer")
        out("viewer  : mujoco.viewer importable")
    except Exception as exc:  # noqa: BLE001
        out(f"viewer  : NOT importable ({exc})")

    root = find_menagerie()
    if root is None:
        out("\n!! MuJoCo Menagerie NOT FOUND. Searched:")
        for r in candidate_roots():
            out(f"     {r}")
        out("   Fix: from the project folder run")
        out("       git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git")
        out("   or pass the path:  python tools/inspect_models.py <path-to-menagerie>")
        return

    out(f"\nMenagerie found at: {root}")
    try:
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        out(f"!! cannot list Menagerie folder: {exc}")
        return

    unitree = [p for p in subdirs if "unitree" in p.name.lower()]
    out("Unitree folders: " + (", ".join(p.name for p in unitree) or "none"))
    for folder in unitree:
        out(f"  {folder.name}: {', '.join(list_dir_xml(folder))}")

    hand_keys = ("shadow", "inspire", "ability", "dex", "leap", "allegro")
    hands = [p for p in subdirs if any(k in p.name.lower() for k in hand_keys)]
    out("Candidate dexterous hands: " + (", ".join(p.name for p in hands) or "none"))
    for folder in hands:
        out(f"  {folder.name}: {', '.join(list_dir_xml(folder))}")

    g1 = root / "unitree_g1"
    for name, full in (("g1_with_hands.xml", True), ("g1.xml", False),
                       ("scene_with_hands.xml", False), ("scene.xml", False)):
        if (g1 / name).exists():
            describe(g1 / name, full=full)
        else:
            out(f"(not present: unitree_g1/{name})")

    for folder in hands:
        for x in sorted(folder.glob("*.xml")):
            if "scene" in x.name.lower():
                continue
            if "hand" in x.name.lower() or len(list_dir_xml(folder)) <= 3:
                describe(x, full=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        out("\n!! Unexpected error:")
        out(traceback.format_exc())
    finally:
        save_report()

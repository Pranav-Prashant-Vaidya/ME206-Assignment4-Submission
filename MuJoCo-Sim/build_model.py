"""
build_model.py
Generates robot_model/g1_shadow_scene.xml from your local MuJoCo Menagerie:

  1. Loads unitree_g1/g1.xml (the 29-DoF G1 without Dex3 hands).
  2. Removes the G1's static "rubber hand" geoms on each wrist.
  3. Takes the five-finger Shadow Hand (shadow_hand/left_hand.xml, right_hand.xml)
     from the palm down (the Shadow forearm and wrist joints are dropped because the
     G1 already has a 3-DoF wrist), scales it to the G1's size, and mounts it on
     left/right_wrist_yaw_link.
  4. Welds the pelvis to the world (config.FIX_PELVIS) at the G1 "stand" height.
  5. Adds a floor, sky, lights, shadows, cameras and tuned physics options.
  6. Compiles the result with MuJoCo to prove the XML is valid.

The merge is done on the XML text with ElementTree, so it does not depend on
version-specific MjSpec functions.

Run:  python build_model.py      (or: python main.py build)
"""

import copy
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import config


class BuildError(RuntimeError):
    pass


# MuJoCo built-in default values, used so the Shadow classes keep their original
# behaviour even if the G1 file sets something in its top-level default.
BUILTIN_DEFAULTS = {
    "geom": {"contype": "1", "conaffinity": "1", "condim": "3", "group": "0",
             "priority": "0", "friction": "1 0.005 0.0001", "solref": "0.02 1",
             "solimp": "0.9 0.95 0.001 0.5 2", "margin": "0", "gap": "0",
             "density": "1000", "rgba": "0.5 0.5 0.5 1", "material": None},
    "joint": {"damping": "0", "armature": "0", "frictionloss": "0", "stiffness": "0",
              "group": "0", "limited": "auto", "actuatorfrclimited": "auto"},
    "site": {"group": "0", "rgba": "0.5 0.5 0.5 1"},
}


# --------------------------------------------------------------------- helpers
def vec(text):
    return [float(x) for x in text.split()]


def fmt(values):
    return " ".join(f"{float(x):.6g}" for x in values)


def load_xml(path):
    try:
        return ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError) as exc:
        raise BuildError(f"Cannot read {path}: {exc}") from exc


def find_or_create(root, tag, index=None):
    el = root.find(tag)
    if el is None:
        el = ET.Element(tag)
        if index is None:
            root.append(el)
        else:
            root.insert(index, el)
    return el


def absolutize_files(root, xml_path):
    """Make every mesh/texture file path absolute so files from two different
    Menagerie folders can live in one generated XML."""
    comp = root.find("compiler")
    assetdir = comp.get("assetdir", "") if comp is not None else ""
    meshdir = comp.get("meshdir", assetdir) if comp is not None else ""
    texdir = comp.get("texturedir", assetdir) if comp is not None else ""
    base = Path(xml_path).resolve().parent
    missing = []
    for el in root.iter():
        if el.tag not in ("mesh", "texture", "hfield", "skin"):
            continue
        sub = texdir if el.tag == "texture" else (meshdir if el.tag in ("mesh", "skin") else assetdir)
        for attr in list(el.attrib):
            if not attr.startswith("file"):
                continue
            p = Path(el.get(attr))
            if not p.is_absolute():
                p = base / sub / p
            p = p.resolve()
            if not p.exists():
                missing.append(str(p))
            el.set(attr, p.as_posix())
    if comp is not None:
        for a in ("assetdir", "meshdir", "texturedir"):
            comp.attrib.pop(a, None)
    if missing:
        raise BuildError("Missing asset files referenced by " + str(xml_path) + ":\n  "
                         + "\n  ".join(missing[:15]))


def default_classes(root):
    """{class_name: (element, parent_name)}; the top-level default is 'main'."""
    out = {}
    top = root.find("default")
    if top is None:
        return out

    def walk(el, name, parent):
        out[name] = (el, parent)
        for ch in el.findall("default"):
            walk(ch, ch.get("class"), name)

    walk(top, top.get("class", "main"), None)
    return out


def resolve_default(classes, cls, tag, attr):
    while cls is not None and cls in classes:
        el, parent = classes[cls]
        t = el.find(tag)
        if t is not None and t.get(attr) is not None:
            return t.get(attr)
        cls = parent
    return None


def scale_element(e, s):
    """Scale lengths by s, masses by s^3 and inertias by s^5."""
    def mul(attr, k):
        v = e.get(attr)
        if v is not None:
            e.set(attr, fmt([x * k for x in vec(v)]))
    if e.tag in ("body", "geom", "site", "joint", "camera", "light", "inertial", "frame"):
        mul("pos", s)
    if e.tag in ("geom", "site"):
        mul("size", s)
        mul("fromto", s)
    if e.tag == "geom":
        mul("mass", s ** 3)
    if e.tag == "inertial":
        mul("mass", s ** 3)
        mul("diaginertia", s ** 5)
        mul("fullinertia", s ** 5)


def rad_to_deg_angles(el):
    k = 180.0 / math.pi
    for e in el.iter():
        if e.tag == "joint":
            for a in ("range", "ref", "springref"):
                if e.get(a) is not None:
                    e.set(a, fmt([x * k for x in vec(e.get(a))]))
        if e.get("euler") is not None:
            e.set("euler", fmt([x * k for x in vec(e.get("euler"))]))


def frame_rotation(a_src, n_src, a_dst, n_dst):
    """Rotation R with R @ a_src = a_dst and R @ n_src ~ n_dst."""
    def ortho(a, n):
        a = np.asarray(a, float)
        a = a / np.linalg.norm(a)
        n = np.asarray(n, float) - np.dot(n, a) * a
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            raise BuildError("Palm normal is parallel to the finger axis; check config.")
        n = n / nn
        return np.column_stack([a, n, np.cross(a, n)])
    return ortho(a_dst, n_dst) @ ortho(a_src, n_src).T


def mat2quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def look_at_xyaxes(pos, target, up=(0, 0, 1)):
    """MuJoCo camera xyaxes for a camera at pos looking at target."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return list(x) + list(y)


# ------------------------------------------------- step 1: G1 pose and wrist frames
def compute_pose_and_mounts(g1_xml):
    """Compile the original G1, put it in the 'stand' keyframe and work out where
    and how each Shadow palm must sit on the wrist."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(g1_xml))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, config.STAND_KEYFRAME)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    else:
        print(f"  ! keyframe '{config.STAND_KEYFRAME}' not found, using default pose")
    mujoco.mj_forward(m, d)

    pose = {"joints": {}, "base_body": None, "base_pos": None, "base_quat": None}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(m.jnt_qposadr[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            pose["base_body"] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))
            pose["base_pos"] = [float(v) for v in d.qpos[adr:adr + 3]]
            pose["base_quat"] = [float(v) for v in d.qpos[adr + 3:adr + 7]]
        elif m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            pose["joints"][name] = float(d.qpos[adr])

    mounts = {}
    for side, inward_y in (("left", -1.0), ("right", 1.0)):
        wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        ej = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_elbow_joint")
        if wb < 0 or ej < 0:
            raise BuildError(f"{side}_wrist_yaw_link or {side}_elbow_joint not in g1.xml")
        R = d.xmat[wb].reshape(3, 3)
        forearm = d.xpos[wb] - d.xanchor[ej]
        forearm /= np.linalg.norm(forearm)
        a_loc = R.T @ forearm                         # forearm direction, wrist frame
        n_loc = R.T @ np.array([0.0, inward_y, 0.0])  # palm faces the body when hanging
        Rm = frame_rotation(config.SHADOW_FINGER_AXIS, config.SHADOW_PALM_NORMAL, a_loc, n_loc)
        mounts[side] = {"pos": list(a_loc * config.HAND_MOUNT_OFFSET),
                        "quat": list(mat2quat(Rm))}
    return pose, mounts


# --------------------------------------------- step 2: extract one Shadow hand
def extract_shadow_hand(xml_path, side_prefix, cls_prefix, scale, to_degrees):
    root = load_xml(xml_path)
    absolutize_files(root, xml_path)
    classes = default_classes(root)
    report = {"dropped_actuators": [], "ignored_main_defaults": []}

    # assets: rename (so left/right do not collide) and bake the scale into meshes
    mesh_map, mat_map, tex_map = {}, {}, {}
    assets = [a for sec in root.findall("asset") for a in sec]
    for a in assets:
        if a.tag == "mesh":
            name = a.get("name") or Path(a.get("file", "mesh")).stem
            sc = a.get("scale") or resolve_default(classes, a.get("class", "main"), "mesh", "scale")
            base = vec(sc) if sc else [1.0, 1.0, 1.0]
            if len(base) == 1:
                base = base * 3
            a.set("scale", fmt([x * scale for x in base]))
            mesh_map[name] = cls_prefix + name
            a.set("name", mesh_map[name])
        elif a.tag in ("material", "texture") and a.get("name"):
            target = mat_map if a.tag == "material" else tex_map
            target[a.get("name")] = cls_prefix + a.get("name")
            a.set("name", target[a.get("name")])
    cls_map = {c: cls_prefix + c for c in classes if c != "main"}

    # palm subtree
    parent = {c: p for p in root.iter() for c in p}
    palm = next((b for b in root.iter("body") if b.get("name") == f"{side_prefix}palm"), None)
    if palm is None:
        raise BuildError(f"No body '{side_prefix}palm' in {xml_path}")
    childclass, node = None, palm
    while node is not None:
        if node.get("childclass"):
            childclass = node.get("childclass")
            break
        node = parent.get(node)
    palm = copy.deepcopy(palm)
    if childclass and not palm.get("childclass"):
        palm.set("childclass", childclass)
    # the G1 already has a 3-DoF wrist: drop any Shadow wrist joint (WRJ*) in the subtree
    report["removed_wrist_joints"] = []
    for body in palm.iter("body"):
        for j in list(body.findall("joint")):
            if "WRJ" in (j.get("name") or ""):
                body.remove(j)
                report["removed_wrist_joints"].append(j.get("name"))

    # default classes (children of the top-level default)
    top = root.find("default")
    defaults = []
    if top is not None:
        for ch in top:
            if ch.tag == "default":
                defaults.append(copy.deepcopy(ch))
            else:
                report["ignored_main_defaults"].append(ch.tag)

    # scale geometry
    for e in palm.iter():
        scale_element(e, scale)
    for dcls in defaults:
        for e in dcls.iter():
            if e.tag not in ("default", "mesh"):
                scale_element(e, scale)

    # make every name in the subtree unique to this hand (e.g. grasp_site)
    name_map = {}
    for e in palm.iter():
        n = e.get("name")
        if n and not n.startswith(side_prefix):
            name_map[n] = side_prefix + n
            e.set("name", name_map[n])
    joints = {j.get("name") for j in palm.iter("joint")}
    bodies = {b.get("name") for b in palm.iter("body")}

    tendons = []
    for sec in root.findall("tendon"):
        for t in sec:
            refs = [name_map.get(j.get("joint"), j.get("joint")) for j in t.iter("joint")]
            if refs and all(r in joints for r in refs):
                t = copy.deepcopy(t)
                for j in t.iter("joint"):
                    j.set("joint", name_map.get(j.get("joint"), j.get("joint")))
                tendons.append(t)
    tendon_names = {t.get("name") for t in tendons}

    actuators = []
    for sec in root.findall("actuator"):
        for a in sec:
            jn = name_map.get(a.get("joint"), a.get("joint"))
            tn = a.get("tendon")
            if (jn and jn in joints) or (tn and tn in tendon_names):
                a = copy.deepcopy(a)
                if jn:
                    a.set("joint", jn)
                actuators.append(a)
            else:
                report["dropped_actuators"].append(a.get("name"))

    excludes = []
    for sec in root.findall("contact"):
        for ex in sec.findall("exclude"):
            b1 = name_map.get(ex.get("body1"), ex.get("body1"))
            b2 = name_map.get(ex.get("body2"), ex.get("body2"))
            if b1 in bodies and b2 in bodies:
                excludes.append(ET.Element("exclude", body1=b1, body2=b2))

    # rename class / mesh / material / texture references everywhere we copy
    refs = {"class": cls_map, "childclass": cls_map, "mesh": mesh_map,
            "material": mat_map, "texture": tex_map}
    for group in [[palm], defaults, tendons, actuators, assets]:
        for el in group:
            for e in el.iter():
                for attr, mp in refs.items():
                    v = e.get(attr)
                    if v in mp:
                        e.set(attr, mp[v])

    if to_degrees:
        rad_to_deg_angles(palm)
        for dcls in defaults:
            rad_to_deg_angles(dcls)

    top_classes = [d.get("class") for d in defaults]
    return {"palm": palm, "defaults": defaults, "assets": assets, "tendons": tendons,
            "actuators": actuators, "excludes": excludes, "report": report,
            "top_classes": top_classes, "n_joints": len(joints)}


# --------------------------------------------------- step 3: scene dressing
def add_scene(root):
    for tag in ("option", "visual", "statistic"):
        for el in root.findall(tag):
            root.remove(el)
    opt = ET.Element("option", timestep=str(config.TIMESTEP), integrator="implicitfast",
                     cone="elliptic", impratio=str(config.IMPRATIO),
                     iterations=str(config.SOLVER_ITERATIONS),
                     ls_iterations=str(config.LS_ITERATIONS))
    root.insert(1, opt)

    vis = ET.Element("visual")
    ET.SubElement(vis, "headlight", diffuse="0.5 0.5 0.5", ambient="0.25 0.25 0.25",
                  specular="0 0 0")
    ET.SubElement(vis, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(vis, "global", azimuth="160", elevation="-15",
                  offwidth="1920", offheight="1080")
    ET.SubElement(vis, "quality", shadowsize="8192")
    root.insert(2, vis)
    root.insert(3, ET.Element("statistic", center="0.1 0 0.9", extent="1.6"))

    asset = find_or_create(root, "asset")
    asset.append(ET.Element("texture", name="scene_sky", type="skybox", builtin="gradient",
                            rgb1="0.35 0.55 0.75", rgb2="0 0 0", width="512", height="3072"))
    asset.append(ET.Element("texture", name="scene_ground", type="2d", builtin="checker",
                            mark="edge", rgb1="0.22 0.26 0.30", rgb2="0.16 0.19 0.22",
                            markrgb="0.7 0.7 0.7", width="300", height="300"))
    asset.append(ET.Element("material", name="scene_ground", texture="scene_ground",
                            texuniform="true", texrepeat="5 5", reflectance="0.15"))

    wb = root.find("worldbody")
    wb.insert(0, ET.Element("light", name="key_light", pos="2 -1.5 4", dir="-0.45 0.3 -1",
                            directional="true", castshadow="true", diffuse="0.75 0.75 0.75",
                            specular="0.2 0.2 0.2"))
    wb.insert(1, ET.Element("light", name="fill_light", pos="-2 2 3", dir="0.5 -0.5 -1",
                            directional="true", castshadow="false", diffuse="0.3 0.3 0.3"))
    wb.insert(2, ET.Element("geom", name="floor", type="plane", size="0 0 0.05",
                            material="scene_ground", condim="3"))
    for name, pos, tgt in (("front", (1.9, -0.7, 1.35), (0.1, 0.0, 0.95)),
                           ("hands", (1.1, 0.0, 1.25), (0.2, 0.0, 1.0)),
                           ("side", (0.4, -2.0, 1.2), (0.2, 0.0, 1.0))):
        wb.insert(3, ET.Element("camera", name=name, pos=fmt(pos),
                                xyaxes=fmt(look_at_xyaxes(pos, tgt))))


# ------------------------------------------------------------------ build
def build(verbose=True):
    import mujoco
    men = config.find_menagerie()
    g1_xml = men / "unitree_g1" / "g1.xml"
    print(f"Menagerie : {men}")
    print(f"MuJoCo    : {mujoco.__version__}")

    print("[1/5] Reading G1 stand pose and wrist frames ...")
    pose, mounts = compute_pose_and_mounts(g1_xml)

    print("[2/5] Loading G1 MJCF ...")
    g1 = load_xml(g1_xml)
    absolutize_files(g1, g1_xml)
    comp = find_or_create(g1, "compiler", 0)
    to_degrees = comp.get("angle", "degree") != "radian"
    comp.set("autolimits", "true")
    for k in g1.findall("keyframe"):
        g1.remove(k)                 # its qpos size no longer matches
    wb = g1.find("worldbody")
    bodies = {b.get("name"): b for b in wb.iter("body")}

    if config.FIX_PELVIS:
        for b in wb.iter("body"):
            for j in list(b):
                if j.tag == "freejoint" or (j.tag == "joint" and j.get("type") == "free"):
                    b.remove(j)
                    b.set("pos", fmt(pose["base_pos"]))
                    b.set("quat", fmt(pose["base_quat"]))
                    print(f"      pelvis '{b.get('name')}' fixed at z={pose['base_pos'][2]:.3f} m")

    g1_main = find_or_create(g1, "default", 1)
    g1_main_settings = {ch.tag: dict(ch.attrib) for ch in g1_main if ch.tag != "default"}
    asset = find_or_create(g1, "asset")
    tendon_sec = find_or_create(g1, "tendon")
    act_sec = find_or_create(g1, "actuator")
    contact_sec = find_or_create(g1, "contact")

    print(f"[3/5] Mounting Shadow Hands (scale {config.HAND_SCALE}) ...")
    summary = {}
    for side, sp, cp, fname in (("left", "lh_", "shL_", "left_hand.xml"),
                                ("right", "rh_", "shR_", "right_hand.xml")):
        wrist = bodies.get(f"{side}_wrist_yaw_link")
        if wrist is None:
            raise BuildError(f"{side}_wrist_yaw_link not found in g1.xml")
        removed = []
        for g in list(wrist.findall("geom")):
            label = ((g.get("mesh") or "") + " " + (g.get("name") or "")).lower()
            if "hand" in label or "rubber" in label:
                wrist.remove(g)
                removed.append(g.get("mesh") or g.get("name") or "unnamed")
        print(f"      {side}: removed G1 hand geoms: {removed or 'none found'}")

        hand = extract_shadow_hand(men / "shadow_hand" / fname, sp, cp,
                                   config.HAND_SCALE, to_degrees)
        palm = hand["palm"]
        for a in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            palm.attrib.pop(a, None)
        palm.set("pos", fmt(mounts[side]["pos"]))
        palm.set("quat", fmt(mounts[side]["quat"]))
        wrist.append(palm)

        # keep Shadow classes independent of anything the G1 sets in its main default
        for dcls in hand["defaults"]:
            for tag, attrs in g1_main_settings.items():
                builtin = BUILTIN_DEFAULTS.get(tag, {})
                child = dcls.find(tag)
                for attr in attrs:
                    if child is not None and child.get(attr) is not None:
                        continue
                    if builtin.get(attr):
                        if child is None:
                            child = ET.SubElement(dcls, tag)
                        child.set(attr, builtin[attr])
                    elif attr not in builtin:
                        print(f"      ! G1 main default sets {tag}.{attr}; Shadow may inherit it")
            g1_main.append(dcls)
        asset.extend(hand["assets"])
        tendon_sec.extend(hand["tendons"])
        act_sec.extend(hand["actuators"])
        contact_sec.extend(hand["excludes"])
        for link in ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link"):
            if f"{side}_{link}" in bodies:
                contact_sec.append(ET.Element("exclude", body1=palm.get("name"),
                                              body2=f"{side}_{link}"))
        if hand["report"]["ignored_main_defaults"]:
            print(f"      ! ignored Shadow top-level defaults: {hand['report']['ignored_main_defaults']}")
        print(f"      {side}: {hand['n_joints']} finger joints, {len(hand['actuators'])} actuators "
              f"(dropped Shadow wrist actuators: {hand['report']['dropped_actuators']})")
        summary[side] = hand["n_joints"]

    for sec in (tendon_sec, contact_sec):
        if len(sec) == 0:
            g1.remove(sec)

    print("[4/5] Adding floor, lights, cameras, physics options and the apple ...")
    add_scene(g1)
    import apple
    apple.add_to_model(g1)
    g1.set("model", "g1_shadow_apple")
    g1.insert(0, ET.Comment(" GENERATED by build_model.py - edit config.py and rebuild, "
                            "do not edit by hand "))

    config.ROBOT_DIR.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(g1)
    ET.indent(tree, space="  ")
    tree.write(str(config.SCENE_XML), encoding="utf-8")

    pose["hand_scale"] = config.HAND_SCALE
    pose["mounts"] = mounts
    pose["fixed_pelvis"] = config.FIX_PELVIS
    config.POSE_JSON.write_text(json.dumps(pose, indent=2), encoding="utf-8")

    print("[5/5] Compiling the generated model with MuJoCo ...")
    try:
        m = mujoco.MjModel.from_xml_path(str(config.SCENE_XML))
    except Exception as exc:  # noqa: BLE001
        raise BuildError(f"MuJoCo rejected the generated XML:\n{exc}") from exc
    print(f"      OK: nq={m.nq} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"mass={mujoco.mj_getTotalmass(m):.2f} kg")
    print(f"      five-finger hands: left {summary['left']} joints, right {summary['right']} joints")
    print(f"Wrote {config.SCENE_XML}")
    print(f"Wrote {config.POSE_JSON}")
    return m


if __name__ == "__main__":
    try:
        build()
    except (BuildError, FileNotFoundError) as exc:
        print(f"\nBUILD FAILED: {exc}")
        sys.exit(1)

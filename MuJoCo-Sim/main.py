"""
main.py
Entry point. Run from the VS Code terminal inside the project folder:

    python main.py build      generate robot_model/g1_shadow_scene.xml
    python main.py check      3 s headless physics test, prints a report
    python main.py view       open the viewer, robot standing still
    python main.py fingers    open the viewer, both hands open and close slowly
    python main.py apple      open the viewer, apple dropped onto its stand repeatedly
    python main.py toss       THE MAIN SIMULATION: pick, toss and catch, forever
    python main.py toss-check same thing headless for 30 s, prints an event log and summary
    python main.py animate    pose the robot with sliders, save keyframes, play animations

Add --recalibrate to force the throw planner to run again.

The model is built automatically the first time if it does not exist yet.
"""

import argparse
import math
import re
import sys

import numpy as np

import config


def ensure_built(force=False):
    if force or not config.SCENE_XML.exists() or not config.POSE_JSON.exists():
        import build_model
        try:
            build_model.build()
        except (build_model.BuildError, FileNotFoundError) as exc:
            print(f"\nBUILD FAILED: {exc}")
            sys.exit(1)


# fingers that flex: MCP (J3), coupled PIP+DIP tendon (J0), thumb flexion
FLEX = re.compile(r"_A_(FFJ3|FFJ0|MFJ3|MFJ0|RFJ3|RFJ0|LFJ3|LFJ0|THJ4|THJ1|THJ2)$")


def finger_wave(sim, t):
    """Step 4 test: curl and open all ten fingers every 4 seconds."""
    alpha = 0.5 - 0.5 * math.cos(2 * math.pi * t / 4.0)   # 0 -> 1 -> 0, smooth
    for side in ("left", "right"):
        for i in sim.hand_actuators(side):
            name = sim.actuator_names[i]
            if FLEX.search(name):
                hi = sim.model.actuator_ctrlrange[i][1]
                sim.data.ctrl[i] = sim.clip_ctrl(i, alpha * 0.8 * hi)


def make_apple_drop_loop(period=4.0, height=0.20):
    """Step 5 viewer test: re-drop the apple onto its stand every few seconds."""
    from apple import apple_rest_position
    state = {"next": 1.0}

    def loop(sim, t):
        if t >= state["next"]:
            sim.apple.set_free_state(np.array(apple_rest_position()) + (0, 0, height))
            state["next"] += period
    return loop


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 apple toss simulation")
    parser.add_argument("mode", nargs="?", default="view",
                        choices=["build", "check", "view", "fingers", "apple",
                                 "toss", "toss-check", "animate"])
    parser.add_argument("--rebuild", action="store_true",
                        help="regenerate the model before running")
    parser.add_argument("--recalibrate", action="store_true",
                        help="delete the cached throw plan and plan again")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="duration of toss-check (simulated seconds)")
    args = parser.parse_args()

    if args.mode == "build":
        ensure_built(force=True)
        return
    ensure_built(force=args.rebuild)
    if args.mode == "animate":
        from animator import Animator
        Animator().run()
        return

    if args.recalibrate:
        cache = config.ROBOT_DIR / "throw_calibration.json"
        if cache.exists():
            cache.unlink()
            print("throw plan cache deleted")

    from simulation import Simulation
    sim = Simulation()
    print(f"Loaded {config.SCENE_XML.name}: {sim.model.nu} actuators, "
          f"{len(sim.hand_actuators('left'))} left-hand / "
          f"{len(sim.hand_actuators('right'))} right-hand")

    if args.mode == "check":
        print("--- holding stand pose ---")
        sim.run_check(3.0)
        print("--- curling fingers ---")
        sim.run_check(2.0, controller=finger_wave)
        print("--- apple ---")
        sim.run_apple_test()
    elif args.mode == "apple":
        print("Viewer open. The apple is dropped onto its stand from 20 cm every 4 s.")
        sim.run_viewer(controller=make_apple_drop_loop())
    elif args.mode == "view":
        print("Viewer open. Drag to orbit, right-drag to pan, scroll to zoom. Close window to exit.")
        sim.run_viewer()
    elif args.mode == "toss":
        from toss import TossSequence
        seq = TossSequence(sim)
        print("Viewer open. The first throw in each direction pauses ~1-3 s while it is planned.")
        sim.run_viewer(controller=seq)
        print("summary:", seq.stats)
    elif args.mode == "toss-check":
        from toss import TossSequence
        seq = TossSequence(sim)
        wall = sim.run_headless(seq, args.seconds)
        st = seq.stats
        print(f"\nSUMMARY: {args.seconds:.0f} s simulated in {wall:.1f} s wall | "
              f"grasps {st['grasps']}, throws {st['throws']}, catches {st['catches']}, "
              f"drops {st['drops']}")
    elif args.mode == "fingers":
        print("Viewer open. Fingers will curl and open every 4 s.")
        sim.run_viewer(controller=finger_wave)


if __name__ == "__main__":
    main()

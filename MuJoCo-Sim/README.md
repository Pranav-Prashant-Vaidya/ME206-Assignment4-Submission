# Unitree G1 Apple Toss in MuJoCo

A physics-based MuJoCo simulation of a **Unitree G1 humanoid with five-finger hands** that
picks up an apple, tosses it into the air and catches it with the other hand.

Everything that moves the apple is real MuJoCo physics: the fingers grip it through
contact and friction, the hand throws it, gravity carries it, and the other hand
catches it. The apple is never teleported or animated along a scripted path.

The project includes an **interactive animator**: pose the robot with sliders, save
keyframes, let the robot throw the apple on command, and render the result to a
smooth video.

<!-- Add your rendered video or a GIF here, for example:
![Demo](videos/FinalSimulation.gif)
-->

---

## Features

- **Unitree G1 humanoid** (29 joints) from Google DeepMind's MuJoCo Menagerie
- **Five-finger Shadow Hands** on both wrists (22 finger joints and 18 motors per hand),
  because the stock G1 hand (Unitree Dex3-1) only has three fingers
- **Physical apple**: 5.4 cm, 70 g, with rolling and twisting friction, a procedurally
  generated apple mesh, stem and leaf
- **Keyframe animator** driven by the MuJoCo viewer's sliders
- **`throw` command**: the hand holding the apple turns palm-up and tosses it. The throw
  is planned by simulating candidate throws, and a green marker shows where it will land
- **Video export**: renders any saved animation to MP4 at a steady 30 or 60 fps, even
  on a slow laptop
- **Automatic toss sequence** (experimental, see [Known limitations](#known-limitations))

---

## Requirements

| Item | Version used | Notes |
|---|---|---|
| Python | 3.13 (Anaconda) | 3.10 or newer should work |
| MuJoCo | 4.0.1 | `pip install mujoco` |
| NumPy, Pillow | any recent | already included in Anaconda |
| imageio + imageio-ffmpeg | optional | only for MP4 video export |
| MuJoCo Menagerie | latest | robot and hand models, **downloaded separately** (see below) |

Tested on Windows 11 with VS Code.

---

## Installation

### 1. Get this repository

```bash
git clone https://github.com/<your-username>/unitree_apple_simulation.git
cd unitree_apple_simulation
```

Or download the ZIP from GitHub and unzip it.

### 2. Install the Python packages

```bash
pip install -r requirements.txt
```

### 3. Download MuJoCo Menagerie

The robot and hand models are **not** included in this repository. They come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
`pip install mujoco-menagerie` does not provide them, so clone the repository:

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

The code automatically looks for a folder named `mujoco_menagerie` (or
`mujoco_menagerie-main`, the name a GitHub ZIP download uses) in any of these places:

- inside this project folder
- next to this project folder
- your home folder (for example `C:\Users\<you>\mujoco_menagerie`)

If you put it somewhere else, set the `MENAGERIE_PATH` environment variable:

```powershell
# Windows PowerShell
$env:MENAGERIE_PATH = "D:\path\to\mujoco_menagerie"
```
```bash
# macOS / Linux
export MENAGERIE_PATH=/path/to/mujoco_menagerie
```

### 4. Build the robot model (required on every new computer)

```bash
python main.py build
```

This generates `robot_model/g1_shadow_scene.xml`, which combines the G1, the two
Shadow Hands, the apple and the scene. The generated file contains absolute paths to
your Menagerie folder, so **every new computer must run this step once**. It should
end with a line like:

```
OK: nq=73 nu=65 nbody=77 ... five-finger hands: left 22 joints, right 22 joints
```

### 5. Check that everything works

```bash
python main.py check
```

This runs a short test without a window and prints a report: the robot holds its
standing pose, the fingers curl, and the apple is dropped onto its stand.

> **macOS users:** MuJoCo's interactive viewer must be started with `mjpython`
> instead of `python`, for example `mjpython main.py animate`.

---

## Quick start

```bash
python main.py view                          # look at the robot
python main.py animate                       # make your own animation
python render_video.py FinalSimulation       # render the included animation to video
```

---

## All commands

Run these from a terminal inside the project folder (in VS Code: Terminal > New Terminal).

| Command | What it does |
|---|---|
| `python main.py build` | Generate the robot model (after downloading or after editing `config.py`) |
| `python main.py check` | Headless physics test with a text report |
| `python main.py view` | Viewer: robot standing still |
| `python main.py fingers` | Viewer: all ten fingers curl and open every 4 s |
| `python main.py apple` | Viewer: apple dropped onto its stand every 4 s |
| `python main.py animate` | **Interactive animator** (sliders, keyframes, throw) |
| `python main.py toss` | Automatic pick, toss and catch sequence (experimental) |
| `python main.py toss-check` | Same as `toss`, without a window, with an event log |
| `python render_video.py <name>` | Render `animations/<name>.json` to a video |
| `python tools/inspect_models.py` | List the Menagerie models, joints and motors on your machine |

---

## Using the animator

```bash
python main.py animate
```

The MuJoCo viewer opens with the robot, a wooden table and the apple. The terminal
shows an `anim>` prompt where you type commands while the viewer stays open.

### How an animation works

1. **Keyframe:** a snapshot of all 65 motor targets (every arm, leg, waist and finger
   joint), recorded with `k`.
2. **Duration:** each keyframe stores how long the robot takes to move from the previous
   pose into it. `k 1.5` means "reach this pose in 1.5 seconds".
3. **Playback:** `play` moves every joint smoothly from one keyframe to the next, with a
   gentle start and stop, and sends the targets to the joint motors 500 times a second.
4. **Physics:** MuJoCo simulates the result. If your keyframes close the fingers around
   the apple and raise the arm, the apple really gets lifted. If the grip is loose, it
   really slips.

### Posing the robot

- **Sliders:** in the viewer's right-hand panel, open **Control**. There is one slider per
  motor, for example `right_shoulder_pitch_joint`, `right_elbow_joint`, or `rh_A_FFJ3`
  (right index finger).
- **Whole hand at once:** `hand right grasp`, `hand right open`, `hand right pregrasp`,
  `hand right cup` (same for `left`).
- **Exact value:** `set right_elbow 1.2`.

Wait for the robot to stop moving before you record a keyframe, especially while it
holds the apple.

### Animator commands

| Command | Description |
|---|---|
| `k [sec]` | Add a keyframe from the current pose (default 1.0 s) |
| `ins N [sec]` | Insert a keyframe at position N |
| `del [N]` | Delete keyframe N (default: the last one) |
| `dur N sec` | Change the duration of keyframe N |
| `list` | List all keyframes |
| `goto N` | Move the robot to keyframe N, to adjust it |
| `play` / `loop` / `stop` | Play once, play forever, stop |
| `reset` | Put the robot and the apple back to the start |
| `hand left\|right open\|pregrasp\|cup\|grasp` | Set all 18 finger motors of a hand |
| `set NAME VALUE` | Set one motor by (part of) its name |
| `pose` | Print the current arm joint targets |
| `throw [height] [sideways]` | Throw the apple (see below) |
| `rewind` | Go back to just before the last throw |
| `ready left\|right` | Move that hand palm-up under the landing marker |
| `speed X` | Slow motion, for example `speed 0.25`; `speed 1` is normal |
| `save [name]` / `load [name]` | Save or load `animations/<name>.json` |
| `help` / `quit` | Show commands / exit |

**Always `save` before closing the viewer.** Unsaved keyframes are lost.

### Example: pick up the apple

```
anim> hand right pregrasp      (use the sliders to bring the right hand above the apple)
anim> k 1.5
                               (lower the hand around the apple with the sliders)
anim> k 1.0
anim> hand right grasp
anim> k 0.6
                               (raise the shoulder and elbow to lift)
anim> k 1.2
anim> play
anim> save pick_apple
```

### Throwing and catching

Once a hand is holding the apple, type `throw`. The robot will:

1. turn that palm up and lift the apple 10 cm,
2. relax the fingers into a cup,
3. plan the throw by simulating up to 48 candidate throws (the viewer pauses for a
   few seconds),
4. throw. The apple leaves the palm on its own as the hand slows down,
5. show a **green marker** where the apple will come back down to hand height.

`throw [height] [sideways]` takes two optional numbers in metres:

- `throw`: 25 cm high, landing 25 cm toward the other hand
- `throw 0.3 0.2`: 30 cm high, 20 cm sideways
- `throw 0.2 0`: straight up, back into the same hand

To catch it with the other hand:

```
anim> throw 0.3 0.25      (watch where the green marker appears)
anim> rewind              (back to the moment before the throw, marker stays)
anim> ready left          (or move the left hand under the marker with the sliders)
anim> throw 0.3 0.25      (same numbers give the same throw)
anim> hand left grasp     (close the fingers after it lands)
```

Note: `throw` runs live and is **not** stored in saved animations, so it will not
appear in rendered videos.

---

## Rendering a video

The live viewer can be slow on a laptop. Rendering produces a smooth video no matter
how slow the computer is, because frames are made without a real-time deadline.

```bash
pip install imageio-ffmpeg                    # once, for MP4 output
python render_video.py FinalSimulation        # animations/FinalSimulation.json
```

The output goes to `videos/<name>.mp4`. Without an MP4 encoder the script writes a
GIF instead.

| Option | Example | Default |
|---|---|---|
| Frame rate | `--fps 60` | 30 |
| Resolution | `--width 1920 --height 1080` | 1280 x 720 |
| Camera | `--camera front` (or `hands`, `side`) | free camera matching the animator view |

---

## Viewer controls and performance

| Action | Control |
|---|---|
| Rotate view | Left-drag |
| Pan | Right-drag |
| Zoom | Scroll |
| Hide left / right panel | `Tab` / `Shift+Tab` |
| Switch camera | `[` and `]` |
| Pause | `Space` |

**If the viewer is slow:**

- `FAST_GRAPHICS = True` in `config.py` (the default) turns off shadows, reflections
  and the sky in the live viewer. Videos always use full quality.
- Hide the side panels with `Tab` and `Shift+Tab`, especially the Control panel.
- Use a smaller viewer window, and keep the laptop plugged in.
- On laptops with an NVIDIA GPU: Windows Settings > System > Display > Graphics, add
  your `python.exe` and choose "High performance".

---

## Project structure

```text
unitree_apple_simulation/
├── main.py              Entry point for every run mode
├── config.py            All settings: Menagerie path, hand size, apple, physics, camera, graphics
├── build_model.py       Builds the model: G1 + Shadow Hands + apple + floor, lights, cameras
├── apple.py             Apple mesh generator, apple and stand bodies, apple state and contacts
├── controllers.py       Arm inverse kinematics, smooth arm trajectories, finger grasp presets
├── simulation.py        Loads the model, standing pose, real-time viewer loop, headless tests
├── animator.py          Interactive keyframe animator (table scene, sliders, terminal commands)
├── throw_action.py      The animator's throw (planned by simulation) and ready commands
├── toss.py              Automatic pick, toss and catch state machine (experimental)
├── render_video.py      Renders saved animations to MP4 or GIF
├── animations/          Saved animations (.json), for example FinalSimulation.json
├── videos/              Rendered videos
├── robot_model/         GENERATED by "python main.py build" (not in the repository)
├── tools/
│   └── inspect_models.py  Lists Menagerie models, joints and motors
├── requirements.txt
└── README.md
```

---

## How it works

**Five-finger hands.** Menagerie's `g1_with_hands.xml` uses the three-finger Unitree
Dex3-1 hand. `build_model.py` starts from the hand-less `g1.xml`, removes its static
rubber hands, and mounts Menagerie's Shadow Hand from the palm down on each
`wrist_yaw_link`, scaled to 0.8 to match the G1's size. The Shadow Hand's own wrist
joints are dropped because the G1 already has a three-joint wrist. Its finger tendons,
which couple each finger's middle and last joints, are kept. The hand's mounting
orientation is computed from the G1's standing pose so the palms face the body.

**Standing.** The pelvis is fixed in place at the G1's standing height, and the leg and
waist motors hold the G1 "stand" pose. There is no balance controller (see below).

**Arm control.** Arm poses are solved with damped least-squares inverse kinematics on
MuJoCo's own Jacobians (hand position and palm orientation, with joint limits).
Motions follow minimum-jerk trajectories, tracked by the G1's position motors.

**Throwing.** The hand cups the apple palm-up and makes a short upward flick. The apple
separates from the palm when the hand decelerates faster than gravity. Before throwing,
candidate flicks are simulated in a copy of the physics state, and the one closest to
the requested height and distance is used.

**Physics settings.** 2 ms timestep, `implicitfast` integrator, elliptic friction cones,
impratio 10 and 50 solver iterations. The apple's collision shape is a sphere with
rolling and twisting friction and a firm, critically damped contact. The detailed apple
mesh is visual only.

---

## Customisation

Edit `config.py`, then run `python main.py build` again for anything that changes the
model (hand size, apple, stand).

| Setting | What it changes |
|---|---|
| `HAND_SCALE` | Size of the Shadow Hands |
| `SHADOW_PALM_NORMAL` | Flip to `(0.0, 1.0, 0.0)` if the palms face outward |
| `APPLE_RADIUS`, `APPLE_MASS`, `APPLE_FRICTION` | The apple |
| `APPLE_START_XY`, `STAND_TOP_Z` | Apple stand position (automatic toss mode) |
| `TIMESTEP`, `SOLVER_ITERATIONS` | Physics accuracy versus speed |
| `FAST_GRAPHICS`, `SHADOW_SIZE` | Live viewer graphics quality |
| `CAM_*` | Default viewer camera |

The table and apple position in the animator are set at the top of `animator.py`
(`TABLE_CENTER`, `TABLE_TOP_Z`, `APPLE_ON_TABLE`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `MuJoCo Menagerie not found` | Clone Menagerie (Installation step 3) or set `MENAGERIE_PATH` |
| Error loading `g1_shadow_scene.xml`, or missing mesh files | Run `python main.py build` on this computer |
| `No module named 'mujoco'` | VS Code is using a different Python. Press `Ctrl+Shift+P` > "Python: Select Interpreter" and pick the one where MuJoCo is installed |
| Palms face outward | Set `SHADOW_PALM_NORMAL = (0.0, 1.0, 0.0)` in `config.py`, then rebuild |
| `throw` says the apple fell out while turning | Grip harder first: `hand right grasp` |
| `throw` says no stroke released the apple | Ask for less height, for example `throw 0.15` |
| Video comes out as a GIF | `pip install imageio-ffmpeg`, then render again |
| Viewer is slow | See [Viewer controls and performance](#viewer-controls-and-performance) |

---

## Known limitations

- **Fixed pelvis.** The robot does not balance by itself. Its pelvis is attached to the
  world, so it cannot fall over. A whole-body balance controller would be needed for a
  free-standing robot.
- **Shadow Hands, not Unitree hands.** Menagerie has no five-finger Unitree hand, so the
  hands look like Shadow Hands rather than Unitree's own.
- **Automatic toss mode is experimental.** `python main.py toss` tries to do the whole
  pick, toss and catch sequence by itself, but in testing it dropped the apple while
  grasping it from the stand. The animator with the `throw` command is the reliable way
  to make the toss.
- **Sphere collision.** The apple collides as a sphere; its apple shape is visual only.
- **`throw` is not recorded** in saved animations.

---

## Credits

- [MuJoCo](https://github.com/google-deepmind/mujoco) by Google DeepMind
- Unitree G1 and Shadow Hand models from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Each model is
  distributed under its own license in its Menagerie folder; they are not redistributed
  in this repository.

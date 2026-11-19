"""
render_video.py
Turn a saved animation (animations/<name>.json) into a smooth video file.

The physics runs at full accuracy and every frame is rendered off-screen, so the
video plays at a steady 30/60 fps no matter how slow the laptop is. Rendering just
takes longer than real time.

    python render_video.py my_animation
    python render_video.py my_animation --fps 60 --width 1920 --height 1080
    python render_video.py my_animation --camera front      (named camera instead of
                                                            the default free view)

Output: videos/<name>.mp4  (or videos/<name>.gif if no MP4 encoder is installed;
        for MP4 run:  pip install imageio-ffmpeg)
"""

import argparse
import sys
import time

import mujoco

import config

VIDEO_DIR = config.PROJECT_DIR / "videos"


# ------------------------------------------------------------------ video writers
class Writer:
    """Tries imageio+ffmpeg, then OpenCV, then falls back to a GIF via Pillow."""

    def __init__(self, stem, fps, width, height):
        VIDEO_DIR.mkdir(exist_ok=True)
        self.kind, self.fps, self.frames = None, fps, []
        mp4 = VIDEO_DIR / f"{stem}.mp4"
        try:
            import imageio.v2 as iio
            import imageio_ffmpeg  # noqa: F401  (without it imageio picks a non-video plugin)
            self.w = iio.get_writer(str(mp4), format="FFMPEG", mode="I", fps=fps,
                                    codec="libx264", quality=8, macro_block_size=1)
            self.kind, self.path = "imageio", mp4
            return
        except Exception:  # noqa: BLE001  (imageio or its ffmpeg plugin missing)
            pass
        try:
            import cv2
            w = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if w.isOpened():
                self.cv2, self.w = cv2, w
                self.kind, self.path = "opencv", mp4
                return
        except Exception:  # noqa: BLE001
            pass
        from PIL import Image  # Pillow ships with Anaconda
        self.Image = Image
        self.kind, self.path = "gif", VIDEO_DIR / f"{stem}.gif"
        print("! no MP4 encoder found, writing a GIF instead "
              "(for MP4:  pip install imageio-ffmpeg)")

    def add(self, rgb):
        if self.kind == "imageio":
            try:
                self.w.append_data(rgb)
                return
            except Exception as exc:  # noqa: BLE001  any encoder problem -> GIF fallback
                print(f"! MP4 writer failed ({exc}); switching to GIF")
                try:
                    self.w.close()
                except Exception:  # noqa: BLE001
                    pass
                from PIL import Image
                self.Image, self.kind = Image, "gif"
                self.path = self.path.with_suffix(".gif")
        if self.kind == "opencv":
            self.w.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        else:
            img = self.Image.fromarray(rgb)
            img.thumbnail((640, 640))                    # keep the GIF a sane size
            self.frames.append(img)

    def close(self):
        if self.kind == "imageio":
            self.w.close()
        elif self.kind == "opencv":
            self.w.release()
        elif self.frames:
            self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:],
                                duration=int(1000 / self.fps), loop=0)
        return self.path


# ------------------------------------------------------------------ render
def render(name, fps=30, width=1280, height=720, camera=None, tail=1.0):
    from animator import Animator, ANIM_DIR
    if not (ANIM_DIR / f"{name}.json").exists():
        sys.exit(f"animations/{name}.json not found. In the animator type  save {name}  first.")
    anim = Animator()
    anim.handle(f"load {name}")
    if not anim.keyframes:
        sys.exit("that animation has no keyframes")
    m, d = anim.m, anim.d
    width = min(width, m.vis.global_.offwidth)
    height = min(height, m.vis.global_.offheight)

    renderer = mujoco.Renderer(m, height=height, width=width)
    opt = mujoco.MjvOption()
    if camera:
        cam = camera
    else:                                        # same view the animator opens with
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.25, 0.0, 0.9)
        cam.distance, cam.azimuth, cam.elevation = 2.0, 160, -18

    anim.start_play(loop=False)
    total = anim.total_time() + tail
    writer = Writer(name, fps, width, height)
    n_frames = int(total * fps) + 1
    print(f"rendering {n_frames} frames at {width}x{height}, {fps} fps ...")
    t_wall, next_frame, frame = time.perf_counter(), 0.0, 0
    while frame < n_frames:
        if d.time - anim.play_t0 >= next_frame:
            renderer.update_scene(d, camera=cam, scene_option=opt)
            writer.add(renderer.render())
            frame += 1
            next_frame += 1.0 / fps
            if frame % fps == 0:
                print(f"  {frame}/{n_frames} frames")
            continue
        if anim.playing:
            anim.step_playback()
        mujoco.mj_step(m, d)
    renderer.close()
    path = writer.close()
    print(f"done in {time.perf_counter() - t_wall:.0f} s -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="Render a saved animation to video")
    p.add_argument("name", nargs="?", default="my_animation")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--camera", default=None, help="front, hands or side")
    a = p.parse_args()
    render(a.name, a.fps, a.width, a.height, a.camera)


if __name__ == "__main__":
    main()

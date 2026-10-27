"""Build the directory layout infer.py expects.

The README's sample (`data/test_long_sample_00502`) is not in the public repo.
This writes one 81-frame clip, three views, depth + robot mask, so the documented
command can run. The pictures are synthetic geometry, not a robot dataset.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

HEIGHT, WIDTH, FRAMES = 240, 416, 81
VIEWS = ("hand_left", "hand_right", "head_color")
PROMPT = "A robot arm on a wooden table reaches for a red mug and lifts it."


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/synthetic_clip81")
    root.mkdir(parents=True, exist_ok=True)
    (root / "original_instruction.txt").write_text(PROMPT + "\n")
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    for view_index, name in enumerate(VIEWS):
        depth_dir = root / name / "depth"
        mask_dir = root / name / "mask"
        depth_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        depth_frames = []
        mask_frames = []
        for t in range(FRAMES):
            cx = int(70 + (WIDTH - 150) * t / (FRAMES - 1))
            cy = 120 + 18 * view_index
            arm = (np.abs(yy - (cy + 36)) < 11) & (xx > 16) & (xx < cx)
            gripper = ((yy - cy) ** 2) / (28 ** 2) + ((xx - cx) ** 2) / (16 ** 2) <= 1
            robot = arm | gripper
            depth = 36 + 70 * (xx / WIDTH) + 24 * view_index + 12 * np.sin((t + 10 * view_index) / 9.0)
            depth = depth + 8 * np.sin(yy / 17.0)
            depth = np.where(robot, 210 - 20 * view_index, depth)
            depth_u8 = np.clip(depth, 0, 255).astype(np.uint8)
            depth_frames.append(np.stack([depth_u8, depth_u8, depth_u8], axis=-1))
            mask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            mask[arm] = (150, 150, 150)
            mask[gripper] = (210, 210, 210)
            mask_frames.append(mask)
        imageio.mimwrite(depth_dir / "original_depth_vda.mp4", depth_frames, fps=15, quality=8, macro_block_size=1)
        imageio.mimwrite(mask_dir / "rgb_robot.mp4", mask_frames, fps=15, quality=8, macro_block_size=1)
        Image.fromarray(depth_frames[0]).save(root / name / "pre_frame.png")
    print(f"SAMPLE_OK {root} frames={FRAMES} views={','.join(VIEWS)}")


if __name__ == "__main__":
    main()

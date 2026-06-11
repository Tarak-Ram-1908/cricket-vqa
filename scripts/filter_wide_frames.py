import cv2, numpy as np, shutil
from pathlib import Path

def is_field_placement_shot(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Must have lots of green grass
    mask_green = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
    green_ratio = mask_green.sum() / 255 / (h * w)
    if green_ratio < 0.40:
        return False

    # Must have sky/crowd at top
    top = hsv[:h//3, :, :]
    mask_sky = cv2.inRange(top, (90, 0, 80), (140, 80, 255))
    sky_ratio = mask_sky.sum() / 255 / (h//3 * w)
    if sky_ratio < 0.08:
        return False

    # Must NOT have a large central figure (close-up batsman)
    # Check center 30% of frame for skin/white clothing domination
    cx1, cx2 = int(w*0.35), int(w*0.65)
    cy1, cy2 = int(h*0.30), int(h*0.85)
    center = img[cy1:cy2, cx1:cx2]
    center_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    # White clothing (high value, low saturation)
    mask_white = cv2.inRange(center_hsv, (0, 0, 180), (180, 40, 255))
    white_ratio = mask_white.sum() / 255 / (center.shape[0] * center.shape[1])
    if white_ratio > 0.15:
        return False

    # Green in bottom half must be high (not stands/crowd dominating)
    bottom = hsv[h//2:, :, :]
    mask_bottom_green = cv2.inRange(bottom, (25, 40, 40), (90, 255, 255))
    bottom_green = mask_bottom_green.sum() / 255 / (h//2 * w)
    if bottom_green < 0.35:
        return False

    return True

frames_dir = Path("data/frames_wide")
out_dir = Path("data/frames_filtered")
out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.jpg"):
    f.unlink()

all_frames = sorted(frames_dir.glob("*.jpg"))
print(f"Filtering {len(all_frames)} wide frames...")
saved = 0
for i, f in enumerate(all_frames):
    if i % 200 == 0:
        print(f"  {i}/{len(all_frames)}, good so far: {saved}")
    if is_field_placement_shot(f):
        shutil.copy(f, out_dir / f.name)
        saved += 1

print(f"Done. Found {saved} field placement frames -> data/frames_filtered/")

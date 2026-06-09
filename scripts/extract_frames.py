import cv2, os
from PIL import Image
import imagehash

def extract_frames(video_path, out_dir, fps=2, min_size=480):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {native_fps}")
    h_vid = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w_vid = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"Video resolution: {w_vid}x{h_vid}")
    step = max(1, int(native_fps / fps))
    seen_hashes = set()
    idx = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            if min(h, w) >= min_size:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                phash = str(imagehash.phash(pil))
                if phash not in seen_hashes:
                    seen_hashes.add(phash)
                    pil.save(f"{out_dir}/{idx:07d}.jpg", quality=92)
                    saved += 1
        idx += 1
    cap.release()
    print(f"Done. Saved {saved} frames from {video_path}")

if __name__ == "__main__":
    import sys
    extract_frames(sys.argv[1], sys.argv[2])

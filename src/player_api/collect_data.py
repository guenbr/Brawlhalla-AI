import cv2
import numpy as np
import os
import sys
import time
from screen_grab.grab import ScreenGrab

# python -m player_api.collect_data capture
# python -m player_api.collect_data label

PATCH_SIZE    = 64
NEG_PER_POS   = 8
SAVE_DIR      = "player_api/data"
RAW_DIR       = "player_api/data/raw_frames"
MONITOR       = 2
CAPTURE_EVERY = 5
CAPTURE_DURATION = 3600

os.makedirs(f"{SAVE_DIR}/p1/pos",  exist_ok=True)
os.makedirs(f"{SAVE_DIR}/p1/neg",  exist_ok=True)
os.makedirs(f"{SAVE_DIR}/cpu/pos", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/cpu/neg", exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

half = PATCH_SIZE // 2

def extract_patch(frame, cx, cy):
    h, w = frame.shape[:2]
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)
    patch = frame[y1:y2, x1:x2]
    return cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

def random_negative(frame_bgr, avoid_cx, avoid_cy):
    h, w = frame_bgr.shape[:2]
    for _ in range(50):
        cx = np.random.randint(half, w - half)
        cy = np.random.randint(half, h - half)
        if abs(cx - avoid_cx) > PATCH_SIZE * 2 and abs(cy - avoid_cy) > PATCH_SIZE * 2:
            return extract_patch(frame_bgr, cx, cy)
    return None

# capture
def auto_capture():
    screen = ScreenGrab(monitor=MONITOR)
    count = 0
    end_time = time.time() + CAPTURE_DURATION
    print(f"Auto-capturing every {CAPTURE_EVERY}s for 1 hour. Play Brawlhalla! Press Ctrl+C to stop early.")
    try:
        while time.time() < end_time:
            raw = screen.grab(greyscale=False)
            bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            ts = int(time.time() * 1000)
            path = os.path.join(RAW_DIR, f"frame_{ts}.png")
            cv2.imwrite(path, bgr)
            count += 1
            remaining = int(end_time - time.time())
            print(f"  Captured frame {count} → {path} ({remaining}s remaining)")
            time.sleep(CAPTURE_EVERY)
    except KeyboardInterrupt:
        pass
    print(f"\nDone. {count} frames saved to {RAW_DIR}")

# label
state = {"mode": None, "frame": None, "bgr": None, "labeled": set()}
counts = {"p1_pos": 0, "p1_neg": 0, "cpu_pos": 0, "cpu_neg": 0}

def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or state["mode"] is None:
        return

    label = state["mode"]
    frame_bgr = state["bgr"]
    ts = int(time.time() * 1000)

    pos_patch = extract_patch(frame_bgr, x, y)
    cv2.imwrite(f"{SAVE_DIR}/{label}/pos/{ts}.png", pos_patch)
    counts[f"{label}_pos"] += 1

    for i in range(NEG_PER_POS):
        neg = random_negative(frame_bgr, x, y)
        if neg is not None:
            cv2.imwrite(f"{SAVE_DIR}/{label}/neg/{ts}_{i}.png", neg)
            counts[f"{label}_neg"] += 1

    state["labeled"].add(label)
    cv2.circle(state["frame"], (x, y), 6,
               (0, 255, 0) if label == "p1" else (0, 0, 255), -1)
    cv2.imshow("Labeler", state["frame"])
    print(f"  Saved {label} at ({x},{y}) | counts: {counts}")

def batch_label():
    frames = sorted([
        f for f in os.listdir(RAW_DIR) if f.endswith(".png")
    ])

    if not frames:
        print(f"No frames found in {RAW_DIR}. Run capture mode first.")
        return

    print(f"Found {len(frames)} frames to label.")
    print("Controls: 1=P1 mode, 2=CPU mode, N=next frame, Q=quit")

    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", on_click)

    idx = 0
    while idx < len(frames):
        path = os.path.join(RAW_DIR, frames[idx])
        bgr = cv2.imread(path)
        state["frame"] = bgr.copy()
        state["bgr"]   = bgr
        state["mode"]  = None
        state["labeled"] = set()

        cv2.imshow("Labeler", state["frame"])
        print(f"\nFrame {idx+1}/{len(frames)}: {frames[idx]}")
        print("  Press 1 (P1) or 2 (CPU), click label, then N for next")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                print(f"\nQuit. Final counts: {counts}")
                cv2.destroyAllWindows()
                return
            elif key == ord('1'):
                state["mode"] = "p1"
                print("  Mode: P1 — click the P1 label")
            elif key == ord('2'):
                state["mode"] = "cpu"
                print("  Mode: CPU — click the CPU label")
            elif key == ord('n'):
                idx += 1
                break

    print(f"\nAll frames labeled! Final counts: {counts}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "label"
    if mode == "capture":
        auto_capture()
    else:
        batch_label()
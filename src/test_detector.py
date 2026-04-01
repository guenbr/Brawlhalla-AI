import cv2
import time
from player_api.player_detector import PlayerDetector

MONITOR = 2
CAPTURE_EVERY = 5
TOTAL_DURATION = 300

detector = PlayerDetector(monitor=MONITOR)
frames = []

total = TOTAL_DURATION // CAPTURE_EVERY
print(f"Capturing {total} frames over {TOTAL_DURATION}s. Go play!")

for i in range(total):
    print(f"  Capture {i+1}/{total}...")
    annotated = detector.debug_frame()
    frames.append(annotated)
    if i < total - 1:
        time.sleep(CAPTURE_EVERY)

print(f"\nDone! Reviewing {len(frames)} frames. Press N/SPACE for next, Q to quit.")

cv2.namedWindow("Detection Review", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection Review", 1280, 720)

for i, frame in enumerate(frames):
    cv2.imshow("Detection Review", frame)
    print(f"Frame {i+1}/{len(frames)}")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('n'), ord(' ')):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Review ended.")
            exit()

cv2.destroyAllWindows()
print("Review complete.")
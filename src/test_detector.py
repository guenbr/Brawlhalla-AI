import cv2
import time
from player_location.player_detector import PlayerDetector

# Which monitor Brawlhalla is running on
MONITOR = 2

# How many seconds between each capture
CAPTURE_EVERY = 5

# Total seconds to capture frames for
TOTAL_DURATION = 60

# Create the detector — loads templates and initializes health tracking
detector = PlayerDetector(monitor=MONITOR)
frames = []

total = TOTAL_DURATION // CAPTURE_EVERY
print(f"Capturing {total} frames over {TOTAL_DURATION}s. Go play!")

# Capture an annotated frame every CAPTURE_EVERY seconds
for i in range(total):
    print(f"  Capture {i+1}/{total}...")
    annotated = detector.debug_frame()
    frames.append(annotated)
    if i < total - 1:
        time.sleep(CAPTURE_EVERY)

print(f"\nDone! Reviewing {len(frames)} frames. Press N/SPACE for next, Q to quit.")

# Open a resizable window to review all captured frames
cv2.namedWindow("Detection Review", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection Review", 1280, 720)

# Step through each frame one at a time
for i, frame in enumerate(frames):
    cv2.imshow("Detection Review", frame)
    print(f"Frame {i+1}/{len(frames)}")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('n'), ord(' ')):
            # Go to next frame
            break
        elif key == ord('q'):
            # Quit the review early
            cv2.destroyAllWindows()
            print("Review ended.")
            exit()

cv2.destroyAllWindows()
print("Review complete.")
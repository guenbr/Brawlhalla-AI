import cv2
import numpy as np
from screen_grab.grab import ScreenGrab

MONITOR = 2

FRAME_W = 2560
FRAME_H = 1440

# platform x bounds — normalized values from environment.py converted to pixels
# 0.32 * 2560 = ~819, 0.68 * 2560 = ~1741
PLATFORM_X_LEFT = int(0.32 * FRAME_W)
PLATFORM_X_RIGHT = int(0.68 * FRAME_W)

# platform y bounds — 0 at top, cuts off at ~850px (normalized 0.59)
PLATFORM_Y_TOP = 0
PLATFORM_Y_BOTTOM = int(0.59 * FRAME_H)

screen = ScreenGrab(monitor=MONITOR)
frame = screen.grab(greyscale=False)
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# draw vertical lines at the left and right platform edges
cv2.line(frame_bgr, (PLATFORM_X_LEFT, 0), (PLATFORM_X_LEFT, FRAME_H), (0, 255, 0), 3)
cv2.line(frame_bgr, (PLATFORM_X_RIGHT, 0), (PLATFORM_X_RIGHT, FRAME_H), (0, 255, 0), 3)

# draw a semi-transparent green fill over the platform zone
overlay = frame_bgr.copy()
cv2.rectangle(overlay,
              (PLATFORM_X_LEFT, PLATFORM_Y_TOP),
              (PLATFORM_X_RIGHT, PLATFORM_Y_BOTTOM),
              (0, 255, 0), -1)
cv2.addWeighted(overlay, 0.15, frame_bgr, 0.85, 0, frame_bgr)

# draw the outline of the platform zone
cv2.rectangle(frame_bgr,
              (PLATFORM_X_LEFT, PLATFORM_Y_TOP),
              (PLATFORM_X_RIGHT, PLATFORM_Y_BOTTOM),
              (0, 255, 0), 3)

# label the zone with its bounds
cv2.putText(frame_bgr, "PLATFORM ZONE (on_platform reward area)",
            (PLATFORM_X_LEFT + 10, PLATFORM_Y_TOP + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.putText(frame_bgr, f"x: {PLATFORM_X_LEFT} to {PLATFORM_X_RIGHT}  (normalized 0.32-0.68)",
            (PLATFORM_X_LEFT + 10, PLATFORM_Y_TOP + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("../test_platform.png", frame_bgr)
print(f"Saved test_platform.png")
print(f"Platform x: {PLATFORM_X_LEFT} to {PLATFORM_X_RIGHT}")
print(f"Platform y scan: {PLATFORM_Y_TOP} to {PLATFORM_Y_BOTTOM}")

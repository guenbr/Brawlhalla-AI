import cv2
from screen_grab.grab import ScreenGrab

MONITOR = 2

# HUD regions to visualize — (x, y, width, height)
REGIONS = {
    'game_end_p1': (2290, 20, 125, 130),
    'game_end_p2': (2400, 20, 125, 130),
}

# health sample pixel coordinates — (x, y)
HEALTH_COORDS = {
    'p1_health': (2350, 135),
    'cpu_health': (2476, 135),
}

screen = ScreenGrab(monitor=MONITOR)
frame = screen.grab(greyscale=False)
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# draw boxes for each HUD region
for name, (x, y, w, h) in REGIONS.items():
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 0, 0), 3)
    cv2.putText(frame_bgr, name, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# draw dots for health sample pixels
for name, (x, y) in HEALTH_COORDS.items():
    cv2.circle(frame_bgr, (x, y), 5, (0, 0, 0), -1)
    cv2.putText(frame_bgr, name, (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.imwrite("../test_coords.png", frame_bgr)
print(f"Saved test_coords.png — frame size: {frame_bgr.shape}")
print("Open test_coords.png to see if the boxes line up with the HUD")

import mss
import numpy as np
import time

with mss.mss() as sct:
    monitor = sct.monitors[2]

def get_pixel(x, y):
    region = {"left": x, "top": y, "width": 1, "height": 1}
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        b, g, r, a = frame[0][0]
        return r, g, b

def main():
    import pyautogui
    print("Move your mouse to the pixel you want. Press Ctrl+C to stop.\n")
    try:
        while True:
            x, y = pyautogui.position()
            r, g, b = get_pixel(x, y)
            print(f"x={x}, y={y} | rgb({r}, {g}, {b}) | hex=#{r:02x}{g:02x}{b:02x}", end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n\nFinal: x={x}, y={y} | rgb({r}, {g}, {b})")
        print(f"Coordinates tuple: ({x}, {y}, 1, 1)")

main()
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

def on_click(x, y, button, pressed):
    if pressed:
        r, g, b = get_pixel(x, y)
        print(f"\nClicked: x={x}, y={y} | rgb({r}, {g}, {b}) | hex=#{r:02x}{g:02x}{b:02x}")
        print(f"Coordinates tuple: ({x}, {y}, 1, 1)", flush=True)

def main():
    from pynput import mouse
    print("Click anywhere to get coordinates. Press Ctrl+C to stop.\n", flush=True)
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

main()
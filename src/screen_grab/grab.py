import mss
import numpy as np
from typing import Optional
import cv2


class ScreenGrab:
    def __init__(self, monitor: int = 1):
        self.monitor_num = monitor

    def process_greyscale(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    def grab(self, coordinates: Optional[tuple] = None, greyscale: bool = False) -> np.ndarray:
        with mss.mss() as sct:
            if coordinates:
                x, y, w, h = coordinates
                region = {"top": y, "left": x, "width": w, "height": h}
            else:
                region = sct.monitors[self.monitor_num]

            screenshot = sct.grab(region)
            frame = np.array(screenshot)
            if greyscale:
                frame = self.process_greyscale(frame)
            return frame
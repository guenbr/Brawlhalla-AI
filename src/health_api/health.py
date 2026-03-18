from health_api.health_constants import RGB_TO_PCT, X_RATIO_P1, X_RATIO_P2, Y_RATIO
from screen_grab.grab import ScreenGrab
import numpy as np
import mss
class HealthAPI():
    
    def __init__(self, monitor: int):
        self.monitor = monitor
        self.v = []
    def process_health(self):
        screen = ScreenGrab(self.monitor)
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor]
            w, h = monitor['width'], monitor['height']
        p1 = screen.grab(coordinates=(100, 127, 1, 1), greyscale=False)
        p2 = screen.grab(coordinates=(2500, 127, 1, 1), greyscale=False)

        self.v.append(RGB_TO_PCT[p1])
        print(self.v)

    def process_lives(self):
        pass

    def get_vector(self):
        return self.v

def main():
    test = HealthAPI(monitor=2)
    test.process_health()

main()
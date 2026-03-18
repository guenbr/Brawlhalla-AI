from screen_grab.grab import ScreenGrab
import numpy as np

class HealthAPI:
    def __init__(self, monitor: int):
        self.monitor = monitor
        self.v = np.array([])

    def process_health(self):
        screen = ScreenGrab(self.monitor)
        p1 = screen.grab(coordinates=(3853, 120, 1, 1), greyscale=False)
        p2 = screen.grab(coordinates=(3969, 127, 1, 1), greyscale=False)

        b1 = p1[0][0][0]
        b2 = p2[0][0][0]

        self.v = np.array([
            round((b1 / 255) * 100),
            round((b2 / 255) * 100)
        ])
        print(self.v)
    def process_lives(self):
        pass

    def get_vector(self):
        return self.v

def main():
    test = HealthAPI(monitor=2)
    test.process_health()

main()
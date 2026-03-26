from screen_grab.grab import ScreenGrab
import numpy as np
import time

class HealthAPI:
    def __init__(self, monitor: int):
        self.v = np.array([])
        self.screen = ScreenGrab(monitor=monitor)
        self.coord1 = (3853, 120, 1, 1)
        self.coord2 = (3969, 127, 1, 1)


    def process_health(self):
        p1 = self.screen.grab(coordinates=self.coord1, greyscale=False)
        p2 = self.screen.grab(coordinates=self.coord2, greyscale=False)

        b1, g1, r1 = p1[0][0][0], p1[0][0][1], p1[0][0][2]
        b2, g2, r2 = p2[0][0][0], p2[0][0][1], p2[0][0][2]

        h1 = self.__class__.rgb_to_health(r1, g1, b1)
        h2 = self.__class__.rgb_to_health(r2, g2, b2)

        self.v = np.array([h1, h2])

    @staticmethod
    def rgb_to_health(r, g, b):
        r, g, b = int(r), int(g), int(b)

        if r < 100 and g > 100 and b > 150:
            return 0
        elif r >= 250 and g >= 250 and b >= 250:
            return 100
        elif r >= 250 and g >= 220 and b < 200:
            return 85 + int((b / 200) * 15)
        elif r >= 250 and 150 <= g < 220 and b < 50:
            return 60 + int(((g - 150) / 70) * 25)
        elif r >= 250 and 40 <= g < 150 and b < 50:
            return 30 + int((g / 150) * 30)
        elif r >= 200 and g < 40 and b < 50:
            return 10 + int(((r - 200) / 55) * 20)
        elif r >= 150 and g < 40 and b < 50:
            return int((r / 255) * 10)
        else:
            return 0

    def get_vector(self):
        return self.v


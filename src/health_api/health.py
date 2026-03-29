from screen_grab.grab import ScreenGrab
import numpy as np
import time


class HealthAPI:
    def __init__(self, monitor: int):
        self.v = np.array([])
        self.screen = ScreenGrab(monitor=monitor)
        self.coord1 = (3853, 120, 1, 1)
        self.coord2 = (3969, 127, 1, 1)
        self.last_valid_health_p1 = 100
        self.last_valid_health_p2 = 100

    def process_health(self):
        p1 = self.screen.grab(coordinates=self.coord1, greyscale=False)
        p2 = self.screen.grab(coordinates=self.coord2, greyscale=False)

        b1, g1, r1 = p1[0][0][0], p1[0][0][1], p1[0][0][2]
        b2, g2, r2 = p2[0][0][0], p2[0][0][1], p2[0][0][2]

        h1 = self.rgb_to_health(r1, g1, b1, player=1)
        h2 = self.rgb_to_health(r2, g2, b2, player=2)

        self.v = np.array([h1, h2])

    def rgb_to_health(self, r, g, b, player):
        r, g, b = int(r), int(g), int(b)

        if r < 100 and g > 100 and b > 150:
            if player == 1:
                self.last_valid_health_p1 = 0
            else:
                self.last_valid_health_p2 = 0
            return 0

        if r < 200:
            if player == 1:
                return self.last_valid_health_p1
            else:
                return self.last_valid_health_p2

        g_norm = g / 255.0
        b_norm = b / 255.0
        health = (g_norm * 0.5 + b_norm * 0.5) * 100
        health = max(1, min(100, int(health)))

        if player == 1:
            self.last_valid_health_p1 = health
        else:
            self.last_valid_health_p2 = health

        return health

    def get_vector(self):
        return self.v


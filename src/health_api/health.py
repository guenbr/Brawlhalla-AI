from health_constants import PLAYER_ONE_HEALTH_BAR_REGION, PLAYER_TWO_HEALTH_BAR_REGION
from health_constants import RGB_TO_PCT
from screen_grab.grab import ScreenGrab
import numpy as np

class HealthAPI():
    
    def __init__(self, monitor: int):
        self.monitor = monitor
        self.v = np.array()
    def process_health(self):
        screen = ScreenGrab(self.monitor)

        p1 = screen.grab(coordinates=PLAYER_ONE_HEALTH_BAR_REGION, greyscale=False)
        p2 = screen.grab(coordinates=PLAYER_TWO_HEALTH_BAR_REGION, greyscale=False)

        self.v.append(RGB_TO_PCT[p1], RGB_TO_PCT[p2])

    def process_lifes(self):
        pass

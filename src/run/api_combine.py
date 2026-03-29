from screen_grab.grab import ScreenGrab
from health_api.health import HealthAPI
#from controls.controls import Controls
import numpy as np
import time


monitor = 0
SCREEN_GRAB = ScreenGrab(monitor=monitor)
HEALTH_API = HealthAPI()


def capture_frame():
    frames = []
    full_frame = None

    for i in range(4):
        full_frame = SCREEN_GRAB.grab(greyscale=False)

        game_area = SCREEN_GRAB.process_greyscale(full_frame[68:68 + 1313, 1472:1472 + 2541])
        frames.append(game_area)

    helper_vectors, is_game_over = get_helper_vectors(full_frame)

    stacked_frames = np.stack(frames, axis=0)
    return stacked_frames, helper_vectors, is_game_over

def get_helper_vectors(frame):
    health_vector, game_over, winner, confidences = HEALTH_API.process_frame(frame)

    # get the x, y coorindates, combine that with the health vector, and total lives left
    # to form matrix
    #return health_vector, is_game_over

def main():
    while True:
        x, y, z = capture_frame()
        print(y, z)
        time.sleep(1)
# p1 is ember, p2 is onyx
main()
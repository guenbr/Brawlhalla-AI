from src.screen_grab.grab import ScreenGrab
from src.health_api.health import HealthAPI
from src.controls.controls import Controls
import numpy as np
import time


MONITOR = 1
STARTING_LIVES = 99

SCREEN_GRAB = ScreenGrab(monitor=MONITOR)
HEALTH_API = HealthAPI(starting_lives=STARTING_LIVES)


def capture_frame():
    full_frame = SCREEN_GRAB.grab(greyscale=False)

    # Process the single frame for stacking
    game_area = SCREEN_GRAB.process_greyscale(full_frame[1:1428, 70:2402])

    # Stack the same frame 4 times (or use a deque to maintain history)
    stacked_frames = np.stack([game_area] * 4, axis=0)

    health_data, is_player_dead, is_game_over = get_helper_vectors(full_frame)

    return stacked_frames, health_data, is_player_dead, is_game_over

def get_helper_vectors(frame):
    health_vector, is_player_dead, winner, lives, is_game_over = HEALTH_API.process_frame(frame)
    # is player dead and is game over used for control sleeping, dont input when is player dead, and call reset game when is game over
    health_data = np.stack([health_vector, lives], axis=0).T
    # get the x, y coorindates, combine that with the health vector, and total lives left
    # to form matrix
    return health_data, is_player_dead, is_game_over


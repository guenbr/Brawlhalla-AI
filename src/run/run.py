from api_combine import *
from src.controls.controls import Controls
import time

CONTROLS = Controls()

def run(episodes):

    for i in range(episodes):
        stacked_frames, health_data, is_player_dead, is_game_over = capture_frame()
        if is_game_over:
            CONTROLS.reset_game()
        elif is_player_dead:
            time.sleep(4.2)






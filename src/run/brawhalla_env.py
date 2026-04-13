import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from src.screen_grab.grab import ScreenGrab
from src.health_api.health import HealthAPI
from src.controls.controls import Controls
import time
from src.player_location.player_detector import PlayerDetector
from src.global_vars import MONITOR


class BrawlhallaEnv:

    def __init__(self, starting_lives, monitor=MONITOR, frame_skip=2, data_size=14):
        self.screen             = ScreenGrab(monitor=monitor)
        self.health_api         = HealthAPI(starting_lives=starting_lives)
        self.player_detector    = PlayerDetector(monitor=monitor)
        self.controls           = Controls()
        self.starting_lives     = starting_lives
        self.prev_health        = np.array([100.0, 100.0])
        self.frame_skip         = frame_skip
        self.first_reset        = True
        self.recent_actions     = deque(maxlen=20)
        self.episode_start_time = None
        self.prev_combined_data = np.zeros(14, dtype=np.float32)  # flat shape
        self.eight_val = True if data_size==8 else False

    def reset(self):
        if self.first_reset:
            self.first_reset = False
        else:
            print("\nResetting game")
            self.controls.release_all()
            self.controls.reset_game()
            self.health_api.health               = np.array([100.0, 100.0])
            self.health_api.lives                = np.array([self.starting_lives,
                                                              self.starting_lives])
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            print("Game reset complete")

        self.prev_health        = np.array([100.0, 100.0])
        self.recent_actions.clear()
        self.episode_start_time = time.time()
        combined_data, _, _ = self.capture_frame()
        self.prev_combined_data = combined_data.copy()

        self.reward_components = {
            'damage_dealt': 0.0,
            'damage_taken': 0.0,
            'time_penalty': 0.0,
            'offstage_penalty': 0.0,
            'suicide_penalty': 0.0,
            'death_penalty': 0.0,
            'kill_reward': 0.0,
        }

        return combined_data

    def capture_frame(self):
        # Grab non-grey scaled frame for downstream API processing
        full_frame = self.screen.grab(greyscale=False)

        # Pass frame into health and player location API's
        health_vector, is_player_dead, winner, lives, is_game_over = \
            self.health_api.process_frame(full_frame)
        location_matrix = self.player_detector.get_positions(full_frame)

        # Normalize health and live values, easier on NN
        normalized_health = health_vector / 100.0
        normalized_lives = lives / float(self.starting_lives)

        location_matrix = np.array(location_matrix, dtype=np.float32)
        location_matrix[:, 0] = location_matrix[:, 0] / 2560.0
        location_matrix[:, 1] = location_matrix[:, 1] / 1440.0
        location_matrix = np.clip(location_matrix, 0, 1)

        if (lives[0] <= 0 or lives[1] <= 0) and not is_game_over:
            print("Forcing game over (a player's lives reached 0)")
            is_game_over = True

        PLATFORM_LEFT  = 0.319
        PLATFORM_RIGHT = 0.678
        PLATFORM_Y     = 0.581

        p1_x, p1_y = location_matrix[0, 0], location_matrix[0, 1]
        p2_x, p2_y = location_matrix[1, 0], location_matrix[1, 1]

        dx = p2_x - p1_x
        dy = p2_y - p1_y
        dist_to_opponent = np.sqrt(dx ** 2 + dy ** 2)
        dist_left_edge   = p1_x - PLATFORM_LEFT
        dist_right_edge  = PLATFORM_RIGHT - p1_x
        on_platform      = float((dist_left_edge > 0) and (dist_right_edge > 0) and (p1_y <= PLATFORM_Y))

        derived = np.array([
            dx, dy,
            dist_to_opponent,
            dist_left_edge,
            dist_right_edge,
            on_platform,
        ], dtype=np.float32)

        scraped_data = np.stack([normalized_health, normalized_lives], axis=0).T
        # 2D combined for reference (2, 4): [health, lives, x, y] per player
        combined_2d  = np.concatenate([scraped_data, location_matrix], axis=1)

        # Flat layout (14,): [h0,l0,x0,y0, h1,l1,x1,y1, dx,dy,dist,dist_left,dist_right,on_platform]
        if self.eight_val:
            combined_data = combined_2d.flatten()
        else:
            combined_data = np.concatenate([combined_2d.flatten(), derived])
        return combined_data, is_player_dead, is_game_over

    def step(self, action):
        total_reward = 0

        # Execute the action using Controls Class
        for _ in range(self.frame_skip):
            self.controls.execute_action(action)
            time.sleep(0.0089)

        # Capture a new frame
        combined_data, is_player_dead, is_game_over = self.capture_frame()

        # Unnormalize health and lives remaining values for reward processing
        health = combined_data[[0, 4]] * 100.0
        lives  = combined_data[[1, 5]] * float(self.starting_lives)

        # Calculate reward
        total_reward += self.calculate_reward(health, lives, is_player_dead,
                                              is_game_over, combined_data, action)

        # Some additional logic for detecting if two players died simultaneously
        if is_player_dead:
            snap_p1 = int(lives[0])
            snap_p2 = int(lives[1])
            # Check for 2.6 seconds, the time length of a respawn
            for check_num in range(26):
                time.sleep(0.1)
                full_frame = self.screen.grab(greyscale=False)
                _, temp_dead, _, temp_lives_raw, _ = \
                    self.health_api.process_frame(full_frame)

                if self.health_api.is_game_over():
                    break
                # if someone died in another players respawn period
                if temp_dead:
                    cur_p1 = int(temp_lives_raw[0])
                    cur_p2 = int(temp_lives_raw[1])
                    if cur_p1 < snap_p1 or cur_p2 < snap_p2:
                        # Processs reward and etc. again
                        temp_health = self.health_api.health.copy()
                        add_r = self.calculate_reward(
                            temp_health, temp_lives_raw, True, False, combined_data, action)
                        total_reward += add_r
                        health  = temp_health
                        lives   = temp_lives_raw
                        snap_p1 = cur_p1
                        snap_p2 = cur_p2
            # Reset necessary game state variables back to full
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            self.prev_combined_data = combined_data.copy()
            self.prev_health        = np.array([100.0, 100.0])
        else:
            self.prev_health = health.copy()

        info = {'health': health, 'lives': lives,
                'winner': None, 'is_player_dead': is_player_dead}
        return combined_data, total_reward, is_game_over, info

    def calculate_reward(self, health, lives, is_player_dead, is_game_over, combined_data, action) -> float:
        reward = 0

        # Calculate health difference from previous frame
        health_diff  = health - self.prev_health
        damage_dealt = abs(health_diff[1]) if health_diff[1] < 0 else 0
        damage_taken = abs(health_diff[0]) if health_diff[0] < 0 else 0

        p1_x = combined_data[2]
        p1_y = combined_data[3]

        # Platform variable, values in vector are normalized, and this fits to screen ratio
        on_platform = (0.319 < p1_x < 0.678) and (p1_y <= 0.581)

        offstage_pen = -.6 if not on_platform else 0.0
        # Reward damage a little higher than taking, encourages fighting
        dealt_r = damage_dealt * 0.05
        taken_r = -(damage_taken * 0.025)

        reward += offstage_pen + dealt_r + taken_r

        # Used for CSV metrics save, see where majority of rewrads came from
        self.reward_components['offstage_penalty'] += offstage_pen
        self.reward_components['damage_dealt']     += dealt_r
        self.reward_components['damage_taken']     += taken_r

        if is_player_dead:
            print(f'  P1 death | damage_taken={damage_taken:.1f}')
            # Punish suicides more than normal deaths, leads to less falling offstage
            if damage_taken > 50:
                reward -= 10
                self.reward_components['suicide_penalty'] -= 10
                print('suicide reward')
            if health[0] <= 1:
                reward -= 3
                self.reward_components['death_penalty'] -= 3
            # Massively reward kills, leads to better objective reward
            if health[1] <= 1:
                reward += 40
                self.reward_components['kill_reward'] += 10
                print('kill reward')
        # Set state var to current, used for diff for next calculation
        self.prev_combined_data = combined_data.copy()
        return reward
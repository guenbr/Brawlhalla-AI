import numpy as np
import cv2
import time
import threading
from collections import deque
from screen_grab.grab import ScreenGrab
from player_location.player_detector import PlayerDetector
from health_api.health import HealthAPI
from controls.controls import Controls

MONITOR = 2
STARTING_LIVES = 15
OBS_SIZE = 8

ACTION_NAMES = ['neutral', 'move_left', 'move_right', 'jump', 'light', 'heavy', 'dodge',
                'left_heavy', 'right_heavy', 'left_light', 'right_light']
NUM_ACTIONS = len(ACTION_NAMES)


class DeathWatcher:
    """Polls the screen in a background thread and sets flags when a player dies.

    Args:
        health_api: HealthAPI instance used for template matching.
        screen: ScreenGrab instance to capture frames from.
        poll_interval: Seconds between each poll (default ~30fps).
    """

    def __init__(self, health_api, screen, poll_interval=0.033):
        self.health_api = health_api
        self.screen = screen
        self.poll_interval = poll_interval

        self.p1_died = False
        self.p2_died = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Starts the background polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the background polling thread."""
        self._running = False

    def consume(self):
        """Returns current death flags and resets them to False.

        Returns:
            Tuple (p1_died, p2_died) as booleans.
        """
        with self._lock:
            p1 = self.p1_died
            p2 = self.p2_died
            self.p1_died = False
            self.p2_died = False
        return p1, p2

    def _run(self):
        """Main loop — grabs a frame and checks for death templates."""
        while self._running:
            try:
                full_frame = self.screen.grab(greyscale=False)
                # must convert BGRA to BGR before template matching
                frame_bgr = cv2.cvtColor(full_frame, cv2.COLOR_BGRA2BGR)

                p1_matched, _ = self.health_api.check_template_match('game_end_p1', frame_bgr)
                p2_matched, _ = self.health_api.check_template_match('game_end_p2', frame_bgr)

                if p1_matched or p2_matched:
                    with self._lock:
                        if p1_matched:
                            self.p1_died = True
                        if p2_matched:
                            self.p2_died = True

            except Exception as e:
                print(f"[DeathWatcher] error: {e}")

            time.sleep(self.poll_interval)


class BrawlhallaEnv:
    """RL environment wrapper for Brawlhalla using an 8-value observation vector.

    Observation vector (normalized to [0, 1]):
        [p1_health, p1_lives, p1_x, p1_y, cpu_health, cpu_lives, cpu_x, cpu_y]
    """

    def __init__(self):
        """Sets up screen, health API, player detector, controls, and death watcher."""
        self.screen = ScreenGrab(monitor=MONITOR)
        self.health_api = HealthAPI(starting_lives=STARTING_LIVES)
        self.detector = PlayerDetector(monitor=MONITOR)
        self.controls = Controls()
        self.starting_lives = STARTING_LIVES

        frame = self.screen.grab(greyscale=False)
        self.frame_h, self.frame_w = frame.shape[:2]

        self.prev_health = np.array([100.0, 100.0])
        self.first_reset = True
        self.recent_actions = deque(maxlen=20)
        self.prev_combined_data = np.zeros((2, 4), dtype=np.float32)

        # running stats for reward normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_n = 0

        self.death_watcher = DeathWatcher(self.health_api, self.screen)
        self.death_watcher.start()

    def reset(self):
        """Resets the environment for a new episode, skipping reset on the very first call.

        Returns:
            combined_data: (2, 4) numpy array with normalized health, lives, x, y.
        """
        if self.first_reset:
            self.first_reset = False
        else:
            print("\nResetting game")
            self.controls.release_all()
            self.controls.reset_game()
            self.health_api.health = np.array([100.0, 100.0])
            self.health_api.lives = np.array([self.starting_lives,
                                              self.starting_lives])
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            print("Game reset complete")

        self.prev_health = np.array([100.0, 100.0])
        self.recent_actions.clear()
        self.death_watcher.consume()  # clear any stale flags from last episode

        combined_data, _ = self.capture_frame()
        self.prev_combined_data = combined_data.copy()
        return combined_data

    def capture_frame(self):
        """Grabs a screen frame and builds the normalized observation matrix.

        Returns:
            combined_data: (2, 4) array — [health, lives, x, y] for each player.
            is_game_over: bool, True if the match is over.
        """
        full_frame = self.screen.grab(greyscale=False)

        health_vector, lives, is_game_over = \
            self.health_api.process_frame(full_frame)

        location_matrix = self.detector.get_positions(full_frame)

        normalized_health = health_vector / 100.0
        normalized_lives = lives / float(self.starting_lives)

        location_matrix = np.array(location_matrix, dtype=np.float32)
        location_matrix[:, 0] = location_matrix[:, 0] / 2560
        location_matrix[:, 1] = location_matrix[:, 1] / 1440
        location_matrix = np.clip(location_matrix, 0, 1)

        if (lives[0] <= 0 or lives[1] <= 0) and not is_game_over:
            print("Forcing game over (lives reached 0)")
            is_game_over = True

        scraped_data = np.stack([normalized_health, normalized_lives], axis=0).T
        combined_data = np.concatenate([scraped_data, location_matrix], axis=1)

        return combined_data, is_game_over

    def get_obs(self, combined_data: np.ndarray) -> np.ndarray:
        """Flattens combined_data into a 1D observation vector for the policy.

        Args:
            combined_data: (2, 4) numpy array.

        Returns:
            1D float32 array of length OBS_SIZE.
        """
        return combined_data.flatten().astype(np.float32)

    def _normalise_reward(self, r: float) -> float:
        """Scales the reward using a running mean/variance estimate.

        Args:
            r: Raw reward value.

        Returns:
            Clipped normalized reward in [-10, 10].
        """
        self._reward_n += 1
        delta = r - self._reward_mean
        self._reward_mean += delta / self._reward_n
        self._reward_var += delta * (r - self._reward_mean)
        std = max(np.sqrt(self._reward_var / max(self._reward_n, 1)), 1.0)
        return np.clip(r / std, -10.0, 10.0)

    def step(self, action: int):
        """Executes an action, captures the result, and computes the reward.

        Args:
            action: Integer action index from ACTION_NAMES.

        Returns:
            obs: Flattened 1D observation array.
            combined_data: (2, 4) raw observation matrix.
            total_reward: Normalized scalar reward.
            is_game_over: bool.
            info: Dict with health, lives, is_player_dead.
        """
        total_reward = 0

        for _ in range(2):
            self.controls.execute_action(action)
            time.sleep(0.0089)

        combined_data, is_game_over = self.capture_frame()

        # death watcher is the only source of death detection
        p1_died, p2_died = self.death_watcher.consume()
        is_player_dead = p1_died or p2_died

        if is_player_dead:
            current_time = time.time()
            if p1_died and (
                    current_time - self.health_api.last_death_time_p1) > self.health_api.death_cooldown:
                self.health_api.lives[0] -= 1
                self.health_api.last_death_time_p1 = current_time
                print(f"  P1 DIED | Lives: {int(self.health_api.lives[0])}")
            if p2_died and (
                    current_time - self.health_api.last_death_time_p2) > self.health_api.death_cooldown:
                self.health_api.lives[1] -= 1
                self.health_api.last_death_time_p2 = current_time
                print(f"  CPU DIED | Lives: {int(self.health_api.lives[1])}")
            if self.health_api.lives[0] <= 0 or self.health_api.lives[1] <= 0:
                is_game_over = True

        health = combined_data[:, 0] * 100.0
        lives = combined_data[:, 1] * float(self.starting_lives)

        total_reward += self.calculate_reward(
            health, lives, is_player_dead, is_game_over, combined_data, action)

        # wait out the death animation and catch any deaths that happen during respawn
        if is_player_dead:
            snap_p1 = int(self.health_api.lives[0])
            snap_p2 = int(self.health_api.lives[1])
            print("Death detected — monitoring respawn period...")

            for check_num in range(26):
                time.sleep(0.1)

                if self.health_api.is_game_over():
                    is_game_over = True
                    break

                pw1, pw2 = self.death_watcher.consume()
                if pw1 or pw2:
                    current_time = time.time()
                    if pw1 and (
                            current_time - self.health_api.last_death_time_p1) > self.health_api.death_cooldown:
                        self.health_api.lives[0] -= 1
                        self.health_api.last_death_time_p1 = current_time
                    if pw2 and (
                            current_time - self.health_api.last_death_time_p2) > self.health_api.death_cooldown:
                        self.health_api.lives[1] -= 1
                        self.health_api.last_death_time_p2 = current_time
                    print(f"  Additional death | P1:{snap_p1}->{int(self.health_api.lives[0])}  "
                          f"CPU:{snap_p2}->{int(self.health_api.lives[1])}")
                    snap_p1 = int(self.health_api.lives[0])
                    snap_p2 = int(self.health_api.lives[1])

            # reset health tracking so next episode starts fresh
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            self.prev_combined_data = combined_data.copy()
            self.prev_health = np.array([100.0, 100.0])
        else:
            self.prev_health = health.copy()

        total_reward = self._normalise_reward(total_reward)

        obs = self.get_obs(combined_data)
        info = {
            'health': health,
            'lives': lives,
            'is_player_dead': is_player_dead
        }
        return obs, combined_data, total_reward, is_game_over, info

    def calculate_reward(self, health, lives, is_player_dead,
                         is_game_over, combined_data, action) -> float:
        """Computes the reward for the current step based on damage, position, and match outcome.

        Args:
            health: Array [p1_health, cpu_health] in raw 0-100 scale.
            lives: Array [p1_lives, cpu_lives].
            is_player_dead: True if a death was detected this step.
            is_game_over: True if the match ended.
            combined_data: (2, 4) observation matrix.
            action: Integer action index taken this step.

        Returns:
            Scalar float reward.
        """
        reward = 0.0

        p1_x = combined_data[0, 2]
        p1_y = combined_data[0, 3]
        p2_x = combined_data[1, 2]
        p2_y = combined_data[1, 3]

        dist = abs(p1_x - p2_x)
        on_platform = 0.32 < p1_x < 0.68 and p1_y < 0.59
        cpu_on_platform = 0.32 < p2_x < 0.68 and p2_y < 0.59
        both_on_platform = on_platform and cpu_on_platform

        health_diff = health - self.prev_health
        damage_dealt = abs(health_diff[1]) if health_diff[1] < 0 else 0
        damage_taken = abs(health_diff[0]) if health_diff[0] < 0 else 0

        cpu_is_left = p2_x < p1_x
        cpu_is_right = p2_x > p1_x

        # true if the directional attack is aimed toward the cpu
        attacked_toward_cpu = (
                (action == 7 and cpu_is_left) or
                (action == 8 and cpu_is_right) or
                (action == 9 and cpu_is_left) or
                (action == 10 and cpu_is_right)
        )
        attacked_away_from_cpu = (
                (action == 7 and cpu_is_right) or
                (action == 8 and cpu_is_left) or
                (action == 9 and cpu_is_right) or
                (action == 10 and cpu_is_left)
        )

        # damage — scale up reward when both players are close on platform
        if both_on_platform:
            proximity_bonus = max(1.0, 3.0 - dist * 6.0)
            direction_mult = 1.5 if attacked_toward_cpu else 1.0
            reward += damage_dealt * 3.0 * proximity_bonus * direction_mult
        else:
            reward += damage_dealt * 1.0

        reward -= damage_taken * 0.5

        # reward attacking toward cpu, penalize attacking away
        if both_on_platform and dist < 0.25:
            if attacked_toward_cpu:
                reward += 0.3
            elif attacked_away_from_cpu:
                reward -= 0.1

        # small bonus for being close to the cpu on platform
        if both_on_platform:
            closeness = max(0.0, 1.0 - dist * 4.0)
            reward += closeness * 0.8

        # reward moving closer to the cpu
        prev_dist = abs(self.prev_combined_data[0, 2] - self.prev_combined_data[1, 2])
        closing = prev_dist - dist
        reward += closing * 1.5

        # small reward for staying on platform, penalty for being off
        if on_platform:
            reward += 0.03
        else:
            reward -= 0.05

        # big reward for killing cpu, penalty for dying
        if is_player_dead:
            if health[1] <= 1:
                reward += 40.0
                print(f"  P1 GOT A KILL | CPU lives: {int(lives[1])}")
            if health[0] <= 1:
                reward -= 10.0
                print(f"  P1 DIED | Lives: {int(lives[0])}")

        # match win/loss/draw bonus at end of episode
        if is_game_over:
            p1, p2 = int(lives[0]), int(lives[1])
            if p1 > p2:
                reward += 60.0 + (p1 - p2) * 5.0
                print(f"  EPISODE WIN  | P1: {p1} CPU: {p2}")
            elif p2 > p1:
                reward -= 30.0
                print(f"  EPISODE LOSS | P1: {p1} CPU: {p2}")
            else:
                print(f"  EPISODE DRAW | Both: {p1}")

        self.prev_combined_data = combined_data.copy()

        return reward

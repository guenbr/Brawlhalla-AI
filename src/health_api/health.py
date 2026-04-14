import numpy as np
import time
import cv2


class HealthAPI:
    """
    Handles extracting health information from a screenshot of a game state
    """
    def __init__(self, starting_lives: int):
        # Initialize varibales
        self.health = np.array([100, 100])
        # Pixel to monitor for health status
        self.coord1 = (2383, 119, 1, 1)
        self.coord2 = (2507, 114, 1, 1)
        self.last_valid_health_p1 = 100
        self.last_valid_health_p2 = 100

        self.last_death_time_p1 = 0
        self.last_death_time_p2 = 0
        self.death_cooldown = 2.0

        # Templates used for detecting death states
        self.templates = {
            'game_end_p2': self.__class__.load_template('../health_api/templates/p1_death_template.png'),
            'game_end_p1': self.__class__.load_template('../health_api/templates/p2_death_template.png')
        }

        # Region to monitor for death state
        self.regions = {
            #'game_end_p1': (2305, 50, 83, 60),
            'game_end_p1': (2305, 50, 84, 71),
            #'game_end_p2': (2431, 50, 84, 71),
            'game_end_p2': (2429, 50, 83, 58),
        }

        # Confidence matching thresholds for detecting death states
        self.thresholds = {
            'game_end_p1': 0.6,
            'game_end_p2': 0.6
        }
        self.lives = np.array([starting_lives, starting_lives])

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool, str, np.ndarray, bool]:
        """
        Process a game frame to extract health, death status, and game state

        Args:
            frame (np.ndarray): colored game frame image as numpy array

        Returns:
            tuple containing:
                - health (np.ndarray): array of [p1_health, p2_health]
                - player_dead (bool): whether a player died this frame
                - winner (str): winner identifier if game over
                - lives (np.ndarray): array of [p1_lives, p2_lives]
                - is_game_over (bool): whether the game has ended
        """
        # Extract RGB pixel from matrix
        p1 = frame[self.coord1[1], self.coord1[0]]
        p2 = frame[self.coord2[1], self.coord2[0]]

        # Extract each layer of RGB
        b1, g1, r1 = p1[0], p1[1], p1[2]
        b2, g2, r2 = p2[0], p2[1], p2[2]

        # Convert to numerical scale, 0-100
        h1 = self.rgb_to_health(r1, g1, b1, player=1)
        h2 = self.rgb_to_health(r2, g2, b2, player=2)

        # Store
        self.health = np.array([h1, h2])

        # Determine other game state variables
        player_dead, winner, confidences = self.is_player_dead(frame)
        is_game_over = self.is_game_over()

        return self.health, player_dead, winner, self.lives, is_game_over

    def rgb_to_health(self, r: int, g: int, b: int, player: int) -> int:
        """
        Applies formula to the color of a health pixel to convert to numerical value from 1-100

        Args:
            r (int): red channel value (0-255)
            g (int): green channel value (0-255)
            b (int): blue channel value (0-255)
            player (int): player number (1 or 2)

        Returns:
            int: health value from 0-100
        """
        r, g, b = int(r), int(g), int(b)

        # Apply formula to convert RGB values to numerical from 0-100
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

    @staticmethod
    def load_template(filepath: str) -> np.ndarray | None:
        """
        Load and preprocess a template image for matching

        Args:
            filepath (str): path to template image file

        Returns:
            np.ndarray | None: grayscale template image as numpy array, or None if loading fails
        """
        try:
            template = cv2.imread(filepath)
            if template is None:
                return None
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            return template
        except:
            return None

    def check_template_match(self, state_name: str, frame: np.ndarray) -> tuple[bool, float]:
        """
        Check if a template matches a region in the current frame

        Args:
            state_name (str): name of the game state to check
            frame (np.ndarray): current game frame as numpy array

        Returns:
            tuple containing:
                - matched (bool): whether template match exceeds threshold
                - max_val (float): confidence score of the match, 0.0-1.0
        """
        if state_name not in self.templates or self.templates[state_name] is None:
            return False, 0.0

        # Crop frame region to match template size
        x, y, width, height = self.regions[state_name]
        screen_region = frame[y:y + height, x:x + width]
        screen_region = cv2.cvtColor(screen_region, cv2.COLOR_BGR2GRAY)
        template = self.templates[state_name]

        # Check template match
        result = cv2.matchTemplate(screen_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # Matched = True if above given threshold for specific template
        threshold = self.thresholds[state_name]
        matched = max_val >= threshold

        return matched, max_val

    def is_player_dead(self, frame: np.ndarray) -> tuple[bool, str | None, tuple[float, float]]:
        """
        Check if any player has died in the current frame

        Args:
            frame (np.ndarray): current game frame as numpy array

        Returns:
            tuple containing:
                - player_dead (bool): whether any player died this frame
                - winner (str | None): 'p1', 'p2', 'draw', or None if no death
                - confidences (tuple[float, float]): match confidence scores for (p1, p2)
        """
        # Check if any player dead using template match
        p1_dead, p1_conf = self.check_template_match('game_end_p1', frame)
        p2_dead, p2_conf = self.check_template_match('game_end_p2', frame)

        # Want to set cooldown if player was discovered dead recently
        # Death template will match for 4 frames, so skip if recently found
        current_time = time.time()

        p1_in_cooldown = (current_time - self.last_death_time_p1) < self.death_cooldown
        p2_in_cooldown = (current_time - self.last_death_time_p2) < self.death_cooldown

        if p1_in_cooldown:
            p1_dead = False
        if p2_in_cooldown:
            p2_dead = False

        if not p1_dead and not p2_dead:
            return False, None, (p1_conf, p2_conf)

        winner = None

        # Determine winners/losers/draw and update metadata
        if p1_dead and p2_dead:
            winner = 'draw'
            self.health[0] = 0
            self.health[1] = 0
            self.lives[0] -= 1
            self.lives[1] -= 1
            self.last_death_time_p1 = current_time
            self.last_death_time_p2 = current_time

        elif p1_dead:
            winner = 'p2'
            self.health[0] = 0
            self.lives[0] -= 1
            self.last_death_time_p1 = current_time

        elif p2_dead:
            winner = 'p1'
            self.health[1] = 0
            self.lives[1] -= 1
            self.last_death_time_p2 = current_time

        return True, winner, (p1_conf, p2_conf)

    def is_game_over(self) -> bool:
        """
        Check if game is over
        """
        if self.lives[0] == 0 or self.lives[1] == 0:
            return True
        return False
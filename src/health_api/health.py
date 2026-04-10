from screen_grab.grab import ScreenGrab
import numpy as np
import time
import cv2


class HealthAPI:
    """
    Handles extracting the health, lives, and death/game-over signals from the game screen

    This class works by: 
    - Sampling specific pixels on the screen to estimate the player's health.
    - Using template matching to detect when a player dies.  
    - Keeping track of remaining lives and the game state.
    - Also includes "cooldowns" to avoid the false detections.
    """

    def __init__(self, starting_lives: int):
        """
        Sets up the HealthAPI for a new match.

        Args: 
            starting_lives: How many lives each player starts with. 
        """
        # Current health values for both players (P1, P2) 
        self.health = np.array([100, 100])

        # Pixel coordinates used to sample health color from the screen
        self.coord1 = (2383, 119, 1, 1)
        self.coord2 = (2507, 114, 1, 1)

        # Store last valid heath values to handle noise
        self.last_valid_health_p1 = 100
        self.last_valid_health_p2 = 100

        # Track last time each player died 
        self.last_death_time_p1 = 0
        self.last_death_time_p2 = 0

        # Prevent multiple detections of the same death
        self.death_cooldown = 2.0  # seconds

        # Templates used to detech death screens through template matching
        self.templates = {
            'game_end_p1': self.__class__.load_template('../health_api/templates/p1_death_template.png'),
            'game_end_p2': self.__class__.load_template('../health_api/templates/p2_death_template.png')
        }
        
        # Screen regions where death indicators will appear 
        self.regions = {
            'game_end_p1': (2305, 50, 83, 60),
            'game_end_p2': (2431, 50, 84, 71),
        }

        # Minimum confidence score for the template match to count as a real detection
        # Match should be at least 69% similar to the template.
        # If it's too high  we'll miss the actual deaths and if too low there can be false positives
        self.thresholds = {
            'game_end_p1': 0.69,
            'game_end_p2': 0.69
        }

        # The remaining lives for both of the players 
        self.lives = np.array([starting_lives, starting_lives])

    def process_frame(self, frame):
        """
        Main function in order to process a single frame of the game 

        It reads both players' heath from their health bar pixels, checks if anyone just died 
        through the template matching, and updates the lives accordingly.

        Args: 
            frame: a full screenshot as a numpy array 
        
        Returns: 
            health: np.array([p1.health, p2_health]), values 0-100
            player_dead: True if anyone died in this frame
            winner: 'p1', 'p2', 'draw', or None if nobody died
            lives: np.array([p1.lives, p2.lives])
            is_game_over: True if either player has hit 0 lives 
        """

        # Grab the pixel values for both players' health bars
        p1 = frame[self.coord1[1], self.coord1[0]]
        p2 = frame[self.coord2[1], self.coord2[0]]

        # Get the RGB values 
        b1, g1, r1 = p1[0], p1[1], p1[2]
        b2, g2, r2 = p2[0], p2[1], p2[2]

        # Convert the pixel color to health value 
        h1 = self.rgb_to_health(r1, g1, b1, player=1)
        h2 = self.rgb_to_health(r2, g2, b2, player=2)

        self.health = np.array([h1, h2])

        # Detect if a player died in this frame 
        player_dead, winner, confidences = self.is_player_dead(frame)

        # Check if the match is over
        is_game_over = self.is_game_over()

        return self.health, player_dead, winner, self.lives, is_game_over

    def rgb_to_health(self, r, g, b, player):
        """
        Converts the RGB values from the health bar and into a numerical health estimate 

        This works by using the color gradient from the health bar. The health bar changes 
        color as the damage goes up, and the pixel that we sample reflects that. This is a 
        heuristic-based approach where: 

            - If the pixel looks blue green with low red, the player is at 0 health 
            - If red is low but doesn't match the 0 health color, the reading isn't trusted and so, 
            we will return the last known valid health value 
            - If not, then we'll calculate the health by looking at how much green and blue are present
            to guess the percentage and then keep that number between 1 and 100. 
        
        Args: 
            r, g, b: individual color channel values (0-255)
            player: 1 or 2, used to track last valid readings separately.

        Returns: 
            health value as an int between 0 and 100
        """

        r, g, b = int(r), int(g), int(b)

        # A very blueish green pixel with almost no red means the health bar is empty (0)
        if r < 100 and g > 100 and b > 150:
            if player == 1:
                self.last_valid_health_p1 = 0
            else:
                self.last_valid_health_p2 = 0
            return 0

        # If red is low but we didn't hit the 0-health case above, then the pixel is not trusted 
        # Return whatever we last knew to be true instead of a random value
        if r < 200:
            if player == 1:
                return self.last_valid_health_p1
            else:
                return self.last_valid_health_p2

        # Red is high and so, we'll calculate the health by looking at how much green and blue are present
        g_norm = g / 255.0
        b_norm = b / 255.0
        health = (g_norm * 0.5 + b_norm * 0.5) * 100

        # We'll make sure to keep that number between 1 and 100 
        health = max(1, min(100, int(health)))

        # Will save this as the last trusted reading or in other words, last valid health value
        if player == 1:
            self.last_valid_health_p1 = health
        else:
            self.last_valid_health_p2 = health

        return health

    @staticmethod
    def load_template(filepath):
        """
        Loads a template image for matching by converting it into grayscale

        Args: 
            filepath: path to template .png file
        
        Returns: 
            the template as a grayscale numpy array, or None if the loading failed
        """
        try:
            template = cv2.imread(filepath)
            if template is None:
                # returns None if the file doesn't exist or can't be read
                return None
            
            # Convert to grayscale if we got a color image 
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            return template
        except:
            return None

    def check_template_match(self, state_name, frame):
        """
        Checks whether the given template (such as the P1 death screen) appears in the frame

        Used for detecting "death" indicators on the screen

        Args: 
            state_name: key into self.templates and self.regions, 
                          e.g. 'game_end_p1' or 'game_end_p2'
            
            frame: the full screenshot as a numpy array 

        Returns: 
            matched: True if the template was found above the confidence threshold 
            max_val: the raw confidence score (0-1) 
        """

        # If we never loaded this template successfully, skip  
        if state_name not in self.templates or self.templates[state_name] is None:
            return False, 0.0

        # Crop the relevant region out of the full frame 
        x, y, width, height = self.regions[state_name]
        screen_region = frame[y:y + height, x:x + width]
        screen_region = cv2.cvtColor(screen_region, cv2.COLOR_BGR2GRAY)
        template = self.templates[state_name]

        # Run the template matching 
        result = cv2.matchTemplate(screen_region, template, cv2.TM_CCOEFF_NORMED)

        # Want the best match location's score
        _, max_val, _, _ = cv2.minMaxLoc(result)

        threshold = self.thresholds[state_name]
        matched = max_val >= threshold
        return matched, max_val

    def is_player_dead(self, frame):
        """
        Detect if either of the players just died this current frame

        Includes cooldown logic to avoid counting the same death multiple time. 

        - The death animation stays on screen for multiple frames, so without a cooldown 
          we'd count one death as many deaths 

        - If a player is on their last life, we skip the cooldown so that we don't accidentally miss 
        the final kill and fail to end the episode. 

        When a death is confirmed, we update the player's lives count and zero out their health.

        Args: 
            frame: the full screenshot as a numpy array
        
        Returns: 
            game_over: True if anyone died this frame
            winner: 'p1' if P1 got the kill, 'p2' if P2 did, 
                    'draw' if both died at the same time, None if no death
            confidences: the tuple of (p1_conf, p2_conf) to the match the scores
        """
        p1_dead, p1_conf = self.check_template_match('game_end_p1', frame)
        p2_dead, p2_conf = self.check_template_match('game_end_p2', frame)

        current_time = time.time()
        
        # Check if this would be a final death (game ending)
        would_be_final_p1 = self.lives[0] == 1
        would_be_final_p2 = self.lives[1] == 1

        # Check cooldowns independently (but skip cooldown if it's final death)
        p1_in_cooldown = (current_time - self.last_death_time_p1) < self.death_cooldown and not would_be_final_p1
        p2_in_cooldown = (current_time - self.last_death_time_p2) < self.death_cooldown and not would_be_final_p2

        # Ignore detections if in cooldown
        if p1_in_cooldown:
            p1_dead = False
        if p2_in_cooldown:
            p2_dead = False

        game_over = p1_dead or p2_dead

        if not game_over:
            return False, None, (p1_conf, p2_conf)

        # Figure out who died and update the state 
        winner = None
        if p1_dead and p2_dead:
            # Both the players dying on the same frame will count as a draw 
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

        return game_over, winner, (p1_conf, p2_conf)
    def is_game_over(self):
        """
        Checks whether the match has ended

        The match ends when either player runs out of lives and so, we'll check both of the players

        Returns
            True if either player is at 0 lives and False if not
        """
        if self.lives[0] == 0 or self.lives[1] == 0:
            return True
        return False
from src.screen_grab.grab import ScreenGrab
import numpy as np
import time
import cv2


class HealthAPI:
    def __init__(self, starting_lives: int):
        self.health = np.array([100, 100])
        self.coord1 = (2383, 119, 1, 1)
        self.coord2 = (2507, 114, 1, 1)
        self.last_valid_health_p1 = 100
        self.last_valid_health_p2 = 100

        self.templates = {
            'game_end_p1': self.__class__.load_template('../health_api/templates/p1_death_template.png'),
            'game_end_p2': self.__class__.load_template('../health_api/templates/p2_death_template.png')
        }

        self.regions = {
            'game_end_p1': (2305, 50, 83, 60),
            'game_end_p2': (2431, 50, 84, 71),
        }

        self.thresholds = {
            'game_end_p1': 0.9,
            'game_end_p2': 0.9
        }
        self.lives = np.array([starting_lives, starting_lives])

    def process_frame(self, frame):
        p1 = frame[self.coord1[1], self.coord1[0]]
        p2 = frame[self.coord2[1], self.coord2[0]]

        b1, g1, r1 = p1[0], p1[1], p1[2]
        b2, g2, r2 = p2[0], p2[1], p2[2]

        h1 = self.rgb_to_health(r1, g1, b1, player=1)
        h2 = self.rgb_to_health(r2, g2, b2, player=2)

        self.health = np.array([h1, h2])

        player_dead, winner, confidences = self.is_player_dead(frame)
        is_game_over = self.is_game_over()

        return self.health, player_dead, winner, self.lives, is_game_over

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

    @staticmethod
    def load_template(filepath):
        try:
            template = cv2.imread(filepath)
            if template is None:
                return None
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            return template
        except:
            return None

    def check_template_match(self, state_name, frame):
        if state_name not in self.templates or self.templates[state_name] is None:
            return False, 0.0

        x, y, width, height = self.regions[state_name]
        screen_region = frame[y:y + height, x:x + width]
        screen_region = cv2.cvtColor(screen_region, cv2.COLOR_BGR2GRAY)
        template = self.templates[state_name]

        result = cv2.matchTemplate(screen_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        threshold = self.thresholds[state_name]
        matched = max_val >= threshold

        return matched, max_val

    def is_player_dead(self, frame):
        p1_dead, p1_conf = self.check_template_match('game_end_p1', frame)
        p2_dead, p2_conf = self.check_template_match('game_end_p2', frame)

        game_over = p1_dead or p2_dead

        winner = None
        if p1_dead and not p2_dead:
            winner = 'p2'
            self.last_valid_health_p1 = 0
            self.health[0] = 0
            self.lives[0] -= 1
        elif p2_dead and not p1_dead:
            winner = 'p1'
            self.last_valid_health_p2 = 0
            self.health[1] = 0
            self.lives[1] -= 1
        elif p1_dead and p2_dead:
            winner = 'draw'
            self.last_valid_health_p1 = 0
            self.last_valid_health_p2 = 0
            self.health[0] = 0
            self.health[1] = 0
            self.lives[0] -= 1
            self.lives[1] -= 1

        return game_over, winner, (p1_conf, p2_conf)

    def is_game_over(self):
        if self.lives[0] == 0 or self.lives[1] == 0:
            return True
        return False
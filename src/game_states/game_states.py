import cv2
import numpy as np
from screen_grab.grab import ScreenGrab
from health_api.health import HealthAPI
import time


class GameStateDetector:
    def __init__(self, screen_grab, health_api):
        self.screen = screen_grab
        self.health_api = health_api
        self.templates = {
            'game_end': self.load_template('templates/results_template.png'),
        }

        self.regions = {
            'game_end': (3761, 38, 300, 150),
        }

        self.thresholds = {
            'game_end': 0.9,
        }

    def load_template(self, filepath):
        try:
            template = cv2.imread(filepath, 0)
            return template
        except:
            print(f"Warning: Could not load {filepath}")
            return None

    def check_template_match(self, state_name):

        if state_name not in self.templates or self.templates[state_name] is None:
            return False, 0.0

        x, y, width, height = self.regions[state_name]
        screen_region = self.screen.grab(coordinates=(x, y, width, height), greyscale=True)

        template = self.templates[state_name]

        result = cv2.matchTemplate(screen_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        threshold = self.thresholds[state_name]
        matched = max_val >= threshold

        return matched, max_val

    def is_game_over(self):
        return self.check_template_match('game_end')


def main():
    """Test the results screen detection"""
    print("=" * 60)
    print("RESULTS SCREEN DETECTOR TEST")
    print("=" * 60)

    # Initialize
    screen = ScreenGrab(monitor=0)
    health_api = HealthAPI(monitor=0)
    detector = GameStateDetector(screen, health_api)

    while True:
        is_results, confidence = detector.is_game_over()

        print(f'{is_results} + {confidence}')
        time.sleep(0.5)
if __name__ == "__main__":
    main()
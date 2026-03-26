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
            'results': self.load_template('templates/results_template.png'),
        }

        self.regions = {
            'results': (600, 30, 200, 60),
        }

        # Confidence thresholds
        self.thresholds = {
            'results': 0.7,
        }

    def load_template(self, filepath):
        """Load template image"""
        try:
            template = cv2.imread(filepath, 0)  # Grayscale
            return template
        except:
            print(f"Warning: Could not load {filepath}")
            return None

    def check_template_match(self, state_name):
        """
        Check if current screen matches a template

        Returns: (matched: bool, confidence: float)
        """
        if state_name not in self.templates or self.templates[state_name] is None:
            return False, 0.0

        # Grab screen region
        x, y, width, height = self.regions[state_name]
        screen_region = self.screen.grab(coordinates=(x, y, width, height), greyscale=True)

        # Template matching
        template = self.templates[state_name]
        result = cv2.matchTemplate(screen_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if confidence exceeds threshold
        threshold = self.thresholds[state_name]
        matched = max_val >= threshold

        return matched, max_val

    def is_results_screen(self):
        """
        Check if currently on results screen

        Returns: (is_results: bool, confidence: float)
        """
        return self.check_template_match('results')

    def wait_for_results(self, timeout=30):
        """
        Wait until results screen appears

        Args:
            timeout: max seconds to wait

        Returns: True if results screen reached, False if timeout
        """
        print(f"Waiting for results screen...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            is_results, confidence = self.is_results_screen()

            if is_results:
                print(f"✓ Results screen detected (confidence: {confidence:.2f})")
                return True

            time.sleep(0.5)

        print(f"✗ Timeout waiting for results screen")
        return False


def main():
    """Test the results screen detection"""
    print("=" * 60)
    print("RESULTS SCREEN DETECTOR TEST")
    print("=" * 60)

    # Initialize
    screen = ScreenGrab(monitor=1)
    health_api = HealthAPI(monitor=1)
    detector = GameStateDetector(screen, health_api)

    # Check if template loaded
    if detector.templates['results'] is None:
        print("\n❌ ERROR: Could not load results_template.png")
        print("Make sure templates/results_template.png exists!")
        return

    print("\n✓ Template loaded successfully")
    print(f"Template shape: {detector.templates['results'].shape}")

    # Continuous monitoring
    print("\nStarting continuous monitoring...")
    print("Navigate to different screens to test detection")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            is_results, confidence = detector.is_results_screen()

            if is_results:
                status = f"✓ RESULTS SCREEN DETECTED | Confidence: {confidence:.3f}"
            else:
                status = f"✗ Not results screen      | Confidence: {confidence:.3f}"

            print(status, end='\r')
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
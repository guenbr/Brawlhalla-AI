import time
import cv2
import numpy as np
import mss
from health_api.health import HealthAPI

STARTING_LIVES = 3


def capture_screen():
    """Grabs a screenshot from monitor 2 and returns it as a BGR frame."""
    with mss.mss() as sct:
        monitor = sct.monitors[2]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def main():
    """Continuously runs template matching and prints match results once per second."""
    api = HealthAPI(starting_lives=STARTING_LIVES)

    print("=== HealthAPI Template Match Tracker ===")
    print("Monitoring check_template_match continuously. Press Ctrl+C to stop.\n")

    last_print = 0

    while True:
        frame = capture_screen()

        p1_matched, p1_conf = api.check_template_match('game_end_p1', frame)
        p2_matched, p2_conf = api.check_template_match('game_end_p2', frame)

        # print status once per second
        current_time = time.time()
        if current_time - last_print >= 1.0:
            p1_status = "MATCHED" if p1_matched else "no match"
            p2_status = "MATCHED" if p2_matched else "no match"
            print(
                f"[P1] {p1_status:<10} conf={p1_conf:.4f}   |   [P2] {p2_status:<10} conf={p2_conf:.4f}")
            last_print = current_time

        if p1_matched or p2_matched:
            print()

        time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")

import numpy as np
import cv2
import os
from screen_grab.grab import ScreenGrab
from player import Player

PLAYER_ONE_ID = 0
PLAYER_TWO_ID = 1

# Minimum confidence needed to count a template match as a valid detection
MATCH_THRESHOLD = 0.6

# File paths relative to this file's location — works on both Mac and Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P1_TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "p1_label.png")
CPU_TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "cpu_label.png")


class PlayerDetector:
    def __init__(self, monitor: int):
        # Create player objects to hold each player's position
        self.player1 = Player(player_id=PLAYER_ONE_ID)
        self.player2 = Player(player_id=PLAYER_TWO_ID)

        # Screen grabber captures frames from the specified monitor
        self.screen = ScreenGrab(monitor=monitor)

        # Load templates and their cyan color masks for both labels
        self.p1_template, self.p1_mask  = self._load_template(P1_TEMPLATE_PATH)
        self.cpu_template, self.cpu_mask = self._load_template(CPU_TEMPLATE_PATH)

    @staticmethod
    def _load_template(path: str) -> tuple[np.ndarray, np.ndarray]:
        # Load the template image from disk
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load template: {path}")

        # Convert to HSV so we can isolate the cyan color of the label
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the HSV range for the cyan text color
        lower_cyan = np.array([80,  80,  80])
        upper_cyan = np.array([100, 255, 255])

        # Create a binary mask — white where cyan pixels are, black everywhere else
        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

        # Expand the mask slightly to include the text outline pixels
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Convert template to grayscale for template matching
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray, mask

    def _find_label(self, frame_bgr: np.ndarray,
                    template: np.ndarray,
                    mask: np.ndarray) -> tuple | None:
        h, w = frame_bgr.shape[:2]

        # Only scan the middle portion of the screen where players can be
        y_start = int(h * 0.15)
        y_end = int(h * 0.70)
        cropped = frame_bgr[y_start:y_end, :]

        # Convert cropped region to grayscale for matching
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Run template matching — mask makes it ignore the background behind the label
        result = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED, mask=mask)

        # Get the best match location and its confidence score
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= MATCH_THRESHOLD:
            th, tw = template.shape

            # Calculate center of the matched region
            cx = max_loc[0] + tw // 2

            # Add y_start back to convert from cropped to full frame coordinates
            cy = max_loc[1] + th // 2 + y_start
            return (cx, cy)

        return None

    def update(self, color_frame: np.ndarray | None = None):
        # Grab a fresh frame if one wasn't passed in
        if color_frame is None:
            color_frame = self.screen.grab(greyscale=False)

        # Convert from BGRA to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        # Search the frame for both player labels
        p1_pos = self._find_label(frame_bgr, self.p1_template, self.p1_mask)
        cpu_pos = self._find_label(frame_bgr, self.cpu_template, self.cpu_mask)

        # Only update position if the label was actually found this frame
        if p1_pos is not None:
            self.player1.update_position(p1_pos)
        if cpu_pos is not None:
            self.player2.update_position(cpu_pos)

    def get_positions(self) -> np.ndarray:
        # Returns a 2x2 matrix of player positions
        # Rows: [P1, CPU] — Columns: [x, y]
        # If a player hasn't been detected yet, their position defaults to (0, 0)
        p1_pos  = self.player1.position if self.player1.position is not None else (0, 0)
        cpu_pos = self.player2.position if self.player2.position is not None else (0, 0)
        return np.array([p1_pos, cpu_pos])

    def get_players(self) -> tuple[Player, Player]:
        # Return both player objects with their current state
        return self.player1, self.player2

    def debug_frame(self, color_frame: np.ndarray | None = None) -> np.ndarray:
        # Grab a frame if one wasn't provided
        if color_frame is None:
            color_frame = self.screen.grab(greyscale=False)

        frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        # Run detection for each player and draw a dot where they were found
        for template, mask, label, color in [
            (self.p1_template,  self.p1_mask,  "P1",  (0, 255, 0)),
            (self.cpu_template, self.cpu_mask, "CPU", (0, 0, 255)),
        ]:
            pos = self._find_label(frame_bgr, template, mask)
            print(f"{label} detected at: {pos}")
            if pos:
                cx, cy = pos
                # Draw a filled circle at the detected position
                cv2.circle(frame_bgr, (cx, cy), 6, color, -1)
                # Draw the label name next to the dot
                cv2.putText(frame_bgr, label, (cx + 8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame_bgr
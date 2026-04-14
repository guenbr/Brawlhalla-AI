import numpy as np
import cv2
import os
from src.screen_grab.grab import ScreenGrab
from player_location.player import Player

PLAYER_ONE_ID = 0
PLAYER_TWO_ID = 1

# Minimum confidence needed to count a template match as a valid detection
MATCH_THRESHOLD = 0.6

# File paths relative to this file's location — works on both Mac and Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P1_TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "Capture.PNG")
CPU_TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "Capture_1.PNG")


class PlayerDetector:
    """
    Detects and tracks player positions on screen using template matching

    Locates each player's name label in the game frame and maps it to an (x, y)
    screen coordinate, which is then passed to the environment as part of the game state
    """

    def __init__(self, monitor: int) -> None:
        """
        Initialize player objects, screen grabber, and load label templates

        Args:
            monitor (int): monitor index to capture frames from
        """
        # Create player objects to hold each player's position
        self.player1 = Player(player_id=PLAYER_ONE_ID)
        self.player2 = Player(player_id=PLAYER_TWO_ID)

        # Screen grabber captures frames from the specified monitor
        self.screen = ScreenGrab(monitor=monitor)

        # Load templates and their cyan color masks for both labels
        self.p1_template, self.p1_mask = self._load_template(P1_TEMPLATE_PATH)
        self.cpu_template, self.cpu_mask = self._load_template(CPU_TEMPLATE_PATH)

    @staticmethod
    def _load_template(path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a label template image and generate its cyan color mask

        Args:
            path (str): file path to the template image

        Returns:
            tuple containing:
                - gray (np.ndarray): grayscale template used for matching
                - mask (np.ndarray): binary mask isolating cyan label pixels
        """
        # Load the template image from disk
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load template: {path}")

        # Convert to HSV so we can isolate the cyan color of the label
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the HSV range for the cyan text color
        lower_cyan = np.array([80, 80, 80])
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
        """
        Search a frame for a player label using masked template matching

        Args:
            frame_bgr (np.ndarray): current game frame in BGR format
            template (np.ndarray): grayscale label template to search for
            mask (np.ndarray): binary mask isolating the label's cyan pixels

        Returns:
            tuple | None: (x, y) center of the matched label, or None if not found
        """
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

    def update(self, color_frame: np.ndarray | None = None) -> None:
        """
        Grab a frame and update both players' positions from label detections

        Args:
            color_frame (np.ndarray | None): frame to process, or None to capture a fresh one
        """
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

    def get_positions(self, color_frame: np.ndarray | None = None) -> np.ndarray:
        """
        Return a 2x2 matrix of current player positions

        Args:
            color_frame (np.ndarray | None): frame to process, or None to capture a fresh one

        Returns:
            np.ndarray: 2x2 array of positions, rows are [P1, CPU], columns are [x, y],
                        defaults to (0, 0) for any player not yet detected
        """
        # Grab a fresh frame and update player positions
        self.update(color_frame)

        # If a player hasn't been detected yet, their position defaults to (0, 0)
        p1_pos = self.player1.position if self.player1.position is not None else (0, 0)
        cpu_pos = self.player2.position if self.player2.position is not None else (0, 0)
        return np.array([p1_pos, cpu_pos])

    def get_players(self) -> tuple[Player, Player]:
        """
        Return both player objects with their current state

        Returns:
            tuple containing:
                - player1 (Player): P1 player object
                - player2 (Player): CPU player object
        """
        return self.player1, self.player2

    def debug_frame(self, color_frame: np.ndarray | None = None) -> np.ndarray:
        """
        Return an annotated frame with detected player positions drawn on it

        Args:
            color_frame (np.ndarray | None): frame to annotate, or None to capture a fresh one

        Returns:
            np.ndarray: BGR frame with colored dots and labels drawn at each player's position
        """
        # Grab a frame if one wasn't provided
        if color_frame is None:
            color_frame = self.screen.grab(greyscale=False)

        frame_bgr = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

        # Pass the same frame into get_positions so dots match what's displayed
        positions = self.get_positions(color_frame)

        # Draw dots for P1 and CPU using positions from the matrix
        for idx, (label, color) in enumerate([("P1", (0, 255, 0)), ("CPU", (0, 0, 255))]):
            x, y = positions[idx]
            print(f"{label} detected at: ({x}, {y})")
            if (x, y) != (0, 0):
                # Draw a filled circle at the detected position
                cv2.circle(frame_bgr, (x, y), 6, color, -1)
                # Draw the label name next to the dot
                cv2.putText(frame_bgr, label, (x + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame_bgr

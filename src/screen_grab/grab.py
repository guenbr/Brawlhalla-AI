import mss
import numpy as np
from typing import Optional
import cv2

# This class turns a screenshot of a current monitor and turns it into a matrix, in greyscale or RGB color
class ScreenGrab:
    """
    Used to grab a screenshot of the Brawhalla screen and turn it into a matrix
    """
    def __init__(self, monitor: int = 1):
        # Initialize with which monitor is running the game
        self.monitor_num = monitor
        self.sct = mss.mss()

    @staticmethod
    def process_greyscale(frame: np.ndarray) -> np.ndarray:
        """
        Turns RGB matrix into greyscale

        Args:
            frame (np.ndarray): screenshot in matrix form
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    def grab(self, coordinates: Optional[tuple] = None, greyscale: bool = False) -> np.ndarray:
        """
        Grabs a screenshot of the monitor and turns it into a matrix

        Args:
            coordinates (tuple): optional coordinates of the screen to grab
            greyscale (bool): toggle if matrix is in greyscale form or RGB form

        Returns:
            frame (np.ndarray): returns frame in matrix form
        """
        # Unpack coordinates if provided, else capture whole screen
        if coordinates:
            x, y, w, h = coordinates
            region = {"top": y, "left": x, "width": w, "height": h}
        else:
            region = self.sct.monitors[self.monitor_num]

        # Take screenshot, and convert to array
        screenshot = self.sct.grab(region)
        frame = np.array(screenshot)

        if greyscale:
            frame = self.__class__.process_greyscale(frame)

        return frame
import numpy as np
from player_api.player import Player
from health_api.health import HealthAPI
# Need to implement YOLO framework for live object detection

# Player ID
PLAYER_ONE_ID = 0
PLAYER_TWO_ID = 1

class PlayerDetector:
    def __init__(self, monitor: int, model_path: str):
        self.player1 = Player(player_id=0)
        self.player2 = Player(player_id=1)

        self.health_api = HealthAPI(monitor=monitor)
        # self.model = Call YOLO model

    def update(self, color_frame: np.ndarray):
        # Call YOLO to get frame and info (cordinates)

        # Need implementation here to get players position from screen
        
        # Then update playeres positon, calling player#.update_position(x, y)

        # Update health for each player
        p1_health, p2_health = self.health_api.process_health()
        if p1_health is not None:
            self.player1.update_health(p1_health)
        if p2_health is not None:
            self.player2.update_health(p2_health)

    def get_players(self) -> tuple[Player, Player]:
        return self.player1, self.player2
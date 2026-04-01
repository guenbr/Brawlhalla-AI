class Player:
    def __init__(self, player_id: int):
        # Since 1v1, player one's player_id = 0, player two's player_id = 1
        self.player_id = player_id

        # Current position as (x, y), starts as None until first detection
        self.position: tuple | None = None

        # Current health value (0-100), starts as None until first reading
        self.health: float | None = None

    def update_position(self, position: tuple):
        # Store the latest detected position of this player
        self.position = position

    def update_health(self, health: float):
        # Store the latest health reading for this player
        self.health = health

    def __repr__(self):
        # Prints a readable summary of the player's current state
        return (f"Player(id={self.player_id}, "
                f"position={self.position}, "
                f"health={self.health})")
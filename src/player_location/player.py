class Player:
    """
    Stores the current state of a single player including position and health
    """

    def __init__(self, player_id: int) -> None:
        # Since 1v1, player one's player_id = 0, player two's player_id = 1
        self.player_id = player_id

        # Current position as (x, y), starts as None until first detection
        self.position: tuple | None = None

        # Current health value (0-100), starts as None until first reading
        self.health: float | None = None

    def update_position(self, position: tuple) -> None:
        """
        Update the current position of this player

        Args:
            position (tuple): detected (x, y) position in screen coordinates
        """
        self.position = position

    def update_health(self, health: float) -> None:
        """
        Update the current health of this player

        Args:
            health (float): health value from 0-100
        """
        self.health = health

    def __repr__(self) -> str:
        # Returns a readable summary of the player's current state
        return (f"Player(id={self.player_id}, "
                f"position={self.position}, "
                f"health={self.health})")

class Player:
    def __init__(self, player_id: int):
        # Since 1v1, player ones player_id = 0, player twos player_id = 1
        self.player_id = player_id

        # Current position as (x, y), set to none init
        self.position: tuple | None = None

        # Current health value, set to none init
        self.health: float | None = None

    def update_position(self, position: tuple):
        self.position = position

    def update_health(self, health: float):
        self.health = health

    def __repr__(self):
        return (f"Player(id={self.player_id}, "
                f"position={self.position}, "
                f"health={self.health})")
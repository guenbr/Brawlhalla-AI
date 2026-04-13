import pyautogui
import time


class Controls:
    """
    Handles all the keyboard input sent to the Brawlhalla game window

    Simulates key presses using pyautogui to control the game based on the actions determined by the agent.

    Movement actions (left, right, up, down) are held for a short duration to simulate continuous movement,
    while attack and dodge actions are pressed briefly.
    """

    def __init__(self):
        """
        Sets up the keybind mapping for all in-game actions. The keys can be customized to match
        the player's preferred controls.
        """

        # Mapping of actions to their corresponding key presses
        self.keys = {
            'neutral': None,
            'light': 'j',
            'heavy': 'k',
            'dodge': 'l',
            'jump': 'w',
            'move_left': 'a',
            'move_right': 'd',
            'move_up': 'w',
            'move_down': 's',
        }

    def release_all(self):
        """
        Releases all keys
        """
        for key in set(self.keys.values()):
            if key is not None:
                try:
                    pyautogui.keyUp(key)
                except:
                    pass

    def execute_action(self, action_id):
        """
        Converts an action ID from the agent into a corresponding key press and executes it.

        This is what connects the agent's output to the actual key presses that control the game.

        Args:
            action_id: An integer representing the action chosen by the agent (e.g., 0 for 'neutral',
                       1 for 'move_left', etc.). The mapping of action IDs to action names is defined
                       in the action_map dictionary within this method.
        """
        # Mapping of action IDs to their corresponding action names
        action_map = {
            0:  'neutral',
            1:  'move_left',
            2:  'move_right',
            3:  'jump',
            4:  'light',
            5:  'heavy',
            6:  'dodge',
            7:  'left_heavy',
            8:  'right_heavy',
            9:  'left_light',
            10: 'right_light',
        }

        action = action_map.get(action_id, 'neutral')
        print(f"Executing action: {action}")

        if action == 'neutral':
            pass

        elif action == 'move_left':
            pyautogui.keyDown('a')
            time.sleep(0.1)
            pyautogui.keyUp('a')

        elif action == 'move_right':
            pyautogui.keyDown('d')
            time.sleep(0.1)
            pyautogui.keyUp('d')

        elif action == 'jump':
            pyautogui.keyDown('w')
            time.sleep(0.05)
            pyautogui.keyUp('w')

        elif action == 'light':
            pyautogui.keyDown('j')
            time.sleep(0.05)
            pyautogui.keyUp('j')

        elif action == 'heavy':
            pyautogui.keyDown('k')
            time.sleep(0.05)
            pyautogui.keyUp('k')

        elif action == 'dodge':
            pyautogui.keyDown('l')
            time.sleep(0.05)
            pyautogui.keyUp('l')

        elif action == 'left_heavy':
            # Hold left first, then press+hold heavy while direction is held,
            # then release both — game must see direction before attack input
            pyautogui.keyDown('a')
            time.sleep(0.05)        # let game register direction
            pyautogui.keyDown('k')
            time.sleep(0.1)         # hold long enough for Ssig to commit
            pyautogui.keyUp('k')
            pyautogui.keyUp('a')

        elif action == 'right_heavy':
            pyautogui.keyDown('d')
            time.sleep(0.05)
            pyautogui.keyDown('k')
            time.sleep(0.1)
            pyautogui.keyUp('k')
            pyautogui.keyUp('d')

        elif action == 'left_light':
            pyautogui.keyDown('a')
            time.sleep(0.05)
            pyautogui.keyDown('j')
            time.sleep(0.1)
            pyautogui.keyUp('j')
            pyautogui.keyUp('a')

        elif action == 'right_light':
            pyautogui.keyDown('d')
            time.sleep(0.05)
            pyautogui.keyDown('j')
            time.sleep(0.1)
            pyautogui.keyUp('j')
            pyautogui.keyUp('d')

    @staticmethod
    def reset_game():
        """
        Resets the game by pressing the 'c' key multiple times with specific delays to navigate
        through the game's menus and start a new match
        """

        # The initial delay allows the player to switch to the game window
        time.sleep(4)

        # Pressing 'c' multiple times with some delays to navigate through the menu and start a new match
        for i in range(5):
            pyautogui.keyDown('c')
            time.sleep(0.2)
            pyautogui.keyUp('c')
            print('pressed c')
            time.sleep(1.5)

        # Final delay to make sure that the match has started before the agent begins taking the actions
        time.sleep(6.5)

        return True
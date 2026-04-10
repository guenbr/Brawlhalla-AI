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

    def press(self, action):
        """
        Press a key corresponding to the given action. For movement actions, the key is held down
        briefly to simulate continuous movement. Attack and dodge actions are just tapped once

        Args: 
            action (str): The name of the action to perform (e.g., 'move_left', 'jump', 'light')
        """

        # Neutral action does not require any key press
        if action == 'neutral' or action is None:
            return

        key = self.keys.get(action)

        # If the action does not have a corresponding key, do nothing
        if key is None:
            return

        # For movement actions, hold the key down briefly to simulate the continuous movement
        # Try and catch is used to ensure that the key is released even if an error occurs 
        # during the sleep period
        if action in {'move_left', 'move_right', 'move_up', 'move_down'}:
            try:
                pyautogui.keyDown(key)
                time.sleep(0.1)
            finally:
                pyautogui.keyUp(key)
        else:
            # For attack and dodge actions, just tap the key once
            pyautogui.press(key)

    def hold(self, action, duration=0.1):
        """ 
        Holds a key down for a specific duration, and then releases it. This can be used for actions 
        that require a longer press, such as charging an attack.

        Args: 
            action: The name of the action to perform (e.g., 'move_left', 'jump', 'light').
            duration: The amount of time in seconds to hold the key down before releasing it. 
        """
        key = self.keys.get(action)
        if key is None:
            return

        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def release(self, action):
        """
        Releases a specific key if it's currently being held down. This can be used to stop a movement 
        or end a charge.

        Args: 
            action: The name of the action to release (e.g., 'move_left', 'jump', 'light').
        """
        key = self.keys.get(action)
        if key is None:
            return

        pyautogui.keyUp(key)

    def press_multiple(self, actions):
        """
        Presses multiple keys in sequence with a small delay between each press. 

        The 0.05 second delay is used to ensure that the game registers each key press properly, 
        especially when multiple keys are pressed rapidly

        Args: 
            actions: A list of action names to perform in sequence (e.g., ['move_left', 'jump', 'light'])
        """
        for action in actions:
            self.press(action)
            # small delay to make sure that the game registers each key press
            time.sleep(0.05) 

    def combo(self, actions, delays=None):
        """
        Executes a sequence of actions with specified delays between them

        This allows for more complex combos of actions, such as a jump followed by an attack 
        or a dodge followed by a movement. The delays can be customized to optimize the timing of the combo

        Args: 
            actions: A list of action names to perform in sequence (e.g., ['jump', 'light', 'dodge'])
            delays: A list of delay times in seconds to wait after each action before performing the next one. 
                    If None, a default delay of 0.05 seconds is used between all actions.

        """
        if delays is None:
            delays = [0.05] * len(actions)

        for action, delay in zip(actions, delays):
            self.press(action)
            time.sleep(delay)

    def release_all(self):
        """
        Releases all keys in the key mapping to ensure that no keys are stuck in a pressed state 
        
        This is important for movement keys, as holding them down can cause the character to keep moving
        even when the agent is trying to perform a different action. 

        The try and catch is used to make sure that all keys are released even if an error occurs 
        during the process, preventing any keys from being stuck in a pressed state 
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
            0: 'neutral',
            1: 'move_left',
            2: 'move_right',
            3: 'jump',
            4: 'light',
            5: 'heavy',
            6: 'dodge',
        }

        action = action_map.get(action_id, 'neutral')
        self.press(action)
        # Helpful for checking which specific actions are being executed by the agent
        print(f"Executing action: {action}")

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
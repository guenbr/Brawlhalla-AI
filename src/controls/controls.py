import pyautogui
import time


class Controls:
    def __init__(self):
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
            'throw': 'h',
            'pick_up': 'u',
            'taunt': 'g'
        }

        self.hold_actions = {'move_left', 'move_right', 'move_up', 'move_down'}
        self.hold_duration = 0.05

    def press(self, action):
        if action == 'neutral' or action is None:
            return

        key = self.keys.get(action)
        if key is None:
            return

        if action in self.hold_actions:
            pyautogui.keyDown(key)
            time.sleep(self.hold_duration)
            pyautogui.keyUp(key)
        else:
            pyautogui.press(key)

    def hold(self, action, duration=0.1):
        key = self.keys.get(action)
        if key is None:
            return

        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def release(self, action):
        key = self.keys.get(action)
        if key is None:
            return

        pyautogui.keyUp(key)

    def press_multiple(self, actions):
        for action in actions:
            self.press(action)
            time.sleep(0.05)

    def combo(self, actions, delays=None):
        if delays is None:
            delays = [0.05] * len(actions)

        for action, delay in zip(actions, delays):
            self.press(action)
            time.sleep(delay)

    def release_all(self):
        for key in set(self.keys.values()):
            if key is not None:
                pyautogui.keyUp(key)

    def execute_action(self, action_id):
        """Map action ID (0-7) to game action"""
        action_map = {
            0: 'neutral',
            1: 'move_left',
            2: 'move_right',
            3: 'jump',
            4: 'light',
            5: 'heavy',
            6: 'dodge',
            7: 'throw'
        }

        action = action_map.get(action_id, 'neutral')
        self.press(action)

    @staticmethod
    def reset_game():
        time.sleep(6)
        for _ in range(5):
            pyautogui.press('c')
            time.sleep(1)
        time.sleep(9)
        return True
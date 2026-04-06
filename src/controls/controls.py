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
        }

    def press(self, action):
        if action == 'neutral' or action is None:
            return

        key = self.keys.get(action)
        if key is None:
            return

        if action in {'move_left', 'move_right', 'move_up', 'move_down'}:
            try:
                pyautogui.keyDown(key)
                time.sleep(0.1)
            finally:
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
                try:
                    pyautogui.keyUp(key)
                except:
                    pass

    def execute_action(self, action_id):
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
        print(f"Executing action: {action}")

    @staticmethod
    def reset_game():
        time.sleep(4)

        for i in range(5):
            pyautogui.keyDown('c')
            time.sleep(0.2)
            pyautogui.keyUp('c')
            print('pressed c')
            time.sleep(1.5)

        time.sleep(6.5)

        return True
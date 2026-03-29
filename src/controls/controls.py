import pyautogui
import time


class Controls:
    def __init__(self):
        self.keys = {
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
        key = self.keys[action]

        if action in self.hold_actions:
            pyautogui.keyDown(key)
            time.sleep(self.hold_duration)
            pyautogui.keyUp(key)
        else:
            pyautogui.press(key)

    def hold(self, action, duration=0.1):
        key = self.keys[action]
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def release(self, action):
        key = self.keys[action]
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
        for key in self.keys.values():
            pyautogui.keyUp(key)

    @staticmethod
    def reset_game():
        time.sleep(2)
        pyautogui.press('c')
        time.sleep(1)
        pyautogui.press('c')
        time.sleep(1)
        pyautogui.press('c')
        time.sleep(1)
        pyautogui.press('c')
        time.sleep(1)
        pyautogui.press('c')
        time.sleep(9)

        # return True, means that we have entered a playing state
        return True

# def main():
#     test = Controls()
#     test.reset_game()
#
# main()
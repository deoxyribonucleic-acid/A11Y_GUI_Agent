import pyautogui
import time
import subprocess
from .utils import launch_app

class MouseAction:
    @staticmethod
    def execute(params):
        action_type = params.get("mouse_action_type")
        mouse_button = params.get("mouse_button", "left")
        pos = params.get("mouse_position", {})
        x = pos.get("width")
        y = pos.get("height")
        scroll_repeat = params.get("scroll_repeat", 0)

        if action_type == "move":
            pyautogui.moveTo(x, y)
        elif action_type == "click":
            pyautogui.moveTo(x, y)
            pyautogui.click(button=mouse_button)
        elif action_type == "double_click":
            pyautogui.moveTo(x, y)
            pyautogui.doubleClick(button=mouse_button)
        elif action_type == "scroll_up":
            pyautogui.scroll(abs(scroll_repeat), x=x, y=y)
        elif action_type == "scroll_down":
            pyautogui.scroll(-abs(scroll_repeat), x=x, y=y)

class KeyboardPressAction:
    @staticmethod
    def execute(params):
        key = params.get("keyboard_key")
        if key:
            pyautogui.press(key)
            return 0
        else:
            print("No key specified for keyboard press action, skipping.")
            return -1
        

class KeyboardCombinationAction:
    @staticmethod
    def execute(params):
        keys = params.get("keyboard_combination", [])
        if isinstance(keys, list):
            for key in keys:
                pyautogui.keyDown(key)
            for key in reversed(keys):
                pyautogui.keyUp(key)
            return 0
        else:
            print("No valid key combination specified for keyboard combination action, skipping.")
            return -1

class KeyboardTextAction:
    @staticmethod
    def execute(params):
        text = params.get("keyboard_text", "")
        pyautogui.typewrite(text)

class WaitAction:
    @staticmethod
    def execute(params):
        wait_time = params.get("wait_time", 1.0)
        time.sleep(wait_time)

class OpenApplicationAction:
    @staticmethod
    def execute(params):
        app_name = params.get("app_name")
        if app_name:
            launch_app(app_name)
        else:
            raise ValueError(f"Application '{app_name}' not found.")
            
class QuitApplicationAction:
    @staticmethod
    def execute(params):
        app_name = params.get("app_name")
        if app_name:
            try:
                subprocess.run(["osascript", "-e", f'tell application "{app_name}" to quit'])
            except Exception as e:
                raise RuntimeError(f"Failed to quit application '{app_name}': {e}")
        else:
            raise ValueError("Application name is required to quit the application.")
        
class CloseActiveWindowAction:
    @staticmethod
    def execute():
        pyautogui.hotkey('command', 'w')
    
class ShowMissionControlAction:
    @staticmethod
    def execute():
        pyautogui.hotkey('control', 'up')

class ShowDesktopAction:
    @staticmethod
    def execute():
        pyautogui.hotkey('command', 'f3')

def dispatch_action(action_dict):
    action_type = action_dict.get("action_type")

    if action_type == "MouseAction":
        MouseAction.execute(action_dict)
    elif action_type == "KeyboardAction":
        if action_dict.get("keyboard_action_type") == "press":
            KeyboardPressAction.execute(action_dict)
        elif action_dict.get("keyboard_action_type") == "comb":
            KeyboardCombinationAction.execute(action_dict)
        elif action_dict.get("keyboard_action_type") == "text":
            KeyboardTextAction.execute(action_dict)
    elif action_type == "WaitAction":
        WaitAction.execute(action_dict)
    elif action_type == "OpenApplicationAction":
        OpenApplicationAction.execute(action_dict)
    elif action_type == "QuitApplicationAction":
        QuitApplicationAction.execute(action_dict)
    elif action_type == "CloseActiveWindowAction":
        CloseActiveWindowAction.execute()
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    
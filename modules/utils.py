import base64
import json
import tempfile
import re
import os
import pyautogui
from PIL import Image

from AppKit import NSWorkspace, NSWorkspace
from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    kAXWindowsAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

from macapptree.apps import windows_for_application, application_for_process_id
from macapptree.window_tools import store_screen_scaling_factor
from macapptree.extractor import extract_window
from macapptree.screenshot_app_window import screenshot_window_to_file
from macapptree.window_tools import segment_window_components
from macapptree.main import get_main_window

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to a base64 string.
    :param image_path: Path to the image file.
    :return: Base64 encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def check_OS_version():
    """
    Check the operating system type and version.
    :return: A string with OS type and version.
    """
    import platform
    system = platform.system()
    if system == "Darwin":
        os_type = "MacOS"
        full_version = platform.mac_ver()[0]
    elif system == "Windows":
        os_type = "Windows"
        full_version = platform.version()
    elif system == "Linux":
        os_type = "Linux"
        full_version = platform.version()
    else:
        os_type = "Unknown"
        full_version = "N/A"
    return os_type + " " + full_version

def get_mouse_position():
    """
    Get the current mouse position.
    :return: A tuple (x, y) representing the mouse position.
    """
    x, y = pyautogui.position()
    return (x, y)

def window_to_global(x, y, width, height):
    """
    Convert window coordinates to global screen coordinates.
    :param x: Window X-coordinate.
    :param y: Window Y-coordinate.
    :param width: Window width.
    :param height: Window height.
    :return: A tuple (global_x, global_y).
    """
    screen_width, screen_height = pyautogui.size()
    global_x = x + (screen_width - width) // 2
    global_y = y + (screen_height - height) // 2
    return (global_x, global_y)

def global_to_window(global_pos, window_pos):
    """
    Convert global screen coordinates to window coordinates.
    :param global_pos: Tuple (global_x, global_y).
    :param window_pos: Tuple (window_x, window_y).
    :return: A tuple (window_x, window_y).
    """
    screen_width, screen_height = pyautogui.size()
    global_x, global_y = global_pos[0], global_pos[1]
    window_x, window_y = window_pos[0], window_pos[1]
    window_x = global_x - (screen_width - window_x) // 2
    window_y = global_y - (screen_height - window_y) // 2
    return (window_x, window_y)

def encode_image_base64(image: Image) -> str:
    """
    Encode a PIL.Image object to a base64 string.
    :param image: PIL.Image object.
    :return: Base64 encoded string.
    """
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



def get_window_position_by_name(app_name):
    """
    Get the position and size of a window by application name.
    :param app_name: Name of the application.
    :return: A tuple (x, y, width, height).
    """
    if app_name is None:
        screen_width, screen_height = pyautogui.size()
        return 0, 0, screen_width, screen_height

    workspace = NSWorkspace.sharedWorkspace()
    pid = None
    for app in workspace.runningApplications():
        if app.localizedName() == app_name:
            pid = app.processIdentifier()
            break

    app = AXUIElementCreateApplication(pid)
    windows_ptr = AXUIElementCopyAttributeValue(app, kAXWindowsAttribute, None)[1]
    window = windows_ptr[0]

    position_ref = AXUIElementCopyAttributeValue(window, kAXPositionAttribute, None)[1]
    size_ref = AXUIElementCopyAttributeValue(window, kAXSizeAttribute, None)[1]

    pos = AXValueGetValue(position_ref, kAXValueCGPointType, None)
    size = AXValueGetValue(size_ref, kAXValueCGSizeType, None)

    _, (x, y) = pos
    _, (width, height) = size
    return int(x), int(y), int(width), int(height)

def get_frontmost_app_info():
    """
    Get information about the frontmost application.
    :return: A tuple (app_name, bundle_id, pid).
    """
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    app_name = app.localizedName()
    bundle_id = app.bundleIdentifier()
    pid = app.processIdentifier()
    return app_name, bundle_id, pid

def get_bundle_id_by_app_name(app_name: str):
    """
    Get the bundle ID of an application by its name.
    :param app_name: Name of the application.
    :return: The bundle ID or None if not found.
    """
    apps = NSWorkspace.sharedWorkspace().runningApplications()
    for app in apps:
        if app.localizedName() == app_name:
            return app.bundleIdentifier()
    return None

def launch_app(app_name, focus=True):
    """
    Launch an application by name.
    :param app_name: Name of the application.
    :param focus: Whether to focus the application after launching.
    """
    from AppKit import NSWorkspace
    import subprocess
    import time

    workspace = NSWorkspace.sharedWorkspace()
    success = workspace.launchApplication_(app_name)

    if not success:
        print(f"❌ Failed to launch {app_name}")
        return

    time.sleep(1.5)

    if focus:
        script = f'tell application "{app_name}" to activate'
        subprocess.run(["osascript", "-e", script])

def focus_app_by_name(app_name):
    """
    Focus an application by name.
    :param app_name: Name of the application.
    """
    import subprocess
    script = f'tell application "{app_name}" to activate'
    subprocess.run(["osascript", "-e", script])

def get_tree_screenshot(app_name, max_depth=None):
    """
    Get the accessibility tree and screenshots of an application window.
    :param app_name: Name of the application.
    :param max_depth: Maximum depth for the accessibility tree.
    :return: A tuple (tree, cropped_image, segmented_image).
    """
    if app_name is None:
        raise ValueError("app_name cannot be None")
    
    a11y_tmp_file = tempfile.NamedTemporaryFile(delete=False)
    screenshot_tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    store_screen_scaling_factor()

    workspace = NSWorkspace.sharedWorkspace()

    running = workspace.runningApplications()
    app = None
    for app in running:
        if app.localizedName() == app_name:
            app_bundle = app.bundleIdentifier()
            pid = app.processIdentifier()
            break

    focus_app_by_name(app_name)

    application = application_for_process_id(pid)

    windows = windows_for_application(application)
    window_element = get_main_window(windows, max_depth)

    extracted = extract_window(
            window_element, app_bundle, a11y_tmp_file.name, False, False, max_depth
        )
    
    if not extracted:
        raise "Couldn't extract accessibility"
    
    if screenshot_tmp_file:
        output_croped, _ = screenshot_window_to_file(app.localizedName(), window_element.name, screenshot_tmp_file.name)
        output_segmented = segment_window_components(window_element, output_croped)
        result = json.dumps({
            "croped_screenshot_path": output_croped,
            "segmented_screenshot_path": output_segmented
        })

    json_match = re.search(r'{.*}', result, re.DOTALL)
    if not json_match:
        print(f"Failed to extract screenshots for {app_name}")
        return json.load(a11y_tmp_file), None, None

    json_str = json_match.group(0)
    screenshots_paths_dict = json.loads(json_str)
    croped_img = Image.open(screenshots_paths_dict["croped_screenshot_path"])
    segmented_img = Image.open(screenshots_paths_dict["segmented_screenshot_path"])

    os.remove(screenshots_paths_dict["croped_screenshot_path"])
    os.remove(screenshots_paths_dict["segmented_screenshot_path"])

    tree = json.load(a11y_tmp_file)

    a11y_tmp_file.close()
    screenshot_tmp_file.close()

    return tree, croped_img, segmented_img

def get_full_screenshot():
    """
    Capture a full screenshot of the screen.
    :return: A resized PIL.Image object of the screenshot.
    """
    screenshot = pyautogui.screenshot()
    screenshot = screenshot.convert("RGB").resize(size=(screenshot.width // 2, screenshot.height // 2))
    return screenshot

def print_accessibility_tree(node, indent=0):
    """Recursively print macOS accessibility tree structure"""
    prefix = "  " * indent
    role = node.get("role", "Unknown")
    name = node.get("name") or node.get("value") or ""
    description = node.get("description") or node.get("role_description") or ""
    bbox = node.get("bbox", [])
    
    print(f"{prefix}- [{role}] '{name}' ({description}) BBox: {bbox}")

    # Print child nodes
    children = node.get("children", [])
    for child in children:
        print_accessibility_tree(child, indent + 1)

if __name__ == "__main__":
    # Example usage
    app_name = "Safari"
    tree, croped_img, segmented_img = get_tree_screenshot(app_name)
    croped_img.show()
    segmented_img.show()
    print_accessibility_tree(tree)
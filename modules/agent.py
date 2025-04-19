from PIL import Image

from .prompts import *
from .utils import *
from .api_tools import LLMTool
from .actions import *

class AgentOrchestrator:
    """
    Orchestrates the planning, execution, and evaluation of tasks using an LLM and memory modules.
    """

    def __init__(self):
        self.llm = LLMTool()
        self.Memory = AgentMemory()

    def plan(self, task_prompt):
        """
        Plans the task based on the given task prompt and updates the memory with planned tasks.
        """
        print("===" * 20)
        print(f"Stage 1/3: Planning task: {task_prompt}")

        self.Memory.update_permanent_memory("task_prompt", task_prompt)
        self.Memory.update_permanent_memory("os_version", check_OS_version())
        self.Memory.update_short_term_memory("app_name", None)

        mouse_global_pos = get_mouse_position()
        self.Memory.update_short_term_memory("mouse_position", mouse_global_pos)

        user_propmpt = user_prompt_info.format(
            task_prompt=task_prompt,
            os_version=self.Memory.permanent_memory["os_version"],
            app_name=self.Memory.short_term_memory["app_name"],
            mouse_position=self.Memory.short_term_memory["mouse_position"],
        )

        response = self.llm.run(sys_prompt_init, user_propmpt)
        result = LLMTool.postprocess_by_prompt(response)

        planned_tasks = [item.get("element") for item in result]
        assigned_apps = [item.get("app_name") for item in result]

        self.Memory.update_permanent_memory("planned_tasks", planned_tasks)
        self.Memory.update_permanent_memory("assigned_apps", assigned_apps)

        print(f"Planned tasks: {planned_tasks}")

    def execute(self, current_task):
        """
        Executes the current task by interacting with the UI and dispatching actions.
        """
        print("===" * 20)
        print(f"Stage 2/3: Executing task: {current_task}")

        self.Memory.update_short_term_memory("current_task", current_task)

        tree, seg_im = None, None
        if self.Memory.short_term_memory["app_name"]:
            try:
                focus_app_by_name(self.Memory.short_term_memory["app_name"])
                app_name, _, _ = get_frontmost_app_info()
                tree, seg_im = get_tree_screenshot(app_name)
            except Exception as e:
                print(f"Error focusing app: {e}, trying to get screenshot without focusing.")

        if seg_im is None:
            seg_im = get_full_screenshot()

        self.Memory.update_short_term_memory("ui_tree", tree)

        mouse_global_pos = get_mouse_position()
        mouse_win_pos = global_to_window(mouse_global_pos, get_window_position_by_name(self.Memory.short_term_memory["app_name"]))
        self.Memory.update_short_term_memory("mouse_position", mouse_win_pos)

        previous_tool_calls = self.Memory.long_term_memory["Attempts"][-1].tool_calls if self.Memory.long_term_memory["Attempts"] else ""

        user_propmpt = user_prompt_action.format(
            task_prompt=self.Memory.permanent_memory["task_prompt"],
            ui_tree=self.Memory.short_term_memory["ui_tree"],
            os_version=self.Memory.permanent_memory["os_version"],
            app_name=self.Memory.short_term_memory["app_name"],
            mouse_position=self.Memory.short_term_memory["mouse_position"],
            current_task=self.Memory.short_term_memory["current_task"],
            previous_tool_calls=previous_tool_calls,
        )

        seg_im_base64 = encode_image_base64(seg_im)
        response = self.llm.run(sys_prompt_action, user_propmpt, screenshot_base64=seg_im_base64)
        result = LLMTool.postprocess_by_prompt(response)

        if isinstance(result, list):
            self.Memory.update_short_term_memory("current_tool_calls", result)
            for action in result:
                if action.get("action_type") == "OpenApplicationAction":
                    self.Memory.update_short_term_memory("app_name", action.get("app_name"))
                    print(f"Opening application: {action.get('app_name')}")
                elif action.get("action_type") == "QuitApplicationAction":
                    self.Memory.update_short_term_memory("app_name", None)
                    print(f"Quitting application: {action.get('app_name')}")
                print(f"Executing action: {action}")
                dispatch_action(action)
                time.sleep(1)

    def judge(self):
        """
        Judges the outcome of the current task and provides feedback or advice for replanning or retrying.
        """
        print("===" * 20)
        print(f"Stage 3/3: Judging task: {self.Memory.short_term_memory['current_task']}")
        print("Current App:", self.Memory.short_term_memory["app_name"])

        if self.Memory.short_term_memory["app_name"]:
            focus_app_by_name(self.Memory.short_term_memory["app_name"])

        im = get_full_screenshot()

        mouse_global_pos = get_mouse_position()
        mouse_win_pos = global_to_window(mouse_global_pos, get_window_position_by_name(self.Memory.short_term_memory["app_name"]))
        self.Memory.update_short_term_memory("mouse_position", mouse_win_pos)

        previous_tool_calls = self.Memory.long_term_memory["Attempts"][-1].tool_calls if self.Memory.long_term_memory["Attempts"] else ""

        user_propmpt = user_prompt_judge.format(
            task_prompt=self.Memory.permanent_memory["task_prompt"],
            ui_tree=self.Memory.short_term_memory["ui_tree"],
            os_version=self.Memory.permanent_memory["os_version"],
            app_name=self.Memory.short_term_memory["app_name"],
            mouse_position=self.Memory.short_term_memory["mouse_position"],
            current_task=self.Memory.short_term_memory["current_task"],
            previous_tool_calls=previous_tool_calls,
            current_tool_calls=self.Memory.short_term_memory["current_tool_calls"],
        )

        seg_im_base64 = encode_image_base64(im)
        response = self.llm.run(sys_prompt_judge, user_propmpt, screenshot_base64=seg_im_base64)
        print("Response:", response)
        result = LLMTool.postprocess_by_prompt(response)

        if isinstance(result, dict):
            print(f"Judging result: {result}, advice: {result.get('advice')}")
            if result.get("status") == "retry":
                self.Memory.update_long_term_memory(self.Memory.short_term_memory["current_task"], self.Memory.short_term_memory["current_tool_calls"], result, im)
            elif result.get("status") == "success":
                self.Memory.drop_long_term_memory()
            elif result.get("status") == "replan":
                self.Memory.drop_long_term_memory()
            return result

    def replan(self, advice):
        """
        Replans the task based on the provided advice and updates the memory with new planned tasks.
        """
        print("===" * 20)
        print(f"Stage 1/3: Replanning task: {self.Memory.short_term_memory['current_task']}, Advice: {advice}")

        mouse_global_pos = get_mouse_position()
        mouse_win_pos = global_to_window(mouse_global_pos, get_window_position_by_name(self.Memory.short_term_memory["app_name"]))
        self.Memory.update_short_term_memory("mouse_position", mouse_win_pos)

        user_propmpt = user_prompt_info.format(
            task_prompt=self.Memory.permanent_memory["task_prompt"],
            os_version=self.Memory.permanent_memory["os_version"],
            app_name=self.Memory.short_term_memory["app_name"],
            mouse_position=self.Memory.short_term_memory["mouse_position"],
        )

        user_propmpt += replan_advice.format(advice=advice)

        response = self.llm.run(sys_prompt_init, user_propmpt)
        result = LLMTool.postprocess_by_prompt(response)

        planned_tasks = [item.get("element") for item in result]
        self.Memory.update_permanent_memory("planned_task", planned_tasks)
        self.Memory.update_short_term_memory("current_task", planned_tasks[0])

        print(f"Planned tasks: {planned_tasks}")

    def run(self, task_prompt, replan_counter=0, advice=""):
        """
        Runs the entire task lifecycle: planning, execution, and judging. Handles replanning if necessary.
        """
        print(f"Running task: {task_prompt}, Replan attempt: {replan_counter}, Advice: {advice}")

        if replan_counter == 0:
            self.plan(task_prompt)
        else:
            self.replan(advice)

        status = None
        for subtask in self.Memory.permanent_memory["planned_tasks"]:
            retry_count = 0
            while True:
                self.execute(subtask)
                result = self.judge()
                status = result.get("status")
                retry_count += 1
                print(f"{subtask} # Attempt {retry_count} - Status: {status}, Advice: {result.get('advice')}")
                if retry_count > 3:
                    print("Retry limit reached. Forcing replan.")
                    break
                if status == "success":
                    print(f"Subtask '{subtask}' completed successfully.")
                    break
                if status == "replan":
                    print(f"Replanning due to advice: {result.get('advice')}")
                    break

            if status == "replan":
                if replan_counter < 3:
                    self.run(task_prompt, replan_counter + 1, result.get("advice"))
                else:
                    print("Replan limit reached. Failed.")
                    break


class PastMemoryUnit:
    """
    Represents a unit of past memory, storing information about a subtask, tool calls, and results.
    """

    def __init__(self, subtask: str, tool_calls: list, result: dict):
        self.subtask = subtask
        self.tool_calls = tool_calls
        self.result = result


class AgentMemory:
    """
    Memory module for the agent, responsible for storing and updating short-term, long-term, and permanent memory.
    """

    def __init__(self):
        self.short_term_mem_attrs = ["ui_tree", "app_name", "mouse_position", "current_task", "current_tool_calls"]
        self.get_from_perm_attrs = ["task_prompt", "planned_tasks", "os_version", "assigned_apps"]

        self.short_term_memory = {}
        self.long_term_memory = {"Attempts": [], "Screenshots": []}
        self.permanent_memory = {
            "task_prompt": "",
            "planned_tasks": [],
            "os_version": "",
            "assigned_apps": [],
        }

        print("Agent memory initialized.")

    def update_short_term_memory(self, attribute: str, value):
        """
        Updates the short-term memory with the given attribute and value.
        """
        if attribute not in self.short_term_mem_attrs:
            raise ValueError(f"Attribute {attribute} not found in short term memory.")
        self.short_term_memory[attribute] = value

    def update_long_term_memory(self, subtask: str, tool_calls: list, judge_result: dict, screenshot: Image = None):
        """
        Updates the long-term memory with the given subtask, tool calls, judge result, and optional screenshot.
        """
        self.long_term_memory["Attempts"].append(PastMemoryUnit(subtask, tool_calls, judge_result))
        self.long_term_memory["Screenshots"].append(screenshot)

    def update_permanent_memory(self, attribute: str, value):
        """
        Updates the permanent memory with the given attribute and value.
        """
        if attribute in self.permanent_memory:
            self.permanent_memory[attribute] = value
        else:
            raise ValueError(f"Attribute {attribute} not found in permanent memory.")

        for attr in self.get_from_perm_attrs:
            if attr not in self.short_term_memory and attr in self.permanent_memory:
                self.short_term_memory[attr] = self.permanent_memory[attr]

    def drop_long_term_memory(self):
        """
        Clears the long-term memory.
        """
        self.long_term_memory = {"Attempts": [], "Screenshots": []}
        print("Long term memory cleared.")

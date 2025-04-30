import os
from PIL import Image

from .prompts import *
from .utils import *
from .api_tools import LLMTool
from .actions import *

class AgentOrchestrator:
    """
    Orchestrates the planning, execution, and evaluation of tasks using an LLM and memory modules.
    """

    def __init__(self, save_result=False, save_path=None):
        self.llm = LLMTool()
        self.Memory = AgentMemory()
        self.save_result = save_result
        self.save_path = save_path

        if self.save_result:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            print(f"Results will be saved to: {self.save_path}")
            # Create subfolders for responses and screenshots if they do not exist
            response_path = os.path.join(self.save_path, "responses")
            screenshots_path = os.path.join(self.save_path, "screenshots")
            if not os.path.exists(response_path):
                os.makedirs(response_path)
            if not os.path.exists(screenshots_path):
                os.makedirs(screenshots_path)

            # Save LLM responses and screenshots
            self.response_path = response_path
            self.screenshots_path = screenshots_path
            print(f"Subfolders created: {response_path}, {screenshots_path}")

    def plan(self, task_prompt):
        """
        Plans the task based on the given task prompt and updates the memory with planned tasks.
        Returns the planned result for saving.
        """
        print("===" * 20)
        print(f"Stage 1/3: Planning task: {task_prompt}")

        self.Memory.update_permanent_memory("task_prompt", task_prompt)
        self.Memory.update_permanent_memory("os_version", check_OS_version())
        self.Memory.update_short_term_memory("app_name", None)

        user_propmpt = user_prompt_info.format(
            task_prompt=task_prompt,
            os_version=self.Memory.permanent_memory["os_version"],
            app_name=self.Memory.short_term_memory["app_name"],
        )

        response = self.llm.run(sys_prompt_init, user_propmpt)
        result = LLMTool.postprocess_by_prompt(response)

        planned_tasks = [item.get("element") for item in result]
        assigned_apps = [item.get("app_name") for item in result]

        self.Memory.update_permanent_memory("planned_tasks", planned_tasks)
        self.Memory.update_permanent_memory("assigned_apps", assigned_apps)

        print(f"Planned tasks: {planned_tasks}")

        return result

    def execute(self, current_task):
        """
        Executes the current task by interacting with the UI and dispatching actions.
        Returns the LLM response, screenshot, and UI tree for saving.
        """
        print("===" * 20)
        print(f"Stage 2/3: Executing task: {current_task}")
        print("Current App:", self.Memory.short_term_memory["app_name"])

        self.Memory.update_short_term_memory("current_task", current_task)

        tree, seg_im = None, None
        tree_compressed = None
        if self.Memory.short_term_memory["app_name"] is not None:
            try:
                app_name = self.Memory.short_term_memory["app_name"]
                focus_app_by_name(app_name)
                tree, _, seg_im = get_tree_screenshot(app_name)
                tree_compressed = compress_ui_tree(tree, compression_level=4)
            except Exception as e:
                print(f"Error focusing app: {e}, trying to get screenshot without focusing.")

        if seg_im is None:
            seg_im = get_full_screenshot()

        self.Memory.update_short_term_memory("ui_tree", tree_compressed)

        mouse_global_pos = get_mouse_position()
        window_bbox = get_window_position_by_name(self.Memory.short_term_memory["app_name"])
        mouse_win_pos = global_to_window(mouse_global_pos, window_bbox)
        self.Memory.update_short_term_memory("mouse_position", mouse_win_pos)

        previous_attempts = self.Memory.long_term_memory["Attempts"] if self.Memory.long_term_memory["Attempts"] else ""

        user_propmpt = user_prompt_action.format(
            task_prompt=self.Memory.permanent_memory["task_prompt"],
            ui_tree=self.Memory.short_term_memory["ui_tree"],
            window_size={"width": window_bbox[2], "height": window_bbox[3]},
            os_version=self.Memory.permanent_memory["os_version"],
            app_name=self.Memory.short_term_memory["app_name"],
            mouse_position=self.Memory.short_term_memory["mouse_position"],
            current_task=self.Memory.short_term_memory["current_task"],
            previous_tool_calls=previous_attempts,
        )

        seg_im_base64 = encode_image_base64(seg_im)
        response = self.llm.run(sys_prompt_action, user_propmpt, screenshot_base64=seg_im_base64)
        result = LLMTool.postprocess_by_prompt(response)

        if isinstance(result, list):
            self.Memory.update_short_term_memory("current_tool_calls", result)
            for action in result:
                if action.get("action_type") == "OpenApplicationAction":
                    self.Memory.update_short_term_memory("app_name", action.get("app_name"))
                    print(f"Opened application: {action.get('app_name')}")
                elif action.get("action_type") == "QuitApplicationAction":
                    self.Memory.update_short_term_memory("app_name", None)
                    print(f"Quitting application: {action.get('app_name')}")
                elif action.get("action_type") == "MouseAction":
                    pos = action.get("mouse_position")
                    if pos is not None:
                        x = pos.get("width")
                        y = pos.get("height")
                        windo_bbox = get_window_position_by_name(self.Memory.short_term_memory["app_name"])
                        x_glob, y_glob = window_to_global(x, y, windo_bbox)
                        print(f"Mapping mouse position: {x}, {y} for app {self.Memory.short_term_memory['app_name']} at {windo_bbox} to {x_glob}, {y_glob}")
                        action["mouse_position"] = {"width": x_glob, "height": y_glob}  # update mouse position to global
                print(f"Executing action: {action}")
                dispatch_action(action)
                time.sleep(2)

        return result, seg_im, tree_compressed

    def judge(self):
        """
        Judges the outcome of the current task and provides feedback or advice for replanning or retrying.
        Returns the screenshot and parsed LLM response for saving.
        """
        print("===" * 20)
        print(f"Stage 3/3: Judging task: {self.Memory.short_term_memory['current_task']}")
        print("Current App:", self.Memory.short_term_memory["app_name"])

        if self.Memory.short_term_memory["app_name"]:
            focus_app_by_name(self.Memory.short_term_memory["app_name"])

        im = get_full_screenshot()

        mouse_global_pos = get_mouse_position()
        window_bbox = get_window_position_by_name(self.Memory.short_term_memory["app_name"])
        mouse_win_pos = global_to_window(mouse_global_pos, window_bbox)
        self.Memory.update_short_term_memory("mouse_position", mouse_win_pos)

        previous_tool_calls = self.Memory.long_term_memory["Attempts"] if self.Memory.long_term_memory["Attempts"] else ""

        user_propmpt = user_prompt_judge.format(
            task_prompt=self.Memory.permanent_memory["task_prompt"],
            ui_tree=self.Memory.short_term_memory["ui_tree"],
            os_version=self.Memory.permanent_memory["os_version"],
            window_size={"width": window_bbox[2], "height": window_bbox[3]},
            app_name=self.Memory.short_term_memory["app_name"],
            mouse_position=self.Memory.short_term_memory["mouse_position"],
            current_task=self.Memory.short_term_memory["current_task"],
            previous_tool_calls=previous_tool_calls,
            current_tool_calls=self.Memory.short_term_memory["current_tool_calls"],
            sub_task_list=self.Memory.permanent_memory["planned_tasks"],
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
            return im, result

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
        self.Memory.update_permanent_memory("planned_tasks", planned_tasks)
        self.Memory.update_short_term_memory("current_task", planned_tasks[0])

        print(f"Planned tasks: {planned_tasks}")

        return result

    def run(self, task_prompt, replan_counter=0, advice=""):
        """
        Runs the entire task lifecycle: planning, execution, and judging. Handles replanning if necessary.
        Saves results to appropriate folders with unique filenames to avoid overwriting.
        """
        print(f"Running task: {task_prompt}, Replan attempt: {replan_counter}, Advice: {advice}")

        if replan_counter == 0:
            plan_result = self.plan(task_prompt)
            if self.save_result:
                # Save plan results
                plan_filename = os.path.join(self.response_path, "plan_result.json")
                with open(plan_filename, "w") as f:
                    json.dump(plan_result, f)
                print(f"Plan result saved to: {plan_filename}")
        else:
            replan_result = self.replan(advice)
            if self.save_result:
            # Save replan results
                replan_filename = os.path.join(self.response_path, f"replan_attempt_{replan_counter}.json")
                with open(replan_filename, "w") as f:
                    json.dump(replan_result, f)
                print(f"Replan result saved to: {replan_filename}")

        status = None
        for idx, subtask in enumerate(self.Memory.permanent_memory["planned_tasks"]):
            retry_count = 0
            while True:
                result, screenshot, ui_tree = self.execute(subtask)
                judge_im, judge_result = self.judge()
                status = judge_result.get("status")
                retry_count += 1
                print(f"{subtask} # Attempt {retry_count} - Status: {status}, Advice: {judge_result.get('advice')}")

                # Save results to appropriate folders
                if self.save_result:
                    # Save execution results
                    exec_filename = os.path.join(self.response_path, f"subtask_{idx}_attempt_{retry_count}_execute.json")
                    with open(exec_filename, "w") as f:
                        json.dump(result, f)
                    print(f"Execution result saved to: {exec_filename}")

                    # Save execution screenshot
                    screenshot_filename = os.path.join(self.screenshots_path, f"subtask_{idx}_attempt_{retry_count}_execute.png")
                    screenshot.save(screenshot_filename)
                    print(f"Execution screenshot saved to: {screenshot_filename}")

                    # Save judge results
                    judge_filename = os.path.join(self.response_path, f"subtask_{idx}_attempt_{retry_count}_judge.json")
                    with open(judge_filename, "w") as f:
                        json.dump(judge_result, f)
                    print(f"Judge result saved to: {judge_filename}")

                    # Save judge screenshot
                    judge_screenshot_filename = os.path.join(self.screenshots_path, f"subtask_{idx}_attempt_{retry_count}_judge.png")
                    if judge_im:
                        judge_im.save(judge_screenshot_filename)
                    print(f"Judge screenshot saved to: {judge_screenshot_filename}")

                if retry_count > 3:
                    print("Retry limit reached. Forcing replan.")
                    status = "replan"
                    break
                if status == "success":
                    print(f"Subtask '{subtask}' completed successfully.")
                    break
                if status == "replan":
                    print(f"Replanning due to advice: {judge_result.get('advice')}")
                    break
                time.sleep(2)

            if status == "replan":
                if replan_counter < 3:
                    self.run(task_prompt, replan_counter + 1, judge_result.get("advice"))
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

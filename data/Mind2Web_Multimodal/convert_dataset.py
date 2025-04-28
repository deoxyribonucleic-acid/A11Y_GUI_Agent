"""
Convert Mind2Web dataset to ScreenAgent format.
Run this script under the ./data/Mind2Web directory.

Mind2Web Data Fields:
"annotation_id" (str): unique id for each task
"website" (str): website name
"domain" (str): website domain
"subdomain" (str): website subdomain
"confirmed_task" (str): task description
"action_reprs" (list[str]): human readable string representation of the action sequence
"actions" (list[dict]): list of actions (steps) to complete the task
    "action_uid" (str): unique id for each action (step)
    "raw_html" (str): raw html of the page before the action is performed
    "cleaned_html" (str): cleaned html of the page before the action is performed
    "operation" (dict): operation to perform
        "op" (str): operation type, one of CLICK, TYPE, SELECT
        "original_op" (str): original operation type, contain additional HOVER and ENTER that are mapped to CLICK, not used
        "value" (str): optional value for the operation, e.g., text to type, option to select
    "pos_candidates" (list[dict]): ground truth elements. Here we only include positive elements that exist in "cleaned_html" after our preprocessing, so "pos_candidates" might be empty. The original labeled element can always be found in the "raw_html".
        "tag" (str): tag of the element
        "is_original_target" (bool): whether the element is the original target labeled by the annotator
        "is_top_level_target" (bool): whether the element is a top level target find by our algorithm. please see the paper for more details.
        "backend_node_id" (str): unique id for the element
        "attributes" (str): serialized attributes of the element, use json.loads to convert back to dict
    "neg_candidates" (list[dict]): other candidate elements in the page after preprocessing, has similar structure as "pos_candidates"
"""


import os
import base64
import glob
import io
import re
import json
import random
from enum import Enum
from tqdm import tqdm

import cv2

import numpy as np
from datasets import load_dataset, DownloadMode
from jinja2 import Template as JinjaTemplate
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from bs4 import BeautifulSoup

# from translation import translate

dataset = load_dataset("osunlp/Multimodal-Mind2Web")

DEBUG=True
OUTPUT_SIZE = (1120, 1120)
TEMPLATES_PATH = "prompts"
SAVE_PATH = "processed_dataset"
if os.path.exists(SAVE_PATH) is False:
    os.mkdir(SAVE_PATH)

def decoder_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    return image

class ImageNotFindException(Exception):
    pass

# def load_templates(template_file_name):
#     with open(os.path.join(TEMPLATES_PATH, template_file_name), "r") as f:
#         template = f.read()
#     return JinjaTemplate(template)


# templates = {
#     "planner_agent_send_prompt_en": load_templates("planner_agent_send_prompt_web_en.txt"),
#     "planner_agent_send_prompt_zh": load_templates("planner_agent_send_prompt_web_zh.txt"),
#     "planner_agent_answer_en": load_templates("planner_agent_answer_web_en.txt"),
#     "planner_agent_answer_zh": load_templates("planner_agent_answer_web_zh.txt"),

#     "actor_agent_send_prompt_en": load_templates("actor_agent_send_prompt_web_en.txt"),
#     "actor_agent_send_prompt_zh": load_templates("actor_agent_send_prompt_web_zh.txt"),
#     "actor_agent_answer_en": load_templates("actor_agent_answer_web_en.txt"),
#     "actor_agent_answer_zh": load_templates("actor_agent_answer_web_zh.txt"),

#     "evaluator_agent_send_prompt_en": load_templates("evaluator_agent_send_prompt_web_en.txt"),
#     "evaluator_agent_send_prompt_zh": load_templates("evaluator_agent_send_prompt_web_zh.txt"),
#     "evaluator_agent_answer_en": load_templates("evaluator_agent_answer_web_en.txt"),
#     "evaluator_agent_answer_zh": load_templates("evaluator_agent_answer_web_zh.txt")
# }


def determining_screen_change_degree(before_action_screenshot: Image, after_action_screenshot: Image):
    before_action_screenshot = before_action_screenshot.convert('L')
    after_action_screenshot = after_action_screenshot.convert('L')
    before_action_screenshot = np.array(before_action_screenshot)
    after_action_screenshot = np.array(after_action_screenshot)

    # Determining image similarity
    score, _ = ssim(before_action_screenshot, after_action_screenshot, full=True)
    return score

def calculate_edge_centroid(image:Image, bbox):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # Edge extraction using Canny algorithm
    edges = cv2.Canny(cropped_image, 100, 200)

    # Get edge pixel
    positions = np.where(edges == 255)

    if len(positions[0]) == 0 or len(positions[1]) == 0:
        # Use the center of the bbox as the center
        centroid_x = (bbox[0] + bbox[2]) // 2
        centroid_y = (bbox[1] + bbox[3]) // 2
        return int(centroid_x), int(centroid_y)
    
    y, x = positions

    # center of gravity
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    # Convert back to the original image coordinate system
    centroid_x += bbox[0]
    centroid_y += bbox[1]

    return int(centroid_x), int(centroid_y)

def resize_by_width(image: Image, width: int):
    wpercent = (width/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((width, hsize))
    return image, wpercent

def crop_and_padding(image: Image, hight_start, hight_end):
    # crop image by high_start and high_end and padding to OUTPUT_SIZE
    assert image.size[0] == OUTPUT_SIZE[0]
    if image.size[1] < OUTPUT_SIZE[1]:
        padding = Image.new(image.mode, (OUTPUT_SIZE[0], OUTPUT_SIZE[1]), (255, 255, 255))
        padding.paste(image, (0, 0))
        return padding
    else:
        return image.crop((0, hight_start, OUTPUT_SIZE[0], hight_end))

def convert_string(string_or_list):
    # Add escaping symbols to English quotes in string
    if isinstance(string_or_list, str):
        return string_or_list.replace('"', '\\"')
    elif isinstance(string_or_list, list):
        return [convert_string(s) for s in string_or_list]


class OpenNewSessionReason(Enum):
    NONE = 0
    NEW_PAGE = 1
    SCREEN_HUGE_CHANGE = 2
    LIST_COMPLETED = 3

def save_example_file(session_id, file_name, data, image, image_file_name):
    if os.path.exists(os.path.join(SAVE_PATH, session_id)) is False:
        os.mkdir(os.path.join(SAVE_PATH, session_id))
        os.mkdir(os.path.join(SAVE_PATH, session_id, "images"))

    with open(os.path.join(SAVE_PATH, session_id, file_name), "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    image.save(os.path.join(SAVE_PATH, session_id, "images", image_file_name))


def save_image_for_check(session_id, image, image_file_name, action_answer_info):
    if os.path.exists(os.path.join(SAVE_PATH, session_id)) is False:
        os.mkdir(os.path.join(SAVE_PATH, session_id))
        os.mkdir(os.path.join(SAVE_PATH, session_id, "images"))

    center_width = action_answer_info["center_width"]
    center_height = action_answer_info["center_height"]

    # copy image
    image_for_check = image.copy()
    draw = ImageDraw.Draw(image_for_check)
    draw.rectangle([(center_width - 10, center_height - 10), (center_width + 10, center_height + 10)], outline ="red")
    if action_answer_info["operation_type"] == "SELECT" or action_answer_info["operation_type"] == "TYPE":
        draw.text((center_width, center_height), action_answer_info["operation_value"], fill=(255, 0, 0))

    image_for_check.save(os.path.join(SAVE_PATH, session_id, "images", image_file_name+"_check.png"))


def is_visible(element):
    bounding_box = element.get('bounding_box_rect')
    return bounding_box != "-1,-1,-1,-1"

def clean_text(text):
    cleaned_text = text.strip()
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\t', ' ')
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    return cleaned_text

def find_semantic_info(element):
    element_text = clean_text(element.get_text(strip=True))
    if element_text:
        return element_text
    
    label = element.find_previous(lambda x: x.name == 'label' and is_visible(x))
    if label:
        label_text = clean_text(label.get_text(strip=True))
        if label_text:
            return label_text
    
    return None


def action_discription(ui_element_name, ui_element_text, operation_type, value):

    ret_en = ""
    ret_zh = ""
    if operation_type == "TYPE":
        if ui_element_text != "":
            ret_en += f'Type text "{value}" into {ui_element_name} with text "{ui_element_text}" on it'
            ret_zh += f'在标有“{ui_element_text}”的{ui_element_name}中输入文本“{value}”'
        else:
            ret_en += f'Type text "{value}" into {ui_element_name}'
            ret_zh += f'在{ui_element_name}中输入文本"{value}"'
    elif operation_type == "SELECT":
        if ui_element_text != "":
            ret_en += f'Select "{value}" from {ui_element_name} with text "{ui_element_text}" on it'
            ret_zh += f'在标有“{ui_element_text}”的{ui_element_name}中选择“{value}”'
        else:
            ret_en += f'Select "{value}" from {ui_element_name}.'
            ret_zh += f'在{ui_element_name}中选择“{value}”'
    elif operation_type == "CLICK":
        if ui_element_text != "":
            ret_en += f'Click the {ui_element_name} element with text "{ui_element_text}" on it'
            ret_zh += f'点击标有“{ui_element_text}”的{ui_element_name}元素'
        else:
            ret_en += f'Click the {ui_element_name} element'
            ret_zh += f'点击{ui_element_name}元素'

    return ret_en, ret_zh


def process_one_task(action):
    annotation_id = action["annotation_id"]

    before_action_screenshot = action.get("screenshot", None)

    task_prompt_en = action["confirmed_task"]
    action_reprs = action["target_action_reprs"]
    action_descriptions_en = []

    action_screens = []

    base_info = {
        "annotation_id": annotation_id,
        "website_en": action["website"],
        "website_zh": action["website"],
        "domain_en": action["domain"],
        "subdomain_en": action["subdomain"],
        "task_prompt_en": task_prompt_en,
    }

    action_uid = action["action_uid"]
    operation = action["operation"]

    # print(f"action_reprs: {action_reprs}")
    # for repr in action_reprs:
    ui_element = action_reprs.split(" -> ")[0]
    ui_element_name, ui_element_text = ui_element.split("] ")
    ui_element_name = ui_element_name[1:]
    ui_element_text = ui_element_text.strip()

    if ui_element_text == "":
        # Trying to find the label of this element
        html_content = action["cleaned_html"]
        soup = BeautifulSoup(html_content, 'html.parser')
        pos_candidates = action["pos_candidates"]
        if isinstance(pos_candidates, str):
                pos_candidates = json.loads(pos_candidates)

        parsed = []

        for item in pos_candidates:
            if isinstance(item, str):
                    item = json.loads(item)
            if "attributes" in item and isinstance(item["attributes"], str):
                    item["attributes"] = json.loads(item["attributes"])
            parsed.append(item)
            pos_candidates = parsed

        if len(pos_candidates) != 0:
            selected_element = soup.find(attrs={"backend_node_id": pos_candidates[0]["backend_node_id"]})
            ui_element_text = find_semantic_info(selected_element)
            if ui_element_text is not None:
                ui_element_text = clean_text(ui_element_text)
                if len(ui_element_text) > 20:
                    ui_element_text = ui_element_text[:10] + "..."
            else:
                print(f"Warning: {annotation_id}, can not find semantic info for {action_uid}")
                print("skipping this action")
                return
                    

    if isinstance(operation, str):
        operation = json.loads(operation)
    action_description_en, _ = action_discription(
        ui_element_name, ui_element_text, operation["op"], operation["value"]
    )
    action_descriptions_en.append(action_description_en)

    # Use the provided screenshot for all actions
    orig_before_action_screenshot, ratio_before = resize_by_width(before_action_screenshot, OUTPUT_SIZE[0])
    # There is no after_action_screenshot in the new format

    bounding_box = action.get("bounding_box", None)
    if bounding_box is None:
        bounding_box = action.get("action", {}).get("bounding_box", None)

    if bounding_box is None:
        # Try extracting bounding box from pos_candidates
        pos_candidates = action.get("pos_candidates", [])
        if isinstance(pos_candidates, str):
            pos_candidates = json.loads(pos_candidates)
        parsed = []
        for item in pos_candidates:
            if isinstance(item, str):
                item = json.loads(item)
            if "attributes" in item and isinstance(item["attributes"], str):
                item["attributes"] = json.loads(item["attributes"])
            parsed.append(item)
        pos_candidates = parsed
        if len(pos_candidates) > 0 and "attributes" in pos_candidates[0]:
            bbox_str = pos_candidates[0]["attributes"].get("bounding_box_rect", None)
            if bbox_str:
                bbox_parts = [float(x.strip()) for x in bbox_str.split(",")]
                bounding_box = {
                    "x": bbox_parts[0],
                    "y": bbox_parts[1],
                    "width": bbox_parts[2],
                    "height": bbox_parts[3]
                }

    # assert bounding_box is not None, f"bounding_box is None in {annotation_id}, {action_uid}"
    if bounding_box is None:
        print(f"Warning: {annotation_id}, {action_uid} has no bounding box, will skip this action")
        return

    # print(bounding_box)
    # print(pos_candidates)
    # exit(0)

    x1, y1, width, height = (
        bounding_box["x"],
        bounding_box["y"],
        bounding_box["width"],
        bounding_box["height"],
    )
    assert x1 >= 0 and y1 >= 0 and width > 0 and height > 0, f"bounding_box is invalid: {x1}, {y1}, {width}, {height} in {annotation_id}, {action_uid}"

    x2, y2 = x1 + width, y1 + height
    x1, y1, x2, y2 = (
        int(x1 * ratio_before),
        int(y1 * ratio_before),
        int(x2 * ratio_before),
        int(y2 * ratio_before),
    )
    L, U, R, D = x1, y1, x2, y2

    # Crop
    if D < OUTPUT_SIZE[1]:
        uu = 0
    else:
        uu_min = max(0, D - OUTPUT_SIZE[1])
        uu_max = U
        uu = random.randint(uu_min, uu_max)

    before_cropped = crop_and_padding(orig_before_action_screenshot, uu, uu + OUTPUT_SIZE[1])
    # There is no after_action_screenshot, so we ignore it

    L2, U2, R2, D2 = L, U - uu, R, D - uu

    # bbox_center_width, bbox_center_height = calculate_edge_centroid(before_cropped, (L2, U2, R2, D2))

    # simply use the center of the bounding box
    bbox_center_width = (L2 + R2) // 2
    bbox_center_height = (U2 + D2) // 2

    # No screen change logic: always set to False
    move_screen_to_top_after_action = False
    if_open_new_session = False
    open_new_session_reason = OpenNewSessionReason.NONE

    action_screens.append({
            "action_uid": action_uid,
            "orig_before_action_screenshot": orig_before_action_screenshot,
            "uu": uu,
            "before_action_screenshot": before_cropped,
            "operation_type": operation["op"],
            "operation_value": operation["value"],
            "orig_bounding_box": (L, U, R, D),
            "LURD_bbox": (L2, U2, R2, D2),
            "center_width": bbox_center_width,
            "center_height": bbox_center_height,
            "if_open_new_session": if_open_new_session,
            "open_new_session_reason": open_new_session_reason,
    })

    # Split session logic: treat all actions as a single session
    split_session = [(0, len(action_screens))]

    advice_en = None
    advice_zh = None

    for sub_session_index, (split_start_index, split_end_index) in enumerate(split_session):
        image = action_screens[split_start_index]["before_action_screenshot"]
        video_info = {
            "video_height": image.size[1],
            "video_width": image.size[0],
        }
        # 1. plan stage
        ## send_prompt
        plan_send_prompt_info = {
            **base_info,
            **video_info,
            "advice_en": advice_en,
            "advice_zh": advice_zh,
        }
        # plan_send_prompt_en = templates["planner_agent_send_prompt_en"].render(**plan_send_prompt_info)
        # plan_send_prompt_zh = templates["planner_agent_send_prompt_zh"].render(**plan_send_prompt_info)

        ## answer
        sub_session_task_list_en = action_descriptions_en[split_start_index:split_end_index]

        plan_answer_info = {
            **base_info,
            "sub_task_list_en": convert_string(sub_session_task_list_en),
        }
        # plan_answer_en = templates["planner_agent_answer_en"].render(**plan_answer_info)
        # plan_answer_zh = templates["planner_agent_answer_zh"].render(**plan_answer_info)

        saved_json_name = f"{annotation_id}_sub_session{sub_session_index}_plan.json"
        saved_image_name = f"{annotation_id}_sub_session{sub_session_index}_plan.png"
        action_uid = action_screens[split_start_index]["action_uid"]
        save_data = {
            "action_uid": action_uid,
            **base_info,
            **video_info,
            "sub_task_list_en": sub_session_task_list_en,
            "saved_image_name": saved_image_name
        }
        # save_example_file(annotation_id, saved_json_name, save_data, image, saved_image_name)

        advice_en = None
        advice_zh = None

        for action_index in range(split_start_index, split_end_index):
            # 2. action stage
            image = action_screens[action_index]["before_action_screenshot"]
            action_uid = action_screens[action_index]["action_uid"]
            video_info = {
                "video_height": image.size[1],
                "video_width": image.size[0],
            }
            current_task_info = {
                "sub_task_list_en": sub_session_task_list_en,
                "current_task_en": action_descriptions_en[action_index],
            }
            ## action send_prompt
            action_send_prompt_info = {
                **base_info,
                **video_info,
                **current_task_info,
                "advice_en": advice_en,
                "advice_zh": advice_zh,
            }
            # action_send_prompt_en = templates["actor_agent_send_prompt_en"].render(**action_send_prompt_info)
            # action_send_prompt_zh = templates["actor_agent_send_prompt_zh"].render(**action_send_prompt_info)

            ## action answer
            is_last_action_in_subsession = action_index == split_end_index - 1

            action_answer_info = {
                # **base_info,
                # **current_task_info,
                "operation_type": action_screens[action_index]["operation_type"],
                "operation_value": convert_string(action_screens[action_index]["operation_value"]),
                "center_width": action_screens[action_index]["center_width"],
                "center_height": action_screens[action_index]["center_height"],
                "is_last_action_in_subsession": is_last_action_in_subsession
            }

            # action_answer_en = templates["actor_agent_answer_en"].render(**action_answer_info)
            # action_answer_zh = templates["actor_agent_answer_zh"].render(**action_answer_info)

            saved_json_name = f"{annotation_id}_sub_session{sub_session_index}_action{action_index}.json"
            saved_image_name = f"{annotation_id}_sub_session{sub_session_index}_action{action_index}.png"
            save_data = {
                "action_uid": action_uid,
                **base_info,
                **video_info,
                **current_task_info,
                # "send_prompt_en": action_send_prompt_en,
                # "send_prompt_zh": action_send_prompt_zh,
                # "answer_en": action_answer_en,
                # "answer_zh": action_answer_zh,
                "answer_en": action_answer_info,
                "saved_image_name": saved_image_name
            }
            save_example_file(annotation_id, saved_json_name, save_data, image, saved_image_name)
            save_image_for_check(annotation_id, image, saved_image_name, action_answer_info)

            advice_en = None
            advice_zh = None

if DEBUG:
    for task in tqdm(dataset["test_task"]): # for debug
        process_one_task(task)
else:
    from multiprocessing import Pool
    import traceback

    def process_one_task_wrapper(task):
        try:
            process_one_task(task)
        except Exception as e:
            print(f"Error: {e}, Annotation ID: {task['annotation_id']}")
            print(traceback.format_exc())
            return None
        return task["annotation_id"]

    def process_dataset(dataset):
        with Pool(processes=20) as pool:
            with tqdm(total=len(dataset)) as pbar:
                for i, _ in enumerate(pool.imap(process_one_task_wrapper, dataset)):
                    pbar.update()
            return 
        
    process_dataset(dataset["test_task"])

print("done")
import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import random
import openai
import base64
import sys
import re
import math

sys.path.append(".")

from modules.api_tools import LLMTool
from modules.prompts import *

import time

# # 直接内置Position定义
# class Position:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height

# 直接内置解析函数
def parse_action_from_text(response_text):
    # 提取 ```json ... ``` 中的内容（如果有）

    positions = []
    for action in response_text:
        if (
            action.get("action_type") == "MouseAction"
            and "mouse_position" in action
            and isinstance(action["mouse_position"], dict)
        ):
            pos = action["mouse_position"]
            if "width" in pos and "height" in pos:
                # 直接使用 dict 而非 Position 类
                positions.append({
                    "width": pos["width"],
                    "height": pos["height"]
                })
    return positions

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

def process_dataset(processed_dataset_folder_path, output_folder_path, sample_start, sample_end, repeat_num):
    os.makedirs(output_folder_path, exist_ok=True)

    subfolders = [d for d in os.listdir(processed_dataset_folder_path) if os.path.isdir(os.path.join(processed_dataset_folder_path, d))]

    if sample_start < sample_end and sample_end <= len(subfolders):
        subfolders = subfolders[sample_start:sample_end]

    for folder_name in tqdm(subfolders, desc="Processing dataset"):
        folder_path = os.path.join(processed_dataset_folder_path, folder_name)

        output_folder_path_name = os.path.join(output_folder_path, folder_name) 
        output_json_folder = os.path.join(output_folder_path_name, "annotations")
        output_image_folder = os.path.join(output_folder_path_name, "images")
        os.makedirs(output_json_folder, exist_ok=True)
        os.makedirs(output_image_folder, exist_ok=True)

        # Get the json file and UI file
        json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith(folder_name + "_sub_session0_action0")), None)
        UI_file = next((f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith(folder_name + "_candidates")), None)
        images_folder = os.path.join(folder_path, "images")
        image_file_clean = os.path.join(images_folder, folder_name + "_sub_session0_action0.png")
        image_file_bounding_box = os.path.join(images_folder, folder_name + "_sub_session0_action0.png_check.png")
        image_file_annotated = os.path.join(images_folder, folder_name + "_sub_session0_action0.png_annotated.png")

        if json_file is None or not os.path.exists(image_file_clean):
            print(f"Skipping {folder_name}, missing json {image_file_clean} or image {image_file_clean}.")
            continue
        if json_file is None or not os.path.exists(image_file_bounding_box):
            print(f"Skipping {folder_name}, missing json {image_file_bounding_box} or image {image_file_bounding_box}.")
            continue
        if json_file is None or not os.path.exists(image_file_annotated):
            print(f"Skipping {folder_name}, missing json {image_file_annotated} or image {image_file_annotated}.")
            continue

        image_clean = Image.open(image_file_clean)
        image_bounding_box = Image.open(image_file_bounding_box)
        image_annotated = Image.open(image_file_annotated)

        json_path = os.path.join(folder_path, json_file)
        UI_path = os.path.join(folder_path, UI_file)
        with open(UI_path, 'r', encoding='utf-8') as f:
            UI_tree = json.load(f)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        video_width = data['video_width']
        video_height = data['video_height']
        task_prompt_en = data['task_prompt_en']
        current_task_en = data['current_task_en']
        annotation_id = data['annotation_id']
        answer = data.get("answer_en", {})
        bounding_box = answer.get("bounding_box", None)
        center_width = answer.get("center_width", None)
        center_height = answer.get("center_height", None)

        Memory_result = []
        current_mouse_position = (0,0)
        scores = []

        draw = ImageDraw.Draw(image_annotated)

        for idx in range(repeat_num):

            user_prompt = user_prompt_action.format(
                task_prompt=task_prompt_en,
                ui_tree="",
                os_version="MacOS 15.1",
                app_name="Safari",
                window_size=(video_width, video_height),
                mouse_position=current_mouse_position,
                current_task=current_task_en,
                previous_tool_calls=Memory_result,
            )

            seg_im_base64 = encode_image_base64(image_clean)
            try:
                response = llm.run(sys_prompt_action, user_prompt, screenshot_base64=seg_im_base64)
            except Exception as e:
                print(f"Error in LLM call: {e}, skipping this iteration.")
                continue
            action = llm.postprocess_by_prompt(response)
            try:
                actions = parse_action_from_text(action)
                current_mouse_position = actions[-1]["width"], actions[-1]["height"]
                Memory_result.append(actions[-1])
            except Exception as e:
                print(f"Error parsing action: {e}")
                continue
            

            time.sleep(5)

            scores.append(math.sqrt(((actions[-1]["width"] - center_width)) ** 2 + ((actions[-1]['height'] - center_height)) ** 2))

            r = 5
            draw.line([
                (actions[-1]["width"] - r, actions[-1]["height"] - r),
                (actions[-1]["width"]+ r, actions[-1]["height"] + r)
            ], fill=(255,0,0), width=3)
            draw.line([
                (actions[-1]["width"] - r, actions[-1]["height"] + r),
                (actions[-1]["width"] + r, actions[-1]["height"] - r)
            ], fill=(255,0,0), width=3)
            
            font = ImageFont.load_default()
            font.size = 32
            draw.text((actions[-1]["width"] + 5, actions[-1]["height"] - 5), str(idx), fill=(255,0,0),font=font)

        with open(os.path.join(output_json_folder, folder_name + ".json"), 'w', encoding='utf-8') as f:
            json.dump({"annotation_id":annotation_id, "Memory_result":Memory_result, "scores":scores}, f, indent=2)
        
    draw.rectangle([
        (bounding_box[0], bounding_box[1]),
        (bounding_box[2], bounding_box[3])
    ], outline=(0,255,0), width=3)

    image_annotated.save(os.path.join(output_image_folder, folder_name + "final.png"))

    print(" All processing done.")
    print("Average score: ", sum(scores) / len(scores))

dataset_path = "processed_dataset"
output_path = "test_ours_abl_cleanim"
sampel_start = 39
sampel_end = 135
repeat_num = 1

llm = LLMTool()

process_dataset(dataset_path, output_path, sampel_start, sampel_end, repeat_num)

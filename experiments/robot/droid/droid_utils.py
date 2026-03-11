from droid.robot_env import RobotEnv
from droid.controllers.oculus_controller import VRPolicy
from droid.user_interface.sensor_manager import SensorManager
import tensorflow as tf
from PIL import Image
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

import os 
import enum
import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import imageio
sensor_manager = SensorManager()


def get_droid_env():
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    return env


def _resize_image(image, size):
    im = Image.fromarray(image)
    height, width = image.shape[0] // 2 - 100, image.shape[1] // 2 - 350
    # print(height, width)
    resized_main = tf.image.crop_to_bounding_box(im, 
                                        height, 
                                        width, 
                                        460, 
                                        700)
    resized_main = tf.image.resize(resized_main, [size[0], size[1]], method=tf.image.ResizeMethod.BILINEAR)
    resized_main = tf.cast(resized_main, tf.uint8).numpy()
    # BRG to RGB
    resized_main = resized_main[:, :, ::-1]
    return resized_main


def get_droid_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["full_image"]
    img = _resize_image(img, resize_size)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_droid_observation(env):
    # Get robot state observation.
    robot_obs = env.get_observation()
    
    # Collect sensor data.
    thermal_raw, thermal_processed, color_img, depth_img, zed_image = sensor_manager.get_sensor_data()
    
    # Merge sensor data with robot state.
    # invert image upside down
    # color_img = cv2.rotate(color_img, cv2.ROTATE_180)
    # color_img = get_droid_image({"full_image": color_img}, (256, 256))
    observation = dict(robot_obs)
    observation.update({
        "full_image": color_img,
        "thermal_processed": thermal_processed,
        "depth_image": depth_img,
        "zed_image": zed_image,
    })

    return observation

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = f"./rollouts/{DATE}/{processed_task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def save_rollot_reasoning(rollout_reasoning, rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = f"./rollouts/{DATE}/{processed_task_description}"
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}/"
    os.makedirs(mp4_path, exist_ok=True)
    for i in range(len(rollout_reasoning)):
        draw_reasoning(rollout_images[i], rollout_reasoning[i]).save(mp4_path + f"reasoning_{i}.png")


# utils

#Define some utils.
def draw_reasoning(image, generated_text):
    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(generated_text, tags)
    text = [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                            ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
    metadata = get_metadata(reasoning)
    bboxes = {}
    for k, v in metadata["bboxes"].items():
        if k[0] == ",":
            k = k[1:]
        bboxes[k.lstrip().rstrip()] = v

    caption = ""
    for t in text:
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False) 
        word_list = wrapper.wrap(text=t) 
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n      '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n\n"

    base = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default(size=14) # big text
    color = (0,0,0) # RGB
    draw.text((30, 30), caption, color, font=font)
    # rescale image to 480 640 3
    # image = image.copy()
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS).copy()
    img_arr = np.array(resized_image)

    # img_arr = np.array(image)
    draw_gripper(img_arr, metadata["gripper"])
    draw_bboxes(img_arr, bboxes)

    text_arr = np.array(base)
    reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))

    return reasoning_img

def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
                # print(tag, s)
            else:
                new_parts[k] = v

    return new_parts

class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    ACTION = "ACTION:"


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.ACTION.value,
    ]

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]


def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

# def get_metadata(reasoning):
#     metadata = {"gripper": [[0, 0]], "bboxes": dict()}

#     if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
#         gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
#         gripper_pos = gripper_pos.split("[")[-1]
#         gripper_pos = gripper_pos.split("]")[0]
#         gripper_pos = [int(x) for x in gripper_pos.split(",")]
#         gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
#         metadata["gripper"] = gripper_pos

#     if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
#         for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
#             obj = sample.split("[")[0]
#             if obj == "":
#                 continue
#             coords = [int(n) for n in sample.split("[")[-1].split(",")]
#             metadata["bboxes"][obj] = coords

#     return metadata

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        # Filter and validate integers
        gripper_pos = [x.strip() for x in gripper_pos.split(",") if x.strip().isdigit()]
        gripper_pos = [int(x) for x in gripper_pos]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            try:
                coords = [x.strip() for x in sample.split("[")[-1].split(",") if x.strip().isdigit()]
                coords = [int(n) for n in coords]
                metadata["bboxes"][obj] = coords
            except (ValueError, IndexError):
                # Handle malformed data gracefully
                print(f"Warning: Skipping malformed data for object {obj}: {sample}")
                continue

    return metadata

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'
        if len(bbox) != 4:
            continue
        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

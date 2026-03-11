"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
import enum

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.utils import TensorType
import time 

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d-%H_%M_%S")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_vla(cfg):
    
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    # print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    elif cfg.norm_stats and os.path.isfile(cfg.norm_stats):
        with open(cfg.norm_stats, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )
    return vla
    

def hf_to_vllm(vla, processor, cfg):
    import vllm
    # Get imbeddings
    vla.input_embds = vla.language_model.get_input_embeddings()

    # Save language model 
    vllm_model_path = f"logs/{cfg.pretrained_checkpoint.replace('/', '_')}-vllm"
    if not os.path.exists(vllm_model_path):
        vla.language_model.save_pretrained(vllm_model_path)
        processor.save_pretrained(vllm_model_path)

    # Load language model with VLLM
    if hasattr(vla, "language_model"):
        del vla.language_model
    # TODO: check vllm load mode, check settings, memory
    # check if async engine is enabled
    if not cfg.async_engine:
        vla.language_model = vllm.LLM(vllm_model_path, 
                                      trust_remote_code=True, 
                                      gpu_memory_utilization=0.7, 
                                      preemption_mode='swap', 
                                      swap_space = 10, 
                                      enable_chunked_prefill = True, 
                                      enable_prefix_caching = True, 
                                      max_num_seqs = 10)
    else:
        vla.language_model  = vllm.AsyncLLMEngine.from_engine_args(
                vllm.AsyncEngineArgs(
                    model="logs/llama-bridge",
                    gpu_memory_utilization=0.64,
                    preemption_mode="swap",
                    swap_space=10,
                    disable_log_requests=True,
                )
        )
    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    
    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, max_new_tokens=None, prompts=None):
    """Generates an action with the VLA policy."""

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")
    infer_time = 0
    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy()) 
        image = image.convert("RGB")
        # print(f'image size: {image.size}')

    # 2. Process original prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    elif "ecot" in base_vla_name: # ECoT
        prompt = f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT: TASK:"
    else:  # OpenVLA 
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # 3. VLLM inference
    if hasattr(vla, 'use_vllm') and vla.use_vllm:
        import vllm # only executed once

        if prompts is None: 
            prompts = [prompt]
            sampling_params = vllm.SamplingParams(temperature=0, max_tokens=max_new_tokens, stop_token_ids=[2])
        else:
            sampling_params = vllm.SamplingParams(temperature=0, max_tokens=max_new_tokens, stop_token_ids=[29901])
        inputs = [processor.tokenizer(p, return_tensors=TensorType.PYTORCH)['input_ids'].to(DEVICE) for p in prompts]
        pixel_values = processor.image_processor(image, return_tensors=TensorType.PYTORCH)["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
        
        start_time = time.perf_counter()
        outputs = vla.vllm_inference(input_ids=inputs, pixel_values=pixel_values, sampling_params=sampling_params)
        infer_time = time.perf_counter() - start_time 
        # --------------------------------------------------
        # TODO: this should be put into modeling_prismatic.py
        generated_ids = []
        for i, o in zip(inputs, outputs):
            generated_ids.append(i[0].cpu().numpy().tolist() + list(o.outputs[0].token_ids))
        # generated_ids = np.array(generated_ids)
        # Fetch normalized actions
        predicted_action_token_ids = np.array(generated_ids[-1][-(vla.get_action_dim(unnorm_key) + 1) : -1])
        discretized_actions = vla.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
        normalized_actions = vla.bin_centers[discretized_actions]
        # Unnormalize actions
        action_norm_stats = vla.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )    
        # --------------------------------------------------
        return infer_time, actions, generated_ids


    # 3. - HF inference
    # Process inputs
    if prompts: # batch style
        processor.tokenizer.padding_side = 'left'
        inputs = processor(prompts, image, padding=True).to(DEVICE, dtype=torch.bfloat16)
    else: 
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    # Get action
    if 'ecot' in base_vla_name: # ECoT
        start_time = time.perf_counter()
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)
        infer_time = time.perf_counter() - start_time
        return infer_time, action # action, generated_ids
    else: # OpenVLA
        start_time = time.perf_counter()
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        infer_time = time.perf_counter() - start_time
        return infer_time, action, [[]]


# M: batch prediction
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


class PromptManager(object):
                        
    def __init__(self, ):
        # Intialize subtask history
        self.subtask_history = dict()
        for t in CotTag:
            self.subtask_history[t.name] = [""]

    def update_history(self, generated_text, index=None):
        """ Update subtask history based on 
            Args:
                - index: if index is specified, only extract that subtask
        """
        if index is None: 
            start_tag_id = 0
            end_tag_id = len(CotTag) - 1
        else:
            start_tag_id = index
            end_tag_id = index + 1
        
        # Extract
        cottag_list = list(CotTag)
        for i in range(start_tag_id, end_tag_id):
            start_idx = generated_text.find(cottag_list[i].value)
            end_idx = generated_text.find(cottag_list[i+1].value)
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                subtask_text =  generated_text[start_idx+len(cottag_list[i].value):end_idx]
                self.subtask_history[cottag_list[i].name].append(subtask_text)


    def generate_prompts(self, task_description):
        """ Generate batch prompts, with history
        """
        prompts = []
        prompt = f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: "
        for i, t in enumerate(CotTag):
            prompt = prompt + t.value 
            prompts.append(prompt)
            if i == len(CotTag) - 1: break
            try:
                prompt = prompt + self.subtask_history[t.name][-1] # Use updated history
            except:
                raise ValueError(f"Subtask {t.name} not found in history, history: {self.subtask_history}")
        return prompts

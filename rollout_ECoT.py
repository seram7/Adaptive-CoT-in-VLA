# Cell 1: imports
import inspect
import os
import textwrap
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor

import sys

sys.path.insert(0, "/home/seram/ut/project/LIBERO")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    save_rollot_reasoning,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_processor

from libero.libero import benchmark

print("libero imported : OK")

ACTION_TOKEN_BEGIN_IDX = 31743
N_ACTION_BINS = 256
ACTION_TOKEN_START = ACTION_TOKEN_BEGIN_IDX + 1
ACTION_TOKEN_END = ACTION_TOKEN_START + N_ACTION_BINS

unnorm_key = {
    "libero_spatial_no_noops": {
        "action": {
            "mean": [
                0.15312467515468597,
                0.1370728462934494,
                -0.15526831150054932,
                -0.005176443140953779,
                -0.011208743788301945,
                -0.020194262266159058,
                0.4578818082809448,
            ],
            "std": [
                0.41272789239883423,
                0.3472437262535095,
                0.5086919665336609,
                0.03726620972156525,
                0.07244434952735901,
                0.057623643428087234,
                0.4982786774635315,
            ],
            "max": [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375, 1.0],
            "min": [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475, 0.0],
            "q01": [
                -0.7454732114076613,
                -0.6616071462631226,
                -0.9375,
                -0.1071428582072258,
                -0.20678570866584778,
                -0.1842857152223587,
                0.0,
            ],
            "q99": [
                0.9375,
                0.8758928775787354,
                0.9321428537368774,
                0.1039285734295845,
                0.17678570747375488,
                0.14571428298950195,
                1.0,
            ],
            "mask": [True, True, True, True, True, True, False],
        },
        "proprio": {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "num_transitions": 52970,
        "num_trajectories": 432,
    }
}


def compute_step_uncertainty_from_action_scores(action_scores):
    """
    action_scores: list/tuple of length 7
        each element shape [1, vocab_size] or [vocab_size]

    returns:
        dict with:
            entropy_per_slot: [7]
            top3_mass_per_slot: [7]
            entropy_mean: scalar
            top3_mass_mean: scalar
            logits_7xV: [7, vocab_size]
    """
    step_logits = torch.stack(
        [x.squeeze(0).float().cpu() for x in action_scores], dim=0
    )  # [7, vocab_size]

    action_logits = step_logits[:, ACTION_TOKEN_START:ACTION_TOKEN_END]  # [7, 256]
    probs = torch.softmax(action_logits, dim=-1)

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [7]
    top3_mass = torch.topk(probs, k=3, dim=-1).values.sum(dim=-1)  # [7]

    return {
        "entropy_per_slot": entropy.numpy(),
        "top3_mass_per_slot": top3_mass.numpy(),
        "entropy_mean": float(entropy.mean().item()),
        "top3_mass_mean": float(top3_mass.mean().item()),
        "logits_7xV": step_logits,
    }


def save_rollout_pt(
    save_dir,
    task_name,
    task_id,
    trial_id,
    success,
    step_ids,
    rollout_logits,
    entropy_series,
    top3_mass_series,
):
    """
    rollout_logits: list of [7, vocab_size] tensors, one per simulation step
    step_ids: list of ints, one per simulation step
    entropy_series: list of floats, one per simulation step
    top3_mass_series: list of floats, one per simulation step
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(rollout_logits) == 0:
        logits = torch.empty(0)
    else:
        logits = torch.stack(rollout_logits, dim=0).float()  # [T, 7, vocab]

    if len(entropy_series) == 0:
        entropy_series = torch.empty(0)
    else:
        entropy_series = torch.tensor(entropy_series, dtype=torch.float32)  # [T]
    if len(top3_mass_series) == 0:
        top3_mass_series = torch.empty(0)
    else:
        top3_mass_series = torch.tensor(top3_mass_series, dtype=torch.float32)  # [T]

    payload = {
        "task_name": str(task_name),
        "task_id": int(task_id),
        "trial_id": int(trial_id),
        "success": int(success),
        "step_ids": np.asarray(step_ids, dtype=np.int32),
        "logits": logits,
        "entropy_series": entropy_series,
        "top3_mass_series": top3_mass_series,
    }

    pt_path = save_dir / f"task{task_id:02d}_trial{trial_id:02d}.pt"
    torch.save(payload, pt_path)
    return pt_path


#####################
def eval_libero_entropy():
    num_trials_per_task = 10

    checkpoint = "leepanic/ecot-libero-spatial-r32"
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )

    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    vla.norm_stats["libero_spatial"] = unnorm_key["libero_spatial_no_noops"]
    vla.norm_stats.update(unnorm_key)

    task_suite_name = "libero_spatial"
    cfg = GenerateConfig(
        pretrained_checkpoint="leepanic/ecot-libero-spatial-r32",
        unnorm_key=task_suite_name,
        center_crop=True,
        num_trials_per_task=2,
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")

    # Get expected image dimensions
    resize_size = 224
    inference_times = []
    # Start evaluation
    total_episodes, total_successes = 0, 0
    num_steps_wait = 10

    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(num_trials_per_task)):
            print(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_reasoning = []
            if task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            rollout_logits = []
            step_ids = []
            entropy_series = []
            top3_mass_series = []
            save_dir = f"./rollouts/{DATE}/{task_description}/trial{episode_idx:02d}"
            success = 0
            while t < max_steps + num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }

                    # Query model to get action
                    inference_time, action, generated_ids, action_scores = get_action(
                        cfg,
                        vla,
                        observation,
                        task_description,
                        processor=processor,
                        max_new_tokens=1024,
                    )
                    inference_times.append(inference_time)
                    generated_text = processor.batch_decode(generated_ids)[0]
                    replay_reasoning.append(generated_text)
                    print(generated_text)

                    action_uncertainty = compute_step_uncertainty_from_action_scores(action_scores)
                    rollout_logits.append(action_uncertainty["logits_7xV"])  # [7, vocab]
                    step_ids.append(t)
                    entropy_series.append(action_uncertainty["entropy_mean"])
                    top3_mass_series.append(action_uncertainty["top3_mass_mean"])

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    action = invert_gripper_action(action)

                    print(f"Inference time: {inference_time:.4f} seconds\n")
                    print(f"Action: {action}")
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        success = 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    break

            save_rollout_pt(
                save_dir=save_dir,
                task_name=task_description,
                task_id=task_id,
                trial_id=episode_idx,
                success=success,
                step_ids=step_ids,
                rollout_logits=rollout_logits,
                entropy_series=entropy_series,
                top3_mass_series=top3_mass_series,
            )
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description
            )
            save_rollot_reasoning(
                replay_reasoning,
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
            )
            env.close()


if __name__ == "__main__":
    eval_libero_entropy()

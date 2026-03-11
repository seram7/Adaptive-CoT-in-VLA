# Fine-Tuning Guide

This guide covers how to fine-tune ECoT models on various datasets using LoRA or full fine-tuning.

## Prerequisites

Make sure you have completed the [installation steps](../README.md#installation) before proceeding.

### Download Base Models and Data

```bash
# Install Git LFS
sudo apt update && sudo apt install git-lfs
git lfs install

# Download the base model
git clone https://huggingface.co/openvla/openvla-7b-prismatic
cd openvla-7b-prismatic && git lfs fetch --all && cd ..

# Download dataset (example: modified LIBERO)
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```

### Prepare Reasoning Annotations

If using ECoT reasoning, place the reasoning JSON file in the dataset directory:

```bash
cp reasoning.json <DATA_ROOT_DIR>/<DATASET_NAME>/reasoning.json
```

## Full Fine-Tuning

Train from scratch on a dataset with reasoning annotations:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <NUM_GPUS> vla-scripts/train.py \
  --pretrained_checkpoint <PATH_TO_BASE_MODEL_CHECKPOINT> \
  --vla.type siglip-224px+mx-libero \
  --data_root_dir <PATH_TO_DATA> \
  --run_root_dir <PATH_TO_OUTPUTS> \
  --image_aug True \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY> \
  --save_interval 40000 \
  --is_resume False
```

## LoRA Fine-Tuning

Parameter-efficient fine-tuning using Low-Rank Adaptation:

### LIBERO Object

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir <PATH_TO_DATA> \
  --dataset_name libero_object_no_noops \
  --run_root_dir <PATH_TO_OUTPUTS> \
  --adapter_tmp_dir <PATH_TO_OUTPUTS>/temp \
  --reasoning_dropout_prob 0 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY> \
  --save_steps 50000
```

### LIBERO Goal

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir <PATH_TO_DATA> \
  --dataset_name libero_goal_no_noops \
  --run_root_dir <PATH_TO_OUTPUTS> \
  --adapter_tmp_dir <PATH_TO_OUTPUTS>/temp \
  --reasoning_dropout_prob 0 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY> \
  --save_steps 50000
```

### Bridge

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir <PATH_TO_DATA> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH_TO_OUTPUTS> \
  --adapter_tmp_dir <PATH_TO_OUTPUTS>/temp \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY> \
  --save_steps 20000
```

### Custom DROID Dataset

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir <PATH_TO_DATA> \
  --dataset_name <CUSTOM_DATASET_NAME> \
  --run_root_dir <PATH_TO_OUTPUTS> \
  --adapter_tmp_dir <PATH_TO_OUTPUTS>/temp \
  --reasoning_dropout_prob 0 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY> \
  --save_steps 50000
```

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `--vla_path` | HuggingFace model path or local checkpoint | `openvla/openvla-7b` |
| `--data_root_dir` | Root directory containing RLDS datasets | — |
| `--dataset_name` | Name of the dataset to fine-tune on | — |
| `--lora_rank` | Rank of LoRA weight matrices | `32` |
| `--learning_rate` | Learning rate | `2e-5` |
| `--batch_size` | Batch size per GPU | `16` |
| `--image_aug` | Enable image augmentations | `True` |
| `--reasoning_dropout_prob` | Dropout for reasoning tokens during training | `0.0` |
| `--action_loss` | Add explicit action-only loss term | `False` |
| `--save_steps` | Checkpoint save interval (gradient steps) | `50000` |
| `--use_quantization` | 4-bit quantization for reduced memory | `False` |

## Memory Requirements

| Setup | GPU Memory |
|---|---|
| LoRA (rank 32, batch 12) | ~48 GB |
| LoRA (rank 32, batch 24) | ~80 GB |
| Full fine-tune | 8× 80 GB GPUs recommended |

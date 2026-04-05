# Adaptive CoT in VLA: Chain-of-Thought for Vision-Language-Action Models with Uncertain Task

[![arXiv](https://img.shields.io/badge/arXiv-2506.07639-df2a2a.svg)](https://arxiv.org/abs/2506.07639)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Zhekai Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan,+Z), [Yuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+Y), [Shikai Geng](https://arxiv.org/search/cs?searchtype=author&query=Geng,+S), [Gaowen Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+G), [Joschka Boedecker](https://arxiv.org/search/cs?searchtype=author&query=Boedecker,+J), [Chris Xiaoxuan Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu,+C+X)

---

**Adaptive CoT in VLA** is an inference-time method for Embodied Chain-of-Thought (ECoT) reasoning in vision-language-action (VLA) models. It brings ECoT policies closer to **practical real-time deployment** without any model changes or additional training.

> Embodied Chain-of-Thought (ECoT) reasoning enhances VLA models by improving performance and interpretability through intermediate reasoning steps. However, its sequential autoregressive token generation introduces significant inference latency, limiting real-time deployment. We propose **Fast ECoT**, which exploits the structured and repetitive nature of ECoT to **(1)** cache and reuse high-level reasoning across timesteps, **(2)** parallelise the generation of modular reasoning steps, and **(3)** decouple reasoning from action decoding via an asynchronous scheduler. Experiments in both simulation (LIBERO) and real-world robot tasks show **up to 7.5× reduction in latency** with comparable or improved task success rate and reasoning faithfulness.

## Demo

<p align="center">
  <img src="media/demo.gif" alt="Fast ECoT Demo" width="100%">
</p>

## Key Features

| Technique | Description |
|---|---|
| **Thought Caching & Reuse** | Caches high-level reasoning (task, plan, subtask) across timesteps and reuses them when unchanged, avoiding redundant generation |
| **Parallel Reasoning Generation** | Parallelises the generation of modular ECoT reasoning steps using batched prompts with cached history |
| **Asynchronous Scheduler** | Decouples reasoning generation from action decoding, enabling the robot to act while new reasoning is being computed |

## Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (16 GB+ VRAM recommended)

### Environment Setup

```bash
# Create and activate conda environment
conda create -n fast-ecot python=3.10 -y
conda activate fast-ecot

# Install PyTorch (adjust for your CUDA version)
# See: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone and install
git clone https://github.com/kevinDuan1/Fast-ECoT.git
cd Fast-ECoT
pip install -e .

# Install Flash Attention 2 for training
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### Optional: vLLM (for accelerated inference)

```bash
pip install vllm
```

## Training

### Full Fine-Tuning

Train from scratch on RLDS-formatted datasets:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH_TO_DATA> \
  --run_root_dir <PATH_TO_CHECKPOINTS> \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY>
```

### LoRA Fine-Tuning

Parameter-efficient fine-tuning with LoRA on specific datasets:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir <PATH_TO_DATA> \
  --dataset_name <DATASET_NAME> \
  --run_root_dir <PATH_TO_CHECKPOINTS> \
  --adapter_tmp_dir <PATH_TO_ADAPTER_TMP> \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project <WANDB_PROJECT> \
  --wandb_entity <WANDB_ENTITY> \
  --save_steps 20000
```

See [finetuning guide](docs/finetuning.md) for detailed instructions on fine-tuning with different datasets (LIBERO, Bridge, DROID).

## Evaluation

### LIBERO Simulation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <CHECKPOINT_PATH> \
  --task_suite_name libero_spatial \
  --center_crop True \
  --reasoning True \
  --use_vllm True
```

**Task suites**: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`

**Fast ECoT flags**:
- `--use_vllm True` — Enable vLLM-accelerated inference
- `--reasoning True` — Enable ECoT reasoning generation
- `--async_engine True` — Enable asynchronous reasoning scheduler

### Batch Evaluation

Run batch evaluation across multiple configs:

```bash
python experiments/robot/libero/run_libero_eval_batch.py \
  --model_family openvla \
  --pretrained_checkpoint <CHECKPOINT_PATH> \
  --task_suite_name libero_object \
  --center_crop True \
  --reasoning True
```

### Real-World Robots

#### Bridge (WidowX)

```bash
python experiments/robot/bridge/run_bridgev2_eval.py \
  --pretrained_checkpoint <CHECKPOINT_PATH>
```

#### DROID

```bash
python experiments/robot/droid/run_droid_eval.py \
  --pretrained_checkpoint <CHECKPOINT_PATH>
```

## Repository Structure

```
Fast-ECoT/
├── prismatic/               # Core package: model loading, training, data utils
│   ├── models/              # VLA model definitions and backbones
│   ├── vla/                 # Action tokenizer, datasets, and VLA utilities
│   └── util/                # CoT utilities, data helpers
├── vla-scripts/             # Training and deployment scripts
│   ├── train.py             # Full model training
│   ├── finetune.py          # LoRA fine-tuning
│   └── deploy.py            # Model deployment
├── experiments/
│   ├── robot/               # Real-world and simulation evaluation
│   │   ├── libero/          # LIBERO simulation eval scripts
│   │   ├── bridge/          # WidowX Bridge eval scripts
│   │   ├── droid/           # DROID eval scripts
│   │   ├── openvla_utils.py # VLA inference utils (vLLM, batching, PromptManager)
│   │   ├── async_utils.py   # Asynchronous inference engine
│   │   └── robot_utils.py   # Shared robot utilities
│   └── bridge/              # Legacy Bridge eval (WidowX)
├── scripts/                 # Data generation and preprocessing
├── docs/                    # Documentation
│   └── finetuning.md        # Detailed finetuning guide
├── media/                   # Demo videos and figures
└── pyproject.toml           # Project configuration and dependencies
```

## Pretrained Models

Our models are based on the [Embodied-CoT](https://huggingface.co/Embodied-CoT) checkpoints:

| Model | Description |
|---|---|
| [`ecot-openvla-7b-bridge`](https://huggingface.co/Embodied-CoT/ecot-openvla-7b-bridge) | ECoT trained on Bridge dataset with reasoning annotations (80k steps) |
| [`ecot-openvla-7b-oxe`](https://huggingface.co/Embodied-CoT/ecot-openvla-7b-oxe) | ECoT trained on OXE, fine-tuned with Bridge reasoning (20k steps) |

**Note on Licensing**: While all code is released under the MIT License, pretrained models may inherit restrictions from Llama-2 and are subject to the [Llama Community License](https://ai.meta.com/llama/license/).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{Duan2025-fast-ecot,
    title={Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse},
    author={Zhekai Duan and Yuan Zhang and Shikai Geng and Gaowen Liu and Joschka Boedecker and Chris Xiaoxuan Lu},
    journal={arXiv preprint arXiv:2506.07639},
    year={2025}
}
```

## Acknowledgements

This codebase is built on top of the following excellent works:

- **[Embodied Chain-of-Thought (ECoT)](https://github.com/MichalZaworski/embodied-CoT)** by Zawalski et al. — The original ECoT reasoning framework for VLA models ([arXiv:2407.08693](https://arxiv.org/abs/2407.08693))
- **[OpenVLA](https://github.com/openvla/openvla)** — The base VLA model architecture and training pipeline
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM inference engine used for accelerated reasoning generation

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
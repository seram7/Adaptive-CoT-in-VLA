#!/usr/bin/env python
"""JSONL stdin/stdout server for DeepThinkVLA action chunks.

Protocol:
  request: {"request_id": "...", "npz_path": "...", "task": "..."}
  npz keys: full_image, wrist_image, state
  response: {"request_id": "...", "ok": true, "actions": [[...]], "reasoning": "..."}
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor


DEFAULT_DEEPTHINK_CHECKPOINT = os.environ.get(
    "DEEPTHINKVLA_CHECKPOINT",
    "yinchenghust/deepthinkvla_libero_cot_sft",
)
DEFAULT_DEEPTHINK_REPO_ROOT = os.environ.get("DEEPTHINKVLA_REPO_ROOT", ".")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def load_worker(args):
    repo_root = Path(args.deepthink_repo_root).resolve()
    for path in (repo_root / "src", repo_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(repo_root)

    import experiments.deepthinkvla_utils as dt_utils  # noqa: PLC0415

    dt_utils.DEVICE = torch.device(args.device)
    checkpoint_path = Path(str(args.checkpoint)).expanduser()
    checkpoint = str(checkpoint_path) if checkpoint_path.exists() else snapshot_download(str(args.checkpoint))
    cfg = argparse.Namespace(
        pretrained_checkpoint=checkpoint,
        compute_dtype=args.compute_dtype,
        num_images_in_input=args.num_images_in_input,
        max_new_tokens=args.max_new_tokens,
    )
    with contextlib.redirect_stdout(sys.stderr):
        model, unnormalize_action = dt_utils.get_vla(cfg)
        processor = AutoProcessor.from_pretrained(checkpoint)
    model.eval()
    return cfg, dt_utils, model, unnormalize_action, processor


def handle_request(req, cfg, dt_utils, model, unnormalize_action, processor, masked_cot=False):
    start = time.perf_counter()
    payload = np.load(req["npz_path"])
    obs = {
        "full_image": payload["full_image"],
        "wrist_image": payload["wrist_image"],
        "state": payload["state"],
    }
    task = req["task"]
    with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
        if masked_cot:
            actions, reasoning = dt_utils.get_vla_action_mask_cot(
                cfg=cfg,
                vla=model,
                unomrmalize_action=unnormalize_action,
                processor=processor,
                obs=obs,
                task_label=task,
            )
        else:
            actions, reasoning = dt_utils.get_vla_action(
                cfg=cfg,
                vla=model,
                unomrmalize_action=unnormalize_action,
                processor=processor,
                obs=obs,
                task_label=task,
            )
    return {
        "request_id": req.get("request_id"),
        "ok": True,
        "actions": np.asarray(actions, dtype=np.float32).tolist(),
        "reasoning": str(reasoning),
        "inference_time": float(time.perf_counter() - start),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_DEEPTHINK_CHECKPOINT)
    parser.add_argument("--deepthink-repo-root", default=DEFAULT_DEEPTHINK_REPO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--masked-cot", action="store_true")
    args = parser.parse_args()

    eprint(f"[deepthink-server] loading checkpoint={args.checkpoint} device={args.device}")
    try:
        cfg, dt_utils, model, unnormalize_action, processor = load_worker(args)
    except Exception as exc:
        eprint(f"[deepthink-server] load failed: {type(exc).__name__}: {exc}")
        raise
    eprint("[deepthink-server] ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("command") == "shutdown":
                print(json.dumps({"request_id": req.get("request_id"), "ok": True, "shutdown": True}), flush=True)
                break
            response = handle_request(
                req,
                cfg=cfg,
                dt_utils=dt_utils,
                model=model,
                unnormalize_action=unnormalize_action,
                processor=processor,
                masked_cot=args.masked_cot,
            )
        except Exception as exc:
            response = {
                "request_id": None,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        print(json.dumps(response, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

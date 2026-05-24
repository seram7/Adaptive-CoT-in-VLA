from pathlib import Path
import json
import re
import gc
import numpy as np
import pandas as pd
import torch

# ============================================================
# Config
# ============================================================

SUITE = "libero_goal"
METHOD = "ECoT_plain"

SCHEDULE_PATH = Path(f"/scratch/donghwa/LIBERO-plus/schedules/libero_plus_l5_{SUITE}.json")
RESULT_ROOT = Path(
    f"/scratch/donghwa/Adaptive-CoT-in-VLA/experiments/libero/rollouts/"
    f"libero_plus_l5/{SUITE}/{METHOD}"
)

# method 이름에 따라 CoT count 계산 방식 지정
FULL_COT_METHODS = {"ECoT_plain"}

# ============================================================
# Load schedule
# ============================================================

schedule = json.loads(SCHEDULE_PATH.read_text())["episodes"]
schedule_df = pd.DataFrame(schedule)
schedule_df["eval_index"] = schedule_df["eval_index"].astype(int)
schedule_df["task_id"] = schedule_df["task_id"].astype(int)
schedule_df["trial_id"] = schedule_df["trial_id"].astype(int)

# ============================================================
# Load rollout .pt files
# ============================================================

records = []

for pt_path in sorted(RESULT_ROOT.rglob("*.pt")):
    m = re.search(r"eval(\d+)_task(\d+)", str(pt_path))
    if m is None:
        continue

    eval_index = int(m.group(1))
    task_id_from_path = int(m.group(2))

    raw = torch.load(pt_path, map_location="cpu", weights_only=False)

    step_ids = np.asarray(raw.get("step_ids", []))
    inference_times = np.asarray(raw.get("inference_times", []), dtype=np.float32)

    prefix_stats = raw.get("prefix_stats", {}) or {}

    num_inference_steps = int(len(step_ids))
    success = int(raw.get("success", 0))

    if METHOD in FULL_COT_METHODS:
        cot_count = num_inference_steps
    else:
        cot_count = int(prefix_stats.get("num_prefix_resets", 0))

    cot_ratio = cot_count / num_inference_steps if num_inference_steps > 0 else np.nan

    mean_inference_time = float(np.mean(inference_times)) if len(inference_times) > 0 else np.nan
    total_inference_time = float(np.sum(inference_times)) if len(inference_times) > 0 else 0.0

    records.append({
        "eval_index": eval_index,
        "task_id": int(raw.get("task_id", task_id_from_path)),
        "trial_id": int(raw.get("trial_id", 0)),
        "success": success,
        "num_inference_steps": num_inference_steps,
        "traj_len": num_inference_steps,
        "mean_inference_time": mean_inference_time,
        "total_inference_time": total_inference_time,
        "cot_count": cot_count,
        "cot_ratio": cot_ratio,
        "num_prefix_updates": int(prefix_stats.get("num_prefix_updates", 0)),
        "num_prefix_resets": int(prefix_stats.get("num_prefix_resets", 0)),
        "num_prefix_changes": int(prefix_stats.get("num_prefix_changes", 0)),
        "num_prefix_decisions": int(prefix_stats.get("num_decisions", 0)),
        "pt_path": str(pt_path),
    })

    del raw

gc.collect()

result_df = pd.DataFrame(records)

# ============================================================
# Completion check
# ============================================================

expected = set(schedule_df["eval_index"].tolist())
completed = set(result_df["eval_index"].tolist())
missing = sorted(expected - completed)

print("=== Completion ===")
print("Expected:", len(expected))
print("Completed:", len(completed))
print("Missing:", len(missing))

if missing:
    missing_df = schedule_df[schedule_df["eval_index"].isin(missing)].copy()
    print("\n=== Missing examples ===")
    print(
        missing_df[
            ["eval_index", "suite", "task_id", "trial_id", "category", "task_name"]
        ].head(50).to_string(index=False)
    )

# ============================================================
# Merge schedule metadata
# ============================================================

df = result_df.merge(
    schedule_df[
        ["eval_index", "suite", "task_id", "trial_id", "task_name", "category", "difficulty_level", "json_id"]
    ],
    on=["eval_index", "task_id", "trial_id"],
    how="left",
)

# ============================================================
# Overall summary
# ============================================================

overall = pd.Series({
    "N": len(df),
    "SR": df["success"].mean(),
    "traj_len": df["traj_len"].mean(),
    "mean_inference_time": df["mean_inference_time"].mean(),
    "total_inference_time": df["total_inference_time"].mean(),
    "cot_count": df["cot_count"].mean(),
    "cot_ratio": df["cot_ratio"].mean(),
})

print("\n=== Overall L5 Summary ===")
print(overall)

# ============================================================
# Category-wise summary
# ============================================================

category_summary = df.groupby("category").agg(
    N=("eval_index", "count"),
    SR=("success", "mean"),
    traj_len=("traj_len", "mean"),
    mean_inference_time=("mean_inference_time", "mean"),
    total_inference_time=("total_inference_time", "mean"),
    cot_count=("cot_count", "mean"),
    cot_ratio=("cot_ratio", "mean"),
).sort_values("N", ascending=False)

print("\n=== Category-wise Summary ===")
print(category_summary)

# ============================================================
# Save CSV
# ============================================================

OUT_DIR = RESULT_ROOT / "_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUT_DIR / "rollout_records.csv", index=False)
category_summary.to_csv(OUT_DIR / "category_summary.csv")
overall.to_frame("value").to_csv(OUT_DIR / "overall_summary.csv")

if missing:
    missing_df.to_csv(OUT_DIR / "missing_episodes.csv", index=False)

print("\nSaved analysis to:", OUT_DIR)

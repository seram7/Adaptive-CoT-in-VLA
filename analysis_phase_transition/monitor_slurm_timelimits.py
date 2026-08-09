#!/usr/bin/env python3
"""Lightweight Slurm monitor for long rollout arrays.

Checks selected array jobs periodically. If a reduced 1h timelimit causes any
TIMEOUT state, the monitor restores that array to a safer timelimit.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import pathlib
import subprocess
import sys
import time


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(fp, message: str) -> None:
    fp.write(f"[{timestamp()}] {message}\n")
    fp.flush()


def parse_sacct(job_id: str) -> list[dict[str, str]]:
    proc = run(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobIDRaw,JobName%32,State,Elapsed,Timelimit,ExitCode",
            "-P",
            "-n",
        ]
    )
    rows: list[dict[str, str]] = []
    if proc.returncode != 0:
        return rows
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        raw_id, name, state, elapsed, timelimit, exit_code = parts[:6]
        if "." in raw_id:
            continue
        rows.append(
            {
                "raw_id": raw_id.strip(),
                "name": name.strip(),
                "state": state.strip().split()[0],
                "elapsed": elapsed.strip(),
                "timelimit": timelimit.strip(),
                "exit_code": exit_code.strip(),
            }
        )
    return rows


def parse_squeue(job_id: str) -> list[dict[str, str]]:
    proc = run(["squeue", "-h", "-j", job_id, "-o", "%i|%j|%T|%M|%l|%R"])
    rows: list[dict[str, str]] = []
    if proc.returncode != 0:
        return rows
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        slurm_id, name, state, elapsed, timelimit, reason = parts[:6]
        rows.append(
            {
                "slurm_id": slurm_id.strip(),
                "name": name.strip(),
                "state": state.strip(),
                "elapsed": elapsed.strip(),
                "timelimit": timelimit.strip(),
                "reason": reason.strip(),
            }
        )
    return rows


def state_counts(rows: list[dict[str, str]]) -> str:
    counts = collections.Counter(row["state"] for row in rows)
    if not counts:
        return "none"
    return ", ".join(f"{state}={counts[state]}" for state in sorted(counts))


def update_timelimit(job_id: str, timelimit: str) -> tuple[bool, str]:
    proc = run(["scontrol", "update", f"JobId={job_id}", f"TimeLimit={timelimit}"])
    detail = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, detail


def monitor(args: argparse.Namespace) -> int:
    log_path = pathlib.Path(args.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    restored: set[str] = set()

    with log_path.open("a", encoding="utf-8") as fp:
        log(fp, f"monitor started jobs={','.join(args.jobs)} interval={args.interval}s")
        log(fp, f"timeout restore target={args.restore_timelimit}")
        while True:
            any_active = False
            for job_id in args.jobs:
                squeue_rows = parse_squeue(job_id)
                sacct_rows = parse_sacct(job_id)
                if squeue_rows:
                    any_active = True

                log(
                    fp,
                    (
                        f"job={job_id} squeue[{state_counts(squeue_rows)}] "
                        f"sacct[{state_counts(sacct_rows)}]"
                    ),
                )

                states = {row["state"] for row in sacct_rows}
                if "TIMEOUT" in states and job_id not in restored:
                    ok, detail = update_timelimit(job_id, args.restore_timelimit)
                    restored.add(job_id)
                    log(
                        fp,
                        (
                            f"job={job_id} detected TIMEOUT; restore "
                            f"TimeLimit={args.restore_timelimit}; ok={ok}; detail={detail}"
                        ),
                    )

                notable = sorted(
                    state for state in states if state in {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL"}
                )
                if notable:
                    log(fp, f"job={job_id} notable terminal states={','.join(notable)}")

            if not any_active:
                log(fp, "no active monitored jobs left; monitor exiting")
                return 0
            time.sleep(args.interval)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--restore-timelimit", default="01:30:00")
    parser.add_argument("--log", required=True)
    return monitor(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

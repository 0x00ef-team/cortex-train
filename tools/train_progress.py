#!/usr/bin/env python3
"""Show live training progress from the log written by `resume_local`.

Usage:
    python3 tools/train_progress.py                 # one-shot snapshot
    watch -n 10 python3 tools/train_progress.py     # live view
    python3 tools/train_progress.py --follow        # follow the log

Works with the already-running 100k-step job: it parses lines like
`step  3900 avg_loss 2.0506` from /tmp/cortex-train-100k.log.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

DEFAULT_LOG = "/tmp/cortex-train-100k.log"
DEFAULT_STATUS = "/tmp/cortex-train-status.json"
LINE = re.compile(r"step\s+(\d+)\s+avg_loss\s+([\d.]+)")


def process_elapsed():
    try:
        pids = subprocess.check_output(["pgrep", "-f", "resume_local"], text=True).split()
        if not pids:
            return None
        secs = subprocess.check_output(
            ["ps", "-o", "etimes=", "-p", pids[0]], text=True).strip()
        return float(secs)
    except Exception:
        return None


def gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def snapshot(log_path, status_path, total):
    step, loss = None, None
    st = {}
    try:
        with open(status_path, encoding="utf-8") as f:
            st = json.load(f)
        step, loss, total = st.get("step", step), st.get("loss", loss), st.get("total", total)
    except Exception:
        pass
    if step is None:
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = LINE.search(line)
                    if m:
                        step, loss = int(m.group(1)), float(m.group(2))
        except FileNotFoundError:
            pass
    if step is None:
        return None

    pct = 100.0 * step / total
    filled = int(30 * step / total)
    bar = "#" * filled + "-" * (30 - filled)
    elapsed = st.get("elapsed_s") if isinstance(st, dict) else None
    if elapsed is None:
        elapsed = process_elapsed()
    if elapsed is None:
        try:
            started = os.path.getmtime(log_path) if os.path.exists(log_path) else time.time()
            elapsed = max(1.0, time.time() - started)
        except OSError:
            elapsed = 0.0
    elapsed = max(1.0, float(elapsed))
    eta = (elapsed / step) * (total - step) if step else 0.0
    temp = gpu_temp()
    temp_s = f"{temp}C" if temp is not None else "n/a"
    return (f"[{bar}] {pct:5.1f}%  step {step}/{total}  "
            f"loss {loss if loss is not None else float('nan'):.4f}  "
            f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m  gpu {temp_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--status", default=DEFAULT_STATUS)
    ap.add_argument("--total", type=int, default=100000)
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()

    if not args.follow:
        s = snapshot(args.log, args.status, args.total)
        print(s if s else "no progress data yet")
        return

    last = None
    while True:
        s = snapshot(args.log, args.status, args.total)
        if s and s != last:
            print(s, flush=True)
            last = s
        time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

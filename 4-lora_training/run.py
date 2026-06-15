#!/usr/bin/env python3
import os
import sys
import traceback
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))

WORLD_SIZE  = 2
MASTER_ADDR = os.environ.get("MASTER_ADDR", "172.23.100.124")  # override with env var if IP changes
MASTER_PORT = os.environ.get("MASTER_PORT", "29501")


def _worker(rank: int) -> None:
    os.environ.update({
        "RANK":                       str(rank),
        "LOCAL_RANK":                 str(rank),
        "WORLD_SIZE":                 str(WORLD_SIZE),
        "MASTER_ADDR":                MASTER_ADDR,
        "MASTER_PORT":                MASTER_PORT,
        "ACCELERATE_MIXED_PRECISION": "fp16",
        # Windows gloo always uses libuv transport (USE_LIBUV=0 is ignored).
        # libuv requires an interface NAME; loopback crashes on the first
        # all-reduce due to Windows socket limitations.
        "GLOO_SOCKET_IFNAME":         "Wi-Fi",
        "PYTORCH_CUDA_ALLOC_CONF":    "max_split_size_mb:128,garbage_collection_threshold:0.8",
    })
    try:
        from train_lora import main
        main()
    except Exception:
        print(f"\n[rank {rank}] Worker failed with:\n", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, nprocs=WORLD_SIZE, join=True)

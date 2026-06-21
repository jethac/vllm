"""
Minimal repro: `gpu_memory_utilization` is not respected on GB10 / DGX Spark
(sm_121) unified-memory systems during startup profiling -- the entire HOST
wedges (not just the engine).

What it does
------------
Starts `LLM(...)` with a conservative `gpu_memory_utilization` (default 0.70)
on a model whose startup profiling transient overruns the implied budget on a
unified-memory device. A background thread samples /proc/meminfo:MemAvailable
every 0.5s. A watchdog (ON by default) kills THIS process if available memory
falls below --watchdog-floor-mb, so it doesn't take the OS down with it.

Run with the watchdog to safely capture the memory collapse trace.
Run with --no-watchdog to reproduce the real failure: the box hangs hard and
typically requires a power-cycle.

Repro (GB10 / DGX Spark, unified memory):
    python repro.py
Expected (buggy) output: MemAvailable collapses during profile_run, far past
`gpu_memory_utilization * total`, then the watchdog fires (or, with
--no-watchdog, the host becomes unresponsive).

Control: the SAME command on a discrete card (e.g. RTX PRO 6000, sm_120) with
a separate VRAM pool either serves fine or fails cleanly with
"No available memory for cache blocks" -- it does NOT wedge the host.
"""
import argparse, os, signal, threading, time


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


def monitor(floor_mb, interval, enable_watchdog, stop):
    pid, lo = os.getpid(), 1 << 62
    while not stop.is_set():
        a = mem_available_mb()
        lo = min(lo, a)
        print(f"[mem] avail={a} MB (min so far {lo})", flush=True)
        if enable_watchdog and a < floor_mb:
            print(f"[watchdog] avail {a} MB < floor {floor_mb} MB -- killing pid {pid} to spare the host", flush=True)
            os.kill(pid, signal.SIGKILL)
            return
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it",
                    help="A Gemma-4 MoE; bf16 weights ~48 GiB. Reproduces with the bf16/Triton default path.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--watchdog-floor-mb", type=int, default=8000)
    ap.add_argument("--no-watchdog", action="store_true",
                    help="Disable the safety kill. WARNING: on a real GB10 this will likely hang the host.")
    ap.add_argument("--sample-interval", type=float, default=0.5)
    a = ap.parse_args()

    if not a.no_watchdog:
        print(f"[watchdog] enabled, floor={a.watchdog_floor_mb} MB. "
              f"Use --no-watchdog to reproduce the hard host wedge (expect a power-cycle).", flush=True)
    stop = threading.Event()
    threading.Thread(target=monitor,
                     args=(a.watchdog_floor_mb, a.sample_interval, not a.no_watchdog, stop),
                     daemon=True).start()

    print(f"[repro] LLM(model={a.model!r}, gpu_memory_utilization={a.gpu_memory_utilization}, "
          f"dtype=bfloat16, kv_cache_dtype=default) ...", flush=True)
    from vllm import LLM
    LLM(model=a.model, gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
        enforce_eager=True, dtype="bfloat16",
        limit_mm_per_prompt={"image": 0, "audio": 0})
    stop.set()
    print("[repro] engine initialized WITHOUT wedging -- bug did NOT reproduce on this config "
          "(e.g. discrete card, or 4-bit weights).", flush=True)


if __name__ == "__main__":
    main()

# Repro: `gpu_memory_utilization` not respected on GB10 unified memory (host wedge)

Minimal, self-contained reproduction for the issue *"`gpu_memory_utilization` is not
respected on GB10 unified-memory systems during startup profiling (host wedges)."*

## TL;DR
On a DGX Spark / GB10 (`sm_121`, 128 GB **unified** memory shared by CPU+GPU), starting
a Gemma-4 MoE with a conservative `gpu_memory_utilization=0.70` drives system memory
**past** the implied budget during startup `profile_run` and **hangs the whole host**
(requires a power-cycle). It reproduces on the **bf16 KV / `TRITON_ATTN` default path**
-- no quantization, no custom kernels -- and is **specific to unified memory** (a discrete
card with a separate VRAM pool fails cleanly instead).

## Run (on a GB10 / DGX Spark)
```bash
# safe: a watchdog kills the process before it can take down the OS, and prints the trace
python repro.py

# reproduce the REAL failure (the box will likely hang and need a power-cycle):
python repro.py --no-watchdog
```

## Expected (buggy) output
After weights load (~48.5 GiB), `MemAvailable` collapses during profiling:
```
[mem] avail=14228 MB ...
[mem] avail=10555 MB ...
[mem] avail=8933 MB ...
[mem] avail=7352 MB ...
[watchdog] avail 7352 MB < floor 8000 MB -- killing pid <...> to spare the host
```
i.e. ~112 GiB in use on a 119 GiB box against an `0.70 * 119 = 83 GiB` budget.

## Control (shows it's unified-memory-specific)
The same command on a discrete card (e.g. RTX PRO 6000, `sm_120`) serves fine or fails
cleanly with `No available memory for cache blocks` -- it does **not** wedge the host.

## Workaround
4-bit weights (NVFP4 / W4A16, ~13-17 GiB instead of 48 GiB bf16) leave the profiling
transient ~100 GiB of headroom; the MoE then starts without tripping the watchdog.

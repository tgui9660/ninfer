#!/usr/bin/env bash
# Serve Qwen3.8-27B NVFP4 on 0.0.0.0:8888, folded from the llama.cpp server
# config: MTP draft window 4, prefill chunk 4096, 4 device checkpoint slots,
# thinking budget 2048 with preserved thinking, and the llama sampling
# overrides. FP8 row-256 KV; host KV tier left
# at the 8192 MiB default. --model-id pins the public OpenAI model ID that
# opencode's ninfer provider requests. Two active lanes plus a 5-minute
# pending-queue deadline absorb opencode's overlapping agent requests (main
# agent, hidden agents, subagents) instead of 503 queue timeouts.
# --kv-capacity auto fills the free VRAM on this 48 GiB card with KV pages
# (1 GiB sizing headroom) up to the max-concurrency x max-context page cap;
# the implicit default would only hold one full 262K context in the shared
# pool, starving the second lane.
#
# --tuning-profile auto resolves the launch-policy set from the detected GPU
# (rtx-5090 on 170 SMs, rtx-pro-5000 on 110 SMs). Card-pinned variants:
# runme_rtx-5090.sh and runme_rtx-pro-5000.sh (require a build with
# --tuning-profile support; see rtx_5k_tune.md).
#
# Server output is teed to logs/serve.log (append) so it survives a dead
# terminal session.
#
# Usage: ./runme.sh
set -euo pipefail

cd "$(dirname "$0")"

MODEL="models/qwen3_8_27b_nvfp4.ninfer"

mkdir -p logs

./build/apps/ninfer-serve "$MODEL" \
    --host 0.0.0.0 --port 8888 \
    --model-id qwen3-coder:30b \
    --max-concurrency 2 --pending-timeout-ms 300000 \
    --max-context 262144 --kv-dtype fp8 --kv-capacity auto \
    --spec mtp --draft-tokens 4 \
    --prefill-chunk 4096 \
    --device-state-slots 4 \
    --default-thinking-budget 2048 \
    --preserve-thinking \
    --temperature 0.7 --top-p 0.95 --top-k 20 --min-p 0.0     --presence-penalty 1.1 \
    --vision \
    --tuning-profile auto 2>&1 | tee -a logs/serve.log

#!/usr/bin/env bash
# Serve Qwen3.8-27B NVFP4 on 0.0.0.0:8888 with the RTX 5090 tuning profile
# (170-SM wave math and 5090-measured dispatch tables). Card: 32 GiB GDDR7.
# Same serving config as runme.sh: MTP draft window 4, prefill chunk 4096,
# 4 device checkpoint slots, thinking budget 2048 with preserved thinking,
# llama sampling overrides, FP8 row-256 KV, 8192 MiB host KV tier.
# --model-id pins the public OpenAI model ID that opencode's ninfer provider
# requests. Two active lanes plus a 5-minute pending-queue deadline absorb
# opencode's overlapping agent requests instead of 503 queue timeouts.
# --kv-capacity auto fills the free VRAM with KV pages (1 GiB sizing
# headroom) up to the max-concurrency x max-context page cap.
#
# Requires a build with --tuning-profile support (rtx_5k_tune.md Phase 2a);
# earlier builds reject the flag at startup. Stop any instance already bound
# to :8888 before starting. Server output is teed to logs/serve-rtx-5090.log
# (append) so it survives a dead terminal session.
#
# Usage: ./runme_rtx-5090.sh
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
    --tuning-profile rtx-5090 2>&1 | tee -a logs/serve-rtx-5090.log

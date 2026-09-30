#!/usr/bin/env bash
# Controlled E2E A/B: same clone binary, two tuning profiles, identical workload.
# Profiles: rtx-5090 (today's 170-SM launch policy on this card — the "before")
# and rtx-pro-5000 (110-SM wave math — the "after").
# Default order is REVERSED (pro-5000 first): the standard-order run
# ab-e2e-20260929-164756 showed a small consistent pro-5000 decode deficit with
# the 5090 leg running from a cooler card; this run supplies the complementary
# cool-first/warm-second data points so both profiles are measured in both
# thermal positions. Pass the profile names to choose the order.
# Workload matches production parity: fp8 KV, MTP draft window 4, prefill chunk 4096.
# Matrix: -pg P,G = timed prefill of P tokens + timed generation of G tokens:
#   512/128, 2048/128, 8192/128  -> small-T decode at shallow/mid context
#   32768/512                    -> deep-context prefill + longer steady decode
#
# VRAM: each leg loads the full model (~25 GiB). The serve instance (~40 GiB)
# cannot coexist with it. STOP the serve instance on :8888 before running this
# (Ctrl+C in its terminal). NOTE: that instance also serves your local AI
# session — expect it to go idle until you restart it (./runme_rtx-pro-5000.sh).
#
# Output: profiles/bench/ab-e2e-<timestamp>/
#   meta.txt                 provenance (git, driver, GPU, exact commands)
#   rtx-5090.table.txt       leg A report
#   rtx-pro-5000.table.txt   leg B report
#   dmon-rtx-5090.log        clocks/power/temp sampled at 5 s during leg A
#   dmon-rtx-pro-5000.log    same during leg B
#
# Usage: ./run_ab_e2e.sh [first_profile second_profile]
#   (default: rtx-pro-5000 rtx-5090)
set -euo pipefail

cd "$(dirname "$0")"

MODEL="models/qwen3_8_27b_nvfp4.ninfer"
BENCH="./build/bench/ninfer_bench"
PG="512,128;2048,128;8192,128;32768,512"
COMMON_ARGS=(--weights "$MODEL" -pg "$PG" --kv-dtype fp8 --spec mtp
             --draft-tokens 4 --prefill-chunk 4096 --max-ctx 65536 -o table)

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="profiles/bench/ab-e2e-${STAMP}"
mkdir -p "$OUT"
# Leg order via positional args (default: reversed, pro-5000 first — see header).
FIRST="${1:-rtx-pro-5000}"
SECOND="${2:-rtx-5090}"
for p in "$FIRST" "$SECOND"; do
    case "$p" in
        rtx-5090|rtx-pro-5000) ;;
        *) echo "ERROR: unknown profile '$p' (use rtx-5090 | rtx-pro-5000)" >&2; exit 1 ;;
    esac
done
[[ "$FIRST" == "$SECOND" ]] && { echo "ERROR: both legs are '$FIRST'" >&2; exit 1; }

DMON_PID=""
cleanup() { [[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "== guards ==================================================="
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE '[:.]8888$'; then
    echo "ERROR: something is listening on :8888 — stop the serve instance first." >&2
    exit 1
fi
USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
if (( USED_MIB > 10000 )); then
    echo "ERROR: GPU has ${USED_MIB} MiB in use (> 10 GiB) — a full-model instance is" >&2
    echo "       resident. Stop it first (the bench needs ~25 GiB free)." >&2
    exit 1
fi
echo "port :8888 free; GPU memory used: ${USED_MIB} MiB"

echo "== provenance ==============================================="
{
    echo "date:        $(date -Is)"
    echo "git:         $(git rev-parse HEAD) $(git status --porcelain | wc -l) dirty paths"
    echo "gpu:         $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader | head -1)"
    echo "toolchain:   $(nvcc --version | grep -oP 'release \K[0-9.]+')"
    echo "leg order:   ${FIRST} then ${SECOND} (30 s gap)"
    echo "matrix:      -pg ${PG} (pp+tg), 5 measured reps + 1 warmup per point"
    echo "common args: ${COMMON_ARGS[*]}"
} | tee "$OUT/meta.txt"

run_leg() {
    local profile="$1"
    local out="$OUT/${profile}.table.txt"
    local dmon="$OUT/dmon-${profile}.log"
    echo
    echo "== leg: --tuning-profile ${profile} ($(date -Is)) =============="
    echo "# dmon start $(date -Is)" > "$dmon"
    nvidia-smi dmon -s pucm -d 5 >> "$dmon" 2>&1 &
    DMON_PID=$!
    "$BENCH" "${COMMON_ARGS[@]}" --tuning-profile "$profile" --output-file "$out"
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=""
    echo "# dmon end   $(date -Is)" >> "$dmon"
    echo "report: $out"
    sed -n '1,40p' "$out"
}

run_leg "$FIRST"
echo; echo "cooldown 30 s (thermal/clock settling between legs)..."
sleep 30
run_leg "$SECOND"

echo
echo "== done ======================================================="
echo "results in $OUT/"
echo "restart serving when finished: ./runme_rtx-pro-5000.sh"

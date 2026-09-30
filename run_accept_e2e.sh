#!/usr/bin/env bash
# Phase 3 ACCEPTANCE E2E A/B — all arguments built in, designed to run unattended.
#
# Purpose: acceptance re-run after ALL Phase 3 landings (linear ladders, gdn SplitK
# routes, rmsnorm prefetch gate, causal small-T wave target + kMax clamp, rope
# block policy, bf16_vector pack boundary). Acceptance bar (rtx_5k_tune.md):
# the rtx-pro-5000 column must at least MATCH the rtx-5090 column on this
# matrix, position-controlled.
#
# BEFORE RUNNING (one-time, manual):
#   1. Stop the serve instance on :8888 (Ctrl+C in its terminal). The bench
#      loads the full model (~25 GiB) and cannot coexist with it (~40 GiB).
#   2. Just run:  ./run_accept_e2e.sh
#      (optionally: nohup ./run_accept_e2e.sh >/dev/null 2>&1 & )
#      Total runtime ~35-40 min (incremental build + 4 legs + cooldowns).
#   3. When finished, restart serving:  ./runme_rtx-pro-5000.sh
#
# Design: two leg orders (round 1: pro-5000 first, round 2: 5090 first) so each
# profile is measured in both thermal positions; the summary averages the two
# rounds per profile, cancelling the small first-leg-cooler artifact
# (+0.1..0.4%, quantified in ab-e2e-20260929-{164756,170232}).
# Workload = production parity: fp8 KV, MTP draft 4, prefill chunk 4096,
# max ctx 65536, synthetic corpus (MTP acceptance saturates), 5 reps + 1 warmup.
#
# Output (single directory, everything inside it): profiles/bench/ab-e2e-accept-<stamp>/
#   run.log                    MASTER LOG — full transcript (this is what gets read)
#   meta.txt                   provenance (git, gpu, power limit, exact commands)
#   round1-rtx-pro-5000.table.txt   leg reports (one per leg, bench -o table)
#   round1-rtx-5090.table.txt
#   round2-rtx-5090.table.txt
#   round2-rtx-pro-5000.table.txt
#   dmon-round*-*.log          power/util/clocks/mem sampled at 5 s during each leg
#   summary.txt                auto-computed position-controlled comparison
set -euo pipefail
cd "$(dirname "$0")"

MODEL="models/qwen3_8_27b_nvfp4.ninfer"
BENCH="./build/bench/ninfer_bench"
PG="512,128;2048,128;8192,128;32768,512"
COOLDOWN_S=30
COMMON_ARGS=(--weights "$MODEL" -pg "$PG" --kv-dtype fp8 --spec mtp
             --draft-tokens 4 --prefill-chunk 4096 --max-ctx 65536 -o table)

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="profiles/bench/ab-e2e-accept-${STAMP}"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

DMON_PID=""
cleanup() { [[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "== guards ==================================================="
if [[ ! -f "$MODEL" ]]; then echo "ERROR: model artifact not found: $MODEL" >&2; exit 1; fi
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
PLIMIT_W="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)"
echo "port :8888 free; GPU memory used: ${USED_MIB} MiB; power limit: ${PLIMIT_W} W (hard-set 300 W)"
echo "output dir: $OUT"

echo
echo "== incremental build (freshness guarantee; no-op if current) ======"
cmake --build build -j"$(nproc)"

echo
echo "== provenance ==============================================="
{
    echo "date:        $(date -Is)"
    echo "git:         $(git rev-parse HEAD) ($(git status --porcelain | wc -l) dirty paths; Phase 3 landings uncommitted)"
    echo "gpu:         $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader | head -1)"
    echo "toolchain:   nvcc $(nvcc --version | grep -oP 'release \K[0-9.]+')"
    echo "leg order:   round1 [rtx-pro-5000, rtx-5090] then round2 [rtx-5090, rtx-pro-5000]; ${COOLDOWN_S} s cooldown between legs"
    echo "matrix:      -pg ${PG} (pp+tg); bench defaults: 5 measured reps + 1 warmup per point"
    echo "common args: ${COMMON_ARGS[*]}"
    echo "acceptance:  rtx-pro-5000 column must at least match rtx-5090 (position-controlled means)"
} | tee "$OUT/meta.txt"

run_leg() {
    local label="$1" profile="$2"
    local out="$OUT/${label}.table.txt"
    local dmon="$OUT/dmon-${label}.log"
    echo
    echo "== ${label}: --tuning-profile ${profile} ($(date -Is)) =============="
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

run_leg round1-rtx-pro-5000 rtx-pro-5000
echo; echo "cooldown ${COOLDOWN_S} s..."
sleep "$COOLDOWN_S"
run_leg round1-rtx-5090 rtx-5090
echo; echo "cooldown ${COOLDOWN_S} s..."
sleep "$COOLDOWN_S"
run_leg round2-rtx-5090 rtx-5090
echo; echo "cooldown ${COOLDOWN_S} s..."
sleep "$COOLDOWN_S"
run_leg round2-rtx-pro-5000 rtx-pro-5000

echo
echo "== summary (position-controlled) =============================="
awk '
    $1 ~ /^pp[0-9]+\+tg[0-9]+$/ {
        f = FILENAME; sub(/^.*\//, "", f)
        prof = (f ~ /pro-5000/) ? "pro5k" : "5090"
        if (f ~ /^round1/) r = "r1"; else r = "r2"
        if (!($1 in seentest)) { seentest[$1] = 1; order[++n] = $1; np[$1] = $2; ng[$1] = $3 }
        val[prof, r, $1, "pre"] = $4
        val[prof, r, $1, "dec"] = $7
    }
    END {
        if (n == 0) { print "ERROR: no pp rows found in the leg tables"; exit 1 }
        print "legend: R1 = round 1 (pro-5000 ran first, cooler card), R2 = round 2 (rtx-5090 first)"
        print "Dmean% = 100 * (mean(pro5k R1,R2) - mean(5090 R1,R2)) / mean(5090 R1,R2); >= 0 means tuned meets the bar"
        printf "%-14s %7s %6s | %-39s | %-39s\n", "test", "pp", "tg", "prefill t/s  R1-5k R1-590 R2-590 R2-5k", "decode t/s   R1-5k R1-590 R2-590 R2-5k"
        for (i = 1; i <= n; i++) {
            t = order[i]
            mp = (val["pro5k","r1",t,"pre"] + val["pro5k","r2",t,"pre"]) / 2
            m5 = (val["5090","r1",t,"pre"]  + val["5090","r2",t,"pre"])  / 2
            md = (val["pro5k","r1",t,"dec"] + val["pro5k","r2",t,"dec"]) / 2
            m5d = (val["5090","r1",t,"dec"] + val["5090","r2",t,"dec"]) / 2
            printf "%-14s %7s %6s | %8s %8s %8s %8s %+7.2f%% | %8s %8s %8s %8s %+7.2f%%\n", \
                t, np[t], ng[t], \
                val["pro5k","r1",t,"pre"], val["5090","r1",t,"pre"], \
                val["5090","r2",t,"pre"],  val["pro5k","r2",t,"pre"],  100*(mp-m5)/m5, \
                val["pro5k","r1",t,"dec"], val["5090","r1",t,"dec"], \
                val["5090","r2",t,"dec"],  val["pro5k","r2",t,"dec"],  100*(md-m5d)/m5d
        }
    }
' "$OUT"/round1-rtx-pro-5000.table.txt \
  "$OUT"/round1-rtx-5090.table.txt \
  "$OUT"/round2-rtx-5090.table.txt \
  "$OUT"/round2-rtx-pro-5000.table.txt | tee "$OUT/summary.txt"

echo
echo "== done ======================================================="
echo "results in $OUT/  (master log: $OUT/run.log)"
echo "restart serving when finished: ./runme_rtx-pro-5000.sh"

#!/usr/bin/env bash
# KV-dtype perplexity sweep — quality side of the KV format decision.
#
# Purpose: fixed-window causal perplexity (docs/perplexity.md KV-format
# protocol: full corpus, --context 65536 --stride 32768) for every Main KV
# representation, so the kv-dtype choice has a quality datum to pair with
# the op-level speed table in profiles/bench/kvdtype-*.txt.
#
# BEFORE RUNNING: stop the serve instance on :8888 (model + KV windows will
# not fit next to it). Just run:  ./run_kvdtype_ppl.sh   (total ~1.5-2 h)
# AFTER: restart serving from ninfer/:  ./runme_rtx-pro-5000.sh
#
# Output (single directory): profiles/perplexity/kvdtype-sweep-<stamp>/
#   run.log            MASTER LOG — full transcript (this is what gets read)
#   meta.txt           provenance (git, gpu, corpus, exact protocol)
#   dmon.log           pwr/util/clocks sampled at 5 s during the whole run
#   <kv>/report.json   full-precision machine report per dtype
#   <kv>.stdout.txt    domain/overall table per dtype
#   <kv>.stderr.txt    startup/corpus/scoring summaries per dtype
#   summary.txt        per-dtype exit status + completion marker
set -euo pipefail
cd "$(dirname "$0")"

MODEL="models/qwen3_8_27b_nvfp4.ninfer"
PPL="./build/apps/ninfer-perplexity"
CORPUS="eval/corpora/perplexity-1m/manifest.json"
KVS=(bf16 fp8 int8 nvfp4 k8v4)
CTX=65536
STRIDE=32768

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="profiles/perplexity/kvdtype-sweep-${STAMP}"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

DMON_PID=""
cleanup() { [[ -n "$DMON_PID" ]] && kill "$DMON_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "== guards ==================================================="
[[ -f "$MODEL" ]]  || { echo "ERROR: model artifact not found: $MODEL" >&2; exit 1; }
[[ -f "$CORPUS" ]] || { echo "ERROR: corpus manifest not found: $CORPUS" >&2; exit 1; }
[[ -x "$PPL" ]]    || { echo "ERROR: perplexity binary missing — build first" >&2; exit 1; }
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE '[:.]8888$'; then
    echo "ERROR: something is listening on :8888 — stop the serve instance first." >&2
    exit 1
fi
USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
if (( USED_MIB > 10000 )); then
    echo "ERROR: GPU has ${USED_MIB} MiB in use — free it first." >&2
    exit 1
fi

{
  echo "date:       $(date -Is)"
  echo "git:        $(git rev-parse HEAD) ($(git status --porcelain | wc -l) dirty paths)"
  echo "gpu:        $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader,nounits | head -1)"
  echo "corpus:     $CORPUS (16 streams, 4 per domain: english_reference, english_long_form, chinese_reference, ninfer_code)"
  echo "protocol:   full corpus, --context $CTX --stride $STRIDE (docs/perplexity.md KV-format profile, no --quick)"
  echo "dtypes:     ${KVS[*]}  (order matters: bf16 runs first/coolest, k8v4 last/hottest — see dmon.log)"
} > "$OUT/meta.txt"
cat "$OUT/meta.txt"

nvidia-smi dmon -s pucm -d 5 > "$OUT/dmon.log" 2>&1 &
DMON_PID=$!

declare -A STATUS=()
for kv in "${KVS[@]}"; do
  echo "== $kv $(date +%T) ==============================================="
  # --output must be an EMPTY directory; keep the transcripts OUTSIDE it
  # (shell redirects open before the binary runs and would fill it).
  mkdir -p "$OUT/$kv"
  if "$PPL" "$MODEL" \
      --corpus "$CORPUS" \
      --context "$CTX" --stride "$STRIDE" \
      --kv-dtype "$kv" \
      --output "$OUT/$kv" \
      > "$OUT/$kv.stdout.txt" 2> "$OUT/$kv.stderr.txt"; then
    STATUS[$kv]="OK"
    echo "$kv OK $(date +%T)  (report: $OUT/$kv/report.json)"
  else
    rc=$?
    STATUS[$kv]="FAILED rc=$rc"
    echo "$kv FAILED rc=$rc $(date +%T) — see $OUT/$kv.stderr.txt; continuing with remaining dtypes"
  fi
done

{
  echo "finished:   $(date -Is)"
  for kv in "${KVS[@]}"; do echo "  $kv: ${STATUS[$kv]}"; done
  echo "marker:     KVDTYPE_PPL_SWEEP_DONE"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"

echo "== done ======================================================="
echo "results in $OUT/  (master log: $OUT/run.log)"
echo "restart serving when finished: cd ../ninfer && ./runme_rtx-pro-5000.sh"

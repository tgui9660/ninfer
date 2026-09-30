# Plan: NInfer tuning for RTX PRO 5000 Blackwell (agent handoff)

Status: Phases S + 2a + 2b + 3 (all rows) done; **no GPU workload without an explicit
per-window go** (2026-09-28). Pre-landing E2E A/B (2026-09-29) in both leg orders
(`profiles/bench/ab-e2e-20260929-{164756,170232}/`). Post-landing ACCEPTANCE A/B
complete (2026-09-30, `./run_accept_e2e.sh`, `profiles/bench/ab-e2e-accept-20260930-093108/`):
tuned column beats rtx-5090 at every point, both metrics, both thermal positions —
acceptance bar met. Temporary plan — remove when done. Single active copy (in this clone).

## Current state (updated 2026-09-29)

### Progress
| Phase | Status |
|---|---|
| S clone setup | **Done.** Configured (dev preset, CUDA 13.2), full build clean 813/813, `models` symlinked, `runme.sh` copied, `ninfer-serve --help` smoke OK. |
| 0 hardware characterization | **Done** — HBM probe + `nvidia-smi dmon` (findings below). |
| 0 Op-bench baseline | **Partial.** 6-bench loop aborted by user. `profiles/bench/phase0-baseline.txt`: rmsnorm section complete; rope header only (no samples); gdn_gating_proj / causal_softmax_attention / nvfp4_linear_add / nvfp4_linear_swiglu never ran. Re-run in a granted window. |
| 2a profile mechanism | **Done** (2026-09-28). `--tuning-profile auto\|rtx-5090\|rtx-pro-5000` in serve/cli/bench; pure resolution in `src/core/tuning_profile.{h,cpp}`; view fields `tuning_profile`/`tuning_sm_count`; `Engine::tuning_summary()`; serve `tuning \|` startup line (+WARN on foreign part); bench report field. Both profiles carry today's constants (behavior-neutral until 2b). Build clean; 4 affected suites + public-API test pass; full ctest 127/128 (one GDN GPU test flaked under `-j 8` contention, passes alone, no overlap with the diff). |
| 2b wave sizing, profile-gated | **Done** (2026-09-29). Wave constants now read the resolved profile's `tuning_sm_count` off `DeviceExecutionView`: rmsnorm prefetch gate (`src/ops/launcher/rmsnorm.cu`), rope large-block wave capacity (`src/ops/launcher/rope.cu`, 6×SMs), small-T 1/2-wave CTA budgets (`small_t.cu` `causal_attention_split_capacity`, scaled 160/170×SMs, floor to multiple of 4). Public op signatures take the view (rmsnorm/gated_rmsnorm/rope/causal_softmax_attention(_cached) + `rmsnorm_dynamic_grouped_conv_prepare`); planning threads `tuning_sm_count` into workspace capacity queries (`startup.{h,cpp}`). rtx-5090 column reproduces today's constants exactly (170/1020/160/320); rtx-pro-5000 yields 110/660/100/200. Build clean; full ctest 128/128. |
| 3 retune sweeps | **All rows done** (2026-09-29: gdn_gating_proj SplitK re-tuned + landed profile-gated; rmsnorm prefetch gate re-tuned + landed profile-gated; rope large-block capacity verified + 192→128 early switch landed; causal small-T wave target + kMax re-measured + landed profile-gated; bf16_vector pack boundary re-measured + landed profile-gated; fp8 A16 ladders re-tuned + landed; nvfp4 A4 and q4 verified no-change; `--route` ownership bug found + fixed — see results below). |
| 4 E2E A/B | **Done** (2026-09-29): serve window 13:55–16:53 (194 reqs) + controlled bench A/B (`profiles/bench/ab-e2e-20260929-164756/`); no regression. |
| 5 Acceptance A/B | **Done** (2026-09-30): post-landing controlled bench A/B via `./run_accept_e2e.sh` (4 legs, two leg orders, both profiles in both thermal positions; `profiles/bench/ab-e2e-accept-20260930-093108/`). Tuned column wins all 16 position-pairwise comparisons (prefill Δmean +0.15…+0.29%, decode +0.01…+0.09%) vs the pre-landing residual of −0.12…−0.40% / −0.05…−0.17%. Acceptance bar met — see §Final. |

### Hard rule (user directive, 2026-09-28)
**Zero GPU workloads without an explicit per-window go.** Coexisting GPU benchmarks steal
SM/clock/bandwidth from the production server that serves the live opencode session — even when
VRAM fits, the latency impact on the user's own requests is unacceptable. CPU-only work (edits,
nvcc compiles, CPU-side unit tests) proceeds freely; each measurement window must be announced
(what runs, how long) and granted.

### Phase 0 findings (HBM probe, one ~10 s run, production server resident)
- 110 SM / 96 MiB L2 / 384-bit confirmed (probe header).
- **300 W software power cap active** under full memory load: SM clock settles ~2220 MHz (max
  3090), memory clock holds ~13,365 MHz, 86–87 °C; cumulative power-capping ~55 h.
- Copy-engine D2D: **1104 GB/s = 82% of the 1344 GB/s spec** — DRAM healthy.
- Kernel uint4 read: 587 GB/s — issue-rate-limited at the power-capped SM clock, not
  DRAM-limited. Treat as context, not a roofline; decision-grade evidence is Op benches under
  identical conditions on both A/B legs.

### Setup gotcha (recorded)
cmake configure needs `PKG_CONFIG_PATH=/home/botman/.local/lib/pkgconfig` (user-local curl 8.22 +
FFmpeg 7.1 static libs; the system has curl 7.81 / FFmpeg 6.1, which breaks the newer media code
in this clone). A *failed* configure caches bad dependency resolutions — wipe `build/` before
reconfiguring if dependencies change.

### Design refinement (supersedes Approved decision 3's fail-fast wording)
Explicit foreign profile on a different part = **startup WARN, not error**. Rationale: the
mechanism exists for A/B validation — `--tuning-profile rtx-5090` must run on the PRO 5000 as
the "before" leg. `auto` (default) never mismatches by construction. The self-labeling startup
log line keeps any misconfiguration visible. Consequences:
- View carries two fields: `tuning_profile` (concrete enum — measured-table selection) and
  `tuning_sm_count` (effective wave-sizing count: the profile's compiled-in constant for
  explicit profiles, *including foreign ones* — runtime SM count for `auto` on unknown parts).
  Existing runtime-count consumers (GDN residency guard, cost models) keep using
  `multiprocessor_count` unchanged.
- This preserves full-gating baseline purity: rtx-5090 on the PRO 5000 uses 170-based wave math
  exactly as today's binary does.

### Wiring points identified (clone)
- Resolution/validation: `src/runtime/engine/engine.cpp:23` `initialize_device()` (after
  DeviceContext construction; props already queried there).
- Startup log line: pattern at `src/serve/operational_log.cpp:441` (`engine_capacity`).
- CLIs: `src/serve/serve_options.cpp` (parse loop ~:176 + usage text), `apps/cli/options.cpp`
  (~:140 + usage), `bench/inference/ninfer_bench_support.cpp` (`parse_args` + usage ~:308),
  EngineOptions wiring at `bench/inference/ninfer_bench.cpp:150`.
- Report self-description: add resolved profile to `BenchEnvironment`
  (`ninfer_bench_support.h`) + report output.
- Call-site counts for the stream→view signature change (Phase 2b): rmsnorm 27, rope 14,
  linear family 41.
- Tests: `tests/test_serve_options.cpp`, `tests/test_cli_options.cpp` (parsing); new
  `tests/test_tuning_profile.cpp` (resolution matrix — pure function, no GPU).

### Open
- Phase 1 concurrency experiment needs a full-instance window — deferred to the user's final
  test (§Final runbook).
- e2e A/B is the user's window (§Final); agent pre-stages everything else.

### Next
1. **Done (2026-09-29):** serve A/B window closed (13:55–16:53); controlled bench A/B run via
   `run_ab_e2e.sh`; results recorded in §Phase 3 sweep inventory. No regression; small consistent
   decode deficit on the synthetic MTP-saturated leg (−0.1…−0.35%) with an unexcluded leg-order
   confound.
2. **Done (2026-09-29 17:05):** reversed-order rerun (`ab-e2e-20260929-170232/`) — confound
   resolved: ~0.1–0.18% order/thermal artifact + residual ≤0.17% decode / ≤0.4% prefill policy
   edge for the 5090 constants (position-controlled). Phase 3 acceptance bar set accordingly.
3. Granted window: finish the partial Phase 0 Op-bench baseline (rope / gdn_gating_proj /
   causal_softmax_attention / nvfp4_linear_add / nvfp4_linear_swiglu) + HBM probe, run the Phase 3
   sweeps (inventory below), land the roofline denominator updates.

## Working tree & environment
- **All work happens in this clone** (`/home/botman/code/ninfer/rtx5k/ninfer`) @ `e31bc99b` (master),
  origin `github.com/Neroued/ninfer.git`. It is **ahead** of the main tree by 7 commits, incl.
  `0784e76f perf(gdn): replace chunked path with two-stage kernels`. The site inventory below is
  against this source.
- Main tree `/home/botman/code/ninfer` and its running server: **do not modify, do not stop**
  (exceptions below).
- Toolchain: `/usr/local/cuda-13.2` (nvcc 13.2) — identical to what built the production binary
  (verified in main-tree `build/CMakeCache.txt`). Only CUDA toolkit installed.
- Python 3.11: `/home/neroued/miniconda3/envs/py311/bin/python`. Build: `cmake --build build -j`.

## Hardware (confirmed from official datasheet)
RTX PRO 5000 Blackwell: `sm_120` (CC 12.0), **110 SMs** (datasheet: 14,080 CUDA cores ÷ 128; the
5090's 21,760 ÷ 128 = 170 validates the method), 48,935 MiB GDDR7 ECC, 384-bit, **1,344 GB/s**
spec, max SM clock 3090 MHz, 300 W. Driver 595.84. Source PDF:
`workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf` from nvidia.com's RTX PRO 5000
page (local copy `/tmp/opencode/rtx-pro-5000-datasheet.pdf`).

| | PRO 5000 | RTX 5090 | Ratio |
|---|---|---|---|
| SMs | 110 | 170 | 0.65× |
| FP32 boost | 65 TFLOPS | ~107.5 | 0.60× |
| Memory | 48 GB, 384-bit, 1,344 GB/s | 32 GB, 512-bit, ~2,240 GB/s | 0.60× |
| Power | 300 W | 575 W | — |

Every 170-SM "one wave" constant is oversized by 170/110 ≈ 1.55× on this card. Decode tok/s cannot
match published 5090 numbers (bandwidth-bound); wins come from wave-overflow elimination,
SplitK/dispatch re-tuning, and more lanes from 48 GiB.

## Production environment (constraint)
- Running: main-tree `./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --host 0.0.0.0
  --port 8888 --model-id qwen3-coder:30b --max-concurrency 2 --kv-capacity auto --spec mtp
  --draft-tokens 4 --prefill-chunk 4096 --device-state-slots 4 ...` (full line in main-tree
  `runme.sh`, which tees output to `logs/serve.log`).
- **The local opencode session is served by that instance.** Never stop it outside a window the
  user explicitly grants.
- It holds ~39–40 GiB VRAM; a second full model instance (weights alone 19.7 GiB) **cannot fit
  alongside it**. Therefore: Op microbenches (MB–GB workspaces) may run anytime; anything loading
  the full model (`ninfer_bench`, a second serve instance) needs the server stopped.

## Approved decisions
1. Full plan; **MoE kernel sites excluded** (dense 27B artifact); q5/q6 linear shape tables out of
   scope.
2. Mechanism: **`--tuning-profile auto|rtx-5090|rtx-pro-5000`**, *full profile gating* — each
   profile is a complete tuning set (wave constants + measured tables). `rtx-5090` must reproduce
   today's launch decisions exactly (regression property).
3. Fail-fast startup validation: profile identity = (CC 12.0, SM count 170/110); explicit mismatch
   → `invalid_argument` naming both. `auto` resolves from device properties; unknown sm_120a part
   → SM-count-derived wave sizing + 5090 measured tables (documented fallback).
4. Resolved profile rides in `DeviceExecutionView` (src/core/device.h:16, one new field); plans
   bake selected constants at plan-build time (no hot-loop branching).
5. **No production handoff**: the user starts and tests the tuned server themselves. Deliverable
   ends at: tuned clone build + evidence + the runbook in §Final.

## Site inventory (this source; dense 27B NVFP4 artifact)
Already device-aware — **no work**: GDN chunked `prepare.cu`/`recurrence.cu`, KDA chunked (new
family), gdn_gating_proj residency (all take `DeviceExecutionView`, runtime wave math).

Profile-gated targets:

| Site | What | Plumbing |
|---|---|---|
| `src/ops/launcher/rmsnorm.cu:18-20` | `kRmsPrefetchBlocks=170` crossing | wrapper + `include/ninfer/ops/rmsnorm.h` + model call sites |
| `src/ops/launcher/rope.cu:16-17` | `kLargeBlockWaveCapacity=1020` (=170×6) | wrapper + `include/ninfer/ops/rope.h` + call sites |
| `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:30-49` | SplitK progressions ("near 192/256 CTAs") | view already flows; table selection only |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cu:65,229` | 160/320 CTA budgets ("1–2 waves") | attention wrapper path |
| `src/ops/linear/{nvfp4,fp8,q4}/shapes/*` | per-shape dispatch tables measured on 5090; this artifact's hot-path linears span all three families (see §Phase 3 scope fix) | linear-family wrappers/plans (largest plumbing item) |
| `src/ops/common/bf16_vector.cuh:22` | pack-width switch "on the registered RTX 5090 target" | evaluate in sweep; gate only if it moves |

Excluded: `sparse_moe/*` (MoE), `linear/q5|q6/shapes/*` (other quantizations).
Reporting-only (never steer execution): `bench/ops/*.cu` roofline literals (1792/1674.5/209.5/419)
+ `tools/hbm_bandwidth_probe.cu:36`.

## Phases

**Phase S — Clone setup.** `cmake` configure (dev preset: tests + benchmarks; CUDA 13.2) + full
build. `ln -s /home/botman/code/ninfer/models models` (23 GB artifact — never copy). Copy
main-tree `runme.sh` (has tee-to-`logs/serve.log`) into the clone; adjust `MODEL` path if needed.
Verify: `ninfer-serve --help`, one smoke request against a throwaway port.

**Phase 0 — Characterize + baseline (no exclusive window needed).**
1. HBM probe: `nvcc -O3 -std=c++17 -arch=sm_120a tools/hbm_bandwidth_probe.cu -o
   build/hbm_bandwidth_probe && ./build/hbm_bandwidth_probe` (per `tools/README.md`) → measured
   DRAM/sustained-read vs 1,344 GB/s spec; note L2/bus width.
2. Sustained clocks under load: `nvidia-smi dmon` during an Op bench burst (300 W limit may hold
   effective clocks below 3090 MHz).
3. Baseline "before": Op-bench numbers on the unmodified clone (== `rtx-5090` profile behavior) +
   production `logs/serve.log` throughput lines as real-world reference. Record GPU/toolchain/
   workload for every number.

**Phase 1 — Concurrency experiment (config only).** In the clone's `runme.sh`, try
`--max-concurrency 3` then `4` with `--kv-capacity auto` on a spare port. KV ceiling =
`concurrency × max-context` (startup.cpp:905); auto mode fails startup rather than shrinking.
Judge: queue times, TTFT, per-request total under overlapping requests. Leave the chosen value
documented in the runbook; the clone `runme.sh` reflects it. (Running a full instance needs the
exclusive window — coordinate with the user.)

**Phase 2a — Profile mechanism (lands before any retuning; independently shippable).**
1. `enum class GpuTuningProfile { Auto, Rtx5090, RtxPro5000 }` + `GpuTuningProfile
   tuning_profile = Auto` in `EngineOptions` (include/ninfer/types.h:151).
2. Parse in `apps/serve`, `apps/cli`, `bench/inference/ninfer_bench`; invalid value → startup error.
3. Resolution/validation as a pure function of (requested profile, detected CC, detected SM
   count) — unit-test the full matrix without a GPU; mismatch throws naming both sides.
4. Resolved profile added to `DeviceExecutionView`; `DeviceContext` carries it (set by the runtime
   after resolution).
5. Startup log line: `tuning | profile rtx-pro-5000 | auto-detected (110 SMs)` (operational log,
   like the existing `capacity |` line).
6. **Both profiles initially carry today's 5090 constants.** Verify `rtx-5090` reproduces current
   launch decisions (plan-output unit tests). Docs/help/tests per project conventions (`--help`
   both apps, `docs/cli.md`, README).

**Phase 2b — Wave sizing, profile-gated. Done (2026-09-29).** Filled the `rtx-pro-5000` column by
reading the resolved profile's `tuning_sm_count` off `DeviceExecutionView` at the three launch-policy
sites (rmsnorm prefetch gate, rope large-block wave capacity 6×SMs, small-T 1/2-wave CTA budgets
scaled 160/170×SMs and floored to a multiple of 4). Public op signatures now take the view; planning
threads the SM count into workspace capacity queries so planning and launch agree. `rtx-5090` column
untouched (reproduces today's constants exactly). Grid caps remain correctness-safe (strided
work-lists). The `kMax=42` split clamp was re-measured and re-landed profile-aware in Phase 3
(rows 4/5 results below).

**Phase 3 — Retune crossovers → `rtx-pro-5000` column.** Per docs/maintainer/op-development.md §7
(task-local candidate sweep compiled as one matrix → winner into dispatch → losers deleted →
re-verify through public Op): gdn_gating_proj SplitK tables; rmsnorm gated crossing; rope block
choice; small-T split counts; **q4 linear shape dispatch tables (10 files)**; bf16_vector pack
width if the sweep shows a delta. Oracle re-qualification wherever reduction order changes (named
suite criteria, not bit-exact). Update bench roofline denominators to this card's measured/spec
values + `bench/README.md` and `docs/maintainer/linear-benchmark.md` citations. All Op benches run
alongside production (coexistence-safe); use median-of-repeats, re-run noise-affected samples in
quiet moments.

**Phase 4 — E2E A/B (user's window; see §Final).** Agent prepares the runbook + pre-stages
everything; the user executes the exclusive window and reports numbers; agent interprets, fixes
regressions, and finalizes docs (README/model-card hardware statements naming both parts).

## GPU coordination rules
- Code edits, compiles, CPU-side tests, Op microbenches: **anytime**, production stays up.
- Full-model loads (`ninfer_bench`, clone serve instances): **only in a user-granted window**;
  never self-serve one.
- If a number looks noise-affected, re-run in a quiet moment before drawing conclusions.

## Evidence bar
- Oracle re-qualification for any route whose arithmetic association changed.
- Public-Op bench curves on this GPU for every retuned boundary (hot interval T≤128 + 512/1024
  anchors per docs/maintainer/linear-tuning.md).
- E2E A/B: same artifact/settings, both profiles, recorded GPU/toolchain/workload; self-labeling
  `tuning |` log lines make attribution trivial.
- `rtx-5090` profile regression check: launch decisions identical to pre-change binary.

## Phase 3 sweep inventory (prepared 2026-09-29; execution needs a granted GPU window)

Method: op-development.md §7.1 route-development transaction — per target, compile candidate
routes as one matrix, sweep through the public Op bench (median of repeats, quiet re-runs for
noise-affected samples), winner into dispatch, losers deleted, re-verify through the public Op;
oracle re-qualification only where reduction order changes. All sweeps run under
`--tuning-profile rtx-pro-5000`; the `rtx-5090` column stays untouched. Linear token grid: hot
interval T≤128 + 512/1024 anchors (linear-tuning.md). Every launch policy here is
correctness-safe (strided work-lists / residency guards), so a wrong winner costs perf only.

Op-bench A/B legs (landed 2026-09-29): the four wave-scaled Op benches (`rmsnorm`, `rope`,
`causal_softmax_attention`, `dynamic_grouped_conv_prepare`) now accept
`--tuning-profile auto|rtx-5090|rtx-pro-5000` (default auto) and self-label the resolved profile
in their output headers, so every Op-level before/after curve is selectable and attributable. The
dgc-prepare bench previously hardcoded the rtx-5090 view; it now resolves like the others.
Builds clean; parse paths smoke-tested (CPU-only).

Scope fix (2026-09-29; evidence: `tools/artifact/inspect.py --objects --bindings` on
`models/qwen3_8_27b_nvfp4.ninfer`, per-binding format + activation policy): the original
inventory assumed the q4 family was the dominant linear cost of this artifact. It is not — the
text decode hot path spans three families:
- **nvfp4** (`src/ops/linear/nvfp4/`): mlp gate+up 34816×5120 ×56 layers (0–55) and mlp down
  5120×17408 ×56, all use-policy AllowA4 → **decode runs the A4 route** (uses_a4 always true:
  T≤32 T32R128, T≤64 T64R128, T≤128 T128R128Pipelined, <256 T128R128Resident, ≥256 TMA).
- **fp8** (`src/ops/linear/fp8/`): GDN in-proj 16384×5120 ×48, full-attn qkv+gate 14336×5120
  ×16, gdn/attn output proj 5120×6144 ×64, mlp gate+up+down of layers 56–63 (34816×5120 /
  5120×17408 ×8), output_head/token_embedding 248320×5120 — all AllowA8 → **A8 route when
  T≥25 (prefill), A16 ladder below that (decode)**. All six registered fp8 shapes are used.
- **q4** (`src/ops/linear/q4/`): proposal head 131072×5120 (A16Only; every MTP round) + vision
  fc1/qkv 4304×1152 / 3456×1152 (A16Only; off the decode path).
Registered-but-unused-by-this-artifact shapes stay untouched (they serve other artifacts): q4
1024/4096/5120×6144/6144/7168/34816/131072×2048; nvfp4 14336/16384/5120×6144. dflash2/MTP
linears are q8 — out of scope per the approved decision. Consequence: rows 6/8/9 below replace
the original "q4 = dominant" row; the nvfp4 A4 route is the primary decode linear path.

| # | Target (site) | Current 5090 policy | Candidate set | Bench |
|---|---|---|---|---|
| 1 | gdn_gating_proj SplitK boundaries (`bf16_gdn_gating_proj_plan.cpp:30-49`) | k27 (48h/5120r): 9–1024→Split8, 1025–2048→Split4, 2049–4096→Split2, 4097+→Unsplit — preferred grid ≈192 CTA; k35 (32h/2048r): 1–127→Split16, 128–1024→Split8, … — ≈256 CTA | Per token bucket, any schedule from {Split16, Split8, Split4, Split2, Unsplit} (+ GemvPairedRows / SmallTSplit10 for small T); shift bucket boundaries so the preferred full grid lands ≈110–165 CTA (1–1.5 waves on 110 SMs) | `bench/ops/gdn_gating_proj_bench.cu`, both shapes |
| 2 | rmsnorm gated prefetch crossing (`launcher/rmsnorm.cu:24`) | Prefetch gate at `blocks/rows > tuning_sm_count`; 5090 sweep put the crossing at 176–192 blocks (≈1.03–1.13× SM) | Multiplier ∈ {1.00, 1.05, 1.10} × SM count, swept over grid size on both gated shapes | `bench/ops/rmsnorm_bench.cu` |
| 3 | rope large-block wave capacity (`launcher/rope.cu:18`) | `kLargeBlockCtasPerSm = 6` × SM count (occupancy property of the fixed kernels — card-invariant) | Verify no change: CtasPerSm ∈ {4, 6, 8} around the fixed-vs-generic crossover; expect 6 to hold | `bench/ops/rope_bench.cu` |
| 4 | small-T split capacity (`small_t.cu` `causal_attention_split_capacity`) | `wave_ctas = (SM×160/170) & ~3` → 100/200 CTA targets for 1/2-wave grids | Target multiplier ∈ {140, 160, 180}/170 scaled (i.e. ≈82/100/118 CTA) or direct targets {80, 96, 110, 120}; sweep B∈{1..8}, T∈{1..6}, representative windows | `bench/ops/causal_softmax_attention_bench.cu` |
| 5 | small-T 8K-window clamp (`small_t.cu:65` region) | `div_up(window, 192/SplitScale)` splits clamped to [4, 42]×SplitScale ("one 170-SM wave") | Divisor ∈ {110, 120, 144, 192}; kMax ∈ {24, 27, 30, 34, 42} (27 = 42×110/170) | same bench, window 5000–8198, Int8Group64, T≥6 |
| 6 | q4 proposal-head table (`src/ops/linear/q4/shapes/n131072_k5120.cu`; vision n3456_k1152 / n4304_k1152 low priority) | Token-threshold ladder selected on 5090 | Sweep the token grid across the instantiations legal for the shape; rebuild the ladder. Runs every MTP round | `bench/ops/linear_bench.cu` (`--qtype q4 --n 131072 --k 5120 --sweep …`) |
| 7 | bf16_vector pack width (`src/ops/common/bf16_vector.cuh:22`) | Pack-width switch "on the registered RTX 5090 target" | Evaluate in the affected sweeps; gate on the profile only if the sweep shows a delta | whichever Op bench exercises the affected route |
| 8 | nvfp4 MLP A4 route (`src/ops/linear/nvfp4/shapes/n34816_k5120.cu`, `n5120_k17408.cu`) | select_a4: T≤32 T32R128, T≤64 T64R128, T≤128 T128R128Pipelined, <256 T128R128Resident, ≥256 TMA (boundaries chosen on 5090) | Schedule set {T32R128, T64R128, T128R128Pipelined, T128R128Resident} (+TMA at prefill sizes) over the decode grid + 512/1024 anchors; primary decode cost (56 layers × 2 linears/round) | `linear_bench --qtype nvfp4 --policy a4 --n 34816 --k 5120 / --n 5120 --k 17408 --sweep …` |
| 9 | fp8 projection ladders (`src/ops/linear/fp8/shapes/n{16384,14336}_k5120.cu`, `n5120_k{6144,17408}.cu`, `n34816_k5120.cu`, `n248320_k5120.cu`) | A16 ladders per shape (decode, T<25) + A8 routes (prefill, T≥25): e.g. 5120×6144 A8 = T≤64 T32R32K128, T≤128 T64R64K128, else T64R128K128 | Per-bucket candidates from each ladder; boundary shifts for 110-SM wave counts; A8 schedule set where the shape defines one | `linear_bench --qtype fp8 --policy a16|a8 --n <N> --k <K> --sweep …` |

Linear sweep results (executed 2026-09-29, serve resident; evidence:
`profiles/bench/phase3-linear-probe.txt` production-policy curves, `phase3-linear-candidates.txt`
forced-route candidate matrices, `phase3-linear-landing-check.txt` post-landing public-dispatch
curves, all `--tuning-profile rtx-pro-5000`):

- **fp8 A16 ladders — re-tuned and landed** (rows 9). Winners by shape (T = decode tokens):
  - 5120×6144: ≤16 sl_16_4_2, ≤32 sl_16_8_2, ≤36 sl_16_4_2, ≤96 sl_32_4_1, else mma_64_128_k64_s2a2
    (dropped mma_64_64_k128). Landing: t=65 cliff +61%→+12% (135.2→94.2 µs); T≥96 −17%.
  - 14336×5120: ≤64 mma_32_64_k128_s1a3 unchanged, ≤95 sl_32_4_1, else mma_64_128_k64_s2a2
    (dropped mma_64_64_k64). Landing: t=65 cliff +74%→+43% (246.1→202.8 µs); T≥96 −9%.
  - 16384×5120: sl_32_4_1 extended ≤32→≤52 (wins 33–52 by ~3% over mma_32_64_k128_s2a2); rest
    unchanged. The t=33 jump (+57%) is intrinsic to both routes' token-capacity discontinuity
    (32-token slice → 2 launches); old ladder jumped +60% there too.
  - 5120×17408: ≤52 sl_32_8_1, ≤64 mma_32_64_k128_s2a2, ≤92 sl_32_8_1, else mma_64_128_k64_s2a2
    (dropped sliced_64_2_2). Landing: t=52 −22% (237.6→188.4 µs), t=65 cliff +56%→+35%, T≥93 −18%.
  - 34816×5120: no change (apparent candidate wins were same-route run-to-run noise).
  Remaining >15% jumps at small T are route-capacity transitions (quantized T-per-route), smaller
  than or equal to the old ladders; absolute µs at every T is better or equal.
- **nvfp4 A4 route (row 8): no change.** Production-policy decode curves (T 1–128) and prefill
  anchors are smooth for both shapes — no positive T-linear jumps; the 5090 boundaries hold.
- **fp8 A8 routes: no change.** Prefill-range curves (T 32–1024) show no flagged losses; mild
  sub-cliffs at T=160/352 on two shapes noted as secondary, not swept.
- **q4 proposal head 131072×5120 (row 6): no change.** The T=33 cliff (+30%) is structural:
  both sliced_r32_t32_w4_s1 (+83% at 33) and mma_r64_t64_k128_s2_a1 (+30% at 33) have grid
  discontinuities there; candidates mma_r64_t48 / mma_r64_t80 lose in their overlap ranges
  (measured t 24–88). Current ladder (≤32 sliced, ≤64 t64_k128, ≤80 t80, then t96/t112/t128) is
  the best available per interval.
- **Bench bug found and fixed (root cause of a mid-sweep crash).** The `--route` extension
  originally instantiated the header-defined fp8 kernels inside `linear_bench.cu`. Kernel
  instantiations are per-TU (vague linkage), so the bench executable carried a second copy of
  routes already instantiated by the owning shape file; the driver then kept per-function state
  (dynamic-shared-memory opt-in via `cudaFuncSetAttribute`) split across the copies, and any
  route needing >48 KB dynamic smem (e.g. `Fp8SlicedInstance<32,8,2>` = 81920 B at 16384×5120,
  T=11–24) failed with `cudaErrorInvalidValue` through the *production* dispatch path while the
  forced path and ctest passed. Fix: named candidate routes now live in the owning shape file's
  translation unit (`Fp8LinearShape.routes`, `Fp8RouteEntry` in `fp8_shapes.h`), resolved by
  `fp8_linear_a16_route(n, k, name)` in `fp8_dispatch.{h,cpp}`; the bench calls that lookup and no
  longer includes `fp8_launch.cuh`. Invariant documented at the struct: each kernel
  instantiation exists in exactly one TU. Q4 launchers are single-TU plain functions and keep
  direct references. Verified: previously failing case passes, full 1:128 sweeps pass on all four
  shapes, forced routes + unknown-route error path intact, linear ctest 22/22.
- **Test coverage:** `tests/ops/linear/test_fp8_a16.cpp` gained boundary invocations 36/37,
  52/53, 92/93 (the new ladder crossings) against the FP32/FP64 oracle for all six shapes.

gdn_gating_proj SplitK results (executed 2026-09-29, serve resident; evidence:
`profiles/bench/phase3-gdn-candidates.txt` 29-token × 5-leg candidate matrix,
`phase3-gdn-boundaries.txt` coarse crossover brackets, `phase3-gdn-fine-crossovers.txt` fine
points + post-landing dispatch verification):

- **Row 1 done — profile-gated 110-SM table landed** (k27 only; k35 is unused by this artifact
  and left at the 5090 policy). The 5090 table is unchanged and still selected when the tuning
  view is unpopulated (`tuning_sm_count = 0`) or ≥170. New `k27RoutesPro5000`:
  {1} GemvPairedRows, {2–8} SmallTSplit10, {9–1152} Split8, {1153–2304} Split4, {2305–2368}
  Split8, {2369–4096} Split2, {4097+} Unsplit.
- **Boundaries sit exactly on the launcher's cooperative-residency wave cliffs** (1152 = 9×BN128
  tiles for Split8, 2304 = 18 for Split4, 4608 = 36 for Split2 at 110 SMs × 2 resident CTAs/SM):
  Split8 wins single-wave through 1152 (24.58 vs 26.62 µs at 1152) and loses the moment its
  second wave appears (30.7 vs 28.7 µs at 1153); Split4 holds 1153–2304; once Split4's second
  wave shrinks below one tile (2305+), Split8's three near-full waves win back until 2368
  (47.10 vs 51.20/49.15 µs at 2368, tied with Split2 at 2400); Split2 then holds to 4096.
  ≥4097 keeps Unsplit: production chunked prefill never lands there (chunks are 4096).
- **Workspace cost of the {2305–2368} Split8 window:** the [1,4096] capacity reservation grows
  3.69→7.27 MB (call-scoped partial buffer at the route's own worst point). Accepted: the window
  is a consistent ~4% win over 64 tokens and the arena is scratch, not model memory.
- **API change (project-owned, no compat):** both public capacity functions in
  `include/ninfer/ops/gdn_gating_proj.h` take `tuning_sm_count`; wrapper forwards it and
  planning (`startup.cpp`) passes `plan.tuning_sm_count`. Capacity is per-profile (the selected
  table's worst point), not a cross-table union, so each profile reserves exactly what it
  dispatches.
- **Bench additions + bug fix:** `gdn_gating_proj_bench.cu` gained `--schedule NAME` (forced
  candidate legs via `bf16_gdn_gating_execute_candidate`) and `--tuning-profile`. Bug found
  during landing verification: the bench's launch lambda rebuilt `DeviceExecutionView` with a
  two-field aggregate initializer, silently dropping the tuning fields — dispatch resolved the
  5090 table while the capacity query used the flagged profile, tripping the workspace
  query/peak mismatch check at T=1025. Fixed by propagating all four view fields. (The earlier
  sweeps predate the flag and measured the 5090 dispatch leg, which was their intent.)
- **Verification:** ctest gdn 7/7 (per-profile capacity contracts at the new boundaries + FP32
  oracle cases under the pro policy at all six new crossovers 1152/1153, 2304/2305, 2368/2369);
  full ctest 128/128; post-landing dispatch reproduces the forced-candidate measurements at
  every boundary under `--tuning-profile rtx-pro-5000`, and the `rtx-5090` column reproduces
  the old table's decisions exactly (regression property).

rmsnorm gated-prefetch gate results (executed 2026-09-29, serve resident; evidence:
`profiles/bench/phase3-rmsnorm-prefetch.txt`, forced-variant matrix over blocks 88..12000 on
both production gated shapes, graph-cold):

- **Row 2 done — profile-gated policy landed.** The 5090 crossing (no-prefetch wins above
  ~176–192 blocks) does not transfer: on the 110-SM part the no-prefetch variant loses below
  ~200 blocks (6.144 vs 8.192 µs quantized medians; min-level delta 0.5–1.6 µs), ties from ~200
  to 12000 blocks, and never wins anywhere in the reachable range (production max = 3×4096 chunk
  tokens = 12288 blocks). Landed policy: `launcher/rmsnorm.cu` keeps the hoisted loads at every
  grid size under the pro-5000 profile (gate threshold raised to the interval maximum); the 5090
  gate is unchanged and verified still flipping at T=57 (blocks 171 > 170).
- **Bench addition:** `rmsnorm_bench.cu` gained `--prefetch auto|on|off` forcing the
  grid-gated D64..256 warp-route instantiation directly (kernels carry no per-function
  attribute state, so bench-side instantiation is safe — unlike the fp8 smem-opt-in case).
- **Gain:** removes a 2 µs (one launch quantum, ~25–33%) cost per call at prefill tail chunks
  T=37..~55 where the old gate misfired; decode (T≤8) was unaffected either way. Norm ctest 6/6.

causal small-T wave target + kMax results (executed 2026-09-29, serve resident; evidence:
`profiles/bench/phase3-causal-wave-{base,c140,c180,c150}.txt` multiplier matrix (k8v4 W∈{1..4},
B∈{1..8}, ctx∈{4096,8192,16384}), `phase3-causal-wave-{c140,c160,c180}-wgap.txt` W-gap points
(W∈{2,3,5}), `phase3-causal-kmax-landing.txt` post-landing dispatch verification):

- **Row 4 done — k8v4 W=1 uplift landed.** No single multiplier captures the (B,W) structure: at
  W=1, 180/170 wins B=2/4 (−4..−7% vs the 160/170 baseline, consistent across median/min/p95);
  at W≥2, B=2 prefers 160 and B=4 prefers 140 (up to −32% at 16K); B=8 ties everywhere (grid floor
  of 4 splits). The fp8e4m3row256 storage measured +6..+13% *worse* under the 180 uplift → the
  uplift is scoped to Fp8KeyNvfp4Value W=1 only. Landed in `causal_attention_split_capacity`: an
  explicit `else if (storage == Fp8KeyNvfp4Value && tokens == 1 && tuning_sm_count < 170)` branch
  raises the target to `(SM×180/170) & ~3` (120/128 CTAs at B=2/4); every other cell is
  byte-identical to prior behavior and the 170-SM part cannot enter the branch.
- **Row 5 done — profile-aware kMax landed.** Int8Group64 T≥6 8K-window clamp: 42 splits = 168
  CTAs = 1.53 waves on 110 SMs; measured kMax=27 (108 CTAs, one wave) wins −5..−28% at B=1
  (same-hardware A/B at 8192: 43.008 vs 59.392 µs). Landed host-side: `causal_small_t_split_count`
  / `causal_small_t_launch_capacity` take `tuning_sm_count`, kMax = (SM<170 ? 27 : 42)×SplitScale;
  the device-side mirror keeps 42×Scale as the hard cap and clamps to `gridDim.y`, so the host
  value binds first on sub-170 profiles. B>1 is unaffected by construction: at 110 SMs the
  grid_limit (≤25 for B≥2) is always below 27. The scale-2 geometry (h16-kv2) inherits the same
  total-CTA argument (2×54 = 4×27 = 108 CTAs); verified by the 84/54 workspace ratio and a −25%
  A/B (36.864 vs 49.152 µs).
- **Verification:** post-landing dispatch reproduces the measured kMax=27 medians exactly
  (36.864/36.864/40.960/43.008/43.008 µs at windows 5184/6000/7000/8064/8192); the rtx-5090 column
  retains kMax=42 (ws 6241664); B=2/4 int8 8K cells unchanged (ws 1238464/1288064). softmax_attention
  ctest 4/4 (oracle-based, both geometries × int8/fp8/k8v4).

rope large-block capacity results (executed 2026-09-29, serve resident; evidence:
`profiles/bench/phase3-rope-landing.txt` forced-block candidate matrix + post-landing dispatch):

- **Row 3 done — CtasPerSm=6 verified; one adjacent landing.** The 256-thread fixed CTA admits
  six per SM on this part: forced b256 is smooth to T=660 (6×110, 97% of roofline) and cliffs at
  661 (+31%, its second wave), while forced b192 wins from 661 up — so the SM-scaled 660 boundary
  is exact and no constant change is needed. Below 660 b256 beats b192 at every point (2.36 vs
  2.53 µs at 440), so lowering the boundary to 4×SM would regress.
- **Adjacent finding landed:** the 192-thread CTA admits eight per SM, so its own second-wave
  cliff lands at 881 (8×110) — a +26% jump (3.09→3.90 µs) that the 5090 policy never exposes
  (its b192 range is only T=1021..1024, below 8×170=1360). Forced b128 beats the b192 second wave
  by −17..−20% across 881..1024 (3.25 vs 3.90 at 881). Landed in `launcher/rope.cu`: sub-170
  profiles end the b192 range at `min(1024, 8×SM)` (880 here) and use the 128-thread block above;
  `launch_fixed` now takes `tuning_sm_count` instead of the precomputed capacity. The 170-SM part
  and unpopulated views keep the 1024 end (8×170 > 1024 makes the branch structurally inert).
  Bench mirror updated (`production_text_block`). Post-landing pro dispatch: b256 ≤660, b192
  661..880, b128 881..1024 (3.34/3.38/3.49 µs at 881/960/1024 vs 3.90/4.11/4.17 pre-landing);
  5090 column unchanged (b256 through 1020, b192 at 1024). Rope ctest 2/2.

bf16_vector pack-width boundary results (executed 2026-09-29, serve resident; evidence:
`profiles/bench/phase3-bf16-vector-landing.txt` forced-pack A/B matrix + post-landing dispatch):

- **Row 7 done — delta found, profile-gated boundary landed.** The shared 32M-element x8→x2
  boundary (AddBias/GELU, vision path only) is too low on this card: the part has a 96 MB L2
  (same as the 5090), and the forced-pack A/B shows the 16-byte x8 route keeps a 60–75% edge over
  the x2 streams until the working set leaves L2 (~48M elements = 96 MB buffer), collapsing to a
  tie beyond (d=4304: x8 +72% at 48.5M, +31% at 50.7M, tie at 52.9M; d=4608: +64% at 47.2M, tie
  at 51.9M; gelu agrees). Vision patch counts are unbounded by image size, so n ∈ (32M, 48M] is
  reachable in production (e.g. d=4304 × 8192 patches = 35.25M). Landed: `bf16_vector.cuh` gains
  `bf16x8_cache_sized_max_elements(tuning_sm_count)` (sub-170 → 48M, else 32M); `add_bias`/`gelu`
  public ops now take `DeviceExecutionView` (launchers read the boundary off it), vision.cpp
  threads `ctx_.execution_view()`, tests pass a default view (5090 column). Post-landing: pro
  dispatches x8 at 35.25M (40.05 vs 90.75 µs add_bias; 50.30 vs 86.64 µs gelu_tanh — 2.3×/1.7×),
  both profiles tie at 70.5M; rtx-5090 column keeps the 32M boundary exactly. Bench additions:
  `--force-pack auto|x8|x2` + `--tuning-profile` on both benches. add_bias/gelu ctest 2/2.

Roofline denominator update (**landed 2026-09-29**, with the sweep results):
- Confirmed references: DRAM spec **1344 GB/s**; measured ceilings (this card) read **1237.4**
  (92.1% of spec), write 1235.5, copy 1091–1104 GB/s; 110 SMs; FP8 dense TC = 1024 FLOP/SM/cycle
  → **348.8 TFLOP/s @ 3090 MHz boost**, **250.3 @ ~2220 MHz** sustained (300 W cap); BF16 dense =
  half (174.4 / 125.2). Bench denominators use the boost-table compute peaks (as the 5090 did)
  and the measured sustained read for `READ_%`.
- Landed: `bench/ops/ninfer_bench_common.h` `kRooflineGBs` 1792→1344 (shared `print_result`);
  per-bench constants renamed `kRtx5090*`/generic → `kRtxPro5000*` with new values across
  `bench/ops/{bf16_linear_add,context_softmax_attention,fp8_linear_add,fp8_linear_swiglu,
  gdn_input_proj,kv_cache_append,linear,linear_pair,packed_softmax_attention,prepare_masked_block,
  q5_linear_add,sliding_window_attention}_bench.cu` + `q4_linear_swiglu_bench.cu` (inline literal)
  + `tools/hbm_bandwidth_probe.cu` default peak; `linear_bench` CSV header now prints
  `reference_gpu=RTX_PRO_5000`. Citations updated in `bench/README.md` and
  `docs/maintainer/linear-benchmark.md`; historical 5090 measurement records kept and labeled
  (incl. a note added to `docs/maintainer/examples/q4-linear.md`, a frozen 5090 report).

A/B results (2026-09-29; serve window 13:55–16:53, 194 reqs + controlled bench):

Controlled bench (`profiles/bench/ab-e2e-20260929-164756/`; same clone binary, production-parity
fp8 KV + MTP draft 4 + chunk 4096, synthetic corpus so MTP acceptance saturates, 5 reps + 1
warmup; both legs pinned at 300 W, pclk ~1900–2460 MHz, mclk 13365):

| point         | prefill 5090 → pro-5k | Δ     | decode 5090 → pro-5k | Δ      |
|---|---|---|---|---|
| pp512+tg128   | 6437.7 → 6374.9 t/s | −1.0% | 85.80 → 85.50   | −0.35% |
| pp2048+tg128  | 7605.3 → 7539.5 t/s | −0.9% | 189.14 → 188.51 | −0.33% |
| pp8192+tg128  | 7299.4 → 7254.6 t/s | −0.6% | 186.01 → 185.35 | −0.35% |
| pp32768+tg512 | 5924.8 → 5907.9 t/s | −0.3% | 178.19 → 178.01 | −0.10% |

Reversed-order rerun (`profiles/bench/ab-e2e-20260929-170232/`; `./run_ab_e2e.sh`, default order
now pro-5000-first) resolves the delta into two components (2×2: profile × thermal position):
- **Order/thermal effect** (first leg starts from a cooler card): +0.10…+0.18% decode,
  +0.26…+0.43% prefill — a measurement artifact, now controlled by alternating leg order.
- **Residual policy effect** (position-controlled): the rtx-5090 constants remain +0.05…+0.17%
  faster on decode and +0.12…+0.40% on prefill at all four points in both runs — small but
  consistently signed. On the hot MTP-saturated path the 170-SM wave math is marginally better
  than the 110-scaled values. Matches the two Phase 3 suspects already in the candidate sets:
  rmsnorm gate at 1.00×SM may cut prefetch early (5090 crossing band was 1.03–1.13×), and
  small-T capacity 100 vs 160 CTA. **Phase 3 acceptance bar: the tuned column must at least
  match the rtx-5090 column on this matrix.**

Real traffic (serve logs, traffic-matched buckets, indicative only): steady decode ctx≥50k —
main tree (170-SM constants) 103.4 tok/s (n=2479, MTP 62.9%) vs clone window 107.8 (n=128,
MTP 63.8%) ≈ +4%; ctx<50k 114.9 (n=212) vs 134.3 (n=17). Raw-prefill bucket still not
comparable (prompt-size mismatch). No regression in either leg.

## Final — acceptance A/B (executed 2026-09-30 via `./run_accept_e2e.sh`)

Controlled bench acceptance re-run after ALL Phase 3 landings: same artifact/settings as the
pre-landing A/B (fp8 KV + MTP draft 4 + chunk 4096, synthetic corpus, 5 reps + 1 warmup,
300 W hard cap), 4 legs in two orders so each profile is measured in both thermal positions
(evidence: `profiles/bench/ab-e2e-accept-20260930-093108/`; dmon: all legs pegged at 300 W,
gtemp ≤90 C, pclk peaks 2385–2415 MHz — conditions comparable across legs):

| point         | prefill Δmean (pro−5090) | decode Δmean (pro−5090) |
|---|---|---|
| pp512+tg128   | +0.20% | +0.09% |
| pp2048+tg128  | +0.25% | +0.08% |
| pp8192+tg128  | +0.29% | +0.08% |
| pp32768+tg512 | +0.15% | +0.01% |

Position-pairwise (same thermal position, per leg·point): pro-5000 wins **16/16**
(8/8 prefill, 8/8 decode). Pre-landing residual was −0.12…−0.40% prefill / −0.05…−0.17%
decode in favor of the rtx-5090 constants; the sign flipped at every point.
**Acceptance bar (tuned ≥ rtx-5090 column, position-controlled) met with margin.**

Optional real-traffic serve comparison (user's call; card-pinned scripts, stop any instance
on :8888 between legs, each tees to its own log):
```bash
./runme_rtx-5090.sh       # baseline leg -> logs/serve-rtx-5090.log
./runme_rtx-pro-5000.sh   # tuned leg   -> logs/serve-rtx-pro-5000.log
# (runme.sh = --tuning-profile auto, logs/serve.log)
# Compare the per-card serve.log throughput lines + perceived latency.
```

## Risks
- 300 W throttling may shift crossovers beyond what SM count predicts — Phase 0 clocks data guards
  against this.
- Concurrency trades per-lane context depth for parallelism — decided empirically in the user's
  test.
- Profile drift for future parts — mitigated by `auto` fallback + trivial new enum value.
- Session dependency — hard rule: no full-model load without an explicit window.

## Read first
`AGENTS.md` · `docs/maintainer/op-development.md` (§6 oracle, §7 performance protocol) ·
`docs/maintainer/linear-tuning.md` · `bench/README.md` · `src/core/device.h` (view pattern) ·
`src/ops/wrapper/gdn_gating_proj.cpp:79` (reference plumbing) · `src/runtime/engine/kv_capacity.cpp`
(capacity math).

#pragma once

// Narrow private BF16 vector storage shared by elementwise kernels. Arithmetic
// remains Op-specific so each contract keeps its exact FP32 operation order and
// BF16 rounding boundary.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

template <int Pairs>
struct alignas(Pairs* static_cast<int>(sizeof(__nv_bfloat162))) Bf16PairPack {
    static_assert(Pairs == 1 || Pairs == 2 || Pairs == 4);
    __nv_bfloat162 pair[Pairs];
};

using Bf16x4Pack = Bf16PairPack<2>;
using Bf16x8Pack = Bf16PairPack<4>;

// Explicit 16-byte packs win while a BF16 activation stays inside the L2-resident regime; above
// the boundary AddBias and GELU switch to their higher-occupancy BF16x2 streams. The 5090
// crossing was measured at 32M elements; on the 110-SM part (measured 2026-09-29) the x8 route
// keeps a 60-75% edge until the working set leaves the 96 MB L2 (~48M elements) and ties the x2
// streams beyond that, so sub-170-SM profiles use the wider L2-sized boundary.
inline constexpr std::int64_t kBf16x8CacheSizedMaxElements5090   = 32LL * 1024LL * 1024LL;
inline constexpr std::int64_t kBf16x8CacheSizedMaxElementsPro5000 = 48LL * 1024LL * 1024LL;

inline std::int64_t bf16x8_cache_sized_max_elements(std::int32_t tuning_sm_count) {
    return (tuning_sm_count > 0 && tuning_sm_count < 170) ? kBf16x8CacheSizedMaxElementsPro5000
                                                          : kBf16x8CacheSizedMaxElements5090;
}

static_assert(sizeof(Bf16x4Pack) == 8);
static_assert(sizeof(Bf16x8Pack) == 16);

} // namespace ninfer::ops

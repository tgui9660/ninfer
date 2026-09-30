#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <string_view>

namespace ninfer {

// Tuning identities of the registered sm_120a parts. Each profile owns a complete launch-policy
// set (wave constants plus measured dispatch tables); resolution selects the set and the
// effective wave-sizing SM count from the detected device facts.
inline constexpr int kRequiredComputeCapability = 120; // sm_120a
inline constexpr int kRtx5090SmCount            = 170;
inline constexpr int kRtxPro5000SmCount         = 110;

struct TuningResolution {
    GpuTuningProfile concrete = GpuTuningProfile::Rtx5090;
    int tuning_sm_count       = kRtx5090SmCount;
    bool foreign              = false;
};

// Accepts "auto", "rtx-5090", and "rtx-pro-5000"; anything else yields nullopt.
[[nodiscard]] std::optional<GpuTuningProfile> parse_tuning_profile(std::string_view text);
[[nodiscard]] const char* tuning_profile_name(GpuTuningProfile profile);

// Pure function of the requested profile and the detected device facts. A compute capability
// other than sm_120a throws invalid_argument naming both sides. Auto resolves from the detected
// SM count and falls back to the RTX 5090 measured tables with the detected SM count on unknown
// parts. Explicit profiles pin their compiled-in SM count even when foreign, so a foreign
// selection reproduces its part's launch decisions exactly; `foreign` marks that case for a
// startup warning.
[[nodiscard]] TuningResolution resolve_tuning_profile(GpuTuningProfile requested,
                                                      int compute_capability,
                                                      int sm_count);

} // namespace ninfer

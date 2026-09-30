#include "tuning_profile.h"

#include <stdexcept>
#include <string>

namespace ninfer {

std::optional<GpuTuningProfile> parse_tuning_profile(std::string_view text) {
    if (text == "auto") { return GpuTuningProfile::Auto; }
    if (text == "rtx-5090") { return GpuTuningProfile::Rtx5090; }
    if (text == "rtx-pro-5000") { return GpuTuningProfile::RtxPro5000; }
    return std::nullopt;
}

const char* tuning_profile_name(GpuTuningProfile profile) {
    switch (profile) {
    case GpuTuningProfile::Auto: return "auto";
    case GpuTuningProfile::Rtx5090: return "rtx-5090";
    case GpuTuningProfile::RtxPro5000: return "rtx-pro-5000";
    }
    return "unknown";
}

TuningResolution resolve_tuning_profile(GpuTuningProfile requested, int compute_capability,
                                        int sm_count) {
    if (compute_capability != kRequiredComputeCapability) {
        throw std::invalid_argument(
            "tuning profile requires compute capability 12.0 (sm_120a), but the detected device "
            "reports " +
            std::to_string(compute_capability / 10) + "." +
            std::to_string(compute_capability % 10));
    }
    if (sm_count <= 0) {
        throw std::invalid_argument("tuning profile resolution requires a positive SM count, but "
                                    "the detected device reports " +
                                    std::to_string(sm_count));
    }
    TuningResolution out;
    switch (requested) {
    case GpuTuningProfile::Auto:
        if (sm_count == kRtx5090SmCount) {
            out = {GpuTuningProfile::Rtx5090, kRtx5090SmCount, false};
        } else if (sm_count == kRtxPro5000SmCount) {
            out = {GpuTuningProfile::RtxPro5000, kRtxPro5000SmCount, false};
        } else {
            // Unknown sm_120a part: keep the RTX 5090 measured tables and size waves from the
            // detected SM count.
            out = {GpuTuningProfile::Rtx5090, sm_count, false};
        }
        break;
    case GpuTuningProfile::Rtx5090:
        out = {GpuTuningProfile::Rtx5090, kRtx5090SmCount, sm_count != kRtx5090SmCount};
        break;
    case GpuTuningProfile::RtxPro5000:
        out = {GpuTuningProfile::RtxPro5000, kRtxPro5000SmCount, sm_count != kRtxPro5000SmCount};
        break;
    }
    return out;
}

} // namespace ninfer

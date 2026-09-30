#include "core/tuning_profile.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace ninfer;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check_resolution(GpuTuningProfile requested, int sm_count, GpuTuningProfile expected_concrete,
                     int expected_sm_count, bool expected_foreign) {
    const TuningResolution resolved = resolve_tuning_profile(requested, 120, sm_count);
    return check(resolved.concrete == expected_concrete &&
                     resolved.tuning_sm_count == expected_sm_count &&
                     resolved.foreign == expected_foreign,
                 "resolution matrix mismatch");
}

} // namespace

int main() {
    int failures = 0;

    failures += check(parse_tuning_profile("auto") == GpuTuningProfile::Auto,
                      "parse: auto");
    failures += check(parse_tuning_profile("rtx-5090") == GpuTuningProfile::Rtx5090,
                      "parse: rtx-5090");
    failures += check(parse_tuning_profile("rtx-pro-5000") == GpuTuningProfile::RtxPro5000,
                      "parse: rtx-pro-5000");
    for (std::string_view text : {"", "banana", "RTX-5090", "rtx_5090", "auto ", " rtx-5090"}) {
        failures += check(!parse_tuning_profile(text), "parse: unknown value accepted");
    }

    failures += check(std::string_view(tuning_profile_name(GpuTuningProfile::Auto)) == "auto",
                      "name: auto");
    failures += check(
        std::string_view(tuning_profile_name(GpuTuningProfile::Rtx5090)) == "rtx-5090",
        "name: rtx-5090");
    failures += check(
        std::string_view(tuning_profile_name(GpuTuningProfile::RtxPro5000)) == "rtx-pro-5000",
        "name: rtx-pro-5000");

    // Auto resolves from the detected SM count; unknown parts keep the RTX 5090 tables with the
    // detected wave-sizing count.
    failures += check_resolution(GpuTuningProfile::Auto, 170, GpuTuningProfile::Rtx5090, 170,
                                 false);
    failures += check_resolution(GpuTuningProfile::Auto, 110, GpuTuningProfile::RtxPro5000, 110,
                                 false);
    failures += check_resolution(GpuTuningProfile::Auto, 96, GpuTuningProfile::Rtx5090, 96,
                                 false);

    // Explicit profiles pin their compiled-in SM count even when foreign.
    failures += check_resolution(GpuTuningProfile::Rtx5090, 170, GpuTuningProfile::Rtx5090, 170,
                                 false);
    failures += check_resolution(GpuTuningProfile::Rtx5090, 110, GpuTuningProfile::Rtx5090, 170,
                                 true);
    failures += check_resolution(GpuTuningProfile::Rtx5090, 96, GpuTuningProfile::Rtx5090, 170,
                                 true);
    failures += check_resolution(GpuTuningProfile::RtxPro5000, 170, GpuTuningProfile::RtxPro5000,
                                 110, true);
    failures += check_resolution(GpuTuningProfile::RtxPro5000, 110, GpuTuningProfile::RtxPro5000,
                                 110, false);
    failures += check_resolution(GpuTuningProfile::RtxPro5000, 96, GpuTuningProfile::RtxPro5000,
                                 110, true);

    // Hard failures name the offending device fact.
    failures += check(
        rejects([] { (void)resolve_tuning_profile(GpuTuningProfile::Auto, 110, 96); }),
        "non-sm_120a compute capability was accepted");
    failures += check(
        rejects([] { (void)resolve_tuning_profile(GpuTuningProfile::Rtx5090, 120, 0); }),
        "zero SM count was accepted");

    if (failures > 0) {
        std::cerr << failures << " tuning profile failure(s)\n";
        return 1;
    }
    return 0;
}

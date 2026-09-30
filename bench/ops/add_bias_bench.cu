#include "core/tuning_profile.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer_bench_common.h"
#include "ops/common/bf16_vector.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

template <int RowsPerBlock>
__global__ void add_bias_payload_control(const uint4* bias, uint4* x, std::int32_t packs,
                                         std::int32_t rows) {
    const int pack = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pack >= packs) { return; }
    const uint4 b       = bias[pack];
    const int first_row = static_cast<int>(blockIdx.y) * RowsPerBlock;
#pragma unroll
    for (int item = 0; item < RowsPerBlock; ++item) {
        const int row = first_row + item;
        if (row >= rows) { return; }
        uint4 value = x[static_cast<std::int64_t>(row) * packs + pack];
        value.x ^= b.x;
        value.y ^= b.y;
        value.z ^= b.z;
        value.w ^= b.w;
        x[static_cast<std::int64_t>(row) * packs + pack] = value;
    }
}

template <int Block>
__global__ void add_bias_pair_payload_control(const unsigned int* bias, unsigned int* x,
                                              std::int32_t pairs, std::int32_t rows) {
    constexpr int pairsPerThread = 4;
    const int first =
        (static_cast<int>(blockIdx.x) * Block + static_cast<int>(threadIdx.x)) * pairsPerThread;
    for (int row = static_cast<int>(blockIdx.y); row < rows; row += static_cast<int>(gridDim.y)) {
        const std::int64_t base = static_cast<std::int64_t>(row) * pairs;
#pragma unroll
        for (int item = 0; item < pairsPerThread; ++item) {
            const int pair = first + item;
            if (pair < pairs) { x[base + pair] ^= bias[pair]; }
        }
    }
}

// Pack-width forcing for the control legs: 0 = production boundary, 1 = Bf16x8 (uint4) streams
// (the higher-occupancy row variant when columns >= 1024, mirroring the launcher), 2 = Bf16x2
// (pair) streams.
int force_pack          = 0;
int g_tuning_sm_count   = 0;
GpuTuningProfile g_profile = GpuTuningProfile::Rtx5090;

bool use_x8_control(std::int64_t n) {
    if (force_pack == 1) { return true; }
    if (force_pack == 2) { return false; }
    return n <= ops::bf16x8_cache_sized_max_elements(g_tuning_sm_count);
}

void run(int d, int columns, bool control) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(columns);
    DeviceBuffer x      = make_bf16(n);
    DeviceBuffer bias   = make_bf16(d);
    Tensor tx(x.p, DType::BF16, {d, columns});
    Tensor tb(bias.p, DType::BF16, {d});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block   = 256;
                const int packs       = d / 8;
                const unsigned grid_x = static_cast<unsigned>((packs + block - 1) / block);
                if (use_x8_control(static_cast<std::int64_t>(n)) && columns >= 1024) {
                    constexpr int rowsPerBlock = 4;
                    const unsigned grid_y =
                        static_cast<unsigned>((columns + rowsPerBlock - 1) / rowsPerBlock);
                    add_bias_payload_control<rowsPerBlock>
                        <<<dim3(grid_x, grid_y), block, 0, stream>>>(
                            static_cast<const uint4*>(bias.p), static_cast<uint4*>(x.p), packs,
                            columns);
                } else if (use_x8_control(static_cast<std::int64_t>(n))) {
                    add_bias_payload_control<1><<<dim3(grid_x, columns), block, 0, stream>>>(
                        static_cast<const uint4*>(bias.p), static_cast<uint4*>(x.p), packs,
                        columns);
                } else {
                    constexpr int pairsPerThread = 4;
                    const int pairs              = d / 2;
                    const unsigned pair_grid_x   = static_cast<unsigned>(
                        (pairs + block * pairsPerThread - 1) / (block * pairsPerThread));
                    const unsigned pair_grid_y = static_cast<unsigned>(std::min(columns, 65535));
                    add_bias_pair_payload_control<block>
                        <<<dim3(pair_grid_x, pair_grid_y), block, 0, stream>>>(
                            static_cast<const unsigned int*>(bias.p),
                            static_cast<unsigned int*>(x.p), pairs, columns);
                }
            } else {
                const DeviceExecutionView execution{.stream          = stream,
                                                    .tuning_profile  = g_profile,
                                                    .tuning_sm_count = g_tuning_sm_count};
                ops::add_bias(tb, tx, execution);
            }
        },
        static_cast<double>(n) * 4.0);

    char tag[80];
    const char* pack_tag =
        (control && force_pack == 1) ? "-x8" : ((control && force_pack == 2) ? "-x2" : "");
    std::snprintf(tag, sizeof(tag), "%s%s [%d,%-5d]", control ? "control" : "add_bias", pack_tag,
                  d, columns);
    print_result(tag, result);
}

void run_matrix(bool control) {
    constexpr std::array<int, 5> patches{8, 256, 4096, 49152, 65536};
    constexpr std::array<int, 5> merged{2, 64, 1024, 12288, 16384};
    for (int d : {1152, 3456, 4304}) {
        for (int p : patches) { run(d, p, control); }
    }
    for (int d : {2048, 4608}) {
        for (int v : merged) { run(d, v, control); }
    }
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_d       = 0;
    int selected_columns = 0;
    bool control         = false;
    GpuTuningProfile requested = GpuTuningProfile::Auto;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            selected_d = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            selected_columns = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else if (!std::strcmp(argv[i], "--force-pack") && i + 1 < argc) {
            const char* pack = argv[++i];
            if (!std::strcmp(pack, "auto")) {
                force_pack = 0;
            } else if (!std::strcmp(pack, "x8")) {
                force_pack = 1;
            } else if (!std::strcmp(pack, "x2")) {
                force_pack = 2;
            } else {
                std::fprintf(stderr, "force-pack must be auto, x8, or x2\n");
                return 2;
            }
        } else if (!std::strcmp(argv[i], "--tuning-profile") && i + 1 < argc) {
            const auto parsed = parse_tuning_profile(argv[++i]);
            if (!parsed) {
                std::fprintf(stderr, "tuning-profile must be auto, rtx-5090, or rtx-pro-5000\n");
                return 2;
            }
            requested = *parsed;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--d D --columns C] [--control] [--force-pack auto|x8|x2] "
                         "[--tuning-profile auto|rtx-5090|rtx-pro-5000]\n",
                         argv[0]);
            return 2;
        }
    }
    int compute_capability   = 0;
    int multiprocessor_count = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        compute_capability   = prop.major * 10 + prop.minor;
        multiprocessor_count = prop.multiProcessorCount;
    }
    const auto tuning =
        resolve_tuning_profile(requested, compute_capability, multiprocessor_count);
    g_profile         = tuning.concrete;
    g_tuning_sm_count = tuning.tuning_sm_count;
    if ((selected_d == 0) != (selected_columns == 0) || selected_d < 0 || selected_columns < 0 ||
        (selected_d > 0 && selected_d % 8 != 0)) {
        std::fprintf(stderr, "d and columns must be supplied together; d must be divisible by 8\n");
        return 2;
    }

    if (selected_d > 0) {
        run(selected_d, selected_columns, control);
        return 0;
    }
    run_matrix(control);
    return 0;
}

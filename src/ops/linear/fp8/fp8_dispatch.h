#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);

// Returns the named candidate A16 route owned by the shape translation unit, or nullptr when the
// shape or route name is unknown. Benchmarks use this to sweep routes without instantiating the
// header-defined kernels a second time.
[[nodiscard]] Fp8Launch fp8_linear_a16_route(std::int32_t output_rows, std::int32_t input_rows,
                                             std::string_view route_name);

} // namespace ninfer::ops::detail

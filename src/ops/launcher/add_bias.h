#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, DeviceExecutionView execution);

} // namespace ninfer::ops::detail

#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/gelu.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, DeviceExecutionView execution);

} // namespace ninfer::ops::detail

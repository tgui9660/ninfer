#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, DeviceExecutionView execution);

} // namespace ninfer::ops::detail

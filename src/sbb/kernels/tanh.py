import triton
import triton.language as tl


@triton.jit
def tanh(x):
    x_f32 = x.to(tl.float32)
    result_f32 = 2 * tl.sigmoid(2 * x_f32) - 1
    return tl.maximum(tl.minimum(result_f32, 1.0), -1.0).to(x.dtype)

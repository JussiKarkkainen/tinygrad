from __future__ import annotations
import functools

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, cdiv
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, UOp

# Handwritten Metal fast path for Transformer.output_norm.
# This is intentionally narrow: Gemma 4 final RMSNorm on METAL with
# float32 activations, float16 weights, hidden size 2560, eps 1e-6.

_OUTPUT_NORM_HIDDEN = 2560
_OUTPUT_NORM_EPS = 1e-6
_VOCAB_DIM = 248320
_OUTPUT_THREADS_PER_TG = 256
_OUTPUT_SIMD_WIDTH = 32
_OUTPUT_ROWS_PER_TG = _OUTPUT_THREADS_PER_TG // _OUTPUT_SIMD_WIDTH

_OUTPUT_NORM_METAL_TEMPLATE = r"""
// Contract:
// - input x:      float32, shape (..., __HIDDEN__), device=METAL
// - input weight: float16, shape (__HIDDEN__,),     device=METAL
// - output out:   float32, same shape/device as x
//
// Numerical behavior to match tinygrad.nn.RMSNorm:
//   x_f = x.float()
//   inv_rms = rsqrt(mean(x_f * x_f, axis=-1) + eps)
//   y = cast(x_f * inv_rms, x.dtype)
//   out = y * weight

#include <metal_stdlib>
using namespace metal;

kernel void output_rmsnorm(
  device float* out,
  device const float* x,
  device const half* weight,

  uint row [[threadgroup_position_in_grid]],
  uint lid [[thread_position_in_threadgroup]],
  uint tpg [[threads_per_threadgroup]]) {

  threadgroup float partial_sums[256];
  const uint hidden_size = __HIDDEN__;
  uint row_start = row * hidden_size;

  const float eps = __EPS__f;

  // Each thread computes a partial sum
  float local_sum = 0.0f;
  for (uint col = lid; col < hidden_size; col += tpg) {
    float v = x[row_start + col];
    local_sum += v * v;
  }

  partial_sums[lid] = local_sum;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce partial sums
  for (uint stride = tpg / 2; stride > 0; stride >>=1) {
    if (lid < stride) {
      partial_sums[lid] += partial_sums[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sum_squares = partial_sums[0];
  float mean_square = sum_squares / float(hidden_size);
  float inv_rms = rsqrt(mean_square + eps);

  // Normalise and write output
  for (uint col = lid; col < hidden_size; col += tpg) {
    float v = x[row_start + col];
    float scale = float(weight[col]);
    float y = v * inv_rms * scale;
    out[row_start + col] = y;
  }

}
"""

_OUTPUT_NORM_METAL_SRC = _OUTPUT_NORM_METAL_TEMPLATE.replace("__HIDDEN__", str(_OUTPUT_NORM_HIDDEN)).replace("__EPS__", f"{_OUTPUT_NORM_EPS:g}")

_OUTPUT_METAL_TEMPLATE = r"""
// Contract:
// - input x:      float32, shape (..., __HIDDEN__), device=METAL
// - input weight: float16, shape (__VOCAB__, __HIDDEN__,),     device=METAL
// - output out:   float32, shape (..., __VOCAB__,), device=METAL

#include <metal_stdlib>
using namespace metal;

kernel void output_vocab_projection(
  device float* out,
  device const float* x,
  device const half* weight,

  uint3 gid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]],
  uint simd_lane [[thread_index_in_simdgroup]]) {

  const uint hidden_size = __HIDDEN__;
  const uint vocab_size = __VOCAB__;
  const uint sg_per_tg = 8;
  const uint threads_per_tg = 256;
  const uint row = gid.y;

  threadgroup float input[hidden_size];

  // Load x into tg memory
  uint x_base = row * hidden_size;
  for (uint k = tid; k < hidden_size; k += threads_per_tg) {
    input[k] = x[x_base + k];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint simd_id = tid / 32;
  uint row_id = gid.x * sg_per_tg + simd_id;
  if (row_id >= vocab_size) return;

  float acc = 0.0f;
  uint w_base = row_id * hidden_size;
  for (uint h = simd_lane; h < hidden_size; h += 32) {
    acc += input[h] * float(weight[w_base + h]);
  }

  acc = simd_sum(acc);

  if (simd_lane == 0) {
    out[row * vocab_size + row_id] = acc;
  }
}
"""

_OUTPUT_METAL_SRC = _OUTPUT_METAL_TEMPLATE.replace("__HIDDEN__", str(_OUTPUT_NORM_HIDDEN)).replace("__VOCAB__", str(_VOCAB_DIM))

@functools.cache
def _compiled_kernel(src:str) -> bytes:
  """
  Compile cache for the handwritten Metal source.
  """
  from tinygrad.runtime.ops_metal import MetalCompiler
  return MetalCompiler().compile_cached(src)

def _output_kernel(out:UOp, x:UOp, weight:UOp) -> UOp:
  assert x.dtype.base == dtypes.float, f"output_vocab_projection expects float32 x, got {x.dtype}"
  assert out.dtype.base == dtypes.float, f"output_vocab_projection expects float32 out, got {out.dtype}"
  assert weight.dtype.base == dtypes.half, f"output_vocab_projection expects float16 weight, got {weight.dtype}"
  assert x.shape[-1] == _OUTPUT_NORM_HIDDEN, f"output_vocab_projection expects hidden={_OUTPUT_NORM_HIDDEN}, got {x.shape}"
  assert weight.shape == (_VOCAB_DIM, _OUTPUT_NORM_HIDDEN), \
    f"output_vocab_projection expects weight shape ({_VOCAB_DIM}, {_OUTPUT_NORM_HIDDEN}), got {weight.shape}"
  assert out.shape == (*x.shape[:-1], _VOCAB_DIM), \
    f"output_vocab_projection expects out shape {(*x.shape[:-1], _VOCAB_DIM)}, got {out.shape}"

  lib = _compiled_kernel(_OUTPUT_METAL_SRC)
  rows = prod(x.shape[:-1])
  groups_x = cdiv(_VOCAB_DIM, _OUTPUT_ROWS_PER_TG)
  ops = rows * _VOCAB_DIM * (2 * _OUTPUT_NORM_HIDDEN)
  mem = rows * (x.shape[-1] * x.dtype.base.itemsize + _VOCAB_DIM * out.dtype.base.itemsize) + \
        prod(weight.shape) * weight.dtype.base.itemsize

  sink = UOp.sink(
    UOp.special(groups_x, "gidx0"),
    UOp.special(rows, "gidx1"),
    UOp.special(_OUTPUT_THREADS_PER_TG, "lidx0"),
    out,
    x,
    weight,
    arg=KernelInfo(name="output_vocab_projection", estimates=Estimates(ops=ops, mem=mem)),
  )
  return UOp(
    Ops.PROGRAM,
    src=(
      sink,
      UOp(Ops.DEVICE, arg=Device.DEFAULT),
      UOp(Ops.LINEAR, src=(*sink.src, sink)),
      UOp(Ops.SOURCE, arg=_OUTPUT_METAL_SRC),
      UOp(Ops.BINARY, arg=lib),
    ),
  )

def custom_output(x:Tensor, weight:Tensor) -> Tensor:
  assert x.device == "METAL", f"custom_output requires METAL, got {x.device}"
  assert x.dtype == dtypes.float, f"custom_output expects float32 x, got {x.dtype}"
  assert x.shape[-1] == _OUTPUT_NORM_HIDDEN, f"custom_output expects hidden={_OUTPUT_NORM_HIDDEN}, got {x.shape[-1]}"
  assert weight.device == x.device, f"weight must be on the same device as x ({x.device}), got {weight.device}"
  assert weight.dtype == dtypes.half, f"custom_output expects float16 weight, got {weight.dtype}"
  assert weight.shape == (_VOCAB_DIM, _OUTPUT_NORM_HIDDEN), \
    f"custom_output expects weight shape ({_VOCAB_DIM}, {_OUTPUT_NORM_HIDDEN}), got {weight.shape}"

  out = Tensor.empty(*x.shape[:-1], _VOCAB_DIM, dtype=x.dtype, device=x.device)
  contig_tensors = tuple(t if t.uop.op is Ops.AFTER else t.contiguous() for t in (out, x, weight))
  params = [t.uop.param_like(i) for i, t in enumerate(contig_tensors)]
  kernel = _output_kernel(*params).call(*[t.uop for t in contig_tensors])
  return Tensor(contig_tensors[0].uop.after(kernel))

def custom_ffn_gate_up(x:Tensor, norm_weight:Tensor, gate_weight:Tensor, up_weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  """
  Handwritten Metal fast path scaffold for:
    gate, up = ffn_gate(rmsnorm(x)), ffn_up(rmsnorm(x))

  Intended contract for the eventual custom kernel:
  - x:           float32, shape (..., dim), device=METAL
  - norm_weight: float16, shape (dim,),     device=METAL
  - gate_weight: float16, shape (hidden, dim), device=METAL
  - up_weight:   float16, shape (hidden, dim), device=METAL
  - returns:     two float32 tensors of shape (..., hidden)
  """
  raise NotImplementedError("custom_ffn_gate_up scaffold is wired, but the Metal kernel has not been implemented yet")

def custom_ffn_down(x:Tensor, weight:Tensor) -> Tensor:
  """
  Handwritten Metal fast path scaffold for:
    out = ffn_down(x)

  Intended contract for the eventual custom kernel:
  - x:      float32, shape (..., hidden), device=METAL
  - weight: float16, shape (dim, hidden), device=METAL
  - returns float32 tensor of shape (..., dim)
  """
  raise NotImplementedError("custom_ffn_down scaffold is wired, but the Metal kernel has not been implemented yet")

def _output_norm_kernel(out:UOp, x:UOp, weight:UOp, eps:float) -> UOp:
  """
  Build the opaque PROGRAM node for the handwritten Metal RMSNorm.
  """
  assert x.dtype.base == dtypes.float, f"output_rmsnorm expects float32 x, got {x.dtype}"
  assert out.dtype.base == dtypes.float, f"output_rmsnorm expects float32 out, got {out.dtype}"
  assert weight.dtype.base == dtypes.half, f"output_rmsnorm expects float16 weight, got {weight.dtype}"
  assert x.shape[-1] == _OUTPUT_NORM_HIDDEN and out.shape[-1] == _OUTPUT_NORM_HIDDEN, \
    f"output_rmsnorm expects hidden={_OUTPUT_NORM_HIDDEN}, got x={x.shape}, out={out.shape}"
  assert weight.shape == (_OUTPUT_NORM_HIDDEN,), \
    f"output_rmsnorm expects weight shape ({_OUTPUT_NORM_HIDDEN},), got {weight.shape}"
  assert abs(eps - _OUTPUT_NORM_EPS) < 1e-12, f"output_rmsnorm expects eps={_OUTPUT_NORM_EPS}, got {eps}"

  lib = _compiled_kernel(_OUTPUT_NORM_METAL_SRC)
  rows = prod(x.shape[:-1])
  elems = prod(x.shape)
  ops = rows * (5 * _OUTPUT_NORM_HIDDEN + 2)
  mem = elems * (x.dtype.base.itemsize * 2 + out.dtype.base.itemsize + weight.dtype.base.itemsize)

  sink = UOp.sink(
    UOp.special(rows, "gidx0"),
    UOp.special(256, "lidx0"),
    out,
    x,
    weight,
    arg=KernelInfo(name="output_rmsnorm", estimates=Estimates(ops=ops, mem=mem)),
  )
  return UOp(
    Ops.PROGRAM,
    src=(
      sink,
      UOp(Ops.DEVICE, arg=Device.DEFAULT),
      UOp(Ops.LINEAR, src=(*sink.src, sink)),
      UOp(Ops.SOURCE, arg=_OUTPUT_NORM_METAL_SRC),
      UOp(Ops.BINARY, arg=lib),
    ),
  )

def custom_output_rmsnorm(x:Tensor, weight:Tensor, eps:float) -> Tensor:
  """
  Tensor-level wrapper for the handwritten Metal replacement for `nn.RMSNorm`.
  """
  assert x.device == "METAL", f"custom_output_rmsnorm requires METAL, got {x.device}"
  assert x.dtype == dtypes.float, f"custom_output_rmsnorm expects float32 x, got {x.dtype}"
  assert len(weight.shape) == 1, f"custom_output_rmsnorm expects 1D weight, got {weight.shape}"
  assert x.shape[-1] == weight.shape[0], f"last dim of {x.shape} must match weight {weight.shape}"
  assert weight.device == x.device, f"weight must be on the same device as x ({x.device}), got {weight.device}"
  assert weight.dtype == dtypes.half, f"custom_output_rmsnorm expects float16 weight, got {weight.dtype}"
  assert x.shape[-1] == _OUTPUT_NORM_HIDDEN, f"custom_output_rmsnorm expects hidden={_OUTPUT_NORM_HIDDEN}, got {x.shape[-1]}"

  out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
  contig_tensors = tuple(t if t.uop.op is Ops.AFTER else t.contiguous() for t in (out, x, weight))
  params = [t.uop.param_like(i) for i, t in enumerate(contig_tensors)]
  kernel = _output_norm_kernel(*params, eps).call(*[t.uop for t in contig_tensors])
  return Tensor(contig_tensors[0].uop.after(kernel))

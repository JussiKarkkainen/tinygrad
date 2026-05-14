from __future__ import annotations
import functools, itertools
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, dtypes, function
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, partial_rotary_factor: float = 1.0,
                         freq_factors:tuple[float, ...]|None=None) -> Tensor:
  rope_angles = int(partial_rotary_factor * dim // 2)
  inv_rot = 1.0 / (theta ** (Tensor.arange(0, 2 * rope_angles, 2) / dim)) if rope_angles > 0 else Tensor.empty(0)
  nope_angles = dim // 2 - rope_angles
  inv_freq = inv_rot.cat(Tensor.zeros(nope_angles), dim=0) if nope_angles > 0 else inv_rot
  if freq_factors is not None:
    ff = Tensor(list(freq_factors), dtype=inv_freq.dtype)
    if ff.shape[0] != inv_freq.shape[0]: raise RuntimeError(f"invalid rope_freqs size: {ff.shape[0]} vs {inv_freq.shape[0]}")
    inv_freq = inv_freq / ff
  freqs = Tensor.arange(end).unsqueeze(dim=1) * inv_freq.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = Tensor.zeros_like(x).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerLayerConfig:
  sliding_window: int|None
  head_dim: int
  rope_theta: float
  partial_rotary_factor: float = 1.0
  rope_freq_factors: tuple[float, ...]|None = None
  kv_shared_source: int|None = None
  store_shared_kv: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  activation: str = "silu"
  final_logit_softcap: float = 0.0
  per_layer_input_dim: int = 0
  layers: tuple[TransformerLayerConfig, ...] = ()

@dataclass
class BlockContext:
  per_layer_inputs: Tensor|None = None
  shared_kv_states: dict[int, tuple[Tensor, Tensor]]|None = None

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.kv_lora_rank > 0: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _apply_ffn_gate_up(self, x:Tensor) -> tuple[Tensor, Tensor]:
    if hasattr(self, 'ffn_gate_exps'): raise RuntimeError("_apply_ffn_gate_up only supports dense FFN blocks")
    use_custom = bool(getenv("CUSTOM_KERNELS"))
    if not use_custom:
      xn = self.ffn_norm(x)
      return self.ffn_gate(xn), self.ffn_up(xn)
    if x.device != "METAL": raise RuntimeError(f"CUSTOM_KERNELS requires METAL for FFN gate/up, got {x.device}")
    assert self.ffn_norm.weight is not None
    assert self.ffn_gate.weight is not None and self.ffn_up.weight is not None
    if x.dtype != dtypes.float or self.ffn_norm.weight.dtype != dtypes.half or self.ffn_gate.weight.dtype != dtypes.half or \
       self.ffn_up.weight.dtype != dtypes.half or x.shape[-1] != self.config.dim or self.ffn_gate.weight.shape != (self.config.hidden_dim, self.config.dim) or \
       self.ffn_up.weight.shape != (self.config.hidden_dim, self.config.dim):
      raise RuntimeError("CUSTOM_KERNELS FFN gate/up contract mismatch")
    from tinygrad.llm.metal_kernels import custom_ffn_gate_up
    return custom_ffn_gate_up(x, self.ffn_norm.weight, self.ffn_gate.weight, self.ffn_up.weight, self.ffn_norm.eps)

  def _apply_ffn_down(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'): raise RuntimeError("_apply_ffn_down only supports dense FFN blocks")
    use_custom = bool(getenv("CUSTOM_KERNELS"))
    if not use_custom: return self.ffn_down(x)
    if x.device != "METAL": raise RuntimeError(f"CUSTOM_KERNELS requires METAL for FFN down, got {x.device}")
    assert self.ffn_down.weight is not None
    if x.dtype != dtypes.float or self.ffn_down.weight.dtype != dtypes.half or x.shape[-1] != self.config.hidden_dim or \
       self.ffn_down.weight.shape != (self.config.dim, self.config.hidden_dim):
      raise RuntimeError("CUSTOM_KERNELS FFN down contract mismatch")
    from tinygrad.llm.metal_kernels import custom_ffn_down
    return custom_ffn_down(x, self.ffn_down.weight)

  def _feed_forward(self, x:Tensor) -> Tensor:
    act = Tensor.gelu if self.config.activation == "gelu" else Tensor.silu
    if hasattr(self, 'ffn_gate_exps'):
      x = self.ffn_norm(x)
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, act(self.ffn_gate_exps(sel, h)) * self.ffn_up_exps(sel, h))  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(act(self.ffn_gate_shexp(x)).contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    gate, up = self._apply_ffn_gate_up(x)
    return self._apply_ffn_down(act(gate).contiguous() * up)

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, context:BlockContext|None=None, block_index:int|None=None):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(h)).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=False)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=False)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=False)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    v_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = Tensor(self.cache_v.uop.after(self.cache_v[:, :, start_pos:start_pos+T, :].uop.store(v_store.uop)))[:, :, 0:start_pos+T, :]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.cache_v = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(Tensor.zeros_like(self.conv_state)),
            self.recurrent_state.assign(Tensor.zeros_like(self.recurrent_state))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Gemma4Block(FFNBlock):
  def __init__(self, config:TransformerConfig, layer_config:TransformerLayerConfig, block_index:int):
    super().__init__(config)
    self.layer_config = layer_config
    self.block_index = block_index
    self.attn_q = nn.Linear(config.dim, layer_config.head_dim * config.n_heads, bias=False)
    self.attn_k = nn.Linear(config.dim, layer_config.head_dim * config.n_kv_heads, bias=False)
    self.attn_v = nn.Linear(config.dim, layer_config.head_dim * config.n_kv_heads, bias=False)
    self.attn_output = nn.Linear(layer_config.head_dim * config.n_heads, config.dim, bias=False)
    self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(layer_config.head_dim, config.norm_eps), nn.RMSNorm(layer_config.head_dim, config.norm_eps)
    self.post_attention_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.post_ffw_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.post_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.inp_gate = nn.Linear(config.dim, config.per_layer_input_dim, bias=False) if config.per_layer_input_dim else None
    self.proj = nn.Linear(config.per_layer_input_dim, config.dim, bias=False) if config.per_layer_input_dim else None
    self.layer_output_scale = {"weight": Tensor.ones(1)}

  def _v_norm(self, x:Tensor) -> Tensor:
    xf = x.float()
    return (xf * (xf.square().mean(axis=-1, keepdim=True) + self.config.norm_eps).pow(-0.5)).cast(x.dtype)

  def _attention_mask(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    qpos = (Tensor.arange(T, device=x.device) + start_pos).reshape(1, 1, T, 1)
    kpos = Tensor.arange(self.config.max_context, device=x.device).reshape(1, 1, 1, self.config.max_context)
    valid = kpos <= qpos
    if self.layer_config.sliding_window is not None:
      valid = valid & (kpos >= qpos - self.layer_config.sliding_window + 1)
    return valid.where(Tensor.zeros(1, 1, T, self.config.max_context, dtype=x.dtype, device=x.device),
                       Tensor.full((1, 1, T, self.config.max_context), float("-inf"), dtype=x.dtype, device=x.device))

  def _attention(self, x:Tensor, start_pos:int|UOp, shared_kv_states:dict[int, tuple[Tensor, Tensor]]) -> Tensor:
    B, T, _ = x.shape

    q = self.attn_q(x).reshape(B, T, self.config.n_heads, self.layer_config.head_dim).transpose(1, 2)
    q = self.attn_q_norm(q)
    q = apply_rope(q, self.freqs_cis[start_pos:start_pos+T])

    if self.layer_config.kv_shared_source is not None:
      k, v = shared_kv_states[self.layer_config.kv_shared_source]
    else:
      k = self.attn_k(x).reshape(B, T, self.config.n_kv_heads, self.layer_config.head_dim).transpose(1, 2)
      v = self.attn_v(x).reshape(B, T, self.config.n_kv_heads, self.layer_config.head_dim).transpose(1, 2)
      k = apply_rope(self.attn_k_norm(k), self.freqs_cis[start_pos:start_pos+T])
      v = self._v_norm(v)
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
      if self.layer_config.store_shared_kv:
        shared_kv_states[self.block_index] = (assigned_kv[0], assigned_kv[1])
      k, v = assigned_kv[0], assigned_kv[1]

    if self.config.n_heads != self.config.n_kv_heads:
      rep = self.config.n_heads // self.config.n_kv_heads
      k = k.unsqueeze(2).expand(B, self.config.n_kv_heads, rep, k.shape[2], k.shape[3]).reshape(B, self.config.n_heads, k.shape[2], k.shape[3])
      v = v.unsqueeze(2).expand(B, self.config.n_kv_heads, rep, v.shape[2], v.shape[3]).reshape(B, self.config.n_heads, v.shape[2], v.shape[3])

    # Mirror tinygrad's working attention precision path:
    # matmul in fp32, then softmax in activation dtype.
    attn = q.matmul(k.transpose(-1, -2), dtype=dtypes.float32)
    attn = attn + self._attention_mask(x, start_pos, T)
    attn = attn.cast(q.dtype).softmax(-1)
    attn = (attn @ v).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.layer_config.head_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.layer_config.head_dim, self.config.max_context, self.layer_config.rope_theta,
                                            self.layer_config.partial_rotary_factor, self.layer_config.rope_freq_factors)

  def __call__(self, x:Tensor, start_pos:int|UOp, context:BlockContext|None=None, block_index:int|None=None):
    per_layer_input = None if context is None or context.per_layer_inputs is None or block_index is None else context.per_layer_inputs[:, :, block_index, :]
    shared_kv_states = {} if context is None or context.shared_kv_states is None else context.shared_kv_states
    self._init_state(x)
    h = x + self.post_attention_norm(self._attention(self.attn_norm(x), start_pos, shared_kv_states))
    h = h + self.post_ffw_norm(self._feed_forward(h))
    if per_layer_input is not None and self.inp_gate is not None:
      act = Tensor.gelu if self.config.activation == "gelu" else Tensor.silu
      h = h + self.post_norm(self.proj(act(self.inp_gate(h)).contiguous() * per_layer_input))
    return (h * self.layer_output_scale["weight"]).contiguous()

class Transformer:
  def __init__(self, config:TransformerConfig):
    if len(config.layers) > 0:
      self.blk:list[FFNBlock|Gemma4Block] = [Gemma4Block(config, lc, i) for i, lc in enumerate(config.layers)]
      self.has_recurrent_block = False
      self._prepare_hidden = self._prepare_hidden_gemma4
    else:
      dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
      if config.ssm: config = replace(config, qk_norm=config.head_dim)
      block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
      self.blk = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                  block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
      self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
      self._prepare_hidden = self._prepare_hidden_default
    self.config = config
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    if len(config.layers) > 0 and config.per_layer_input_dim:
      self.per_layer_token_embd = nn.Embedding(config.vocab_size, config.num_blocks * config.per_layer_input_dim)
      self.per_layer_proj_norm = nn.RMSNorm(config.per_layer_input_dim, config.norm_eps)
      self.per_layer_model_proj = nn.Linear(config.dim, config.num_blocks * config.per_layer_input_dim, bias=False)
    self.max_context = config.max_context
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def _sample_from_logits(self, logits:Tensor, temperature:Tensor) -> Tensor:
    if float(temperature.item()) == 0.0: return logits.float().argmax(-1, keepdim=True).cast("int32")
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    # Keep sampling math in fp32 to avoid fp16 overflow at very low temperatures.
    logits_f = logits.float()
    gumbel = (Tensor.rand_like(logits_f).maximum(1e-12).log().neg()).log()
    return (logits_f / temperature.float().maximum(1e-12) - gumbel).argmax(-1, keepdim=True).cast("int32")

  def _prepare_hidden_default(self, tokens:Tensor) -> tuple[Tensor, BlockContext]:
    return self.token_embd(tokens).float(), BlockContext()

  def _prepare_hidden_gemma4(self, tokens:Tensor) -> tuple[Tensor, BlockContext]:
    x = (self.token_embd(tokens) * (self.config.dim ** 0.5)).float()
    if hasattr(self, "per_layer_token_embd"):
      ple = (self.per_layer_token_embd(tokens) * (self.config.per_layer_input_dim ** 0.5)).reshape(tokens.shape[0], tokens.shape[1], self.config.num_blocks, self.config.per_layer_input_dim).float()
      proj = self.per_layer_proj_norm((self.per_layer_model_proj(x) * (self.config.dim ** -0.5)).reshape(tokens.shape[0], tokens.shape[1], self.config.num_blocks, self.config.per_layer_input_dim))
      ple = (ple + proj) * (2.0 ** -0.5)
    else:
      ple = None
    return x, BlockContext(per_layer_inputs=ple, shared_kv_states={})

  def _run_blocks(self, x:Tensor, start_pos:int|UOp, context:BlockContext) -> Tensor:
    for i, block in enumerate(self.blk):
      x = block(x, start_pos, context, i)
    return x

  def _apply_output_norm(self, x:Tensor) -> Tensor:
    use_custom = bool(getenv("CUSTOM_KERNELS"))
    if not use_custom: return self.output_norm(x)
    if x.device != "METAL": raise RuntimeError(f"CUSTOM_KERNELS requires METAL for output_norm, got {x.device}")
    assert self.output_norm.weight is not None
    if x.dtype != dtypes.float or self.output_norm.weight.dtype != dtypes.half or x.shape[-1] != 2560 or abs(self.output_norm.eps - 1e-6) > 1e-12:
      raise RuntimeError(f"CUSTOM_KERNELS output_norm contract mismatch: x.dtype={x.dtype}, weight.dtype={self.output_norm.weight.dtype}, "
                         f"x.shape={x.shape}, eps={self.output_norm.eps}")
    from tinygrad.llm.metal_kernels import custom_output_rmsnorm
    return custom_output_rmsnorm(x, self.output_norm.weight, self.output_norm.eps)

  def _apply_output(self, x:Tensor) -> Tensor:
    use_custom = bool(getenv("CUSTOM_KERNELS"))
    if not use_custom: return self.output(x)
    if x.device != "METAL": raise RuntimeError(f"CUSTOM_KERNELS requires METAL for output projection, got {x.device}")
    assert self.output.weight is not None
    if x.dtype != dtypes.float or self.output.weight.dtype != dtypes.half or x.shape[-1] != 2560:
      raise RuntimeError(f"CUSTOM_KERNELS output contract mismatch: x.dtype={x.dtype}, weight.dtype={self.output.weight.dtype}, x.shape={x.shape}")
    from tinygrad.llm.metal_kernels import custom_output
    return custom_output(x, self.output.weight)
  
  def logits(self, tokens:Tensor, start_pos:int|UOp):
    x, context = self._prepare_hidden(tokens)
    x = self._run_blocks(x, start_pos, context)
    logits = self._apply_output(self._apply_output_norm(x))[:, -1, :]
    if self.config.final_logit_softcap: logits = (logits / self.config.final_logit_softcap).tanh() * self.config.final_logit_softcap
    return logits

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return self.logits(tokens, start_pos)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  def reset_cache(self):
    for block in self.blk:
      for name in ("cache_kv", "cache_k", "cache_v", "full_kv_cache", "conv_state", "recurrent_state"):
        if hasattr(block, name): delattr(block, name)
    self._cached_tokens = []

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    if not self._cached_tokens:
      for block in self.blk:
        for name in ("cache_kv", "cache_k", "cache_v", "full_kv_cache", "conv_state", "recurrent_state"):
          if hasattr(block, name): delattr(block, name)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor(temperature).contiguous()
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
      logits = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      out = self._sample_from_logits(logits, temp).reshape(1, 1).contiguous().realize()
      start_pos += nt.val
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize())
    arch = kv['general.architecture']

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    if arch == 'gemma4':
      pattern = kv[f'{arch}.attention.sliding_window_pattern']
      first_shared = kv[f'{arch}.block_count'] - kv.get(f'{arch}.attention.shared_kv_layers', 0)
      prev_layers = ["sliding_attention" if x else "full_attention" for x in pattern[:first_shared]]
      rope_freq_factors = None
      if (rf:=state_dict.get("rope_freqs.weight")) is not None:
        rope_freq_factors = tuple(float(x) for x in rf.float().tolist())
      else:
        for k, v in state_dict.items():
          if k.endswith(".rope_freqs.weight"):
            rope_freq_factors = tuple(float(x) for x in v.float().tolist())
            break
      config = TransformerConfig(
        num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'], hidden_dim=kv[f'{arch}.feed_forward_length'],
        n_heads=kv[f'{arch}.attention.head_count'], n_kv_heads=kv[f'{arch}.attention.head_count_kv'],
        norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'], vocab_size=len(kv['tokenizer.ggml.tokens']),
        head_dim=0, rope_theta=0.0, rope_dim=0, v_head_dim=0, max_context=max_context,
        activation="gelu", final_logit_softcap=kv.get(f'{arch}.final_logit_softcapping', 0.0),
        per_layer_input_dim=kv.get(f'{arch}.embedding_length_per_layer_input', 0),
        layers=tuple(
          TransformerLayerConfig(
            sliding_window=kv[f'{arch}.attention.sliding_window'] if is_sliding else None,
            head_dim=kv[f'{arch}.attention.key_length_swa'] if is_sliding else kv[f'{arch}.attention.key_length'],
            rope_theta=kv[f'{arch}.rope.freq_base_swa'] if is_sliding else kv[f'{arch}.rope.freq_base'],
            # GGUF Gemma4 uses rope_freqs factors on full-attention layers; when present, run full-dim RoPE and
            # let those factors null out unrotated components.
            partial_rotary_factor=1.0 if (is_sliding or rope_freq_factors is not None) else 0.25,
            rope_freq_factors=None if is_sliding else rope_freq_factors,
            kv_shared_source=None if i < first_shared else len(prev_layers) - 1 - prev_layers[::-1].index("sliding_attention" if is_sliding else "full_attention"),
            store_shared_kv=(i < first_shared and i == len(prev_layers) - 1 - prev_layers[::-1].index("sliding_attention" if is_sliding else "full_attention")))
          for i, is_sliding in enumerate(pattern)))
    else:
      n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
      ssm = None
      if arch in ('qwen35', 'qwen35moe'):
        ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
        state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

      kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
      head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
      rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

      # Permute RoPE weights from interleaved to half-split layout.
      for name in state_dict:
        if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
          w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
          prefix = head_dim-rope_dim
          state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
        elif arch == 'llama' and 'attn_k.weight' in name:
          w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
          state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
        elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
          state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
      config = TransformerConfig(
        num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
        hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
        n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
        vocab_size=len(kv['tokenizer.ggml.tokens']),
        head_dim=head_dim,
        rope_theta=kv[f'{arch}.rope.freq_base'],
        rope_dim=rope_dim,
        v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
        max_context=max_context,
        qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
        num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
        norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
        kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
        leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
        shared_expert_dim=kv.get(
          f'{arch}.expert_shared_feed_forward_length',
          kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
        shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
        dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
        routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
        full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
        final_logit_softcap=kv.get(f'{arch}.final_logit_softcapping', 0.0))
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

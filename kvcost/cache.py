import torch
from transformers.cache_utils import Cache, CacheLayerMixin

# Symmetric int8 range
INT8_MAX = 127.0
PING_PONG_SLOTS = 2


def quantize_int8(x):
    abs_max = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8)
    scale = abs_max / INT8_MAX
    q = torch.round(x.float() / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale.to(x.dtype)


def dequantize_int8(q, scale, dtype):
    with torch.profiler.record_function("kvcost::dequant"):
        return q.to(dtype) * scale


def offloaded_layer_ids(num_layers, fraction):
    n_offload = int(round(fraction * num_layers))
    if n_offload <= 0:
        return []

    picked = set()
    for j in range(n_offload):
        layer_idx = int((j + 1) * num_layers / n_offload) - 1
        picked.add(min(num_layers - 1, layer_idx))
    return sorted(picked)


class StagingPool:
    def __init__(self, capacity, kind):
        self.capacity = capacity
        self.kind = kind
        self.buffers = []

    def get_slot(self, slot_idx, ref_tensor, dtype):
        # Allocate on first forward pass 
        if not self.buffers:
            batch, kv_heads, _, head_dim = ref_tensor.shape
            cache_dtype = torch.int8 if self.kind == "int8" else dtype

            for _ in range(PING_PONG_SLOTS):
                buf = {
                    "k": torch.empty(self.capacity, batch, kv_heads, head_dim, dtype=cache_dtype, device=ref_tensor.device),
                    "v": torch.empty(self.capacity, batch, kv_heads, head_dim, dtype=cache_dtype, device=ref_tensor.device),
                }
                if self.kind == "int8":
                    buf["ks"] = torch.empty(self.capacity, batch, kv_heads, 1, dtype=dtype, device=ref_tensor.device)
                    buf["vs"] = torch.empty(self.capacity, batch, kv_heads, 1, dtype=dtype, device=ref_tensor.device)
                self.buffers.append(buf)

        return self.buffers[slot_idx % PING_PONG_SLOTS]


class OffloadLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, kind, capacity, pool, slot):
        super().__init__()
        self.kind = kind
        self.capacity = capacity
        self.pool = pool
        self.slot = slot
        self.seen = 0
        self.stored = 0
        self.pending_event = None
        self.host = {}

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_cuda = self.device.type == "cuda"

        batch, kv_heads, _, head_dim = key_states.shape
        cache_dtype = torch.int8 if self.kind == "int8" else self.dtype

        self.host = {
            "k": torch.zeros(self.capacity, batch, kv_heads, head_dim, dtype=cache_dtype, pin_memory=self.is_cuda),
            "v": torch.zeros(self.capacity, batch, kv_heads, head_dim, dtype=cache_dtype, pin_memory=self.is_cuda),
        }
        if self.kind == "int8":
            self.host["ks"] = torch.zeros(self.capacity, batch, kv_heads, 1, dtype=self.dtype, pin_memory=self.is_cuda)
            self.host["vs"] = torch.zeros(self.capacity, batch, kv_heads, 1, dtype=self.dtype, pin_memory=self.is_cuda)

        self.is_initialized = True

    def issue_prefetch(self, stream):
        if self.pending_event is not None or not self.is_initialized:
            return

        stage = self.pool.get_slot(self.slot, self._ref, self.dtype)
        if not self.is_cuda:
            for name in self.host:
                stage[name][: self.stored].copy_(self.host[name][: self.stored])
            self.pending_event = "cpu_ready"
            return

        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for name in self.host:
                stage[name][: self.stored].copy_(self.host[name][: self.stored], non_blocking=True)
            dma_event = torch.cuda.Event()
            dma_event.record(stream)
        self.pending_event = dma_event

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self._ref = key_states
            self.lazy_initialization(key_states, value_states)

        n_new = key_states.shape[-2]
        if self.stored + n_new > self.capacity:
            raise RuntimeError(
                f"KV cache full on offloaded layer (stored={self.stored}, incoming={n_new}, "
                f"capacity={self.capacity}). Increase cache capacity / max_seq_len."
            )

        stage = self.pool.get_slot(self.slot, self._ref, self.dtype)

        # Wait for in-flight H2D prefetch or copy synchronously if prefetch wasn't scheduledxw
        if isinstance(self.pending_event, torch.cuda.Event):
            torch.cuda.current_stream(self.device).wait_event(self.pending_event)
        elif self.pending_event is None and self.stored > 0:
            for name in self.host:
                stage[name][: self.stored].copy_(self.host[name][: self.stored], non_blocking=self.is_cuda)
        self.pending_event = None

        start = self.stored
        end = start + n_new

        if self.kind == "int8":
            qk, sk = quantize_int8(key_states)
            qv, sv = quantize_int8(value_states)
            payload = {"k": qk, "v": qv, "ks": sk, "vs": sv}
        else:
            payload = {"k": key_states, "v": value_states}

        for name, tensor in payload.items():
            tok_major = tensor.permute(2, 0, 1, 3).contiguous()
            stage[name][start:end] = tok_major
            self.host[name][start:end].copy_(tok_major, non_blocking=self.is_cuda)

        prev_len = self.stored
        self.stored += n_new
        self.seen += n_new
        total_len = self.stored

        if self.kind != "int8":
            return stage["k"][:total_len].permute(1, 2, 0, 3), stage["v"][:total_len].permute(1, 2, 0, 3)
        if n_new > 1:
            k_prev = dequantize_int8(
                stage["k"][:prev_len].permute(1, 2, 0, 3),
                stage["ks"][:prev_len].permute(1, 2, 0, 3),
                self.dtype,
            )
            v_prev = dequantize_int8(
                stage["v"][:prev_len].permute(1, 2, 0, 3),
                stage["vs"][:prev_len].permute(1, 2, 0, 3),
                self.dtype,
            )
            return torch.cat([k_prev, key_states], dim=-2), torch.cat([v_prev, value_states], dim=-2)

        return (
            dequantize_int8(
                stage["k"][:total_len].permute(1, 2, 0, 3),
                stage["ks"][:total_len].permute(1, 2, 0, 3),
                self.dtype,
            ),
            dequantize_int8(
                stage["v"][:total_len].permute(1, 2, 0, 3),
                stage["vs"][:total_len].permute(1, 2, 0, 3),
                self.dtype,
            ),
        )

    def get_seq_length(self):
        return self.seen

    def get_max_length(self):
        return -1

    def get_mask_sizes(self, query_length):
        kv_length = self.stored + query_length
        return kv_length, self.seen + query_length - kv_length

    def reset(self):
        self.host = {}
        self.is_initialized = False
        self.seen = 0
        self.stored = 0
        self.pending_event = None


class KVSpec:
    def __init__(self, kind="full", offload=0.0, prefetch=True):
        if kind not in ("full", "int8"):
            raise ValueError(f"Unsupported KV format '{kind}' (expected 'full' or 'int8')")
        if not (0.0 <= offload <= 1.0):
            raise ValueError(f"Offload fraction must be between 0.0 and 1.0, got {offload}")

        self.kind = kind
        self.offload = offload
        self.prefetch = prefetch

    def to_dict(self):
        cfg = {"kind": self.kind}
        if self.offload > 0.0:
            cfg["offload"] = self.offload
            cfg["prefetch"] = self.prefetch
        return cfg


class FullLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.seen = 0
        self.stored = 0
        self.keys = None
        self.values = None

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        batch, kv_heads, _, head_dim = key_states.shape

        self.keys = torch.zeros(batch, kv_heads, self.capacity, head_dim, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(batch, kv_heads, self.capacity, head_dim, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        n_new = key_states.shape[-2]
        if self.stored + n_new > self.capacity:
            raise RuntimeError(
                f"KV cache full (stored={self.stored}, incoming={n_new}, capacity={self.capacity}). "
                f"Increase cache capacity / max_seq_len."
            )

        start = self.stored
        end = start + n_new
        self.keys[:, :, start:end] = key_states
        self.values[:, :, start:end] = value_states
        self.stored += n_new
        self.seen += n_new
        return self.keys[:, :, : self.stored], self.values[:, :, : self.stored]

    def get_seq_length(self):
        return self.seen

    def get_max_length(self):
        return -1

    def get_mask_sizes(self, query_length):
        kv_length = self.stored + query_length
        return kv_length, self.seen + query_length - kv_length

    def reset(self):
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.seen = 0
        self.stored = 0

class Int8Layer(CacheLayerMixin):

    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.seen = 0
        self.stored = 0
        self.keys = None
        self.values = None
        self.k_scale = None
        self.v_scale = None

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        batch, kv_heads, _, head_dim = key_states.shape

        self.keys = torch.zeros(batch, kv_heads, self.capacity, head_dim, dtype=torch.int8, device=self.device)
        self.values = torch.zeros(batch, kv_heads, self.capacity, head_dim, dtype=torch.int8, device=self.device)
        self.k_scale = torch.zeros(batch, kv_heads, self.capacity, 1, dtype=self.dtype, device=self.device)
        self.v_scale = torch.zeros(batch, kv_heads, self.capacity, 1, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        n_new = key_states.shape[-2]
        if self.stored + n_new > self.capacity:
            raise RuntimeError(
                f"KV cache full (stored={self.stored}, incoming={n_new}, capacity={self.capacity}). "
                f"Increase cache capacity / max_seq_len."
            )

        start = self.stored
        if n_new > 1:
            out_k = torch.cat([dequantize_int8(self.keys[:, :, :start], self.k_scale[:, :, :start], self.dtype), key_states], dim=-2)
            out_v = torch.cat([dequantize_int8(self.values[:, :, :start], self.v_scale[:, :, :start], self.dtype), value_states], dim=-2)

        end = start + n_new
        self.keys[:, :, start:end], self.k_scale[:, :, start:end] = quantize_int8(key_states)
        self.values[:, :, start:end], self.v_scale[:, :, start:end] = quantize_int8(value_states)
        self.stored += n_new
        self.seen += n_new

        if n_new > 1:
            return out_k, out_v

        return (
            dequantize_int8(self.keys[:, :, : self.stored], self.k_scale[:, :, : self.stored], self.dtype),
            dequantize_int8(self.values[:, :, : self.stored], self.v_scale[:, :, : self.stored], self.dtype),
        )

    def get_seq_length(self):
        return self.seen

    def get_max_length(self):
        return -1

    def get_mask_sizes(self, query_length):
        kv_length = self.stored + query_length
        return kv_length, self.seen + query_length - kv_length

    def reset(self):
        self.keys = None
        self.values = None
        self.k_scale = None
        self.v_scale = None
        self.is_initialized = False
        self.seen = 0
        self.stored = 0


class KVCostCache(Cache):
    def __init__(self, spec, num_layers, capacity):
        self.offloaded = offloaded_layer_ids(num_layers, spec.offload)
        self.pool = StagingPool(capacity, spec.kind) if self.offloaded else None

        layers = []
        slot = 0
        for i in range(num_layers):
            if i in self.offloaded:
                layers.append(OffloadLayer(spec.kind, capacity, self.pool, slot=slot))
                slot += 1
            elif spec.kind == "full":
                layers.append(FullLayer(capacity))
            else:
                layers.append(Int8Layer(capacity))

        super().__init__(layers=layers)
        self.spec = spec
        self.decoding = False
        self._dma_stream = None

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.offloaded and self.spec.prefetch and self.decoding:
            self._schedule_next_prefetch(layer_idx, key_states.device)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def _schedule_next_prefetch(self, layer_idx, device):
        target = self.offloaded[0]
        for idx in self.offloaded:
            if idx > layer_idx:
                target = idx
                break

        # Guard against ping-pong buffer collision
        if layer_idx in self.offloaded:
            curr_slot = self.offloaded.index(layer_idx)
            target_slot = self.offloaded.index(target)
            if target_slot % PING_PONG_SLOTS == curr_slot % PING_PONG_SLOTS:
                return

        if device.type == "cuda" and self._dma_stream is None:
            self._dma_stream = torch.cuda.Stream(device)
        self.layers[target].issue_prefetch(self._dma_stream)

    def start_decode(self):
        self.decoding = True

    def n_stored(self):
        return self.layers[0].stored


def build_cache(model_config, spec, capacity):
    return KVCostCache(spec, model_config.num_hidden_layers, capacity)
import gc
import hashlib
import importlib.util
import json
import os
import random
import statistics
import time
from itertools import product
from pathlib import Path
from functools import partial

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from .cache import KVSpec, build_cache, offloaded_layer_ids


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device, method="auto"):
    if method == "auto":
        method = "events" if device.type == "cuda" else "wall"

    if method == "events":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        return out, start.elapsed_time(end)

    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, elapsed_ms


def empty_cache(device):
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()




def load_model(name, dtype, device, attn="sdpa", seed=0, **extra):
    if os.environ.get("KVCOST_EXPERTS"):
        extra.setdefault("experts_implementation", os.environ["KVCOST_EXPERTS"])

   
    device_map = None
    if device.type == "cuda" and importlib.util.find_spec("accelerate") is not None:
        device_map = str(device)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        attn_implementation=attn,
        device_map=device_map,
        **extra,
    )

    return model.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def prefill(model, input_ids, cache, chunk=None):
    seq_len = input_ids.shape[1]
    offset = cache.get_seq_length()
    step = chunk or seq_len
    out = None

    for start in range(0, seq_len, step):
        end = min(start + step, seq_len)
        pos = torch.arange(offset + start, offset + end, device=input_ids.device).unsqueeze(0)
        out = model(
            input_ids=input_ids[:, start:end],
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    cache.start_decode()
    return out.logits[:, -1]


@torch.inference_mode()
def decode_step(model, token, cache):
    pos = torch.full((token.shape[0], 1), cache.get_seq_length(), device=token.device, dtype=torch.long)
    out = model(
        input_ids=token,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return out.logits[:, -1]


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand_trials(cfg):
    trials = []
    models = [os.path.expandvars(m) for m in cfg["models"]]
    grids = cfg.get("grids") or [cfg["grid"]]
    repeats = cfg.get("repeats", 3)

    for grid in grids:
        chunks = grid.get("prefill_chunk", cfg.get("prefill_chunk", 4096))
        if not isinstance(chunks, list):
            chunks = [chunks]

        combos = product(models, grid["n_ctx"], grid["batch"], grid["kv"], chunks, range(repeats))
        for model, n_ctx, batch, kv, chunk, rep in combos:
            trial = {
                "model": model,
                "n_ctx": int(n_ctx),
                "batch": int(batch),
                "kv": KVSpec(**kv).to_dict(),
                "repeat": rep,
                "dtype": cfg.get("dtype", "bfloat16"),
                "attn": cfg.get("attn", "sdpa"),
                "warmup_steps": cfg.get("warmup_steps", 10),
                "measure_steps": cfg.get("measure_steps", 50),
                "prefill_chunk": int(chunk),
                "timer": cfg.get("timer", "auto"),
            }
            raw_id = json.dumps(trial, sort_keys=True).encode()
            trial["trial_id"] = hashlib.sha1(raw_id).hexdigest()[:16]
            trials.append(trial)

    return trials


def _done_ids(out):
    ids = set()
    if not out.exists():
        return ids

    with open(out) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def _is_oom(err):
    if isinstance(err, torch.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "invalid buffer size" in msg


def _kv_bytes_per_token(cfg, kind, elem):
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads

    if kind == "full":
        per_head = head_dim * elem
    else:
        #1 byte per element + one scale in model dtype per head
        per_head = head_dim + elem

    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * per_head


def gpu_bytes_needed(cfg, spec, n_ctx, batch, capacity, chunk, elem):
    num_layers = cfg.num_hidden_layers
    kv = batch * capacity * _kv_bytes_per_token(cfg, spec.kind, elem)
    n_off = len(offloaded_layer_ids(num_layers, spec.offload))
    if n_off:
        kv = kv * (num_layers - n_off + 2) // num_layers

    one_layer_full = batch * n_ctx * _kv_bytes_per_token(cfg, "full", elem) // num_layers
    act = batch * min(chunk, n_ctx) * (3 * cfg.intermediate_size + 6 * cfg.hidden_size) * elem
    return int(kv + 2 * one_layer_full + act + 2**30)


def host_bytes_needed(cfg, spec, batch, capacity, elem):
    n_off = len(offloaded_layer_ids(cfg.num_hidden_layers, spec.offload))
    total_kv = batch * capacity * _kv_bytes_per_token(cfg, spec.kind, elem)
    return total_kv * n_off // cfg.num_hidden_layers


def host_available_bytes():
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass

    cgroup_paths = (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    )
    for limit_f, usage_f in cgroup_paths:
        try:
            with open(limit_f) as lf, open(usage_f) as uf:
                limit_raw = lf.read().strip()
                usage = int(uf.read().strip())
        except (OSError, ValueError):
            continue

        if limit_raw != "max" and int(limit_raw) < (1 << 60):
            free = int(limit_raw) - usage
            avail = free if avail is None else min(avail, free)
            break

    return avail


@torch.inference_mode()
def run_trial(model, t, device, seed):
    spec = KVSpec(**t["kv"])
    n_ctx = t["n_ctx"]
    batch = t["batch"]
    warm = t["warmup_steps"]
    steps = t["measure_steps"]

    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (batch, n_ctx), generator=gen).to(device)

    empty_cache(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cache = build_cache(model.config, spec, capacity=n_ctx + warm + steps + 1)
    logits, prefill_ms = timed(partial(prefill, model, ids, cache, chunk=t["prefill_chunk"]), device, "wall")
    tok = logits.argmax(-1, keepdim=True)

    step_ms = []
    for i in range(warm + steps):
        logits, ms = timed(partial(decode_step, model, tok, cache), device, t["timer"])
        tok = logits.argmax(-1, keepdim=True)
        if i >= warm:
            step_ms.append(ms)

    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return {
        "status": "ok",
        "prefill_ms": prefill_ms,
        "step_ms": step_ms,
        "step_median_ms": statistics.median(step_ms),
        "offloaded_layers": len(cache.offloaded),
        "mem_peak_bytes": peak_mem,
    }


def run(cfg, limit=None):
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda"))
    seed = cfg.get("seed", 0)

    done = _done_ids(out)
    trials = [t for t in expand_trials(cfg) if t["trial_id"] not in done]
    random.Random(seed).shuffle(trials)
    if limit:
        trials = trials[:limit]

    print(f"{len(trials)} trials to run on {device} -> {out}")
    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }

    by_model = {}
    for t in trials:
        by_model.setdefault(t["model"], []).append(t)

    for model_name, model_trials in by_model.items():
        dtype = DTYPES[model_trials[0]["dtype"]]
        model = load_model(model_name, dtype, device, model_trials[0]["attn"], seed=seed)
        mc = model.config
        elem = next(model.parameters()).element_size()
        empty_cache(device)

        budget = None
        if device.type == "cuda":
            free_mem = torch.cuda.mem_get_info(device)[0]
            budget = int(free_mem * cfg.get("memory_fraction", 0.85))

        for k, t in enumerate(model_trials, 1):
            row = dict(t)
            row["env"] = env
            row["time"] = time.time()

            spec = KVSpec(**t["kv"])
            capacity = t["n_ctx"] + t["warmup_steps"] + t["measure_steps"] + 1
            need = gpu_bytes_needed(mc, spec, t["n_ctx"], t["batch"], capacity, t["prefill_chunk"], elem)
            host_need = host_bytes_needed(mc, spec, t["batch"], capacity, elem)
            host_avail = host_available_bytes() if host_need else None

            if budget is not None and need > budget:
                row["status"] = "skipped_memory"
                row["mem_estimate_bytes"] = need
                row["mem_budget_bytes"] = budget
            elif host_avail is not None and 2 * host_need > 0.8 * host_avail:
                row["status"] = "skipped_host_memory"
                row["host_need_bytes"] = host_need
                row["host_available_bytes"] = host_avail
            else:
                try:
                    row.update(run_trial(model, t, device, seed + t["repeat"]))
                except Exception as e:
                    if not _is_oom(e):
                        raise
                    row["status"] = "oom"
                    row["error"] = str(e)[:300]
                    empty_cache(device)

                if cfg.get("cooldown_s"):
                    time.sleep(cfg["cooldown_s"])

            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")

            med = row.get("step_median_ms")
            status_str = f"{med:.2f} ms/step" if med is not None else row["status"]
            print(f"[{k}/{len(model_trials)}] n={t['n_ctx']} B={t['batch']} kv={t['kv']} rep={t['repeat']}: {status_str}", flush=True)

        del model
        empty_cache(device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(load_config(args.config), limit=args.limit)
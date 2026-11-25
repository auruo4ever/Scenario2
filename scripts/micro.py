import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvcost.bench import DTYPES, TINY, decode_step, empty_cache, load_model, prefill, timed, tiny_config  # noqa: E402
from kvcost.cache import KVSpec, OffloadLayer, build_cache, offloaded_layer_ids  # noqa: E402

KINDS = ("full", "int8")


def shrunk_model(name, dtype, device, attn):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = tiny_config() if name == TINY else AutoConfig.from_pretrained(name)
    cfg.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size = 256, 512, 1024
    if getattr(cfg, "moe_intermediate_size", None):
        cfg.moe_intermediate_size = 64
    cfg.tie_word_embeddings = False
    cfg._attn_implementation = attn
    if os.environ.get("KVCOST_EXPERTS"):
        cfg._experts_implementation = os.environ["KVCOST_EXPERTS"]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def step_ms(model, spec, n, warm, steps, device):
    ids = torch.randint(0, model.config.vocab_size, (1, n), device=device)
    cache = build_cache(model.config, spec, capacity=n + warm + steps + 1)
    tok = prefill(model, ids, cache).argmax(-1, keepdim=True)
    ms = []
    for i in range(warm + steps):
        logits, t = timed(lambda c=cache: decode_step(model, tok, c), device, "auto")
        tok = logits.argmax(-1, keepdim=True)
        if i >= warm:
            ms.append(t)
    del cache
    empty_cache(device)
    return statistics.median(ms)


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float((x @ y) / (x @ x)) if len(x) and x @ x > 0 else float("nan")


def cpu_main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--offloads", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--real", action="store_true")
    p.add_argument("--json", required=True)
    a = p.parse_args()
    dev, dtype = torch.device(a.device), DTYPES[a.dtype]
    model = shrunk_model(a.model, dtype, dev, a.attn)
    L = model.config.num_hidden_layers
    sp = []
    for kind in KINDS:
        sp.append(KVSpec(kind))
        sp += [KVSpec(kind, offload=f, prefetch=pf) for f in a.offloads for pf in (True, False)]
    times = {s: [] for s in sp}
    for _ in range(a.repeats):
        for s in sp:
            times[s].append(step_ms(model, s, a.n, a.warmup, a.steps, dev))
    rows = [dict(kind=s.kind, offload=s.offload, prefetch=s.prefetch, L_off=len(offloaded_layer_ids(L, s.offload)),
                 step_ms=statistics.median(times[s]), step_ms_all=times[s]) for s in sp]
    base = {r["kind"]: r["step_ms"] for r in rows if r["offload"] == 0}

    def k(prefetch):
        sel = [r for r in rows if r["offload"] > 0 and r["prefetch"] == prefetch]
        return slope([r["L_off"] for r in sel], [r["step_ms"] - base[r["kind"]] for r in sel])

    derived = {
        "c_cpu_ms_per_layer": base["full"] / L,
        "c_int8_cpu_ms_per_layer": (base["int8"] - base["full"]) / L,
        "k_off_prefetch_ms_per_layer": k(True),
        "k_off_sync_ms_per_layer": k(False),
    }
    for r in rows:
        print(f"{r['kind']:5s} offload {r['offload']:4.2f} {'prefetch' if r['prefetch'] else 'sync    '} "
              f"{r['step_ms']:8.2f} ms/step")
    print(derived)
    real = None
    if a.real and a.model != TINY:
        del model
        empty_cache(dev)
        rm = load_model(a.model, dtype, dev, a.attn)
        real = {kind: step_ms(rm, KVSpec(kind), a.n, a.warmup, a.steps, dev) for kind in KINDS}
        print(f"real model at n={a.n}: {real}")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"model": a.model, "n_layers": L, "n_ctx": a.n, "derived": derived,
                                        "real_short_context_ms": real, "rows": rows}, indent=2))


def config(name, n_layers, experts):
    from transformers import AutoConfig, Qwen3MoeConfig

    if name == "tiny-qwen-moe":
        c = Qwen3MoeConfig(vocab_size=1000, hidden_size=256, intermediate_size=512, moe_intermediate_size=64,
                           num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
                           num_experts=16, num_experts_per_tok=4)
    else:
        c = AutoConfig.from_pretrained(name)
    c.num_hidden_layers = n_layers
    if getattr(c, "layer_types", None):
        c.layer_types = c.layer_types[:n_layers]
    if getattr(c, "mlp_only_layers", None):
        c.mlp_only_layers = [i for i in c.mlp_only_layers if i < n_layers]
    c._attn_implementation = "sdpa"
    if experts:
        c._experts_implementation = experts
    return c


def build(name, n_layers, experts, dtype, dev):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config(name, n_layers, experts)).to(device=dev, dtype=dtype).eval()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def free(dev):
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def filled(model, spec, B, kv, extra, chunk, dev):
    cache = build_cache(model.config, spec, capacity=kv + extra + 1)
    if kv:
        prefill(model, torch.randint(0, model.config.vocab_size, (B, kv), device=dev), cache, chunk=chunk)
    return cache


@torch.inference_mode()
def decode_times(model, spec, B, kv, dev, steps, chunk):
    cache = filled(model, spec, B, kv, 4 + steps * 2, chunk, dev)
    tok = torch.randint(0, model.config.vocab_size, (B, 1), device=dev)
    for _ in range(3):
        tok = decode_step(model, tok, cache).argmax(-1, keepdim=True)
    wall = []
    for _ in range(steps):
        sync(dev)
        t = time.perf_counter()
        tok = decode_step(model, tok, cache).argmax(-1, keepdim=True)
        sync(dev)
        wall.append((time.perf_counter() - t) * 1e3)
    gpu = copy = float("nan")
    if dev.type == "cuda":
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                tok = decode_step(model, tok, cache).argmax(-1, keepdim=True)
            sync(dev)
        k = c = 0.0
        for e in prof.key_averages():
            t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)
            if e.key.lower().startswith(("memcpy", "memset")):
                c += t
            else:
                k += t
        gpu, copy = k / 1e3 / steps, c / 1e3 / steps
    del cache
    free(dev)
    return statistics.median(wall), gpu, copy


@torch.inference_mode()
def chunk_time(model, spec, B, q, kv_prev, dev, reps, chunk_prev):
    ts = []
    for _ in range(reps):
        cache = filled(model, spec, B, kv_prev, q, chunk_prev, dev)
        ids = torch.randint(0, model.config.vocab_size, (B, q), device=dev)
        sync(dev)
        t = time.perf_counter()
        prefill(model, ids, cache, chunk=q)
        sync(dev)
        ts.append((time.perf_counter() - t) * 1e3)
        del cache
        free(dev)
    return statistics.median(ts)


def h2d_rate(gib, pieces=48, reps=3):
    n = int(gib * 2**30) // pieces
    host = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for _ in range(pieces)]
    for h in host:
        h.fill_(1)
    dev = [torch.empty(n, dtype=torch.uint8, device="cuda") for _ in range(pieces)]
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for d, h in zip(dev, host):
            d.copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return n * pieces / best / 1e9


def layers_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", default=os.environ.get("KVCOST_EXPERTS", "grouped_mm"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    q = a.quick
    Bs = [1, 2] if q else [1, 2, 4]
    dec_kv = [64, 256] if q else [1024, 8192, 32768, 131072]
    pf_q = [32] if q else [2048, 8192]
    pf_kv = [0, 64] if q else [0, 8192, 32768, 98304]
    places = [("gpu", KVSpec), ("cpu_prefetch", lambda k: KVSpec(k, offload=1.0, prefetch=True)),
              ("cpu_sync", lambda k: KVSpec(k, offload=1.0, prefetch=False))]
    if dev.type != "cuda":
        places = places[:1]
    steps = 4 if q else 12
    pchunk = 32 if q else 8192
    path = Path(a.json)
    out = {"model": a.model, "experts": a.experts, "device": str(dev), "decode": [], "prefill": [], "h2d": []}
    if path.exists():
        out = json.loads(path.read_text())
    if not out.get("h2d") and dev.type == "cuda":
        out["h2d"] = [{"gib": g, "gbs": h2d_rate(g)} for g in (0.0625, 0.25, 1.0, 4.0, 12.0)]
    done_dec = {(r["kind"], r["place"], r["B"], r["kv"]) for r in out["decode"]}
    done_pf = {(r["kind"], r["place"], r["B"], r["q"], r["kv_prev"]) for r in out["prefill"]}

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))

    models = {n: build(a.model, n, a.experts, dt, dev) for n in (1, 2)}
    for kind in ("full", "int8"):
        for pname, mk in places:
            for B in Bs:
                for kv in dec_kv:
                    if (pname != "gpu" and kv < (0 if q else 8192)) or (kind, pname, B, kv) in done_dec:
                        continue
                    r = {n: decode_times(models[n], mk(kind), B, kv, dev, steps, pchunk) for n in (1, 2)}
                    out["decode"].append(dict(kind=kind, place=pname, B=B, kv=kv,
                                              wall_layer=r[2][0] - r[1][0], wall_once=2 * r[1][0] - r[2][0],
                                              gpu_layer=r[2][1] - r[1][1], gpu_once=2 * r[1][1] - r[2][1],
                                              copy_layer=r[2][2] - r[1][2], wall_1=r[1][0], wall_2=r[2][0]))
                    save()
                    print(f"decode {kind} {pname} B={B} kv={kv}: {out['decode'][-1]['wall_layer']:.2f} ms/layer",
                          flush=True)
        for pname, mk in places:
            for B in Bs:
                for qq in pf_q + ([32768] if (B == 1 and not q and pname == "gpu") else []):
                    for kvp in pf_kv:
                        if (pname != "gpu" and not q and qq != 8192) or (kind, pname, B, qq, kvp) in done_pf:
                            continue
                        r = {n: chunk_time(models[n], mk(kind), B, qq, kvp, dev, 1 if q else 2, pchunk)
                             for n in (1, 2)}
                        out["prefill"].append(dict(kind=kind, place=pname, B=B, q=qq, kv_prev=kvp,
                                                   layer=r[2] - r[1], once=2 * r[1] - r[2], t1=r[1], t2=r[2]))
                        save()
                        print(f"prefill {kind} {pname} B={B} q={qq} kv_prev={kvp}: "
                              f"{out['prefill'][-1]['layer']:.2f} ms/layer", flush=True)
    save()


def one(kind, B, cap, n_layers, heads, head_dim, dev):
    ref = torch.empty(B, heads, 1, head_dim, dtype=torch.bfloat16, device=dev)
    layers, ts = [], []
    for _ in range(n_layers):
        lay = OffloadLayer(kind, cap, pool=None, slot=0)
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        lay.lazy_initialization(ref, ref)
        ts.append((time.perf_counter() - t) * 1e3)
        layers.append(lay)
    nbytes = sum(t.numel() * t.element_size() for t in layers[0].host.values())
    del layers
    empty_cache(dev)
    return nbytes, ts


def alloc_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--budget-gib", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--caps", default="8448,33024,65792,131328")
    ap.add_argument("--batches", default="1,2,4")
    a = ap.parse_args()
    dev = torch.device("cuda")
    caps = [int(x) for x in a.caps.split(",")]
    Bs = [int(x) for x in a.batches.split(",")]
    rows = {}
    for _ in range(a.reps):
        for kind in ("full", "int8"):
            for B in Bs:
                for cap in caps:
                    per = cap * B * a.heads * a.head_dim * (4 if kind == "full" else 2)
                    n = max(1, min(24, int(a.budget_gib * 2**30 // per)))
                    nbytes, ts = one(kind, B, cap, n, a.heads, a.head_dim, dev)
                    r = rows.setdefault((kind, B, cap), dict(kind=kind, B=B, capacity=cap, bytes_per_layer=nbytes,
                                                             ms=[]))
                    r["ms"] += ts
                    print(f"{kind} B={B} cap={cap}: {statistics.median(ts):.1f} ms/layer", flush=True)
    for r in rows.values():
        r["ms_per_layer"] = statistics.median(r["ms"])
        r["gbs"] = r["bytes_per_layer"] / r["ms_per_layer"] / 1e6
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"rows": list(rows.values())}, indent=1))

if __name__ == "__main__":
    cmd = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    mains = {"cpu": cpu_main, "layers": layers_main, "alloc": alloc_main}
    if cmd not in mains:
        raise SystemExit("usage: micro.py {cpu,layers,alloc} [options]")
    mains[cmd]()

import math

import numpy as np


def interp(x, xs, ys, log=False):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    if len(xs) == 1:
        return float(ys[0])

    if log:
        xs = np.log(xs)
        ys = np.log(np.maximum(ys, 1e-9))
        x = math.log(max(x, 1e-9))

    i = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
    y = ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    if log:
        return float(np.exp(y))
    return float(y)


def n_offloaded(L, frac):
    # unique CPU layers
    k = int(round(frac * L))
    if k <= 0:
        return 0

    layers = set()
    for j in range(k):
        idx = int((j + 1) * L / k) - 1
        layers.add(min(L - 1, idx))
    return len(layers)


class CostModel:
    def __init__(self, layers, cpu, n_layers, kv_bytes_token, alloc=None, alloc_by_batch=True):
        self.L = n_layers
        self.bytes_per_layer = {}
        for kind, total_bytes in kv_bytes_token.items():
            self.bytes_per_layer[kind] = total_bytes / n_layers

        self.decode_rows = layers["decode"]
        self.prefill_rows = layers["prefill"]
        self.alloc_rows = (alloc or {}).get("rows", [])
        self.alloc_by_batch = alloc_by_batch
        self.missing = set()

        h2d_curve = []
        for r in layers.get("h2d", []):
            h2d_curve.append((r["gib"] * (2**30), r["gbs"]))
        h2d_curve.sort()
        self.h2d = h2d_curve

        shrunk = {}
        for r in cpu["rows"]:
            if r["offload"] == 0:
                shrunk[r["kind"]] = r["step_ms"]

        real = cpu.get("real_short_context_ms") or {}
        if real.get("full"):
            self.anchor = real["full"] / shrunk["full"]
        else:
            self.anchor = 1.0

        d = cpu["derived"]
        self.cpu_layer = d["c_cpu_ms_per_layer"]
        self.cpu_int8 = d["c_int8_cpu_ms_per_layer"]
        self.k_prefetch = d["k_off_prefetch_ms_per_layer"]

    def decode_step(self, kind, B, n, offload=0.0, prefetch=True):
        L = self.L
        L_off = n_offloaded(L, offload)

        per_layer_cpu = self.cpu_layer + (self.cpu_int8 if kind == "int8" else 0.0)
        cpu = per_layer_cpu * L * self.anchor

        ref_vals = sorted(
            r["wall_layer"]
            for r in self.decode_rows
            if r["kind"] == kind and r["place"] == "gpu" and r["B"] == 1
        )
        ref = ref_vals[len(ref_vals) // 2]
        gpu = L * max(0.0, self._decode_table(kind, B, n, "wall_layer") - ref)
        base = cpu + gpu

        copy = 0.0
        extra = 0.0
        if L_off:
            copy = self._decode_table(kind, B, n, "copy_layer", ("cpu_prefetch", "cpu_sync"))
            if math.isnan(copy):
                nbytes = B * n * self.bytes_per_layer[kind]
                if self.h2d:
                    xs = [p[0] for p in self.h2d]
                    ys = [p[1] for p in self.h2d]
                    rate = interp(min(max(nbytes, xs[0]), xs[-1]), xs, ys)
                else:
                    rate = 25.0
                copy = nbytes / (rate * 1e6)

            if prefetch:
                overlap = (L / L_off) * base / L
                extra = self.k_prefetch + max(0.0, copy - overlap)
            else:
                extra = self._sync_extra(kind, B, n)

        return {
            "step": base + L_off * extra,
            "cpu": cpu,
            "gpu": gpu,
            "copy": L_off * copy,
            "offload_extra": L_off * extra,
        }

    def prefill(self, kind, B, n, chunk, offload=0.0, prefetch=True, capacity=None):
        L = self.L
        L_off = n_offloaded(L, offload)
        place = "cpu_prefetch" if prefetch else "cpu_sync"

        total = 0.0
        kv = 0
        while kv < n:
            q = min(chunk, n - kv)
            layer = self._prefill_table(kind, "gpu", B, q, kv, "layer")
            t = self._prefill_table(kind, "gpu", B, q, kv, "once") + (L - L_off) * layer
            if L_off:
                pf_off = self._prefill_table(kind, place, B, 8192, kv, "layer")
                pf_gpu = self._prefill_table(kind, "gpu", B, 8192, kv, "layer")
                extra = pf_off - pf_gpu
                t += L_off * (layer + extra * (q + kv) / (8192 + kv))
            total += t
            kv += q

        if L_off:
            cap = capacity if capacity is not None else n
            nbytes = cap * B * self.bytes_per_layer[kind]
            total += L_off * self.alloc_ms(kind, nbytes, B)
        return total

    def request(self, kind, B, n, chunk, n_out=256, offload=0.0, prefetch=True):
        ttft = self.prefill(kind, B, n, chunk, offload, prefetch, capacity=n + n_out)
        d = self.decode_step(kind, B, n + n_out // 2, offload, prefetch)
        tpot = d["step"]
        total_sec = (ttft + (n_out - 1) * tpot) / 1e3

        return {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "throughput_tok_s": B * n_out / total_sec,
            "decode_cpu_ms": d["cpu"],
            "decode_gpu_growth_ms": d["gpu"],
            "decode_copy_ms": d["copy"],
            "decode_offload_extra_ms": d["offload_extra"],
        }

    def alloc_ms(self, kind, nbytes, B):
        rows = [r for r in self.alloc_rows if r["kind"] == kind]
        if not rows:
            rows = [r for r in self.alloc_rows if r["kind"] != kind]
            if not rows:
                return 0.0
            self.missing.add(("alloc", kind))

        def interp_alloc(target_bytes, pts):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if target_bytes <= xs[0]:
                return ys[0] * target_bytes / xs[0]
            if target_bytes >= xs[-1]:
                return ys[-1] * target_bytes / xs[-1]
            return interp(target_bytes, xs, ys)

        if not self.alloc_by_batch:
            pts = [(r["bytes_per_layer"], r["ms_per_layer"]) for r in rows]
            return interp_alloc(nbytes, pts)

        batches = sorted({r["B"] for r in rows})
        per_b = []
        for b in batches:
            pts = [(r["bytes_per_layer"] / b, r["ms_per_layer"]) for r in rows if r["B"] == b]
            per_b.append(interp_alloc(nbytes / B, pts))
        return max(0.0, interp(B, batches, per_b))

    def _decode_table(self, kind, B, n, col, places=("gpu",)):
        rows = [
            r
            for r in self.decode_rows
            if r["kind"] == kind
            and r["place"] in places
            and r.get(col) is not None
            and not math.isnan(r[col])
        ]
        if not rows:
            return float("nan")

        batches = sorted({r["B"] for r in rows})
        per_b = []
        for b in batches:
            by_kv = {}
            for r in rows:
                if r["B"] == b:
                    by_kv.setdefault(r["kv"], []).append(r[col])
            kvs = sorted(by_kv)
            means = [sum(by_kv[k]) / len(by_kv[k]) for k in kvs]
            per_b.append(interp(n, kvs, means))
        return interp(B, batches, per_b)

    def _prefill_table(self, kind, place, B, q, kv, col):
        rows = [r for r in self.prefill_rows if r["kind"] == kind and r["place"] == place]
        if not rows:
            self.missing.add((kind, place))
            rows = [r for r in self.prefill_rows if r["place"] == place]

        batches = sorted({r["B"] for r in rows})
        per_b = []
        for b in batches:
            chunks = sorted({r["q"] for r in rows if r["B"] == b})
            per_q = []
            for qq in chunks:
                pts = sorted((r["kv_prev"], r[col]) for r in rows if r["B"] == b and r["q"] == qq)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                per_q.append(interp(kv, xs, ys))

            if len(chunks) > 1:
                clamped = [max(v, 1e-6) for v in per_q]
                per_b.append(interp(q, chunks, clamped, log=True))
            else:
                per_b.append(per_q[0])
        return interp(B, batches, per_b)

    def _sync_extra(self, kind, B, n):
        rows = [r for r in self.decode_rows if r["kind"] == kind]
        on_gpu = {}
        for r in rows:
            if r["place"] == "gpu":
                on_gpu[(r["B"], r["kv"])] = r["wall_1"]

        pts = {}
        for r in rows:
            key = (r["B"], r["kv"])
            if r["place"] == "cpu_sync" and key in on_gpu:
                diff = max(0.0, r["wall_1"] - on_gpu[key])
                pts.setdefault(r["B"], []).append((r["kv"], diff))

        batches = sorted(pts)
        per_b = []
        for b in batches:
            b_pts = [(0, 0.0)] + sorted(pts[b])
            xs = [p[0] for p in b_pts]
            ys = [p[1] for p in b_pts]
            per_b.append(interp(n, xs, ys))
        return max(0.0, interp(B, batches, per_b))
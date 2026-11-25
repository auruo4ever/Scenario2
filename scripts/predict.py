import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvcost.bench import expand_trials, load_config  # noqa: E402
from kvcost.costmodel import CostModel  # noqa: E402


def config_key(trial):
    kv = trial["kv"]
    return (
        trial["n_ctx"],
        trial["batch"],
        kv["kind"],
        round(kv.get("offload", 0.0), 4),
        bool(kv.get("prefetch", True)),
        trial["prefill_chunk"],
    )


def predict(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")

    cfg = load_config(args.config)
    with open(args.cpu) as f:
        cpu = json.load(f)
    with open(args.layers) as f:
        layers = json.load(f)
    with open(args.alloc) as f:
        alloc = json.load(f)

    kv_bytes_per_tok = {"full": args.kv_full, "int8": args.kv_int8}
    model = CostModel(
        layers,
        cpu,
        cpu["n_layers"],
        kv_bytes_per_tok,
        alloc,
        alloc_by_batch=(args.alloc_interp == "batch"),
    )

    n_out = cfg.get("warmup_steps", 5) + cfg.get("measure_steps", 250) + 1

    seen = set()
    rows = []
    for trial in expand_trials(cfg):
        k = config_key(trial)
        if k in seen:
            continue
        seen.add(k)

        n_ctx, batch, kind, offload, prefetch, chunk = k
        est = model.request(kind, batch, n_ctx, chunk, n_out, offload, prefetch)
        row = {
            "n_ctx": n_ctx,
            "batch": batch,
            "kind": kind,
            "offload": offload,
            "prefetch": prefetch,
            "chunk": chunk,
        }
        row.update(est)
        rows.append(row)

    if model.missing:
        print(f"WARNING: tables missing for {sorted(model.missing)}, the other KV kind's rows were used")

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"# {stamp} {args.config}; {args.layers}, {args.cpu}, {args.alloc} "
            f"(allocation interpolated by {args.alloc_interp}); decode {n_out} tokens\n"
        )
        df.to_csv(f, index=False)

    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))
    print(f"\nwrote {out} ({len(df)} configurations)")


def check(args):
    pred_df = pd.read_csv(args.predictions, comment="#")

    all_runs = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                all_runs.append(json.loads(line))

    # Group successful benchmarks by configuration and compute per-run metrics
    measured = defaultdict(list)
    for r in all_runs:
        if r.get("status") != "ok":
            continue
        n_out = r["warmup_steps"] + r["measure_steps"] + 1
        tpot = statistics.median(r["step_ms"])
        total_sec = (r["prefill_ms"] + (n_out - 1) * tpot) / 1e3
        measured[config_key(r)].append({
            "ttft": r["prefill_ms"],
            "tpot": tpot,
            "thr": r["batch"] * n_out / total_sec,
        })

    rows = []
    for p in pred_df.itertuples():
        k = (p.n_ctx, p.batch, p.kind, round(p.offload, 4), bool(p.prefetch), p.chunk)
        reps = measured.get(k)
        if not reps:
            continue

        rows.append({
            "n_ctx": p.n_ctx,
            "batch": p.batch,
            "kind": p.kind,
            "offload": p.offload,
            "mode": "prefetch" if p.prefetch else "sync",
            "chunk": p.chunk,
            "reps": len(reps),
            "ttft_meas": statistics.median(x["ttft"] for x in reps),
            "ttft_pred": p.ttft_ms,
            "tpot_meas": statistics.median(x["tpot"] for x in reps),
            "tpot_pred": p.tpot_ms,
            "thr_meas": statistics.median(x["thr"] for x in reps),
            "thr_pred": p.throughput_tok_s,
        })

    df = pd.DataFrame(rows)
    for metric in ("ttft", "tpot", "thr"):
        df[f"{metric}_err"] = (df[f"{metric}_pred"] / df[f"{metric}_meas"] - 1.0) * 100.0

    pd.set_option("display.width", 220)
    print(df.round(1).to_string(index=False))
    print(f"\n{len(df)} of {len(pred_df)} predicted configurations measured; mean / max |error| (%):")
    for metric, label in (("ttft", "TTFT"), ("tpot", "TPOT"), ("thr", "throughput")):
        err_col = df[f"{metric}_err"].abs()
        print(f"  {label:10s} {err_col.mean():5.1f} / {err_col.max():5.1f}")

    for r in all_runs:
        if r.get("status") != "ok":
            print(f"not measured ({r.get('status')}): {r['n_ctx']} B={r['batch']} {r['kv']}")

    # ??
    main_chunk = df["chunk"].mode().iloc[0]
    sub = df[df["chunk"] == main_chunk]
    group_cols = ["n_ctx", "batch", "kind"]
    best_meas = sub.loc[sub.groupby(group_cols)["tpot_meas"].idxmin()].set_index(group_cols)
    best_pred = sub.loc[sub.groupby(group_cols)["tpot_pred"].idxmin()].set_index(group_cols)
    matched = ((best_meas["offload"] == best_pred["offload"]) & (best_meas["mode"] == best_pred["mode"])).sum()
    print(f"fastest decode plan per (context, batch, KV) picked right: {matched} of {len(best_meas)}")


def summary(args):
    steps = defaultdict(list)
    for path in args.results:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                kv = r["kv"]
                if r["status"] != "ok":
                    off = kv.get("offload", 0.0)
                    print(f"infeasible: {r['n_ctx']} B={r['batch']} {kv['kind']} offload {off} ({r['status']}, {Path(path).name})")
                    continue
                if r["prefill_chunk"] == 8192:
                    k = (r["n_ctx"], r["batch"], kv["kind"], kv.get("offload", 0.0), kv.get("prefetch", True))
                    steps[k].append(statistics.median(r["step_ms"]))

    tpot = {}
    for k, vals in steps.items():
        tpot[k] = statistics.median(vals)

    print(
        f"\n{'n_ctx':>6s} {'B':>2s}  half BF16 offloaded (prefetch / sync)  INT8 vs BF16 on GPU  "
        f"BF16 / INT8 all offloaded"
    )

    pairs = sorted({(k[0], k[1]) for k in tpot})
    for n_ctx, batch in pairs:
        gpu_bf16 = tpot.get((n_ctx, batch, "full", 0.0, True))
        half_pf = tpot.get((n_ctx, batch, "full", 0.5, True))
        half_sync = tpot.get((n_ctx, batch, "full", 0.5, False))
        gpu_int8 = tpot.get((n_ctx, batch, "int8", 0.0, True))
        all_bf16 = tpot.get((n_ctx, batch, "full", 1.0, True))
        all_int8 = tpot.get((n_ctx, batch, "int8", 1.0, True))

        if gpu_bf16 and half_pf and half_sync:
            pf_delta = 100.0 * (half_pf / gpu_bf16 - 1.0)
            sync_delta = 100.0 * (half_sync / gpu_bf16 - 1.0)
            half_col = f"{pf_delta:+6.1f}% / {sync_delta:+6.1f}%"
        else:
            half_col = "-"

        if gpu_bf16 and gpu_int8:
            quant_col = f"{100.0 * (gpu_int8 / gpu_bf16 - 1.0):+6.1f}%"
        else:
            quant_col = "-"

        if all_bf16 and all_int8:
            offload_col = f"{all_bf16 / all_int8:5.2f}x"
        else:
            offload_col = "-"

        print(f"{n_ctx:6d} {batch:2d}  {half_col:>36s}  {quant_col:>19s}  {offload_col:>25s}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("config")
    p_pred.add_argument("--layers", required=True)
    p_pred.add_argument("--cpu", required=True)
    p_pred.add_argument("--alloc", required=True)
    p_pred.add_argument("--alloc-interp", choices=("batch", "bytes"), default="batch")
    # Default KV footprint per token for a 24-layer GQA-8 head_dim=128 model (BF16 vs INT8+scales)
    p_pred.add_argument("--kv-full", type=float, default=98304)
    p_pred.add_argument("--kv-int8", type=float, default=49920)
    p_pred.add_argument("--out", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("predictions")
    p_check.add_argument("results")

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("results", nargs="+")

    args = parser.parse_args()
    if args.cmd == "predict":
        predict(args)
    elif args.cmd == "check":
        check(args)
    elif args.cmd == "summary":
        summary(args)


if __name__ == "__main__":
    main()
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant(weights, scales, block_size):
    rows = weights.shape[-2]
    cols = weights.shape[-1]
    block_rows, block_cols = block_size

    scales = scales.repeat_interleave(block_rows, dim=-2)
    scales = scales.repeat_interleave(block_cols, dim=-1)

    scales = scales[..., :rows, :cols]

    result = weights.to(torch.float32) * scales.to(torch.float32)
    return result.to(torch.bfloat16)


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    with open(src / "config.json") as f:
        cfg = json.load(f)

    q = cfg.pop("quantization_config", None)
    if q is None:
        q = {}
    block = q.get("weight_block_size", (128, 128))

    dst.mkdir(parents=True, exist_ok=True)
    skip_files = ("config.json", "model.safetensors.index.json")
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in skip_files:
            shutil.copy(f, dst / f.name)

    cfg["torch_dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    weight_map = {}
    total = 0
    t0 = time.time()

    skip_suffixes = ("_scale_inv", ".input_scale", ".activation_scale")
    for shard in sorted(src.glob("*.safetensors")):
        out = {}
        with safe_open(str(shard), framework="pt", device=args.device) as f:
            keys = set(f.keys())
            for k in sorted(keys):
                if k.endswith(skip_suffixes):
                    continue

                t = f.get_tensor(k)
                sk = k + "_scale_inv"
                if sk in keys:
                    t = dequant(t, f.get_tensor(sk), block)
                elif t.is_floating_point():
                    t = t.to(torch.bfloat16)

                t = t.cpu().contiguous()
                out[k] = t
                total += t.numel() * t.element_size()
                weight_map[k] = shard.name

        save_file(out, str(dst / shard.name), metadata={"format": "pt"})
        elapsed = time.time() - t0
        print(f"{shard.name}: {len(out)} tensors, {total / 1e9:.1f} GB so far, {elapsed:.0f} s", flush=True)
        del out

    index_data = {
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)

    leftover = []
    for k in weight_map:
        if "scale" in k:
            leftover.append(k)
    if leftover:
        sys.exit(f"scale tensors left in the output: {leftover[:5]}")

    elapsed = time.time() - t0
    print(f"wrote {dst}: {total / 1e9:.1f} GB BF16 in {elapsed:.0f} s")


if __name__ == "__main__":
    main()
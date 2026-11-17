"""VRAM budget model for llama.cpp partial offload.

Predicts GPU memory for a GGUF model at a given number of offloaded layers
(`-ngl`) and context size, so we can pick settings that fit *before* loading
with an empirical margin for unmodelled allocations.

    vram = weight_scale * offloaded weights + KV cache + fixed overhead

The empirical defaults describe Qwen3-8B on the measured RTX 4060 only.
`calibrate_overhead` refits the intercept with the weight scale held fixed.

CLI:
    python -m inference.bench.vram MODEL.gguf --ctx 8192 --budget-mib 7000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from .gguf import GGUFHeader, read_header

MIB = 1024 * 1024

# bytes per element of the KV cache
KV_BYTES = {"f32": 4, "f16": 2, "q8_0": 1.0625, "q4_0": 0.5625}

# OLS on research/results/phase2/bench.jsonl, including ngl=0; see vram-model-v2.md.
DEFAULT_OVERHEAD_MIB = 984.4145632964137
DEFAULT_WEIGHT_SCALE = 0.9313736266040277


@dataclass(frozen=True)
class Estimate:
    ngl: int
    n_ctx: int
    weights_mib: float
    kv_mib: float
    overhead_mib: float
    weight_scale: float = DEFAULT_WEIGHT_SCALE

    @property
    def total_mib(self) -> float:
        return self.weight_scale * self.weights_mib + self.kv_mib + self.overhead_mib


def offloaded_weight_bytes(h: GGUFHeader, ngl: int) -> int:
    """llama.cpp offloads the *last* `ngl` blocks; ngl > n_layer adds the output head."""
    n_layer = h.n_layer
    blocks = h.block_bytes()
    k = max(0, min(ngl, n_layer))
    total = sum(blocks[n_layer - k :]) if k else 0
    if ngl > n_layer:
        total += h.non_block_bytes()["output"]
    return total


def kv_cache_bytes(h: GGUFHeader, n_ctx: int, n_layers: int, kv_type: str = "f16") -> float:
    """K and V tensors for `n_layers` layers: 2 * ctx * head_dim * n_head_kv each."""
    if kv_type not in KV_BYTES:
        raise ValueError(f"unknown KV cache type {kv_type!r}")
    per_layer = 2 * n_ctx * h.head_dim * h.n_head_kv * KV_BYTES[kv_type]
    return per_layer * n_layers


def estimate(
    h: GGUFHeader,
    ngl: int,
    n_ctx: int,
    *,
    kv_type: str = "f16",
    overhead_mib: float = DEFAULT_OVERHEAD_MIB,
    weight_scale: float = DEFAULT_WEIGHT_SCALE,
) -> Estimate:
    if ngl < 0 or n_ctx <= 0:
        raise ValueError("ngl must be >= 0 and n_ctx > 0")
    k = min(ngl, h.n_layer)
    return Estimate(
        ngl=ngl,
        n_ctx=n_ctx,
        weights_mib=offloaded_weight_bytes(h, ngl) / MIB,
        kv_mib=kv_cache_bytes(h, n_ctx, k, kv_type) / MIB,
        overhead_mib=overhead_mib,
        weight_scale=weight_scale,
    )


def max_ngl(
    h: GGUFHeader,
    n_ctx: int,
    budget_mib: float,
    *,
    kv_type: str = "f16",
    overhead_mib: float = DEFAULT_OVERHEAD_MIB,
    weight_scale: float = DEFAULT_WEIGHT_SCALE,
) -> Estimate:
    """Largest ngl that fits; raise if even the fixed CPU-run overhead exceeds budget."""
    best = estimate(h, 0, n_ctx, kv_type=kv_type, overhead_mib=overhead_mib, weight_scale=weight_scale)
    if best.total_mib > budget_mib:
        raise ValueError("budget is below the fixed overhead, even at ngl=0")
    for ngl in range(1, h.n_layer + 2):
        e = estimate(h, ngl, n_ctx, kv_type=kv_type, overhead_mib=overhead_mib, weight_scale=weight_scale)
        if e.total_mib > budget_mib:
            break
        best = e
    return best


def calibrate_overhead(samples: list[tuple[Estimate, float]]) -> float:
    """Fit the fixed overhead from (estimate, measured_mib) pairs.

    Returns the mean residual after removing the derived terms, i.e. the
    overhead value that makes the model unbiased on these runs.
    """
    residuals = [measured - (e.weight_scale * e.weights_mib + e.kv_mib) for e, measured in samples]
    if not residuals:
        raise ValueError("need at least one sample to calibrate")
    return sum(residuals) / len(residuals)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Estimate VRAM use of a GGUF model")
    p.add_argument("model")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", type=int, help="estimate this ngl instead of searching")
    p.add_argument("--budget-mib", type=float, default=7000.0)
    p.add_argument("--kv-type", default="f16", choices=sorted(KV_BYTES))
    p.add_argument("--overhead-mib", type=float, default=DEFAULT_OVERHEAD_MIB)
    p.add_argument("--weight-scale", type=float, default=DEFAULT_WEIGHT_SCALE)
    args = p.parse_args(argv)

    h = read_header(args.model)
    if args.ngl is not None:
        e = estimate(
            h,
            args.ngl,
            args.ctx,
            kv_type=args.kv_type,
            overhead_mib=args.overhead_mib,
            weight_scale=args.weight_scale,
        )
    else:
        e = max_ngl(
            h,
            args.ctx,
            args.budget_mib,
            kv_type=args.kv_type,
            overhead_mib=args.overhead_mib,
            weight_scale=args.weight_scale,
        )
    out = asdict(e) | {
        "total_mib": round(e.total_mib, 1),
        "n_layer": h.n_layer,
        "fits_fully": e.ngl > h.n_layer,
        "budget_mib": args.budget_mib,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

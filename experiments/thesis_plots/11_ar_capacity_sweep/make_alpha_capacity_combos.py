from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIGDIR = TMP_ROOT / "x01_mplconfig"
XDG_CACHE_HOME = TMP_ROOT / "x01_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
X01_SRC = ROOT / "x01" / "src"
sys.path.insert(0, str(X01_SRC))

from x01.ar import KoopmanAE2D

DEFAULT_OUTPUT = HERE / "alpha_capacity_combos.npz"
DEFAULT_CSV = HERE / "alpha_capacity_combos.csv"
DEFAULT_CHUNK_CSV = HERE / "alpha_capacity_chunks.csv"

DIMS = (16, 32, 48, 64)
HEADS = (1, 2, 4, 8)
ENCODER_PROFILES = (
    ("E1_baseline", (4, 6, 6, 8)),
    ("E2_deeper", (6, 8, 8, 10)),
    ("E3_deepest", (8, 10, 10, 12)),
)

FIXED_OBJECTIVE = {
    "source": "07_ar_further_attempts latent_step_consistency recipe",
    "dynamics_mode": "single",
    "lambda_recon": 1.0,
    "lambda_fwd": 1.0,
    "lambda_bwd": 1.0,
    "lambda_latent_fwd": 1.0,
    "lambda_ortho": 0.1,
    "lr": 1e-4,
    "batch_size": 32,
    "epochs_alpha": 10,
    "selection_split": "validation",
    "selection_metric": "autocorr_score_validation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chunk-csv", type=Path, default=DEFAULT_CHUNK_CSV)
    parser.add_argument("--num-chunks", type=int, default=5)
    return parser.parse_args()


def unique_param_count(module) -> int:
    seen = set()
    total = 0
    for param in module.parameters():
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        total += param.numel()
    return total


def count_model(dims: int, blocks: tuple[int, int, int, int]) -> dict[str, int | float]:
    model = KoopmanAE2D(
        in_channels=1,
        out_channels=1,
        dims=dims,
        num_blocks=blocks,
        num_heads=HEADS,
        dynamics_mode="single",
        dynamics_rank=None,
    )
    encoder = unique_param_count(model.encoder)
    decoder = unique_param_count(model.decoder)
    koopman = unique_param_count(model.dynamics)
    total = unique_param_count(model)
    encdec = encoder + decoder
    return {
        "total_params": total,
        "encoder_params": encoder,
        "decoder_params": decoder,
        "encoder_decoder_params": encdec,
        "koopman_params": koopman,
        "koopman_share": koopman / total,
        "koopman_to_encdec": koopman / encdec,
    }


def make_rows(num_chunks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for encoder_level, (profile, blocks) in enumerate(ENCODER_PROFILES, start=1):
        for dims in DIMS:
            counts = count_model(dims, blocks)
            rows.append(
                {
                    "combo_index": idx,
                    "combo_id": f"alpha_e{encoder_level}_d{dims:03d}",
                    "encoder_profile": profile,
                    "encoder_level": encoder_level,
                    "dims": dims,
                    "num_blocks": blocks,
                    "num_heads": HEADS,
                    **counts,
                }
            )
            idx += 1

    chunk_loads = [0 for _ in range(num_chunks)]
    for row in sorted(rows, key=lambda item: int(item["total_params"]), reverse=True):
        chunk_id = min(range(num_chunks), key=lambda cid: chunk_loads[cid])
        row["chunk_id"] = chunk_id
        chunk_loads[chunk_id] += int(row["total_params"])
    return sorted(rows, key=lambda item: int(item["combo_index"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "combo_index",
        "chunk_id",
        "combo_id",
        "encoder_profile",
        "encoder_level",
        "dims",
        "num_blocks",
        "num_heads",
        "total_params",
        "encoder_params",
        "decoder_params",
        "encoder_decoder_params",
        "koopman_params",
        "koopman_share",
        "koopman_to_encdec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["num_blocks"] = list(out["num_blocks"])
            out["num_heads"] = list(out["num_heads"])
            writer.writerow(out)


def write_chunk_csv(path: Path, rows: list[dict[str, Any]], num_chunks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_id", "n_combos", "estimated_total_params", "combo_ids"])
        writer.writeheader()
        for chunk_id in range(num_chunks):
            chunk_rows = [row for row in rows if row["chunk_id"] == chunk_id]
            writer.writerow(
                {
                    "chunk_id": chunk_id,
                    "n_combos": len(chunk_rows),
                    "estimated_total_params": sum(int(row["total_params"]) for row in chunk_rows),
                    "combo_ids": " ".join(str(row["combo_id"]) for row in chunk_rows),
                }
            )


def save_npz(path: Path, rows: list[dict[str, Any]], num_chunks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "Alpha",
        "purpose": "AR capacity screen with fixed latent-step consistency objective",
        "dims": list(DIMS),
        "encoder_profiles": {name: list(blocks) for name, blocks in ENCODER_PROFILES},
        "num_heads": list(HEADS),
        "num_chunks": num_chunks,
        "chunking": "greedy balance by total parameter count",
        "fixed_objective": FIXED_OBJECTIVE,
    }
    np.savez_compressed(
        path,
        combo_index=np.asarray([row["combo_index"] for row in rows], dtype=np.int32),
        chunk_id=np.asarray([row["chunk_id"] for row in rows], dtype=np.int32),
        combo_id=np.asarray([row["combo_id"] for row in rows]),
        encoder_profile=np.asarray([row["encoder_profile"] for row in rows]),
        encoder_level=np.asarray([row["encoder_level"] for row in rows], dtype=np.int32),
        dims=np.asarray([row["dims"] for row in rows], dtype=np.int32),
        num_blocks=np.asarray([row["num_blocks"] for row in rows], dtype=np.int32),
        num_heads=np.asarray([row["num_heads"] for row in rows], dtype=np.int32),
        total_params=np.asarray([row["total_params"] for row in rows], dtype=np.int64),
        encoder_params=np.asarray([row["encoder_params"] for row in rows], dtype=np.int64),
        decoder_params=np.asarray([row["decoder_params"] for row in rows], dtype=np.int64),
        encoder_decoder_params=np.asarray([row["encoder_decoder_params"] for row in rows], dtype=np.int64),
        koopman_params=np.asarray([row["koopman_params"] for row in rows], dtype=np.int64),
        koopman_share=np.asarray([row["koopman_share"] for row in rows], dtype=np.float32),
        koopman_to_encdec=np.asarray([row["koopman_to_encdec"] for row in rows], dtype=np.float32),
        lambda_recon=np.full(len(rows), FIXED_OBJECTIVE["lambda_recon"], dtype=np.float32),
        lambda_fwd=np.full(len(rows), FIXED_OBJECTIVE["lambda_fwd"], dtype=np.float32),
        lambda_bwd=np.full(len(rows), FIXED_OBJECTIVE["lambda_bwd"], dtype=np.float32),
        lambda_latent_fwd=np.full(len(rows), FIXED_OBJECTIVE["lambda_latent_fwd"], dtype=np.float32),
        lambda_ortho=np.full(len(rows), FIXED_OBJECTIVE["lambda_ortho"], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def print_summary(rows: list[dict[str, Any]], num_chunks: int) -> None:
    print("| combo | chunk | dims | encoder | params M | K M |")
    print("|---|---:|---:|---|---:|---:|")
    for row in rows:
        print(
            f"| {row['combo_id']} | {row['chunk_id']} | {row['dims']} | {row['encoder_profile']} | "
            f"{row['total_params'] / 1e6:.2f} | {row['koopman_params'] / 1e6:.2f} |"
        )
    print("\nchunks:")
    for chunk_id in range(num_chunks):
        chunk_rows = [row for row in rows if row["chunk_id"] == chunk_id]
        load = sum(int(row["total_params"]) for row in chunk_rows) / 1e6
        ids = ", ".join(str(row["combo_id"]) for row in chunk_rows)
        print(f"- {chunk_id}: {len(chunk_rows)} combos, approx {load:.1f}M params total: {ids}")


def main() -> None:
    args = parse_args()
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    rows = make_rows(args.num_chunks)
    save_npz(args.output, rows, args.num_chunks)
    write_csv(args.csv, rows)
    write_chunk_csv(args.chunk_csv, rows, args.num_chunks)
    print_summary(rows, args.num_chunks)
    print(f"\nwrote {args.output}")
    print(f"wrote {args.csv}")
    print(f"wrote {args.chunk_csv}")


if __name__ == "__main__":
    main()

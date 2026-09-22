from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ_OUTPUT = HERE / "gamma_loss_combos.npz"
CSV_OUTPUT = HERE / "gamma_loss_combos.csv"
CHUNKS_OUTPUT = HERE / "gamma_loss_chunks.csv"

COEFF_VALUES = np.asarray([0.0, 1e-3, 1e-2, 1e-1, 1.0], dtype=np.float32)
DEFAULT_ROW = {
    "lambda_fwd": 1.0,
    "lambda_bwd": 1.0,
    "lambda_recon": 1.0,
    "lambda_latent_fwd": 1.0,
    "lambda_ortho_a": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=NPZ_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=CSV_OUTPUT)
    parser.add_argument("--chunks-output", type=Path, default=CHUNKS_OUTPUT)
    parser.add_argument("--n-random", type=int, default=24)
    parser.add_argument("--n-chunks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=314)
    return parser.parse_args()


def coeff_label(value: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    if np.isclose(value, 1.0):
        return "1"
    return f"1e{int(round(np.log10(float(value))))}"


def row_signature(row: dict[str, float]) -> tuple[float, ...]:
    return (
        row["lambda_fwd"],
        row["lambda_bwd"],
        row["lambda_recon"],
        row["lambda_latent_fwd"],
        row["lambda_ortho_a"],
    )


def is_valid(row: dict[str, float]) -> bool:
    aux = np.asarray(
        [
            row["lambda_bwd"],
            row["lambda_recon"],
            row["lambda_latent_fwd"],
            row["lambda_ortho_a"],
        ],
        dtype=np.float32,
    )
    stabilizers = np.asarray(
        [row["lambda_bwd"], row["lambda_latent_fwd"], row["lambda_ortho_a"]],
        dtype=np.float32,
    )
    if row["lambda_fwd"] != 1.0:
        return False
    if not np.any(stabilizers > 0.0):
        return False
    if np.all(aux == 0.0):
        return False
    if np.all(aux == 1.0):
        return False
    if np.count_nonzero(aux > 0.0) < 2:
        return False
    return True


def sample_row(rng: np.random.Generator) -> dict[str, float]:
    return {
        "lambda_fwd": 1.0,
        "lambda_bwd": float(rng.choice(COEFF_VALUES, p=[0.08, 0.14, 0.23, 0.30, 0.25])),
        "lambda_recon": float(rng.choice(COEFF_VALUES, p=[0.12, 0.22, 0.27, 0.28, 0.11])),
        "lambda_latent_fwd": float(rng.choice(COEFF_VALUES, p=[0.08, 0.15, 0.24, 0.28, 0.25])),
        "lambda_ortho_a": float(rng.choice(COEFF_VALUES, p=[0.05, 0.18, 0.30, 0.34, 0.13])),
    }


def make_rows(n_random: int, seed: int) -> list[dict[str, float | str | int | bool]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str | int | bool]] = [
        {
            "gamma_combo_index": 0,
            "gamma_combo_id": "gamma_default",
            "is_default": True,
            **DEFAULT_ROW,
        }
    ]
    seen = {row_signature(DEFAULT_ROW)}
    attempts = 0
    while len(rows) < n_random + 1:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("could not sample enough unique valid gamma combinations")
        row = sample_row(rng)
        signature = row_signature(row)
        if signature in seen or not is_valid(row):
            continue
        seen.add(signature)
        labels = "_".join(
            [
                f"b{coeff_label(row['lambda_bwd'])}",
                f"r{coeff_label(row['lambda_recon'])}",
                f"l{coeff_label(row['lambda_latent_fwd'])}",
                f"o{coeff_label(row['lambda_ortho_a'])}",
            ]
        ).replace("-", "m")
        rows.append(
            {
                "gamma_combo_index": len(rows),
                "gamma_combo_id": f"gamma_{len(rows):02d}_{labels}",
                "is_default": False,
                **row,
            }
        )
    return rows


def chunk_rows(rows: list[dict[str, float | str | int | bool]], n_chunks: int) -> None:
    if n_chunks < 1:
        raise ValueError("--n-chunks must be >= 1")
    for idx, row in enumerate(rows):
        row["gamma_chunk_id"] = idx % n_chunks


def write_csv(path: Path, rows: list[dict[str, float | str | int | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gamma_combo_index",
        "gamma_chunk_id",
        "gamma_combo_id",
        "is_default",
        "lambda_fwd",
        "lambda_bwd",
        "lambda_recon",
        "lambda_latent_fwd",
        "lambda_ortho_a",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_chunks(path: Path, rows: list[dict[str, float | str | int | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = sorted(set(int(row["gamma_chunk_id"]) for row in rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gamma_chunk_id", "n_combos", "gamma_combo_ids"])
        writer.writeheader()
        for chunk_id in chunks:
            ids = [str(row["gamma_combo_id"]) for row in rows if int(row["gamma_chunk_id"]) == chunk_id]
            writer.writerow(
                {
                    "gamma_chunk_id": chunk_id,
                    "n_combos": len(ids),
                    "gamma_combo_ids": " ".join(ids),
                }
            )


def write_npz(path: Path, rows: list[dict[str, float | str | int | bool]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "gamma_combo_index": np.asarray([row["gamma_combo_index"] for row in rows], dtype=np.int32),
        "gamma_chunk_id": np.asarray([row["gamma_chunk_id"] for row in rows], dtype=np.int32),
        "gamma_combo_id": np.asarray([row["gamma_combo_id"] for row in rows]),
        "is_default": np.asarray([row["is_default"] for row in rows], dtype=bool),
        "lambda_fwd": np.asarray([row["lambda_fwd"] for row in rows], dtype=np.float32),
        "lambda_bwd": np.asarray([row["lambda_bwd"] for row in rows], dtype=np.float32),
        "lambda_recon": np.asarray([row["lambda_recon"] for row in rows], dtype=np.float32),
        "lambda_latent_fwd": np.asarray([row["lambda_latent_fwd"] for row in rows], dtype=np.float32),
        "lambda_ortho_a": np.asarray([row["lambda_ortho_a"] for row in rows], dtype=np.float32),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "stage": "Gamma",
                    "n_random": int(args.n_random),
                    "n_total": len(rows),
                    "n_chunks": int(args.n_chunks),
                    "seed": int(args.seed),
                    "values": [float(x) for x in COEFF_VALUES],
                    "constraints": [
                        "lambda_fwd fixed at 1.0",
                        "at least one stabilizer among bwd, latent_fwd, ortho_a is active",
                        "at least two auxiliary terms are active",
                        "auxiliary terms are not all 1.0",
                    ],
                },
                sort_keys=True,
            )
        ),
    }
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    rows = make_rows(args.n_random, args.seed)
    chunk_rows(rows, args.n_chunks)
    write_npz(args.output, rows, args)
    write_csv(args.csv_output, rows)
    write_chunks(args.chunks_output, rows)
    print(args.output)
    print(args.csv_output)
    print(args.chunks_output)


if __name__ == "__main__":
    main()

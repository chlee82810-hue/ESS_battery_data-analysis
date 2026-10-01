from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WORK_DIR = ROOT / "work"

BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Official cleaning rules from the authors' Load Data.ipynb.
EXCLUSIONS = {
    "Batch 1": {8, 10, 12, 13, 22},
    "Batch 2": set(),
    "Batch 3": {2, 23, 32, 37, 42, 43},
}

# Batch 1 cells 0-4 were paused and continued in the authors' second run.
# Their final life is corrected using the official continuation lengths. Only
# early-cycle features (<=100) are used, so no future signal is introduced.
BATCH1_CYCLE_LIFE_ADD = {0: 662, 1: 981, 2: 1060, 3: 208, 4: 482}


def deref_array(handle: h5py.File, ref: h5py.Reference) -> np.ndarray:
    return np.asarray(handle[ref][()]).squeeze()


def decode_policy(array: np.ndarray) -> str:
    values = np.asarray(array).astype(int).flatten()
    return "".join(chr(value) for value in values if value)


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.ptp(x[mask]) == 0:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def parse_policy(policy: str) -> tuple[float, float, float]:
    match = re.search(r"([0-9.]+)C\(([0-9.]+)%\)(?:-([0-9.]+)C)?", policy)
    if not match:
        return np.nan, np.nan, np.nan
    c1 = float(match.group(1))
    switch_soc = float(match.group(2))
    c2 = float(match.group(3)) if match.group(3) else c1
    return c1, switch_soc, c2


def piecewise_knee(cycles: np.ndarray, qd: np.ndarray) -> float:
    mask = np.isfinite(cycles) & np.isfinite(qd)
    x = cycles[mask]
    y = qd[mask]
    if len(x) < 120:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    candidates = np.linspace(max(60, int(len(x) * 0.2)), int(len(x) * 0.9), 90).astype(int)
    best_idx, best_sse = None, np.inf
    for idx in np.unique(candidates):
        if idx < 20 or len(x) - idx < 20:
            continue
        p1 = np.polyfit(x[:idx], y[:idx], 1)
        p2 = np.polyfit(x[idx:], y[idx:], 1)
        sse = float(np.square(y[:idx] - np.polyval(p1, x[:idx])).sum())
        sse += float(np.square(y[idx:] - np.polyval(p2, x[idx:])).sum())
        if sse < best_sse:
            best_sse, best_idx = sse, idx
    return float(x[best_idx]) if best_idx is not None else np.nan


def downsample_curve(cycles: np.ndarray, qd: np.ndarray, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(cycles) & np.isfinite(qd)
    x, y = cycles[mask], qd[mask]
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx], y[idx]


def extract_batch(batch_label: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    delta_curves: dict[str, np.ndarray] = {}

    with h5py.File(path, "r") as handle:
        batch = handle["batch"]
        num_cells = batch["summary"].shape[0]
        for cell_idx in range(num_cells):
            if cell_idx in EXCLUSIONS[batch_label]:
                continue
            cell_id = f"{batch_label.replace('Batch ', 'b')}c{cell_idx}"
            cycle_life = float(deref_array(handle, batch["cycle_life"][cell_idx, 0]))
            continued = batch_label == "Batch 1" and cell_idx in BATCH1_CYCLE_LIFE_ADD
            if continued:
                cycle_life += BATCH1_CYCLE_LIFE_ADD[cell_idx]

            policy = decode_policy(deref_array(handle, batch["policy_readable"][cell_idx, 0]))
            c1, switch_soc, c2 = parse_policy(policy)

            summary = handle[batch["summary"][cell_idx, 0]]
            series = {
                "IR": np.asarray(summary["IR"][0, :]).squeeze(),
                "QC": np.asarray(summary["QCharge"][0, :]).squeeze(),
                "QD": np.asarray(summary["QDischarge"][0, :]).squeeze(),
                "Tavg": np.asarray(summary["Tavg"][0, :]).squeeze(),
                "Tmin": np.asarray(summary["Tmin"][0, :]).squeeze(),
                "Tmax": np.asarray(summary["Tmax"][0, :]).squeeze(),
                "chargetime": np.asarray(summary["chargetime"][0, :]).squeeze(),
                "cycle": np.asarray(summary["cycle"][0, :]).squeeze(),
            }
            min_len = min(map(len, series.values()))
            series = {key: np.asarray(value[:min_len], dtype=float) for key, value in series.items()}
            early = (series["cycle"] >= 10) & (series["cycle"] <= 100)

            cycles_group = handle[batch["cycles"][cell_idx, 0]]
            first_curve = deref_array(handle, cycles_group["Qdlin"][0, 0])
            offset = 1 if np.allclose(first_curve, 0) else 0
            qd10 = deref_array(handle, cycles_group["Qdlin"][9 + offset, 0]).astype(float)
            qd100 = deref_array(handle, cycles_group["Qdlin"][99 + offset, 0]).astype(float)
            voltage = deref_array(handle, batch["Vdlin"][cell_idx, 0]).astype(float)
            delta_q = qd100 - qd10
            finite_delta = delta_q[np.isfinite(delta_q)]
            if finite_delta.size < 10:
                continue

            delta_curves[f"{cell_id}__delta_q"] = delta_q
            delta_curves[f"{cell_id}__voltage"] = voltage
            x_ds, qd_ds = downsample_curve(series["cycle"], series["QD"])
            for cycle, qd in zip(x_ds, qd_ds):
                curve_rows.append(
                    {
                        "cell_id": cell_id,
                        "batch": batch_label,
                        "cycle": float(cycle),
                        "qd": float(qd),
                        "cycle_life": cycle_life,
                        "continued": continued,
                    }
                )

            def early_stat(name: str, stat: str = "mean") -> float:
                values = series[name][early]
                values = values[np.isfinite(values)]
                if not len(values):
                    return np.nan
                return float(np.nanmean(values) if stat == "mean" else np.nanstd(values))

            rows.append(
                {
                    "cell_id": cell_id,
                    "batch": batch_label,
                    "source_file": path.name,
                    "cycle_life": cycle_life,
                    "log_cycle_life": np.log10(cycle_life),
                    "charge_policy": policy,
                    "c_rate_1": c1,
                    "switch_soc": switch_soc,
                    "c_rate_2": c2,
                    "continued": continued,
                    "n_summary_cycles": min_len,
                    "knee_cycle_descriptive": np.nan if continued else piecewise_knee(series["cycle"], series["QD"]),
                    "delta_q_mean": float(np.mean(finite_delta)),
                    "delta_q_min": float(np.min(finite_delta)),
                    "delta_q_max": float(np.max(finite_delta)),
                    "delta_q_var": float(np.var(finite_delta, ddof=1)),
                    "log_delta_q_var": float(np.log10(max(np.var(finite_delta, ddof=1), 1e-16))),
                    "delta_q_range": float(np.ptp(finite_delta)),
                    "delta_q_skew": float(skew(finite_delta, bias=False)),
                    "delta_q_kurtosis": float(kurtosis(finite_delta, fisher=True, bias=False)),
                    "delta_q_l1": float(np.mean(np.abs(finite_delta))),
                    "delta_q_l2": float(np.sqrt(np.mean(np.square(finite_delta)))),
                    "qd_early_mean": early_stat("QD"),
                    "qd_early_std": early_stat("QD", "std"),
                    "qd_early_slope": safe_slope(series["cycle"][early], series["QD"][early]),
                    "ir_early_mean": early_stat("IR"),
                    "ir_early_slope": safe_slope(series["cycle"][early], series["IR"][early]),
                    "tavg_early_mean": early_stat("Tavg"),
                    "tmax_early_mean": early_stat("Tmax"),
                    "temp_span_early_mean": float(np.nanmean((series["Tmax"] - series["Tmin"])[early])),
                    "charge_time_early_mean": early_stat("chargetime"),
                    "charge_time_early_slope": safe_slope(series["cycle"][early], series["chargetime"][early]),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(curve_rows), delta_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=list(BATCH_FILES), action="append")
    args = parser.parse_args()
    selected = args.batch or list(BATCH_FILES)

    metadata = {"batches": {}, "cleaning": {"exclusions": {k: sorted(v) for k, v in EXCLUSIONS.items()}}}
    for batch_label in selected:
        path = DATA_DIR / BATCH_FILES[batch_label]
        if not path.exists():
            raise FileNotFoundError(path)
        features, curves, deltas = extract_batch(batch_label, path)
        slug = batch_label.lower().replace(" ", "_")
        features.to_csv(WORK_DIR / f"features_{slug}.csv", index=False)
        curves.to_csv(WORK_DIR / f"curves_{slug}.csv", index=False)
        np.savez_compressed(WORK_DIR / f"delta_curves_{slug}.npz", **deltas)
        metadata["batches"][batch_label] = {
            "source": str(path),
            "cells_after_cleaning": int(len(features)),
            "cycle_life_min": float(features["cycle_life"].min()),
            "cycle_life_max": float(features["cycle_life"].max()),
        }
        print(batch_label, metadata["batches"][batch_label], flush=True)

    with (WORK_DIR / "extraction_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

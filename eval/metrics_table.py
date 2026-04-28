"""
eval/metrics_table.py
=====================
Formats evaluation results into comparison tables for the paper.

Produces:
  - Per-round mAP table (our method vs baselines)
  - Best-round summary table across folds
  - CSV export for further analysis

Usage:
  python eval/metrics_table.py \\
      --results_dir outputs/eval \\
      --output_dir outputs/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path regardless of how the script is invoked
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import EvalResult, load_results


# ── Table formatting ──────────────────────────────────────────────────────────

def format_per_round_table(method_results: Dict[str, List[EvalResult]],
                            metric:        str = "map50") -> str:
    """
    Format a per-round comparison table.

    Args:
        method_results: dict mapping method_name → List[EvalResult]
        metric:         'map50' or 'map50_95'

    Returns:
        Formatted string table.
    """
    # Collect all round_ids
    all_rounds = sorted({r.round_id
                         for results in method_results.values()
                         for r in results})

    # Build lookup: method → round_id → metric value
    lookup: Dict[str, Dict[int, float]] = {}
    for method, results in method_results.items():
        lookup[method] = {r.round_id: getattr(r, metric) for r in results}

    methods = sorted(method_results.keys())
    col_w   = 12

    header  = f"{'Round':<8}" + "".join(f"{m:<{col_w}}" for m in methods)
    divider = "-" * len(header)
    lines   = [header, divider]

    for round_id in all_rounds:
        row = f"{round_id:<8}"
        for method in methods:
            val = lookup[method].get(round_id, None)
            row += f"{val:<{col_w}.4f}" if val is not None else f"{'N/A':<{col_w}}"
        lines.append(row)

    return "\n".join(lines)


def format_best_round_table(method_results: Dict[str, List[EvalResult]]) -> str:
    """
    Format a summary table showing best-round mAP50 and mAP50-95 per method.
    """
    lines = [
        f"{'Method':<30} {'Best Round':<12} {'mAP50':<10} {'mAP50-95':<12}",
        "-" * 65,
    ]
    for method, results in sorted(method_results.items()):
        if not results:
            continue
        best = max(results, key=lambda r: r.map50)
        lines.append(
            f"{method:<30} {best.round_id:<12} {best.map50:<10.4f} {best.map50_95:<12.4f}"
        )
    return "\n".join(lines)


def summarise_folds(fold_results: List[List[EvalResult]],
                    metric:       str = "map50") -> Tuple[float, float]:
    """
    Compute mean and std of best-round metric across folds.

    Args:
        fold_results: list of per-fold result lists
        metric:       'map50' or 'map50_95'

    Returns:
        (mean, std) across folds
    """
    import statistics
    best_per_fold = []
    for results in fold_results:
        if results:
            best = max(results, key=lambda r: getattr(r, metric))
            best_per_fold.append(getattr(best, metric))

    if not best_per_fold:
        return 0.0, 0.0
    if len(best_per_fold) == 1:
        return best_per_fold[0], 0.0
    return statistics.mean(best_per_fold), statistics.stdev(best_per_fold)


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(method_results: Dict[str, List[EvalResult]],
               output_path:   str) -> None:
    """Export all results to CSV for further analysis."""
    rows = []
    for method, results in method_results.items():
        for r in results:
            rows.append({
                "method":    method,
                "round_id":  r.round_id,
                "map50":     r.map50,
                "map50_95":  r.map50_95,
                "checkpoint": r.checkpoint,
                "n_images":  r.n_images,
            })

    if not rows:
        print("No results to export.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {output_path}")


# ── Results loader ────────────────────────────────────────────────────────────

def load_all_results(results_dir: str) -> Dict[str, List[EvalResult]]:
    """
    Load all JSON result files from a directory.
    File naming convention: {method_name}_{fold}_{ratio}.json
    or simply {method_name}.json

    Returns dict: method_name → List[EvalResult]
    """
    results_dir = Path(results_dir)
    method_results: Dict[str, List[EvalResult]] = {}

    for json_file in sorted(results_dir.glob("*.json")):
        method_name = json_file.stem
        try:
            results = load_results(str(json_file))
            method_results[method_name] = results
        except Exception as e:
            print(f"Warning: could not load {json_file.name}: {e}")

    return method_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparison tables from evaluation results"
    )
    parser.add_argument("--results_dir", required=True,
        help="Directory containing eval JSON files")
    parser.add_argument("--output_dir", default="outputs/tables",
        help="Directory to write table files")
    parser.add_argument("--metric", default="map50",
        choices=["map50", "map50_95"],
        help="Primary metric for per-round table")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir} ...")
    method_results = load_all_results(args.results_dir)

    if not method_results:
        print("No result files found.")
        exit(1)

    print(f"Found {len(method_results)} method(s): {list(method_results.keys())}")
    print()

    # Per-round table
    print(f"=== Per-Round {args.metric.upper()} ===")
    table = format_per_round_table(method_results, metric=args.metric)
    print(table)
    print()
    (output_dir / f"per_round_{args.metric}.txt").write_text(table)

    # Best-round summary
    print("=== Best Round Summary ===")
    summary = format_best_round_table(method_results)
    print(summary)
    print()
    (output_dir / "best_round_summary.txt").write_text(summary)

    # CSV export
    export_csv(method_results, str(output_dir / "all_results.csv"))
    print(f"\nTables written to {output_dir}/")
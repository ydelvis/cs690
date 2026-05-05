"""
evaluate_permissions.py

Evaluates the quality of permission inference by comparing:

  baseline      — permissions from base_permissions only (no param info)
  full_static   — permissions inferred using all observed parameters (oracle)
  llm_assisted  — permissions produced by the LLM validation step (Step 6)
  rag_assisted  — permissions produced by RAG + LLM (Step 6b)

Ground truth = full_static_permissions (best purely-static answer).

Metrics per function: Precision, Recall, F1.
Aggregate metrics: macro-averaged over all functions.

Output:
  evaluation/results.json    — per-function details
  evaluation/summary.json    — aggregate metrics
  (printed report)
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
STATIC_PERMS_DIR = "static_permissions"
LLM_PERMS_DIR    = "llm_permissions"
RAG_PERMS_DIR    = "rag_permissions"
OUTPUT_DIR       = "evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(predicted: Set[str], ground_truth: Set[str]) -> dict:
    if not ground_truth and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "false_positives": sorted(predicted - ground_truth),
        "false_negatives": sorted(ground_truth - predicted),
    }

def macro_average(metrics_list: List[dict]) -> dict:
    if not metrics_list:
        return {}
    keys = ("precision", "recall", "f1")
    return {k: round(sum(m[k] for m in metrics_list) / len(metrics_list), 4) for k in keys}

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate_app(app_name: str) -> List[dict]:
    sp_path = os.path.join(STATIC_PERMS_DIR, f"{app_name}.json")
    if not os.path.exists(sp_path):
        return []

    with open(sp_path) as f:
        static_data = json.load(f)

    llm_path = os.path.join(LLM_PERMS_DIR, f"{app_name}.json")
    llm_data: dict = {}
    if os.path.exists(llm_path):
        with open(llm_path) as f:
            llm_data = json.load(f)

    rag_path = os.path.join(RAG_PERMS_DIR, f"{app_name}.json")
    rag_data: dict = {}
    if os.path.exists(rag_path):
        with open(rag_path) as f:
            rag_data = json.load(f)

    rows: List[dict] = []

    for func_name, func_data in static_data.items():
        ground_truth = set(func_data.get("full_static_permissions", []))
        baseline     = set(func_data.get("baseline_permissions", []))

        llm_entry = llm_data.get(func_name, {})
        llm_perms = set(llm_entry.get("llm_permissions", []))

        rag_entry = rag_data.get(func_name, {})
        rag_perms = set(rag_entry.get("rag_permissions", []))

        row = {
            "app":           app_name,
            "function":      func_name,
            "ground_truth":  sorted(ground_truth),
            "baseline":      sorted(baseline),
            "llm":           sorted(llm_perms),
            "rag":           sorted(rag_perms),
            "baseline_metrics": compute_metrics(baseline, ground_truth),
            "llm_metrics":      compute_metrics(llm_perms, ground_truth) if llm_perms else None,
            "rag_metrics":      compute_metrics(rag_perms, ground_truth) if rag_perms else None,
            "call_count":    len(func_data.get("call_details", [])),
        }
        rows.append(row)

    return rows

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_rows: List[dict] = []

    for app in sorted(os.listdir(STATIC_PERMS_DIR)):
        if not app.endswith(".json"):
            continue
        app_name = app[:-5]
        rows = evaluate_app(app_name)
        all_rows.extend(rows)

    if not all_rows:
        print("No evaluation data found. Run infer_static_permissions.py first.")
        exit(1)

    # Save per-function results
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    # Compute aggregates
    baseline_metrics = [r["baseline_metrics"] for r in all_rows]
    llm_metrics      = [r["llm_metrics"] for r in all_rows if r.get("llm_metrics")]
    rag_metrics      = [r["rag_metrics"] for r in all_rows if r.get("rag_metrics")]

    baseline_avg = macro_average(baseline_metrics)
    llm_avg      = macro_average(llm_metrics) if llm_metrics else {}
    rag_avg      = macro_average(rag_metrics) if rag_metrics else {}

    summary = {
        "total_functions":    len(all_rows),
        "functions_with_llm": len(llm_metrics),
        "functions_with_rag": len(rag_metrics),
        "baseline": baseline_avg,
        "llm":      llm_avg,
        "rag":      rag_avg,
        "per_app":  {},
    }

    # Per-app aggregates
    for app in sorted({r["app"] for r in all_rows}):
        app_rows = [r for r in all_rows if r["app"] == app]
        app_bm   = [r["baseline_metrics"] for r in app_rows]
        app_llm  = [r["llm_metrics"] for r in app_rows if r.get("llm_metrics")]
        app_rag  = [r["rag_metrics"]  for r in app_rows if r.get("rag_metrics")]
        summary["per_app"][app] = {
            "n_functions": len(app_rows),
            "baseline":    macro_average(app_bm),
            "llm":         macro_average(app_llm) if app_llm else {},
            "rag":         macro_average(app_rag)  if app_rag  else {},
        }

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # PRINT REPORT
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("PERMISSION INFERENCE EVALUATION REPORT")
    print("=" * 70)
    print(f"Total functions evaluated:  {summary['total_functions']}")
    print(f"Functions with LLM output:  {summary['functions_with_llm']}")
    print(f"Functions with RAG output:  {summary['functions_with_rag']}")
    print()
    print("AGGREGATE METRICS (macro-average over all functions)")
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    if baseline_avg:
        print(f"{'Baseline':20} {baseline_avg['precision']:>10.4f} "
              f"{baseline_avg['recall']:>10.4f} {baseline_avg['f1']:>10.4f}")
    if llm_avg:
        print(f"{'LLM-assisted':20} {llm_avg['precision']:>10.4f} "
              f"{llm_avg['recall']:>10.4f} {llm_avg['f1']:>10.4f}")
    if rag_avg:
        print(f"{'RAG-assisted':20} {rag_avg['precision']:>10.4f} "
              f"{rag_avg['recall']:>10.4f} {rag_avg['f1']:>10.4f}")
    print()
    print("PER-FUNCTION DETAILS")
    print("-" * 70)
    for row in all_rows:
        bm = row["baseline_metrics"]
        print(f"\n[{row['app']}] {row['function']}")
        print(f"  Ground truth: {row['ground_truth']}")
        print(f"  Baseline:     {row['baseline']} "
              f"(P={bm['precision']:.2f} R={bm['recall']:.2f} F1={bm['f1']:.2f})")
        if row.get("llm_metrics"):
            lm = row["llm_metrics"]
            print(f"  LLM:          {row['llm']} "
                  f"(P={lm['precision']:.2f} R={lm['recall']:.2f} F1={lm['f1']:.2f})")
        if row.get("rag_metrics"):
            rm = row["rag_metrics"]
            print(f"  RAG:          {row['rag']} "
                  f"(P={rm['precision']:.2f} R={rm['recall']:.2f} F1={rm['f1']:.2f})")
        if bm.get("false_negatives"):
            print(f"  Missing (baseline): {bm['false_negatives']}")

    print()
    print(f"Full results → {results_path}")
    print(f"Summary      → {summary_path}")

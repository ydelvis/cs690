#!/usr/bin/env python3
"""
run_pipeline.py — Pipeline Orchestrator

Runs the complete AWS Lambda permission inference pipeline in dependency order,
parallelising independent steps where possible.

Dependency graph:
  1 ──────────────────────────────┐
  2 ──┐                           │
  3 ──┴──► 4 ──► 5 ──► 6  ──► 7
                   └──► 6b ──► 7

Parallel groups:
  Steps 2 and 3  run concurrently (both feed into 4)
  Steps 6 and 6b run concurrently (both feed into 7)

Usage:
  python3 run_pipeline.py                   # run all steps
  python3 run_pipeline.py --skip-llm        # skip step 6  (LLM without RAG)
  python3 run_pipeline.py --skip-rag        # skip step 6b (RAG pipeline)
  python3 run_pipeline.py --force           # re-run even if outputs already exist
  python3 run_pipeline.py --steps 4,5,7     # run specific steps only
  python3 run_pipeline.py --dry-run         # print what would run, then exit
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# STEP REGISTRY
# ---------------------------------------------------------------------------
# Each entry: script path, human name, output path(s) used for skip-check,
# and which other step keys must complete first.
STEPS = {
    "1": {
        "name":   "Build permission database",
        "script": "src/build_permission_db.py",
        "deps":   [],
        "output": "dataset/permissions.json",
    },
    "2": {
        "name":   "Extract app configurations",
        "script": "src/extract_configs.py",
        "deps":   [],
        "output": "configs",
    },
    "3": {
        "name":   "Build call graphs",
        "script": "src/build_call_graphs.py",
        "deps":   [],
        "output": "call_graphs",
    },
    "4": {
        "name":   "Extract AWS SDK calls",
        "script": "src/extract_aws_calls.py",
        "deps":   ["2", "3"],
        "output": "aws_calls",
    },
    "5": {
        "name":   "Infer static permissions",
        "script": "src/infer_static_permissions.py",
        "deps":   ["1", "4"],
        "output": "static_permissions",
    },
    "6": {
        "name":   "LLM validation (raw source)",
        "script": "src/llm_validate_permissions.py",
        "deps":   ["5"],
        "output": "llm_permissions",
        "optional": True,
    },
    "6b": {
        "name":   "RAG validation (semantic retrieval)",
        "script": "src/rag_pipeline.py",
        "deps":   ["3", "5"],
        "output": "rag_permissions",
        "optional": True,
    },
    "7": {
        "name":   "Evaluate (all methods)",
        "script": "src/evaluate_permissions.py",
        "deps":   ["5"],
        "output": "evaluation",
    },
}

# Steps that can run concurrently (same parallel group → same integer key)
PARALLEL_GROUPS = {
    "2": 1, "3": 1,     # extract configs + build call graphs
    # Steps 6 and 6b run sequentially to avoid simultaneous Ollama load
}

# ---------------------------------------------------------------------------
# TERMINAL COLOURS
# ---------------------------------------------------------------------------
_BOLD  = "\033[1m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_CYAN  = "\033[36m"
_GRAY  = "\033[90m"
_RESET = "\033[0m"

def _fmt(msg: str, colour: str) -> str:
    return f"{colour}{msg}{_RESET}"


# ---------------------------------------------------------------------------
# OUTPUT-EXISTS CHECK
# ---------------------------------------------------------------------------
def _output_exists(step_key: str) -> bool:
    """Return True if the step's declared output already contains files."""
    out = STEPS[step_key]["output"]
    p = Path(out)
    if not p.exists():
        return False
    if p.is_file():
        return True
    # directory: check for at least one .json file inside
    return any(p.glob("*.json"))


# ---------------------------------------------------------------------------
# STEP RUNNER
# ---------------------------------------------------------------------------
_status: dict = {}   # step_key → "ok" | "skip" | "fail" | "skipped_by_user"

def _run_step(key: str, force: bool) -> bool:
    """
    Execute one pipeline step. Returns True on success.
    Prints timing and status. Updates _status in-place.
    """
    step = STEPS[key]
    label = f"Step {key:>2} — {step['name']}"

    if not force and _output_exists(key):
        print(f"  {_fmt('SKIP', _GRAY)}  {label}  {_fmt('(output exists)', _GRAY)}")
        _status[key] = "skip"
        return True

    print(f"  {_fmt('RUN ', _CYAN)}  {label}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, step["script"]],
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"  {_fmt('OK  ', _GREEN)}  {label}  {_fmt(f'({elapsed:.1f}s)', _GRAY)}")
            _status[key] = "ok"
            return True
        else:
            print(f"  {_fmt('FAIL', _RED)}  {label}  (exit {proc.returncode})")
            _status[key] = "fail"
            return False
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  {_fmt('FAIL', _RED)}  {label}  ({exc})")
        _status[key] = "fail"
        return False


# ---------------------------------------------------------------------------
# DEPENDENCY CHECK
# ---------------------------------------------------------------------------
def _deps_ok(key: str) -> bool:
    for dep in STEPS[key]["deps"]:
        if _status.get(dep) not in ("ok", "skip", "skipped_by_user"):
            return False
    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Step 6 (LLM without RAG)")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip Step 6b (RAG pipeline)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs already exist")
    parser.add_argument("--steps", metavar="STEPS",
                        help="Comma-separated list of step keys to run (e.g. 4,5,7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without running anything")
    args = parser.parse_args()

    # ---- Build the set of steps to run ----
    if args.steps:
        requested = set(args.steps.split(","))
        unknown   = requested - set(STEPS)
        if unknown:
            print(f"Unknown step(s): {unknown}.  Valid keys: {sorted(STEPS)}")
            sys.exit(1)
        run_set = requested
    else:
        run_set = set(STEPS.keys())

    if args.skip_llm:
        run_set.discard("6")
    if args.skip_rag:
        run_set.discard("6b")

    # Mark user-skipped steps so dependency checks still pass
    for key in set(STEPS) - run_set:
        _status[key] = "skipped_by_user"

    print(f"\n{_BOLD}AWS Lambda Permission Inference Pipeline{_RESET}")
    print(f"Steps to run: {sorted(run_set)}\n")

    if args.dry_run:
        for key in STEPS:   # preserve insertion (topological) order
            if key not in run_set:
                tag = _fmt("SKIP (user)", _GRAY)
            elif not args.force and _output_exists(key):
                tag = _fmt("SKIP (exists)", _GRAY)
            else:
                tag = _fmt("RUN", _CYAN)
            print(f"  {tag:30s}  Step {key:>2} — {STEPS[key]['name']}")
        return

    t_start = time.time()

    # ---- Execution order: honour dependencies + parallelise where safe ----
    # We process keys in topological order. Within the same parallel group,
    # we launch concurrent threads.

    # Determine the ordered sequence of groups to process
    # Solo steps (not in any parallel group) each form their own singleton group
    def _group_key(k):
        pg = PARALLEL_GROUPS.get(k)
        # Use a tuple so same-group steps sort together, preserving step order
        # within the overall sequence defined by STEPS ordering
        step_order = list(STEPS.keys()).index(k)
        if pg is not None:
            return (step_order - (step_order % 2), pg)   # group together
        return (step_order, 0)

    # Sequential execution units: each is a list of step keys that can run together
    processed: set = set()
    execution_plan: list = []   # list of lists (batches)

    for key in STEPS:
        if key in processed or key not in run_set:
            processed.add(key)
            continue
        # Find sibling steps in the same parallel group, also in run_set
        pg = PARALLEL_GROUPS.get(key)
        if pg is not None:
            siblings = [k for k, v in PARALLEL_GROUPS.items()
                        if v == pg and k in run_set and k not in processed]
        else:
            siblings = [key]
        execution_plan.append(siblings)
        processed.update(siblings)

    # ---- Run each batch ----
    for batch in execution_plan:
        # Check deps for all steps in this batch
        runnable = []
        for key in batch:
            if not _deps_ok(key):
                failed_deps = [d for d in STEPS[key]["deps"]
                               if _status.get(d) not in ("ok", "skip", "skipped_by_user")]
                is_optional = STEPS[key].get("optional", False)
                if is_optional:
                    print(f"  {_fmt('SKIP', _GRAY)}  Step {key:>2} — {STEPS[key]['name']}  "
                          f"{_fmt(f'(deps failed: {failed_deps})', _GRAY)}")
                    _status[key] = "skip"
                else:
                    print(f"  {_fmt('FAIL', _RED)}  Step {key:>2} — {STEPS[key]['name']}  "
                          f"(required deps failed: {failed_deps})")
                    _status[key] = "fail"
            else:
                runnable.append(key)

        if not runnable:
            continue

        if len(runnable) == 1:
            _run_step(runnable[0], args.force)
        else:
            # Parallel execution
            print(f"  {_fmt('....', _CYAN)}  Running steps {runnable} in parallel")
            with ThreadPoolExecutor(max_workers=len(runnable)) as ex:
                futures = {ex.submit(_run_step, k, args.force): k for k in runnable}
                for fut in as_completed(futures):
                    fut.result()   # re-raise if exception (already logged)

    # ---- Summary ----
    total = time.time() - t_start
    ok    = sum(1 for s in _status.values() if s == "ok")
    skip  = sum(1 for s in _status.values() if s in ("skip", "skipped_by_user"))
    fail  = sum(1 for s in _status.values() if s == "fail")

    print(f"\n{_BOLD}Pipeline complete{_RESET}  "
          f"({ok} ran, {skip} skipped, {_fmt(str(fail)+' failed', _RED if fail else _GRAY)}"
          f"  —  {total:.1f}s total)\n")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

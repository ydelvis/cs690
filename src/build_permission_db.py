"""
build_permission_db.py

Compresses the dataset/database folder into a single queryable JSON file.

Each service folder in dataset/database/<service>/ contains JSON files,
one per API action.  Each file is a list of {request, policy, apiName}
entries capturing real AWS IAM evaluation results for various parameter
combinations.

Output: dataset/permissions.json
  {
    "s3.putObject": {
      "base_permissions": ["s3:PutObject"],
      "param_permissions": {
        "ACL":                      ["s3:PutObjectAcl"],
        "ObjectLockLegalHoldStatus": ["s3:PutObjectLegalHold"],
        "ObjectLockMode":            ["s3:PutObjectRetention"]
      },
      "ambiguous_params": ["TaggingDirective"],
      "no_impact_params": ["Bucket", "BucketKeyEnabled", "CacheControl", ...],
      "sample_count": 12,
      "raw_cases": [
        {"params": ["ACL", "Bucket", ...], "permissions": ["s3:GetObject", ...]},
        ...
      ]
    },
    ...
  }

Query helper (importable):
  from build_permission_db import query_permissions
  perms = query_permissions(db, "s3", "putObject", {"Bucket", "Key", "ACL"})
  # → {"s3:GetObject", "s3:PutObject", "s3:PutObjectAcl"}
"""

import os
import re
import json
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATABASE_DIR = "dataset/database"
OUTPUT_PATH = "dataset/permissions.json"

# Confidence thresholds for calling a param a trigger vs. ambiguous
TRIGGER_MIN_COVERAGE = 0.75    # param appears in >= 75% of entries WITH the permission
TRIGGER_MAX_NOISE    = 0.20    # param appears in <= 20% of entries WITHOUT the permission

# ---------------------------------------------------------------------------
# REQUEST STRING PARSER
# ---------------------------------------------------------------------------
def parse_request_params(request_str: str) -> FrozenSet[str]:
    """
    Extract TOP-LEVEL parameter names from a Java-style SDK request string.

    Example:
      GetItemRequest(TableName=t, Key={Id=AttributeValue(S=1)}, ConsistentRead=true)
      → frozenset({'TableName', 'Key', 'ConsistentRead'})

    Nested structures (inside {}, (), []) are skipped so that attribute
    value keys like 'S', 'Id' don't pollute the top-level param set.
    """
    start = request_str.find("(")
    if start == -1:
        return frozenset()

    params: Set[str] = set()
    depth = 0
    key_buf: List[str] = []
    reading_key = True
    i = start

    while i < len(request_str):
        c = request_str[i]

        if c == "(" and depth == 0:
            depth = 1
            reading_key = True
            key_buf = []
            i += 1
            continue

        if depth == 1:
            if c == "=":
                key = "".join(key_buf).strip()
                if key and re.match(r"^[A-Za-z]\w*$", key):
                    params.add(key)
                key_buf = []
                reading_key = False
            elif c == ",":
                reading_key = True
                key_buf = []
            elif c in "({[":
                depth += 1
                reading_key = False
            elif c in ")}]":
                depth -= 1
                if depth == 0:
                    break
            elif reading_key and c not in "({[)}]":
                key_buf.append(c)
        else:
            if c in "({[":
                depth += 1
            elif c in ")}]":
                depth -= 1
                if depth == 0:
                    break

        i += 1

    return frozenset(params)


# ---------------------------------------------------------------------------
# PERMISSION EXTRACTION
# ---------------------------------------------------------------------------
def _extract_permissions(policy: dict) -> FrozenSet[str]:
    perms: Set[str] = set()
    for stmt in policy.get("Statement", []):
        if stmt.get("Effect", "Allow") != "Allow":
            continue
        for action in stmt.get("Action", []):
            perms.add(action)
    return frozenset(perms)


# ---------------------------------------------------------------------------
# CORE ANALYSIS
# ---------------------------------------------------------------------------
def analyze_action(entries: List[dict]) -> Optional[dict]:
    """
    Given all database entries for a single API action, produce a
    compressed, queryable permission mapping.
    """
    if not entries:
        return None

    # Parse each entry into (param_set, permission_set)
    cases: List[Tuple[FrozenSet[str], FrozenSet[str]]] = []
    for entry in entries:
        params = parse_request_params(entry.get("request", ""))
        perms  = _extract_permissions(entry.get("policy", {}))
        cases.append((params, perms))

    all_perm_sets = [perms for _, perms in cases]

    # Base permissions: granted in every case regardless of params
    base_perms = frozenset.intersection(*all_perm_sets)

    # Extra permissions: present in some but not all cases
    extra_perms = frozenset.union(*all_perm_sets) - base_perms

    # All top-level params ever observed for this action
    all_params = frozenset.union(*[params for params, _ in cases]) if cases else frozenset()

    # For each extra permission, determine which params trigger it
    param_to_perms: Dict[str, Set[str]] = {}    # param → permissions it triggers
    ambiguous: Set[str] = set()

    for perm in extra_perms:
        with_perm    = [params for params, ps in cases if perm in ps]
        without_perm = [params for params, ps in cases if perm not in ps]

        total_with    = len(with_perm)
        total_without = len(without_perm)

        if total_with == 0:
            continue

        params_ever_with    = frozenset.union(*with_perm)
        params_ever_without = frozenset.union(*without_perm) if without_perm else frozenset()

        # Params exclusive to entries with this permission → definite triggers
        exclusive = params_ever_with - params_ever_without
        for param in exclusive:
            param_to_perms.setdefault(param, set()).add(perm)

        # Params shared between both groups → check correlation strength
        shared = params_ever_with & params_ever_without
        for param in shared:
            freq_with    = sum(1 for p in with_perm    if param in p) / total_with
            freq_without = sum(1 for p in without_perm if param in p) / total_without

            if freq_with >= TRIGGER_MIN_COVERAGE and freq_without <= TRIGGER_MAX_NOISE:
                param_to_perms.setdefault(param, set()).add(perm)
            elif freq_with > freq_without * 1.5:
                # Weaker signal — flag as ambiguous rather than asserting
                ambiguous.add(param)

    # No-impact: never control any extra permission
    trigger_params = frozenset(param_to_perms.keys())
    no_impact = all_params - trigger_params - ambiguous

    return {
        "base_permissions":  sorted(base_perms),
        "param_permissions": {
            param: sorted(perms)
            for param, perms in sorted(param_to_perms.items())
        },
        "ambiguous_params":  sorted(ambiguous),
        "no_impact_params":  sorted(no_impact),
        "sample_count":      len(cases),
        "raw_cases": [
            {
                "params":      sorted(params),
                "permissions": sorted(perms),
            }
            for params, perms in cases
        ],
    }


# ---------------------------------------------------------------------------
# QUERY HELPER  (importable by rag_pipeline.py)
# ---------------------------------------------------------------------------
def load_permission_db(path: str = OUTPUT_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def query_permissions(
    db: dict,
    service: str,
    action: str,
    observed_params: Set[str],
) -> Set[str]:
    """
    Estimate required IAM permissions for a given SDK call.

    Args:
        db:              The loaded permissions.json dict.
        service:         AWS service name, e.g. "s3" or "dynamodb".
        action:          SDK method name, e.g. "putObject" or "getItem".
        observed_params: Set of top-level parameter names observed in the call.

    Returns:
        Set of IAM permission strings (e.g. {"s3:PutObject", "s3:PutObjectAcl"}).
    """
    key = f"{service}.{action}"
    entry = db.get(key)
    if entry is None:
        return set()

    required: Set[str] = set(entry["base_permissions"])

    for param in observed_params:
        extra = entry.get("param_permissions", {}).get(param, [])
        required.update(extra)

    return required


def query_permissions_exact(
    db: dict,
    service: str,
    action: str,
    observed_params: Set[str],
) -> Optional[List[str]]:
    """
    Exact lookup against raw_cases.  Returns permissions for the entry
    whose param set best (largest subset) matches observed_params.
    Falls back to None if no case is a subset of observed_params.
    """
    key = f"{service}.{action}"
    entry = db.get(key)
    if entry is None:
        return None

    best_match = None
    best_overlap = -1
    for case in entry.get("raw_cases", []):
        case_params = set(case["params"])
        if case_params <= observed_params:  # case is a subset of observed
            overlap = len(case_params)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = case["permissions"]

    return best_match


# ---------------------------------------------------------------------------
# BUILD DATABASE
# ---------------------------------------------------------------------------
def build_permission_db(database_dir: str) -> dict:
    db: dict = {}

    for service in sorted(os.listdir(database_dir)):
        service_dir = os.path.join(database_dir, service)
        if not os.path.isdir(service_dir):
            continue

        print(f"Processing service: {service}")

        for fname in sorted(os.listdir(service_dir)):
            if not fname.endswith(".json"):
                continue

            action = fname[:-5]  # strip .json
            key    = f"{service}.{action}"
            fpath  = os.path.join(service_dir, fname)

            with open(fpath, "r", encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON error in {fpath}: {e}")
                    continue

            # Some files are a single entry dict instead of a list
            entries = [raw] if isinstance(raw, dict) else raw
            # Drop any non-dict items that may appear in malformed files
            entries = [e for e in entries if isinstance(e, dict)]

            result = analyze_action(entries)
            if result is None:
                continue

            db[key] = result
            print(
                f"  {key}: base={result['base_permissions']}, "
                f"triggers={len(result['param_permissions'])} params, "
                f"samples={result['sample_count']}"
            )

    return db


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Building permission database from {DATABASE_DIR}...")
    db = build_permission_db(DATABASE_DIR)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(db, f, indent=2)

    total_actions = len(db)
    total_samples = sum(e["sample_count"] for e in db.values())
    print(f"\nDone. {total_actions} actions, {total_samples} total samples → {OUTPUT_PATH}")

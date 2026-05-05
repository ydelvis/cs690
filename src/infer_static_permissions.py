"""
infer_static_permissions.py

Derives two permission sets for each serverless function:

  baseline  — only base_permissions from permissions.json, independent of
               which parameters were observed.  Represents what we'd know
               from just seeing "s3.putObject was called" with no param info.

  full_static — base_permissions + any param-triggered permissions deduced
                from the *observed* parameters extracted by extract_aws_calls.
                Represents the best purely-static answer (the oracle /
                ground truth for evaluation).

Output: static_permissions/<app>.json
  {
    "<func_name>": {
      "baseline_permissions":    ["s3:PutObject"],
      "full_static_permissions": ["s3:PutObject", "s3:PutObjectAcl"],
      "call_details": [
        {
          "service": "s3", "method": "putObject",
          "params_present": ["ACL", "Bucket", "Key"],
          "baseline":    ["s3:PutObject"],
          "full_static": ["s3:PutObject", "s3:PutObjectAcl"]
        }
      ]
    }
  }
"""

import json
import os
import re
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
AWS_CALLS_DIR    = "aws_calls"
PERMISSIONS_PATH = "dataset/permissions.json"
OUTPUT_DIR       = "static_permissions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOAD PERMISSION DB
# ---------------------------------------------------------------------------
def load_permission_db(path: str = PERMISSIONS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# PERMISSION LOOKUP
# ---------------------------------------------------------------------------
def _method_to_permission(service: str, method: str) -> str:
    """
    Derive the IAM permission string from the SDK method name.
    Convention: service:MethodName with first letter uppercased.
    Paginator wrappers (e.g. scanPaginator) strip the suffix so the
    result matches the real IAM action (dynamodb:Scan).
    """
    m = re.sub(r"Paginator$", "", method)
    return f"{service}:{m[0].upper()}{m[1:]}"


def baseline_permissions(db: dict, service: str, method: str) -> List[str]:
    """
    Only base_permissions — no param info needed.
    If the DB has no base_permissions for this method, fall back to
    deriving the permission directly from the method name (e.g.
    s3.deleteObject → s3:DeleteObject).  This ensures methods like
    deleteObject and getObject get a meaningful baseline even when the
    database samples only cover non-default parameter combinations.
    """
    key   = f"{service}.{method}"
    entry = db.get(key)
    if not entry:
        return []
    base = list(entry.get("base_permissions", []))
    if not base:
        base = [_method_to_permission(service, method)]
    return base


def full_static_permissions(
    db: dict,
    service: str,
    method: str,
    params_present: List[str],
) -> List[str]:
    """base_permissions + param-triggered permissions from observed params."""
    key   = f"{service}.{method}"
    entry = db.get(key)
    if not entry:
        return []

    required: Set[str] = set(entry.get("base_permissions", []))
    if not required:
        required = {_method_to_permission(service, method)}
    param_perms = entry.get("param_permissions", {})
    for param in params_present:
        required.update(param_perms.get(param, []))

    return sorted(required)

# ---------------------------------------------------------------------------
# PER-APP
# ---------------------------------------------------------------------------
def infer_for_app(app_name: str, db: dict) -> dict:
    aws_calls_path = os.path.join(AWS_CALLS_DIR, f"{app_name}.json")
    if not os.path.exists(aws_calls_path):
        return {}

    with open(aws_calls_path) as f:
        app_calls = json.load(f)

    result: dict = {}

    for func_name, func_data in app_calls.items():
        baseline_set: Set[str]    = set()
        full_static_set: Set[str] = set()
        call_details: List[dict]  = []

        for call in func_data.get("aws_calls", []):
            svc    = call["service"]
            method = call["method"]
            params = call.get("params_present", [])

            base  = baseline_permissions(db, svc, method)
            full  = full_static_permissions(db, svc, method, params)

            baseline_set.update(base)
            full_static_set.update(full)

            call_details.append({
                "service":        svc,
                "method":         method,
                "params_present": params,
                "caller_func":    call.get("caller_func", ""),
                "file":           call.get("file", ""),
                "line":           call.get("line"),
                "baseline":       sorted(base),
                "full_static":    sorted(full),
            })

        result[func_name] = {
            "handler_file":             func_data.get("handler_file"),
            "handler_func":             func_data.get("handler_func"),
            "language":                 func_data.get("language"),
            "baseline_permissions":     sorted(baseline_set),
            "full_static_permissions":  sorted(full_static_set),
            "call_details":             call_details,
        }

    return result

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    db = load_permission_db()

    for app in sorted(os.listdir(AWS_CALLS_DIR)):
        if not app.endswith(".json"):
            continue
        app_name = app[:-5]

        print(f"Inferring static permissions: {app_name}")
        result = infer_for_app(app_name, db)
        if not result:
            print("  (no AWS calls found)")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{app_name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        for func, data in result.items():
            print(
                f"  {func}: baseline={data['baseline_permissions']} "
                f"full={data['full_static_permissions']}"
            )

"""
extract_configs.py

Parses serverless.yml (and any .env files) from each app and emits a
normalized config JSON used by the RAG pipeline to resolve environment
variable references found in source code.

Output (one file per app):
  configs/<app_name>.json
  {
    "service": "my-service",
    "provider": {"name": "aws", "runtime": "...", "region": "..."},
    "environment": {"ENV_VAR": "value", ...},       # merged: provider-level + all function-level
    "functions": {
      "myFunc": {
        "handler": "path/to/handler.method",
        "environment": {"FUNC_VAR": "value"},
        "events": [...]
      }
    },
    "iam_statements": [
      {"Effect": "Allow", "Action": ["s3:PutObject"], "Resource": "..."}
    ],
    "custom": {...},
    "source_files": ["serverless.yml", ".env"]
  }

Requires:
  pip install pyyaml python-dotenv
"""

import os
import json
import glob
from typing import Any, Dict, List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Run: pip install pyyaml")

try:
    from dotenv import dotenv_values
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset/apps"
OUTPUT_DIR = "configs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SERVERLESS.YML PARSER
# ---------------------------------------------------------------------------
def _resolve_sls_refs(obj: Any, service: str, provider: dict, custom: dict) -> Any:
    """
    Best-effort resolution of ${self:...}, ${env:...}, ${opt:...} references.
    Unresolvable refs are left as-is (string).
    """
    if isinstance(obj, str):
        import re
        def _replace(m):
            ref = m.group(1)
            if ref.startswith("self:service"):
                return service
            if ref.startswith("self:provider.stage"):
                return provider.get("stage", "dev")
            if ref.startswith("self:provider.region"):
                return provider.get("region", "us-east-1")
            if ref.startswith("self:custom."):
                key = ref[len("self:custom."):]
                return str(custom.get(key, m.group(0)))
            if ref.startswith("env:"):
                env_key = ref[4:]
                return os.environ.get(env_key, m.group(0))
            return m.group(0)
        return re.sub(r"\$\{([^}]+)\}", _replace, obj)
    if isinstance(obj, dict):
        return {k: _resolve_sls_refs(v, service, provider, custom) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_sls_refs(v, service, provider, custom) for v in obj]
    return obj


def parse_serverless_yml(yml_path: str) -> Optional[dict]:
    if not _YAML_AVAILABLE:
        return None

    # Register CloudFormation-specific YAML tags so they parse as plain scalars
    # instead of raising a constructor error.
    _CF_TAGS = [
        "!Ref", "!Sub", "!Join", "!Select", "!Split", "!If", "!Equals",
        "!Not", "!And", "!Or", "!GetAtt", "!FindInMap", "!Base64",
        "!Condition", "!ImportValue", "!Transform",
    ]
    loader_class = type(
        "_CFLoader",
        (yaml.SafeLoader,),
        {},
    )
    def _cf_constructor(loader, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)

    for _tag in _CF_TAGS:
        loader_class.add_constructor(_tag, _cf_constructor)

    with open(yml_path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            raw = yaml.load(f, Loader=loader_class)  # noqa: S506
        except yaml.YAMLError as e:
            print(f"  Warning: YAML parse error in {yml_path}: {e}")
            return None

    if not isinstance(raw, dict):
        return None

    service = raw.get("service", "")
    provider = raw.get("provider", {}) or {}
    custom = raw.get("custom", {}) or {}

    # Resolve interpolations
    provider = _resolve_sls_refs(provider, service, provider, custom)
    custom = _resolve_sls_refs(custom, service, provider, custom)

    # Provider-level environment
    provider_env: Dict[str, str] = {}
    if isinstance(provider.get("environment"), dict):
        provider_env = {
            k: str(v) for k, v in provider["environment"].items() if v is not None
        }

    # IAM statements
    iam_statements: List[dict] = []
    iam_raw = provider.get("iamRoleStatements") or []
    for stmt in iam_raw:
        if isinstance(stmt, dict):
            iam_statements.append(stmt)

    # Functions
    functions: Dict[str, dict] = {}
    merged_env: Dict[str, str] = dict(provider_env)  # start with provider-level

    for func_name, func_cfg in (raw.get("functions") or {}).items():
        if not isinstance(func_cfg, dict):
            continue
        func_cfg = _resolve_sls_refs(func_cfg, service, provider, custom)

        func_env: Dict[str, str] = {}
        if isinstance(func_cfg.get("environment"), dict):
            func_env = {k: str(v) for k, v in func_cfg["environment"].items() if v is not None}

        # Merge into the global env (function-level overrides provider-level)
        merged_env.update(func_env)

        functions[func_name] = {
            "handler": func_cfg.get("handler", ""),
            "runtime": func_cfg.get("runtime", provider.get("runtime", "")),
            "environment": func_env,
            "events": func_cfg.get("events", []),
        }

    return {
        "service": service,
        "provider": {
            "name": provider.get("name", "aws"),
            "runtime": provider.get("runtime", ""),
            "region": provider.get("region", ""),
            "stage": provider.get("stage", "dev"),
            "memory_mb": provider.get("memorySize"),
            "timeout_s": provider.get("timeout"),
        },
        "environment": merged_env,
        "functions": functions,
        "iam_statements": iam_statements,
        "custom": custom,
    }


# ---------------------------------------------------------------------------
# .env FILE PARSER
# ---------------------------------------------------------------------------
def parse_dotenv(env_path: str) -> Dict[str, str]:
    if _DOTENV_AVAILABLE:
        return {k: v for k, v in dotenv_values(env_path).items() if v is not None}

    # Minimal fallback parser for KEY=VALUE format
    env_vars: Dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    env_vars[key] = val
    return env_vars


# ---------------------------------------------------------------------------
# PER-APP CONFIG EXTRACTION
# ---------------------------------------------------------------------------
def extract_app_config(app_path: str) -> dict:
    config = {
        "service": os.path.basename(app_path),
        "provider": {},
        "environment": {},
        "functions": {},
        "iam_statements": [],
        "custom": {},
        "source_files": [],
    }

    # 1. serverless.yml (may be nested — take the first one found)
    sls_files = glob.glob(os.path.join(app_path, "**/serverless.yml"), recursive=True)
    sls_files += glob.glob(os.path.join(app_path, "**/serverless.yaml"), recursive=True)

    for sls_path in sls_files:
        rel = os.path.relpath(sls_path, app_path)
        parsed = parse_serverless_yml(sls_path)
        if parsed:
            config["service"] = parsed.get("service") or config["service"]
            config["provider"].update(parsed.get("provider", {}))
            config["environment"].update(parsed.get("environment", {}))
            config["functions"].update(parsed.get("functions", {}))
            config["iam_statements"].extend(parsed.get("iam_statements", []))
            config["custom"].update(parsed.get("custom", {}))
            config["source_files"].append(rel)

    # 2. .env files
    for env_file in glob.glob(os.path.join(app_path, "**/.env*"), recursive=True):
        rel = os.path.relpath(env_file, app_path)
        env_vars = parse_dotenv(env_file)
        # .env values don't override serverless.yml (lower precedence)
        for k, v in env_vars.items():
            if k not in config["environment"]:
                config["environment"][k] = v
        config["source_files"].append(rel)

    return config


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for app in sorted(os.listdir(DATASET_DIR)):
        app_path = os.path.join(DATASET_DIR, app)
        if not os.path.isdir(app_path):
            continue

        print(f"Extracting config for {app}...")
        config = extract_app_config(app_path)

        out_path = os.path.join(OUTPUT_DIR, f"{app}.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        n_env = len(config["environment"])
        n_funcs = len(config["functions"])
        n_iam = len(config["iam_statements"])
        print(
            f"  {n_env} env vars, {n_funcs} functions, "
            f"{n_iam} IAM statements → {out_path}"
        )

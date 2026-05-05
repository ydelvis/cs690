"""
llm_validate_permissions.py

Takes the baseline static permissions for each serverless function
and asks a local LLM (via Ollama) to validate, add, or remove
permissions based on the actual source code.

Input:
  static_permissions/<app>.json  (from infer_static_permissions.py)
  dataset/apps/<app>/            (source files for context)
  configs/<app>.json             (env vars for additional context)

Output: llm_permissions/<app>.json
  {
    "<func_name>": {
      "initial_permissions":  ["s3:PutObject"],
      "llm_permissions":      ["s3:PutObject", "s3:GetObject"],
      "reasoning":            "...",
      "raw_response":         "..."
    }
  }

Requires Ollama running at OLLAMA_URL with OLLAMA_MODEL loaded.
"""

import json
import os
import re
from typing import List

import requests

# ---------------------------------------------------------------------------
# CONFIG  (mirror rag_pipeline.py settings)
# ---------------------------------------------------------------------------
OLLAMA_URL   = "http://128.119.245.39:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:27b"

STATIC_PERMS_DIR = "static_permissions"
DATASET_DIR      = "dataset/apps"
CONFIGS_DIR      = "configs"
OUTPUT_DIR       = "llm_permissions"

# How many lines of source to include per file (avoid prompt overflow)
MAX_LINES_PER_FILE = 120

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SOURCE CODE COLLECTOR
# ---------------------------------------------------------------------------
_EXT_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go"}


def collect_source(app_path: str, handler_file: str, max_lines: int = MAX_LINES_PER_FILE) -> str:
    """
    Returns up to max_lines of source code from the handler file and any
    files it directly imports/requires (one level deep).
    """
    chunks: List[str] = []

    def add_file(fp: str):
        if not os.path.exists(fp):
            return
        ext  = os.path.splitext(fp)[1].lower()
        lang = _EXT_LANG.get(ext, "")
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            return
        content = "".join(lines[:max_lines])
        rel     = os.path.relpath(fp, app_path)
        chunks.append(f"# File: {rel}\n```{lang}\n{content}\n```")

    primary = os.path.join(app_path, handler_file) if handler_file else None
    if primary:
        add_file(primary)

        # One-level dependency scan
        ext = os.path.splitext(primary)[1].lower()
        try:
            src = open(primary, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            src = ""

        base_dir = os.path.dirname(primary)
        if ext in (".js", ".ts"):
            for rel_req in re.findall(r"require\(['\"](\.[^'\"]+)['\"]\)", src):
                for e in (".js", ".ts"):
                    candidate = os.path.normpath(os.path.join(base_dir, rel_req + e))
                    if os.path.exists(candidate):
                        add_file(candidate)
                        break
        elif ext == ".py":
            for mod in re.findall(r"^from\s+\.(\w+)\s+import|^import\s+\.(\w+)", src, re.M):
                name = (mod[0] or mod[1]).strip()
                candidate = os.path.join(base_dir, name + ".py")
                if os.path.exists(candidate):
                    add_file(candidate)

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an AWS IAM security analyst. /no_think\n"
    "Given a Lambda function's source code and a list of statically inferred "
    "IAM permissions, determine the COMPLETE and MINIMAL set of permissions the "
    "function actually requires. "
    "Rules: include a permission only if the code demonstrably uses that AWS API; "
    "remove over-provisioned permissions; add any missing ones you detect, including "
    "indirect SDK usage (wrappers, helper modules, env var references). "
    "Reply with STRICT JSON only — no markdown, no explanation outside the JSON:\n"
    '{"permissions": ["service:Action", ...], "reasoning": "brief explanation"}'
)


def build_prompt(
    func_name: str,
    initial_permissions: List[str],
    source_code: str,
    env_vars: dict,
    call_details: List[dict],
) -> str:
    env_block = "\n".join(f"  {k}={v}" for k, v in sorted(env_vars.items())) or "  (none)"
    calls_block = "\n".join(
        f"  {c['service']}.{c['method']}({', '.join(c.get('params_present', []))})"
        for c in call_details
    ) or "  (none detected statically)"
    perms_block = "\n".join(f"  {p}" for p in initial_permissions) or "  (none)"

    return (
        f"FUNCTION: {func_name}\n\n"
        f"ENVIRONMENT VARIABLES:\n{env_block}\n\n"
        f"AWS SDK CALLS DETECTED:\n{calls_block}\n\n"
        f"INITIAL PERMISSIONS (static analysis):\n{perms_block}\n\n"
        f"SOURCE CODE:\n{source_code}"
    )


# ---------------------------------------------------------------------------
# OLLAMA CALL  (/api/chat — applies the model's native chat template)
# ---------------------------------------------------------------------------
def query_ollama(system: str, user: str) -> str:
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def safe_json_parse(text: str) -> dict:
    # Strip qwen3 chain-of-thought blocks before trying to parse
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"error": "parse_failed", "raw": text}


# ---------------------------------------------------------------------------
# PER-APP
# ---------------------------------------------------------------------------
def validate_app(app_name: str) -> dict:
    sp_path = os.path.join(STATIC_PERMS_DIR, f"{app_name}.json")
    if not os.path.exists(sp_path):
        print(f"  No static permissions found for {app_name}.")
        return {}

    with open(sp_path) as f:
        static_perms = json.load(f)

    config_path = os.path.join(CONFIGS_DIR, f"{app_name}.json")
    env_vars = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            env_vars = json.load(f).get("environment", {})

    app_path = os.path.join(DATASET_DIR, app_name)
    results: dict = {}

    for func_name, func_data in static_perms.items():
        initial = func_data.get("baseline_permissions", [])
        handler_file = func_data.get("handler_file") or ""
        call_details = func_data.get("call_details", [])

        source     = collect_source(app_path, handler_file)
        user_msg   = build_prompt(func_name, initial, source, env_vars, call_details)

        print(f"  Querying LLM for {func_name}...")
        try:
            raw    = query_ollama(_SYSTEM_PROMPT, user_msg)
            parsed = safe_json_parse(raw)
        except Exception as e:
            parsed = {"error": str(e)}
            raw = ""

        results[func_name] = {
            "initial_permissions": initial,
            "llm_permissions":     parsed.get("permissions", []),
            "reasoning":           parsed.get("reasoning", ""),
            "raw_response":        raw[:2000] if isinstance(raw, str) else "",
        }

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for app in sorted(os.listdir(STATIC_PERMS_DIR)):
        if not app.endswith(".json"):
            continue
        app_name = app[:-5]

        out_path = os.path.join(OUTPUT_DIR, f"{app_name}.json")
        if os.path.exists(out_path):
            print(f"LLM validation: {app_name}  (skipping — output exists)")
            continue

        print(f"LLM validation: {app_name}")
        results = validate_app(app_name)
        if not results:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{app_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → {out_path}")

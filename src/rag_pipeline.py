"""
rag_pipeline.py — Step 6b

RAG-assisted IAM permission validation.

Builds a per-app FAISS vector index over source-level function chunks, then
for each Lambda entry function retrieves the most semantically relevant code
and asks the LLM to validate / add / remove permissions.

Retrieval strategy (hybrid):
  1. Semantic: top-K chunks per query (one for the handler, one per permission)
  2. Call-graph: code chunks belonging to functions reachable from the entry point

This complements Step 6 (llm_validate_permissions.py), which injects the raw
handler file.  RAG scales better to large apps and finds helpers in other files.

Input:
  static_permissions/<app>.json   (from infer_static_permissions.py — Step 5)
  call_graphs/<app>.json          (from build_call_graphs.py — Step 3)
  dataset/apps/<app>/             (source files)

Output: rag_permissions/<app>.json
  {
    "<func_name>": {
      "initial_permissions":  ["s3:PutObject"],
      "rag_permissions":      ["s3:PutObject", "s3:GetObject"],
      "reasoning":            "...",
      "raw_response":         "..."
    }
  }

Requires Ollama running at OLLAMA_URL with OLLAMA_MODEL loaded.
"""

import hashlib
import importlib
import json
import os
import pickle
import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EMBED_MODEL      = "BAAI/bge-base-en"
OLLAMA_URL       = "http://128.119.245.39:11434/api/chat"
OLLAMA_MODEL     = "qwen3.5:27b"
TOP_K            = 5   # chunks retrieved per query
CACHE_DIR        = ".cache"

STATIC_PERMS_DIR = "static_permissions"
CALLGRAPHS_DIR   = "call_graphs"
DATASET_DIR      = "dataset/apps"
OUTPUT_DIR       = "rag_permissions"

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)

# ---------------------------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------------------------
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------------------------------------------------------------------
# TREE-SITTER SETUP
# ---------------------------------------------------------------------------
EXT_TO_LANG = {".js": "javascript", ".ts": "javascript", ".py": "python", ".go": "go"}

LANG_FUNCTION_TYPES = {
    "javascript": ["function_declaration", "method_definition",
                   "function_expression", "arrow_function"],
    "python":     ["function_definition"],
    "go":         ["function_declaration", "method_declaration"],
}

_LANG_PACKAGES = {
    "javascript": "tree_sitter_javascript",
    "python":     "tree_sitter_python",
    "go":         "tree_sitter_go",
}

parsers: Dict[str, Parser] = {}
for _lang, _pkg in _LANG_PACKAGES.items():
    try:
        _mod  = importlib.import_module(_pkg)
        _lang_obj = Language(_mod.language())
        parsers[_lang] = Parser(_lang_obj)
    except Exception as _e:
        print(f"Warning: could not load parser for {_lang}: {_e}")

# ---------------------------------------------------------------------------
# FUNCTION NAME EXTRACTION
# ---------------------------------------------------------------------------
def _get_function_name(node, code: str, parent=None) -> str:
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "field_identifier"):
            return code[child.start_byte:child.end_byte]

    # const/let/var name = function()/() => {}
    if parent is not None and parent.type == "variable_declarator":
        for child in parent.children:
            if child.type == "identifier":
                return code[child.start_byte:child.end_byte]

    # exports.xxx = function() or module.exports.xxx = function()
    if parent is not None and parent.type == "assignment_expression":
        for child in parent.children:
            if child.type == "member_expression":
                for mc in reversed(child.children):
                    if mc.type in ("identifier", "property_identifier"):
                        return code[mc.start_byte:mc.end_byte]
                break

    return "<anonymous>"

# ---------------------------------------------------------------------------
# FALLBACK EXTRACTORS (used when tree-sitter parser is unavailable)
# ---------------------------------------------------------------------------
def _extract_functions_python_ast(file_path: str) -> List[Dict]:
    import ast as _ast
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = _ast.parse(source)
    except Exception:
        return []
    lines  = source.splitlines()
    funcs  = []
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end   = getattr(node, "end_lineno", start + 1)
            funcs.append({
                "file":       file_path,
                "code":       "\n".join(lines[start:end]),
                "name":       node.name,
                "language":   "python",
                "start_line": node.lineno,
            })
    return funcs


def _extract_functions_go_regex(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except Exception:
        return []
    funcs   = []
    pattern = re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE)
    for m in pattern.finditer(source):
        name       = m.group(1)
        start_pos  = m.start()
        start_line = source[:start_pos].count("\n") + 1
        body_start = source.find("{", m.end())
        if body_start == -1:
            continue
        depth, pos = 1, body_start + 1
        while pos < len(source) and depth > 0:
            if source[pos] == "{":
                depth += 1
            elif source[pos] == "}":
                depth -= 1
            pos += 1
        funcs.append({
            "file":       file_path,
            "code":       source[start_pos:pos],
            "name":       name,
            "language":   "go",
            "start_line": start_line,
        })
    return funcs


# ---------------------------------------------------------------------------
# CHUNK EXTRACTION
# ---------------------------------------------------------------------------
def extract_functions(file_path: str) -> List[Dict]:
    ext  = os.path.splitext(file_path)[1].lower()
    lang = EXT_TO_LANG.get(ext)
    if lang is None:
        return []

    # Use tree-sitter when available, fall back to language-specific parsers
    if lang in parsers:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        tree   = parsers[lang].parse(bytes(code, "utf8"))
        ftypes = LANG_FUNCTION_TYPES[lang]
        funcs  = []

        def walk(node, parent=None):
            if node.type in ftypes:
                name = _get_function_name(node, code, parent)
                funcs.append({
                    "file":       file_path,
                    "code":       code[node.start_byte:node.end_byte],
                    "name":       name,
                    "language":   lang,
                    "start_line": node.start_point[0] + 1,
                })
            for child in node.children:
                walk(child, node)

        walk(tree.root_node)
        return funcs

    if lang == "python":
        return _extract_functions_python_ast(file_path)
    if lang == "go":
        return _extract_functions_go_regex(file_path)
    return []


def build_chunks(app_path: str) -> List[Dict]:
    chunks = []
    for root, _, files in os.walk(app_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXT_TO_LANG:
                chunks.extend(extract_functions(os.path.join(root, f)))
    return chunks

# ---------------------------------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------------------------------
def _cache_path(app_path: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.md5(app_path.encode()).hexdigest() + "_rag.pkl")

def load_cache(app_path: str):
    p = _cache_path(app_path)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def save_cache(app_path: str, data):
    with open(_cache_path(app_path), "wb") as f:
        pickle.dump(data, f)

# ---------------------------------------------------------------------------
# VECTOR INDEX
# ---------------------------------------------------------------------------
def build_index(chunks: List[Dict]) -> Tuple:
    texts = [
        f"[{c['language']}] Function {c['name']} in {c['file']}:\n{c['code']}"
        for c in chunks
    ]
    embs = embedder.encode(texts, show_progress_bar=True).astype("float32")
    idx  = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx, embs

# ---------------------------------------------------------------------------
# CALL GRAPH LOADER
# ---------------------------------------------------------------------------
def load_call_graph(app_name: str) -> Dict[str, List[str]]:
    path = os.path.join(CALLGRAPHS_DIR, f"{app_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("adjacency", {})

def reachable_from(graph: Dict[str, List[str]], entry: str) -> Set[str]:
    visited: Set[str] = set()
    queue = deque([entry])
    while queue:
        fn = queue.popleft()
        if fn in visited:
            continue
        visited.add(fn)
        for callee in graph.get(fn, []):
            queue.append(callee)
            queue.append(callee.split(".")[-1])
    return visited

# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL
# ---------------------------------------------------------------------------
def retrieve_chunks(
    func_name:   str,
    permissions: List[str],
    chunks:      List[Dict],
    index,
    call_graph:  Dict[str, List[str]],
) -> List[Dict]:
    """
    Combines semantic search with call-graph expansion.
    Runs one query per permission plus one for the handler context.
    """
    queries = [
        f"Lambda handler function {func_name}. AWS SDK calls, helpers, env vars.",
    ]
    for perm in permissions:
        queries.append(
            f"AWS permission {perm}. SDK calls, wrappers, environment variables."
        )

    seen:      Set[int] = set()
    retrieved: List[Dict] = []

    for q in queries:
        q_emb = embedder.encode([q]).astype("float32")
        _, I  = index.search(q_emb, TOP_K)
        for i in I[0]:
            if i >= 0 and i not in seen:
                seen.add(i)
                retrieved.append(chunks[i])

    # Augment with chunks from call-graph-reachable functions
    reachable = reachable_from(call_graph, func_name)
    for c in chunks:
        if id(c) not in seen and c["name"] in reachable:
            seen.add(id(c))
            retrieved.append(c)

    # Cap total context to avoid prompt overflow
    return retrieved[: TOP_K * (len(permissions) + 1)]

# ---------------------------------------------------------------------------
# PROMPT  (split into system + user for /api/chat)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an AWS IAM security analyst. /no_think\n"
    "Given a Lambda function's statically detected SDK calls, initial permissions, "
    "and semantically retrieved source code chunks, determine the COMPLETE and MINIMAL "
    "set of IAM permissions the function actually requires. "
    "Rules: include a permission only if the code demonstrably uses that AWS API; "
    "remove over-provisioned permissions; add any missing ones, including indirect "
    "SDK usage via wrappers, helper modules, or environment variables. "
    "Reply with STRICT JSON only — no markdown, no explanation outside the JSON:\n"
    '{"permissions": ["service:Action", ...], "reasoning": "brief explanation"}'
)


def build_user_message(
    func_name:    str,
    permissions:  List[str],
    chunks:       List[Dict],
    call_details: List[Dict],
) -> str:
    context_text = "\n\n".join(
        f"[FILE: {c['file']} | LANG: {c['language']} | FUNC: {c['name']}]\n{c['code']}"
        for c in chunks
    )
    perms_block = "\n".join(f"  {p}" for p in permissions) or "  (none)"
    calls_block = "\n".join(
        f"  {cd['service']}.{cd['method']}({', '.join(cd.get('params_present', []))})"
        for cd in call_details
    ) or "  (none detected statically)"

    return (
        f"FUNCTION: {func_name}\n\n"
        f"AWS SDK CALLS DETECTED (static analysis):\n{calls_block}\n\n"
        f"INITIAL PERMISSIONS (static analysis):\n{perms_block}\n\n"
        f"RETRIEVED CODE CONTEXT (semantically relevant functions):\n{context_text}"
    )


# ---------------------------------------------------------------------------
# OLLAMA  (/api/chat — applies the model's native chat template)
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

    app_path   = os.path.join(DATASET_DIR, app_name)
    call_graph = load_call_graph(app_name)

    # Build / load FAISS index for this app
    cached = load_cache(app_path)
    if cached:
        print("  Loaded cached RAG index")
        chunks, index = cached
    else:
        chunks = build_chunks(app_path)
        if not chunks:
            print(f"  No source chunks found in {app_path}, skipping.")
            return {}
        print(f"  Building index over {len(chunks)} chunks...")
        index, _ = build_index(chunks)
        save_cache(app_path, (chunks, index))

    results: dict = {}

    for func_name, func_data in static_perms.items():
        initial     = func_data.get("baseline_permissions", [])
        call_details = func_data.get("call_details", [])

        context = retrieve_chunks(func_name, initial, chunks, index, call_graph)
        user_msg = build_user_message(func_name, initial, context, call_details)

        print(f"  Querying LLM for {func_name} ({len(context)} chunks retrieved)...")
        try:
            raw    = query_ollama(_SYSTEM_PROMPT, user_msg)
            parsed = safe_json_parse(raw)
        except Exception as e:
            parsed = {"error": str(e)}
            raw    = ""

        results[func_name] = {
            "initial_permissions": initial,
            "rag_permissions":     parsed.get("permissions", []),
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
            print(f"RAG validation: {app_name}  (skipping — output exists)")
            continue

        print(f"RAG validation: {app_name}")
        results = validate_app(app_name)
        if not results:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{app_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → {out_path}")

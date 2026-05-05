"""
build_call_graphs.py

Generates a per-app call graph for JavaScript, Python, and Go source files.

Output (one file per app):
  call_graphs/<app_name>.json
  {
    "adjacency": {"callerFunc": ["callee1", "callee2"], ...},
    "nodes": [{"name": "...", "file": "...", "language": "...", "line": N}, ...],
    "edges": [{"caller": "...", "callee": "...", "caller_file": "..."}, ...]
  }

Prerequisites (for JS/Go):
  tree-sitter grammar repos cloned into the project root:
    git clone https://github.com/tree-sitter/tree-sitter-javascript
    git clone https://github.com/tree-sitter/tree-sitter-python   # optional — stdlib ast is used instead
    git clone https://github.com/tree-sitter/tree-sitter-go
"""

import os
import ast
import json
import re
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset/apps"
OUTPUT_DIR = "call_graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXT_TO_LANG = {
    ".js": "javascript",
    ".ts": "javascript",
    ".py": "python",
    ".go": "go",
}

# ---------------------------------------------------------------------------
# TREE-SITTER SETUP (JS + Go only; Python uses stdlib ast)
# ---------------------------------------------------------------------------
_ts_parsers: Dict[str, object] = {}

def _init_tree_sitter():
    try:
        from tree_sitter import Language, Parser
    except ImportError:
        print("Warning: tree-sitter not installed. JS/Go call graphs will be skipped.")
        return

    lang_pkg_map = {
        "javascript": "tree_sitter_javascript",
        "go":         "tree_sitter_go",
    }

    for lang, pkg_name in lang_pkg_map.items():
        try:
            import importlib
            pkg = importlib.import_module(pkg_name)
            lang_obj = Language(pkg.language())
            _ts_parsers[lang] = Parser(lang_obj)
        except Exception as e:
            print(f"Warning: could not load tree-sitter parser for {lang}: {e}")

_init_tree_sitter()

# ---------------------------------------------------------------------------
# JAVASCRIPT / TYPESCRIPT  (tree-sitter)
# ---------------------------------------------------------------------------
_JS_FUNC_TYPES = {
    "function_declaration",
    "method_definition",
    "function_expression",
    "arrow_function",
}

def _js_func_name(node, code: str, parent=None) -> str:
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
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
                # get the last property in the member chain (e.g. "handler" from exports.handler)
                for mc in reversed(child.children):
                    if mc.type in ("identifier", "property_identifier"):
                        return code[mc.start_byte:mc.end_byte]
                break

    return "<anonymous>"

def _js_call_name(node, code: str) -> str:
    func_child = next((c for c in node.children if c.type == "function"), None)
    if func_child is None:
        # call_expression first child is the function
        if node.children:
            func_child = node.children[0]
    if func_child is None:
        return ""
    if func_child.type == "identifier":
        return code[func_child.start_byte:func_child.end_byte]
    if func_child.type == "member_expression":
        parts = []
        for c in func_child.children:
            if c.type in ("identifier", "property_identifier"):
                parts.append(code[c.start_byte:c.end_byte])
        return ".".join(parts)
    return ""

def _extract_js(file_path: str) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """Returns (function_nodes, call_edges)."""
    parser = _ts_parsers.get("javascript")
    if parser is None:
        return [], []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    tree = parser.parse(bytes(code, "utf8"))
    func_nodes = []
    call_edges = []  # (caller_name, callee_name)

    def walk(node, parent=None, enclosing_func: str = None):
        if node.type in _JS_FUNC_TYPES:
            name = _js_func_name(node, code, parent)
            func_nodes.append({
                "name": name,
                "file": file_path,
                "language": "javascript",
                "line": node.start_point[0] + 1,
            })
            # Recurse into the function body with updated enclosing context
            for child in node.children:
                walk(child, node, name)
            return  # already recursed

        if node.type == "call_expression" and enclosing_func:
            callee = _js_call_name(node, code)
            if callee and callee != enclosing_func:
                call_edges.append((enclosing_func, callee))

        for child in node.children:
            walk(child, node, enclosing_func)

    walk(tree.root_node)
    return func_nodes, call_edges


# ---------------------------------------------------------------------------
# PYTHON  (stdlib ast — always available)
# ---------------------------------------------------------------------------
def _extract_python(file_path: str) -> Tuple[List[dict], List[Tuple[str, str]]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return [], []

    func_nodes = []
    call_edges = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self._stack: List[str] = []  # enclosing function name stack

        def _caller(self) -> str:
            return self._stack[-1] if self._stack else None

        def _call_name(self, node: ast.Call) -> str:
            if isinstance(node.func, ast.Name):
                return node.func.id
            if isinstance(node.func, ast.Attribute):
                parts = []
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                return ".".join(reversed(parts))
            return ""

        def _visit_funcdef(self, node):
            func_nodes.append({
                "name": node.name,
                "file": file_path,
                "language": "python",
                "line": node.lineno,
            })
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        visit_FunctionDef = _visit_funcdef
        visit_AsyncFunctionDef = _visit_funcdef

        def visit_Call(self, node):
            caller = self._caller()
            if caller:
                callee = self._call_name(node)
                if callee and callee != caller:
                    call_edges.append((caller, callee))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return func_nodes, call_edges


# ---------------------------------------------------------------------------
# GO  (tree-sitter)
# ---------------------------------------------------------------------------
_GO_FUNC_TYPES = {"function_declaration", "method_declaration"}

def _go_func_name(node, code: str) -> str:
    for child in node.children:
        if child.type in ("identifier", "field_identifier"):
            return code[child.start_byte:child.end_byte]
    return "<anonymous>"

def _go_call_name(node, code: str) -> str:
    if not node.children:
        return ""
    func_child = node.children[0]
    if func_child.type == "identifier":
        return code[func_child.start_byte:func_child.end_byte]
    if func_child.type == "selector_expression":
        parts = []
        for c in func_child.children:
            if c.type in ("identifier", "field_identifier", "package_identifier", "type_identifier"):
                parts.append(code[c.start_byte:c.end_byte])
        return ".".join(parts)
    return ""

def _extract_go(file_path: str) -> Tuple[List[dict], List[Tuple[str, str]]]:
    parser = _ts_parsers.get("go")
    if parser is None:
        return [], []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    tree = parser.parse(bytes(code, "utf8"))
    func_nodes = []
    call_edges = []

    def walk(node, enclosing_func: str = None):
        if node.type in _GO_FUNC_TYPES:
            name = _go_func_name(node, code)
            func_nodes.append({
                "name": name,
                "file": file_path,
                "language": "go",
                "line": node.start_point[0] + 1,
            })
            for child in node.children:
                walk(child, name)
            return

        if node.type == "call_expression" and enclosing_func:
            callee = _go_call_name(node, code)
            if callee and callee != enclosing_func:
                call_edges.append((enclosing_func, callee))

        for child in node.children:
            walk(child, enclosing_func)

    walk(tree.root_node)
    return func_nodes, call_edges


# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------
_EXTRACTORS = {
    "javascript": _extract_js,
    "python": _extract_python,
    "go": _extract_go,
}

def extract_file(file_path: str) -> Tuple[List[dict], List[Tuple[str, str]]]:
    ext = os.path.splitext(file_path)[1].lower()
    lang = EXT_TO_LANG.get(ext)
    if lang is None:
        return [], []
    return _EXTRACTORS[lang](file_path)


# ---------------------------------------------------------------------------
# PER-APP CALL GRAPH
# ---------------------------------------------------------------------------
def build_call_graph(app_path: str) -> dict:
    all_nodes: List[dict] = []
    all_edges: List[Tuple[str, str]] = []  # (caller, callee, caller_file)
    edge_records: List[dict] = []

    for root, _, files in os.walk(app_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXT_TO_LANG:
                continue
            file_path = os.path.join(root, fname)
            nodes, edges = extract_file(file_path)
            all_nodes.extend(nodes)
            for caller, callee in edges:
                edge_records.append({
                    "caller": caller,
                    "callee": callee,
                    "caller_file": file_path,
                })

    # Build known function names for intra-app filtering
    known_funcs: Set[str] = {n["name"] for n in all_nodes}

    # Adjacency: only keep callees that are defined within the app
    adjacency: Dict[str, Set[str]] = {}
    filtered_edges = []
    for rec in edge_records:
        callee_base = rec["callee"].split(".")[-1]  # strip obj prefix for matching
        if callee_base in known_funcs or rec["callee"] in known_funcs:
            adjacency.setdefault(rec["caller"], set()).add(rec["callee"])
            filtered_edges.append(rec)

    return {
        "adjacency": {k: sorted(v) for k, v in sorted(adjacency.items())},
        "nodes": all_nodes,
        "edges": filtered_edges,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for app in sorted(os.listdir(DATASET_DIR)):
        app_path = os.path.join(DATASET_DIR, app)
        if not os.path.isdir(app_path):
            continue

        print(f"Building call graph for {app}...")
        graph = build_call_graph(app_path)

        out_path = os.path.join(OUTPUT_DIR, f"{app}.json")
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)

        n_funcs = len(graph["nodes"])
        n_edges = len(graph["edges"])
        print(f"  {n_funcs} functions, {n_edges} intra-app edges → {out_path}")

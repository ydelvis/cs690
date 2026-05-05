"""
extract_aws_calls.py

AST-based extraction of AWS SDK calls from each serverless app.

For every entry function defined in serverless.yml, this script:
  1. Resolves the handler to a source file + function name
  2. Traverses reachable functions via the call graph (BFS)
  3. Finds all AWS SDK calls in those functions
  4. Extracts parameter names and resolves static values where possible
     (literals, env vars from serverless.yml)

Output: aws_calls/<app>.json
  {
    "<func_name>": {
      "handler_file": "...",
      "handler_func": "...",
      "aws_calls": [
        {
          "service": "s3",
          "method": "copyObject",
          "params_present": ["Bucket", "CopySource", "Key"],
          "resolved_params": {"Bucket": "my-bucket"},
          "caller_func": "handler",
          "file": "handler.py",
          "line": 42
        }
      ]
    }
  }
"""

import ast
import json
import os
import re
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_DIR   = "dataset/apps"
CONFIGS_DIR   = "configs"
CALLGRAPHS_DIR = "call_graphs"
OUTPUT_DIR    = "aws_calls"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maps AWS service import/package name → canonical service key used in permissions.json
_SERVICE_ALIASES = {
    "s3":             "s3",
    "dynamodb":       "dynamodb",
    "kinesis":        "kinesis",
    "lambda":         "lambda",
    "sns":            "sns",
    "sqs":            "sqs",
    "ec2":            "ec2",
    "iam":            "iam",
    "sts":            "sts",
    "cloudwatch":     "cloudwatch",
    "cloudwatchlogs": "cloudwatch",
    "logs":           "cloudwatch",
    "elastictranscoder": "elastictranscoder",
    "rekognition":    "rekognition",
    "ses":            "ses",
    "s3manager":      "s3",
    "s3control":      "s3",
}

# ---------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------
def snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])

# DynamoDB DocumentClient uses short method names that map to the underlying API
_DOCLIENT_METHOD_MAP = {
    "put":         "putItem",
    "get":         "getItem",
    "delete":      "deleteItem",
    "update":      "updateItem",
    "scan":        "scan",
    "query":       "query",
    "batchGet":    "batchGetItem",
    "batchWrite":  "batchWriteItem",
    "transactGet": "transactGetItems",
    "transactWrite": "transactWriteItems",
}

def normalize_method(service: str, method: str) -> str:
    """Map short DocumentClient method names to full DynamoDB API names."""
    if service == "dynamodb":
        return _DOCLIENT_METHOD_MAP.get(method, method)
    return method

def normalize_service(s: str) -> str:
    return _SERVICE_ALIASES.get(s.lower(), s.lower())

def _strip_input_suffix(type_name: str) -> str:
    """'CopyObjectInput' -> 'copyObject'"""
    if type_name.endswith("Input"):
        base = type_name[:-5]
        return base[0].lower() + base[1:]
    return type_name[0].lower() + type_name[1:]

# ---------------------------------------------------------------------------
# CALL GRAPH LOADER
# ---------------------------------------------------------------------------
def load_call_graph(app_name: str) -> Dict[str, List[str]]:
    path = os.path.join(CALLGRAPHS_DIR, f"{app_name}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("adjacency", {})

def reachable_from(graph: Dict[str, List[str]], entry: str) -> Set[str]:
    visited: Set[str] = set()
    queue = deque([entry])
    while queue:
        func = queue.popleft()
        if func in visited:
            continue
        visited.add(func)
        for callee in graph.get(func, []):
            # strip obj prefix: "self.foo" -> "foo", "obj.method" -> "method"
            callee_base = callee.split(".")[-1]
            queue.append(callee)
            queue.append(callee_base)
    return visited

# ---------------------------------------------------------------------------
# HANDLER STRING RESOLUTION
# ---------------------------------------------------------------------------
def find_go_lambda_entry(go_file: str) -> Optional[str]:
    """Return the function name passed to lambda.Start() in a Go file."""
    with open(go_file, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    # lambda.Start(Handler) or lambda.Start(epsagon.WrapLambdaHandler(cfg, Handler))
    m = re.search(r"lambda\.Start\(\s*(?:\w+\.WrapLambdaHandler\([^,]+,\s*)?(\w+)", src)
    return m.group(1) if m else None


def resolve_handler(app_path: str, handler_str: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns (file_path, func_name, language).
    handler_str examples:
      'api/predict.handler'  -> api/predict.py, handler, python
      'handler.hello'        -> handler.js, hello, javascript
      'bin/handler'          -> main.go (Go), go
      'write/main'           -> write/main.go (Go), go
    """
    # Go: no dot in basename (binary path like "bin/handler" or "write/main")
    if "/" in handler_str and "." not in os.path.basename(handler_str):
        # 1. Direct match: handler_str + ".go" (e.g. "write/main" → "write/main.go")
        direct = os.path.join(app_path, handler_str + ".go")
        if os.path.exists(direct):
            return direct, find_go_lambda_entry(direct), "go"

        # 2. main.go in the handler's own directory (e.g. "write/main" → "write/main.go" already tried)
        # 3. Source dir named after the binary basename (e.g. "bin/rest" → "rest/main.go")
        binary_base = os.path.basename(handler_str)
        for src_dir in [binary_base, handler_str.split("/")[0]]:
            candidate = os.path.join(app_path, src_dir, "main.go")
            if os.path.exists(candidate):
                return candidate, find_go_lambda_entry(candidate), "go"

        # 4. Fallback: walk tree for any main.go
        for root, _, files in os.walk(app_path):
            for f in files:
                if f == "main.go":
                    fp = os.path.join(root, f)
                    return fp, find_go_lambda_entry(fp), "go"
        return None, None, "go"

    # split on LAST dot
    if "." in handler_str:
        file_part, func_name = handler_str.rsplit(".", 1)
    else:
        file_part = handler_str
        func_name = "handler"

    # Try each extension
    for ext, lang in [(".py", "python"), (".js", "javascript"), (".ts", "javascript")]:
        candidate = os.path.join(app_path, file_part + ext)
        if os.path.exists(candidate):
            return candidate, func_name, lang

    # Extension already included?
    candidate = os.path.join(app_path, file_part)
    if os.path.exists(candidate):
        if candidate.endswith(".py"):
            return candidate, func_name, "python"
        if candidate.endswith((".js", ".ts")):
            return candidate, func_name, "javascript"
        if candidate.endswith(".go"):
            return candidate, func_name, "go"

    return None, func_name, "unknown"

# ---------------------------------------------------------------------------
# PYTHON AWS CALL EXTRACTOR  (stdlib ast)
# ---------------------------------------------------------------------------
class _PyAWSExtractor(ast.NodeVisitor):
    def __init__(self, env_vars: dict):
        self.env_vars = env_vars
        self.module_clients: Dict[str, str] = {}  # var -> service
        self.func_calls: Dict[str, List[dict]] = {}  # func_name -> [call]
        self._stack: List[Tuple[str, Dict[str, str]]] = []  # (func_name, local_clients)

    # ------------------------------------------------------------------
    def _boto3_service(self, call_node: ast.Call) -> Optional[str]:
        """If call_node is boto3.client('svc') or boto3.resource('svc'), return svc."""
        func = call_node.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "boto3"
                and func.attr in ("client", "resource")):
            return None
        args = call_node.args
        if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            return normalize_service(args[0].value)
        # keyword: boto3.client(service_name='s3')
        for kw in call_node.keywords:
            if kw.arg == "service_name" and isinstance(kw.value, ast.Constant):
                return normalize_service(kw.value.value)
        return None

    def _resolve(self, node) -> Optional[str]:
        if isinstance(node, ast.Constant):
            return str(node.value)
        # os.environ['KEY'] or os.getenv('KEY')
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "os"
                    and node.value.attr == "environ"):
                slc = node.slice
                if isinstance(slc, ast.Constant):
                    return self.env_vars.get(slc.value, "${%s}" % slc.value)
        if isinstance(node, ast.Call):
            func = node.func
            # os.environ.get('KEY') or os.getenv('KEY')
            if isinstance(func, ast.Attribute) and func.attr in ("get", "getenv"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    key = node.args[0].value
                    return self.env_vars.get(key, "${%s}" % key)
        return None

    def _extract_params(self, call_node: ast.Call) -> Tuple[Set[str], dict]:
        """Extract keyword argument names + resolved values."""
        params_present: Set[str] = set()
        resolved: dict = {}
        for kw in call_node.keywords:
            if kw.arg:
                params_present.add(kw.arg)
                v = self._resolve(kw.value)
                if v is not None:
                    resolved[kw.arg] = v
        # first positional arg as dict literal
        for arg in call_node.args:
            if isinstance(arg, ast.Dict):
                for k in arg.keys:
                    if isinstance(k, ast.Constant):
                        params_present.add(str(k.value))
        return params_present, resolved

    def _current_clients(self) -> Dict[str, str]:
        if self._stack:
            return self._stack[-1][1]
        return self.module_clients

    def _current_func(self) -> Optional[str]:
        if self._stack:
            return self._stack[-1][0]
        return None

    # ------------------------------------------------------------------
    def visit_Assign(self, node: ast.Assign):
        clients = self._current_clients()
        if isinstance(node.value, ast.Call):
            svc = self._boto3_service(node.value)
            if svc:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        clients[t.id] = svc
            # Also: var = boto3.client('svc').get_paginator('method')
            # → treat the outer call's chained boto3.client as the service
            if (isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Call)):
                inner = node.value.func.value
                svc = self._boto3_service(inner)
                if svc:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            clients[t.id] = svc
        self.generic_visit(node)

    def _visit_funcdef(self, node):
        local_clients = dict(self.module_clients)
        self._stack.append((node.name, local_clients))
        self.func_calls.setdefault(node.name, [])
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef       = _visit_funcdef
    visit_AsyncFunctionDef  = _visit_funcdef

    def visit_Call(self, node: ast.Call):
        clients = self._current_clients()
        func_name = self._current_func() or "<module>"

        # ---- Pattern 1: client.method(...) where client is a tracked var ----
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var   = node.func.value.id
            meth  = node.func.attr
            if var in clients:
                svc = clients[var]
                params, resolved = self._extract_params(node)
                self.func_calls.setdefault(func_name, []).append({
                    "service":          svc,
                    "method":           snake_to_camel(meth),
                    "params_present":   sorted(params),
                    "resolved_params":  resolved,
                    "line":             node.lineno,
                })

        # ---- Pattern 2: boto3.client('svc').method(...) inline ----
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)):
            inner_call = node.func.value
            svc = self._boto3_service(inner_call)
            if svc:
                meth = node.func.attr
                params, resolved = self._extract_params(node)
                self.func_calls.setdefault(func_name, []).append({
                    "service":          svc,
                    "method":           snake_to_camel(meth),
                    "params_present":   sorted(params),
                    "resolved_params":  resolved,
                    "line":             node.lineno,
                })

        # ---- Pattern 3: boto3.client('svc').get_paginator('method') ----
        # handled by Pattern 2 above (meth = 'get_paginator', but we can
        # pull the actual method from the first arg)
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_paginator"
                and isinstance(node.func.value, ast.Call)):
            inner_call = node.func.value
            svc = self._boto3_service(inner_call)
            if svc and node.args and isinstance(node.args[0], ast.Constant):
                real_meth = node.args[0].value
                # Overwrite the last appended entry (from Pattern 2 above)
                entries = self.func_calls.get(func_name, [])
                if entries and entries[-1]["method"] == snake_to_camel("get_paginator"):
                    entries[-1]["method"] = snake_to_camel(real_meth)

        self.generic_visit(node)


def extract_python_aws_calls(file_path: str, env_vars: dict) -> Dict[str, List[dict]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=file_path)
    except SyntaxError:
        return {}
    visitor = _PyAWSExtractor(env_vars)
    # First pass: collect module-level clients
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            svc = visitor._boto3_service(node.value)
            if svc:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        visitor.module_clients[t.id] = svc
    visitor.visit(tree)
    return {k: v for k, v in visitor.func_calls.items() if v}

# ---------------------------------------------------------------------------
# JAVASCRIPT AWS CALL EXTRACTOR  (regex-based; tree-sitter used if available)
# ---------------------------------------------------------------------------
_AWS_INST_RE = re.compile(
    r"(?:const|let|var|this\.)\s*(\w+)\s*=\s*new\s+"
    r"(?:aws|AWS)\."
    r"([\w.]+)"
    r"\s*\(",
    re.MULTILINE,
)
_REQUIRE_INST_RE = re.compile(
    r"(?:const|let|var)\s*(\w+)\s*=\s*new\s+\("
    r"require\(['\"]aws-sdk['\"](?:\.clients\.\w+)?['\"]\)"
    r"(?:\.\s*([\w.]+))?\)\s*\(",
    re.MULTILINE,
)
_METHOD_CALL_RE = re.compile(
    r"\b(\w+)\s*\.\s*(\w+)\s*\(\s*(\{[^)]*?\})\s*\)",
    re.DOTALL,
)
_OBJECT_KEY_RE = re.compile(r"\b(\w+)\s*:")

# Known AWS service type names (from SDK v2 class names)
_JS_SERVICE_MAP = {
    "S3":                     "s3",
    "DynamoDB":               "dynamodb",
    "DynamoDB.DocumentClient": "dynamodb",
    "DocumentClient":         "dynamodb",
    "Kinesis":                "kinesis",
    "Lambda":                 "lambda",
    "SNS":                    "sns",
    "SQS":                    "sqs",
    "EC2":                    "ec2",
    "IAM":                    "iam",
    "CloudWatch":             "cloudwatch",
    "CloudWatchLogs":         "cloudwatch",
    "ElasticTranscoder":      "elastictranscoder",
    "Rekognition":            "rekognition",
    "SES":                    "ses",
}

def _js_module_exports_service(file_path: str) -> Optional[str]:
    """
    If a JS file does `module.exports = <aws_client_var>`, return the service name.
    Used to resolve cross-module require() patterns.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
    except OSError:
        return None

    clients: Dict[str, str] = {}
    for m in _AWS_INST_RE.finditer(src):
        svc = _JS_SERVICE_MAP.get(m.group(2).strip())
        if svc:
            clients[m.group(1)] = svc

    # module.exports = clientVar
    for m in re.finditer(r"module\.exports\s*=\s*(\w+)", src):
        var = m.group(1)
        if var in clients:
            return clients[var]
    return None


def _extract_js_object_params(src: str, open_brace_pos: int):
    """
    Given the position of an opening '{', return (param_names, resolved_params)
    by walking until the matching '}' at depth 0, handling nested braces.
    """
    params: Set[str] = set()
    resolved: dict = {}
    depth = 0
    i = open_brace_pos
    n = len(src)
    key_buf: List[str] = []
    reading_key = True

    while i < n:
        c = src[i]
        if c == "{":
            depth += 1
            if depth > 1:
                reading_key = False
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            reading_key = False
        elif depth == 1:
            if c in (":", ) and reading_key:
                key = "".join(key_buf).strip().strip("'\"")
                if key and re.match(r"^[A-Za-z]\w*$", key):
                    params.add(key)
                    # Try to resolve value: scan to next comma/} at depth 1
                    val_start = i + 1
                    j = i + 1
                    val_depth = 0
                    while j < n:
                        c2 = src[j]
                        if c2 in "{[(": val_depth += 1
                        elif c2 in "}])": val_depth -= 1
                        if val_depth < 0 or (val_depth == 0 and c2 == ","):
                            break
                        j += 1
                    val_str = src[val_start:j].strip()
                    # process.env.KEY
                    env_m = re.search(r"process\.env\.(\w+)", val_str)
                    if env_m:
                        pass  # resolved later with env_vars
                    # string literal
                    str_m = re.match(r"^['\"`]([^'\"` ]+)['\"`]$", val_str)
                    if str_m:
                        resolved[key] = str_m.group(1)
                key_buf = []
                reading_key = False
            elif c == ",":
                reading_key = True
                key_buf = []
            elif reading_key and c not in "{}[]()\n\r\t":
                key_buf.append(c)
        i += 1

    return sorted(params), resolved


def extract_js_aws_calls(file_path: str, env_vars: dict) -> Dict[str, List[dict]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()

    # Track: var_name -> service
    clients: Dict[str, str] = {}

    for m in _AWS_INST_RE.finditer(src):
        var_name  = m.group(1)
        type_path = m.group(2).strip()
        svc = _JS_SERVICE_MAP.get(type_path)
        if svc:
            clients[var_name] = svc

    # this.db = new AWS.DynamoDB.DocumentClient()
    for m in re.finditer(
        r"this\.(\w+)\s*=\s*new\s+(?:aws|AWS)\.([\w.]+)\s*\(",
        src, re.MULTILINE
    ):
        prop, type_path = m.group(1), m.group(2).strip()
        svc = _JS_SERVICE_MAP.get(type_path)
        if svc:
            clients[f"this.{prop}"] = svc
            clients[prop] = svc

    # const mod = require('./path') where path.js exports an AWS client
    base_dir = os.path.dirname(file_path)
    for m in re.finditer(
        r"(?:const|let|var)\s+(\w+)\s*=\s*require\(['\"](\.[^'\"]+)['\"]\)",
        src, re.MULTILINE
    ):
        var_name = m.group(1)
        rel_path = m.group(2)
        for ext in (".js", ".ts", ""):
            candidate = os.path.normpath(os.path.join(base_dir, rel_path + ext))
            if os.path.exists(candidate):
                svc = _js_module_exports_service(candidate)
                if svc:
                    clients[var_name] = svc
                break

    if not clients:
        return {}

    result: Dict[str, List[dict]] = {}
    _JS_SKIP_WORDS = frozenset([
        "if", "else", "return", "const", "let", "var", "true", "false",
        "null", "undefined", "new", "this", "function", "async", "await",
    ])

    # Find var.method( — with either an object literal OR a variable argument
    var_alt = "|".join(re.escape(v) for v in clients)
    call_pat = re.compile(
        rf"\b({var_alt})\s*\.\s*(\w+)\s*\(",
        re.MULTILINE,
    )

    for m in call_pat.finditer(src):
        var    = m.group(1)
        method = m.group(2)
        svc    = clients.get(var)
        if not svc or method.startswith("_") or method in ("promise", "then", "catch"):
            continue

        params: List[str] = []
        resolved: dict = {}

        # Look at what follows the opening paren
        after_paren = src[m.end():].lstrip()
        if after_paren.startswith("{"):
            # Object literal — extract keys
            brace_pos = m.end() + (src[m.end():].index("{"))
            params, resolved = _extract_js_object_params(src, brace_pos)
            params = [p for p in params if p not in _JS_SKIP_WORDS]
        # else: variable arg — params list stays empty (we still record the call)

        # Resolve env vars from nearby context
        # Scan ahead to find process.env references
        call_end = m.end()
        depth = 1
        i = call_end
        while i < len(src) and depth > 0:
            if src[i] == "(": depth += 1
            elif src[i] == ")": depth -= 1
            i += 1
        call_text = src[m.end():i]
        for param_name, env_key in re.findall(
            r"\b(\w+)\s*:\s*process\.env\.(\w+)", call_text
        ):
            resolved[param_name] = env_vars.get(env_key, "${%s}" % env_key)

        func_name = _find_enclosing_js_func(src, m.start()) or "<module>"

        result.setdefault(func_name, []).append({
            "service":         svc,
            "method":          normalize_method(svc, method),
            "params_present":  params,
            "resolved_params": resolved,
            "line":            src[:m.start()].count("\n") + 1,
        })

    return result

def _find_enclosing_js_func(src: str, pos: int) -> Optional[str]:
    """Heuristic: find nearest function definition before pos."""
    chunk = src[:pos]
    last_func = None
    for m in re.finditer(
        r"(?:function\s+(\w+)"
        r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()"
        r"|(?:exports|module\.exports)\.(\w+)\s*=\s*(?:async\s+)?(?:function|\())",
        chunk,
    ):
        last_func = m.group(1) or m.group(2) or m.group(3)
    return last_func

# ---------------------------------------------------------------------------
# GO AWS CALL EXTRACTOR  (tree-sitter or regex)
# ---------------------------------------------------------------------------
def _extract_go_struct_body(src: str, open_brace_pos: int) -> str:
    """Return the content between matching braces at open_brace_pos."""
    depth = 0
    i = open_brace_pos
    n = len(src)
    while i < n:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace_pos + 1:i]
        i += 1
    return src[open_brace_pos + 1:]


def _go_struct_top_level_fields(body: str) -> List[str]:
    """
    Extract top-level field names from a Go struct literal body.
    Skips nested struct literals to avoid picking up inner field names.
    """
    fields: List[str] = []
    depth = 0
    reading_key = True
    key_buf: List[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c in "{([":
            depth += 1
            reading_key = False
        elif c in "})]":
            depth -= 1
            if depth == 0:
                reading_key = True
                key_buf = []
        elif depth == 0:
            if c == ":" and reading_key:
                key = "".join(key_buf).strip()
                if key and re.match(r"^[A-Z]\w*$", key):
                    fields.append(key)
                key_buf = []
                reading_key = False
            elif c == ",":
                reading_key = True
                key_buf = []
            elif reading_key and c not in "\n\r\t":
                key_buf.append(c)
        i += 1
    return fields


def extract_go_aws_calls(file_path: str, env_vars: dict) -> Dict[str, List[dict]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()

    result: Dict[str, List[dict]] = {}

    # Find import aliases: "github.com/aws/aws-sdk-go/service/s3" -> "s3"
    # Also handles sub-packages (s3/s3manager) and aws-sdk-go-v2
    import_aliases: Dict[str, str] = {}
    for m in re.finditer(
        r'(?:(\w+)\s+)?"github\.com/aws/aws-sdk-go(?:-v2)?/service/([\w/]+)"',
        src
    ):
        full_path = m.group(2)            # e.g. "s3", "s3/s3manager", "dynamodb"
        pkg_name  = full_path.split("/")[-1]  # last component: "s3", "s3manager", "dynamodb"
        alias     = m.group(1) or pkg_name
        service   = normalize_service(pkg_name)
        import_aliases[alias] = service

    if not import_aliases:
        return {}

    # Find struct literals: &servicePkg.MethodNameInput{...}
    for service_pkg, service in import_aliases.items():
        struct_pattern = re.compile(
            rf"&?{re.escape(service_pkg)}\.(\w+Input)\s*\{{",
        )
        for m in struct_pattern.finditer(src):
            type_name = m.group(1)
            method    = _strip_input_suffix(type_name)
            body      = _extract_go_struct_body(src, m.end() - 1)
            params    = _go_struct_top_level_fields(body)

            resolved: dict = {}
            for field, env_key in re.findall(
                r'\b(\w+)\s*:.*?os\.Getenv\(["\'](\w+)["\']\)', body
            ):
                resolved[field] = env_vars.get(env_key, "${%s}" % env_key)
            for field, val in re.findall(
                r'\b(\w+)\s*:.*?aws\.String\(["\']([^"\']+)["\']\)', body
            ):
                resolved[field] = val

            func_name = _find_enclosing_go_func(src, m.start()) or "main"
            result.setdefault(func_name, []).append({
                "service":         service,
                "method":          method,
                "params_present":  sorted(set(params)),
                "resolved_params": resolved,
                "line":            src[:m.start()].count("\n") + 1,
            })

    return result

def _find_enclosing_go_func(src: str, pos: int) -> Optional[str]:
    chunk = src[:pos]
    last = None
    for m in re.finditer(r"\bfunc\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", chunk):
        last = m.group(1)
    return last

# ---------------------------------------------------------------------------
# LANGUAGE DISPATCH
# ---------------------------------------------------------------------------
def extract_aws_calls_from_file(
    file_path: str,
    env_vars: dict,
    language: str,
) -> Dict[str, List[dict]]:
    """Returns {func_name: [aws_call_dict, ...]}"""
    try:
        if language == "python":
            return extract_python_aws_calls(file_path, env_vars)
        if language == "javascript":
            return extract_js_aws_calls(file_path, env_vars)
        if language == "go":
            return extract_go_aws_calls(file_path, env_vars)
    except Exception as e:
        print(f"    Warning: error parsing {file_path}: {e}")
    return {}

_EXT_LANG = {".py": "python", ".js": "javascript", ".ts": "javascript", ".go": "go"}

# ---------------------------------------------------------------------------
# APP-LEVEL ANALYSIS
# ---------------------------------------------------------------------------
def analyze_app(app_name: str) -> dict:
    app_path = os.path.join(DATASET_DIR, app_name)

    # Load config
    config_path = os.path.join(CONFIGS_DIR, f"{app_name}.json")
    if not os.path.exists(config_path):
        print(f"  No config found for {app_name}, skipping.")
        return {}
    with open(config_path) as f:
        config = json.load(f)

    env_vars   = config.get("environment", {})
    functions  = config.get("functions", {})
    call_graph = load_call_graph(app_name)

    # Pre-index: all source files + their AWS calls (lazy, file-level)
    file_calls_cache: Dict[str, Dict[str, List[dict]]] = {}

    def get_file_calls(fp: str, lang: str) -> Dict[str, List[dict]]:
        if fp not in file_calls_cache:
            file_calls_cache[fp] = extract_aws_calls_from_file(fp, env_vars, lang)
        return file_calls_cache[fp]

    # Build a global function -> (file, lang) index for the whole app
    func_to_file: Dict[str, Tuple[str, str]] = {}
    for root, _, files in os.walk(app_path):
        for fname in files:
            ext  = os.path.splitext(fname)[1].lower()
            lang = _EXT_LANG.get(ext)
            if not lang:
                continue
            fp = os.path.join(root, fname)
            calls_in_file = get_file_calls(fp, lang)
            for func in calls_in_file:
                func_to_file[func] = (fp, lang)

    results: dict = {}

    for sls_func_name, func_cfg in functions.items():
        handler_str = func_cfg.get("handler", "")
        if not handler_str:
            continue

        file_path, handler_func, lang = resolve_handler(app_path, handler_str)
        if not file_path:
            continue

        # Reachable functions from the entry point
        reachable = reachable_from(call_graph, handler_func or sls_func_name)
        reachable.add(handler_func or sls_func_name)

        # For Python: class __init__ methods may be invoked at module load time via
        # decorators (e.g. @SetupModel applies SetupModel.__init__ before the handler
        # is ever called). Include them so decorator-time AWS calls are captured.
        if lang == "python":
            reachable |= reachable_from(call_graph, "__init__")

        # Pre-extract the handler file so we know which func names live there
        handler_file_calls = get_file_calls(file_path, lang)
        handler_file_funcs = set(handler_file_calls.keys())

        # Module-level calls always come from the handler file only
        reachable_for_handler = set(reachable) | {"<module>"}

        all_aws_calls: List[dict] = []

        for func_name in reachable_for_handler:
            # Use the handler file for:
            #  (a) functions actually defined there, OR
            #  (b) <module> scope, OR
            #  (c) functions whose name is also in the handler file (avoid cross-contamination)
            if func_name in handler_file_funcs or func_name == "<module>":
                fp, fl = file_path, lang
            elif func_name in func_to_file:
                fp, fl = func_to_file[func_name]
            else:
                fp, fl = file_path, lang

            calls_in_file = get_file_calls(fp, fl)
            for call in calls_in_file.get(func_name, []):
                all_aws_calls.append({
                    **call,
                    "caller_func": func_name,
                    "file":        os.path.relpath(fp, app_path),
                })

        if all_aws_calls:
            results[sls_func_name] = {
                "handler_file": os.path.relpath(file_path, app_path) if file_path else None,
                "handler_func": handler_func,
                "language":     lang,
                "aws_calls":    all_aws_calls,
            }

    return results

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for app in sorted(os.listdir(DATASET_DIR)):
        app_path = os.path.join(DATASET_DIR, app)
        if not os.path.isdir(app_path):
            continue

        print(f"Extracting AWS calls: {app}")
        result = analyze_app(app)

        out_path = os.path.join(OUTPUT_DIR, f"{app}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        total_calls = sum(len(v["aws_calls"]) for v in result.values())
        print(f"  {len(result)} entry functions, {total_calls} AWS calls → {out_path}")

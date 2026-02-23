from __future__ import annotations

import os
import re
import json
import ast
import operator
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


EXTERNAL_PATTERNS = [
    r"wikipedia",
    r"youtube",
    r"google",
    r"\bsearch\b",
    r"according to",
    r"website",
    r"openreview",
    r"\bnih\b",
    r"nature\.com",
    r"arxiv",
    r"view history",
    r"pdf online",
    r"uploaded by",
    r"\bhttp(s)?://",
    r"\bwww\.",
]

AUDIO_PATTERNS = [
    r"\.mp3\b",
    r"\.wav\b",
    r"\baudio\b",
    r"\brecording\b",
]

IMAGE_PATTERNS = [
    r"\.jpg\b",
    r"\.png\b",
    r"\.jpeg\b",
    r"\bimage\b",
    r"\bphoto\b",
]



FINAL_RE = re.compile(
    r"(?:^|\n)\s*(?:FINAL\s+ANSWER|Final\s+Answer|Answer)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

TOOL_CALL_RE = re.compile(
    r"^\s*(?:CALL|call)\s*:\s*([a-zA-Z_]\w*)\s*:\s*(.+?)\s*$"
)

PLAN_LINE_RE = re.compile(r"^\s*\d+\.\s*(.+)", re.MULTILINE)


def extract_final_answer(text: str) -> str:
    """Extracts the final answer line from model output or returns last non-empty line as fallback."""
    if not text:
        return ""
    m = FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_plan_steps(text: str) -> List[str]:
    """Extracts numbered plan steps from the model-generated planning text."""
    return PLAN_LINE_RE.findall(text or "")


def extract_tool_calls(text: str) -> List[Tuple[str, str]]:
    """Parses and returns tool calls emitted by the model in CALL: format."""
    calls: List[Tuple[str, str]] = []
    if not text:
        return calls
    for ln in text.splitlines():
        m = TOOL_CALL_RE.match(ln.strip())
        if m:
            calls.append((m.group(1), m.group(2)))
    return calls


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Any:
    """Safely evaluates an AST node using a restricted operator allowlist."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsafe / unsupported expression")


def safe_calculator(expr: str) -> Any:
    """Parses and safely evaluates a mathematical expression string."""
    tree = ast.parse(expr.strip(), mode="eval")
    return _safe_eval(tree)

class Scratchpad:
    def __init__(self, max_chars: int = 14000):
        self.entries: List[str] = []
        self.max_chars = int(max_chars)

    def add(self, text: str):
        """Adds text to the scratchpad while enforcing max character limit."""
        if text is None:
            return
        self.entries.append(str(text))
        self._truncate()

    def clear(self):
        """Clears all scratchpad entries for a new task."""
        self.entries = []

    def _truncate(self):
        """Truncates scratchpad content to maintain max character constraint."""
        s = "\n".join(self.entries)
        if len(s) <= self.max_chars:
            return
        # keep only the tail
        tail = s[-self.max_chars :]
        self.entries = [tail]

    def __str__(self) -> str:
        """Returns full scratchpad content as a single string."""
        return "\n".join(self.entries)


def _looks_like_json(s: str) -> bool:
    """Checks whether a string appears to be JSON formatted."""
    s = (s or "").strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _parse_arg_maybe_json(arg: str) -> Any:
    """Attempts to parse a tool argument as JSON, otherwise returns raw string."""
    arg = (arg or "").strip()
    if _looks_like_json(arg):
        try:
            return json.loads(arg)
        except Exception:
            return arg
    return arg


def _is_safe_query(expr: str) -> bool:
    """
    Conservative allowlist for pandas .query() expressions.
    Blocks: backticks, semicolons, brackets/braces, @, __, import, eval/exec, os/sys access.
    """
    if not expr:
        return False
    bad = ["`", ";", "{", "}", "[", "]", "@", "__", "import", "os.", "sys.", "eval", "exec"]
    low = expr.lower()
    if any(b in low for b in bad):
        return False
    allowed = re.compile(r"^[\w\s\.\'\"\(\)\=\!\<\>\&\|\+\-\*\/%]+$")
    return bool(allowed.match(expr.strip()))


class DataFrameStore:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.source: Optional[str] = None

    def load(self, path: str, filename: str, sheet: Optional[str] = None) -> str:
        """Loads a CSV/TSV/Excel file into memory and returns summary information."""
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
        elif ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            self.df = pd.read_csv(path, sep=sep)
        else:
            return "[ERROR: Not a tabular file (.xlsx/.xls/.csv/.tsv)]"

        self.source = filename
        return self.summary()

    def summary(self) -> str:
        """Returns basic summary including shape, columns, and head preview."""
        if self.df is None:
            return "[No dataframe loaded]"
        df = self.df
        cols = list(df.columns)
        shape = df.shape
        head = df.head(20).to_string(index=False)
        return (
            f"[DF LOADED from {self.source}] shape={shape}\n"
            f"columns={cols}\n"
            f"head(20):\n{head}"
        )

    def head(self, n: int = 20) -> str:
        """Returns the first n rows of the loaded dataframe."""
        if self.df is None:
            return "[No dataframe loaded]"
        n = max(1, min(int(n), 100))
        return self.df.head(n).to_string(index=False)

    def columns(self) -> str:
        """Returns dataframe column names as JSON string."""
        if self.df is None:
            return "[No dataframe loaded]"
        return json.dumps(list(self.df.columns), ensure_ascii=False)

    def shape(self) -> str:
        """Returns dataframe shape as string tuple."""
        if self.df is None:
            return "[No dataframe loaded]"
        return str(tuple(self.df.shape))

    def filter_query(self, expr: str, limit: int = 50) -> str:
        """Filters dataframe rows using a validated pandas query expression."""
        if self.df is None:
            return "[No dataframe loaded]"
        if not _is_safe_query(expr):
            return "[ERROR: Unsafe filter expression]"
        try:
            out = self.df.query(expr)
        except Exception as e:
            return f"[ERROR: query failed: {e}]"
        limit = max(1, min(int(limit), 200))
        return out.head(limit).to_string(index=False)

    def value_counts(self, column: str, limit: int = 50) -> str:
        """Returns value frequency counts for a specified column."""
        if self.df is None:
            return "[No dataframe loaded]"
        if column not in self.df.columns:
            return f"[ERROR: Unknown column '{column}']"
        vc = self.df[column].value_counts(dropna=False).head(max(1, min(int(limit), 200)))
        return vc.to_string()

    def sum_column(self, column: str, where: Optional[str] = None) -> str:
        """Computes sum of a numeric column optionally filtered by condition."""
        if self.df is None:
            return "[No dataframe loaded]"
        if column not in self.df.columns:
            return f"[ERROR: Unknown column '{column}']"
        df = self.df
        if where:
            if not _is_safe_query(where):
                return "[ERROR: Unsafe where expression]"
            try:
                df = df.query(where)
            except Exception as e:
                return f"[ERROR: where query failed: {e}]"
        try:
            s = pd.to_numeric(df[column], errors="coerce").sum()
        except Exception as e:
            return f"[ERROR: sum failed: {e}]"
        return str(s)

    def count_rows(self, where: Optional[str] = None) -> str:
        """Returns row count optionally filtered by condition."""
        if self.df is None:
            return "[No dataframe loaded]"
        df = self.df
        if where:
            if not _is_safe_query(where):
                return "[ERROR: Unsafe where expression]"
            try:
                df = df.query(where)
            except Exception as e:
                return f"[ERROR: where query failed: {e}]"
        return str(len(df))



SUPPORTED_TEXT_EXT = {".txt", ".md", ".json", ".log", ".csv", ".tsv", ".xlsx", ".xls"}


def _read_text_snippet(path: str, max_chars: int = 8000) -> str:
    """Reads and returns a safe preview snippet from supported file types."""
    ext = os.path.splitext(path.lower())[1]
    if ext not in SUPPORTED_TEXT_EXT:
        return f"[Skipped unsupported file type: {os.path.basename(path)}]"

    if ext == ".json":
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
            return json.dumps(obj, indent=2, ensure_ascii=False)[:max_chars]
        except Exception as e:
            return f"[ERROR reading json: {e}]"

    if ext in [".txt", ".md", ".log"]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(max_chars)
        except Exception as e:
            return f"[ERROR reading text: {e}]"

    # tabular preview
    if ext in [".csv", ".tsv"]:
        try:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            return df.head(50).to_string(index=False)[:max_chars]
        except Exception as e:
            return f"[ERROR reading table: {e}]"

    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(path)
            return df.head(50).to_string(index=False)[:max_chars]
        except Exception as e:
            return f"[ERROR reading table: {e}]"

    return f"[Unsupported read_file for ext {ext}]"


class ToolAPIProxy:
    """
    Tools (emit in model output as):
      CALL: calculator: <expr>
      CALL: read_file: <filename>
      CALL: load_table: <filename>   (or JSON {"filename":"...","sheet":"..."})
      CALL: df_summary: {}
      CALL: df_head: {"n": 20}
      CALL: df_columns: {}
      CALL: df_shape: {}
      CALL: df_filter: {"expr":"...","limit":50}
      CALL: df_value_counts: {"column":"...","limit":50}
      CALL: df_sum: {"column":"...","where":"..."}
      CALL: df_count: {"where":"..."}
    """

    def __init__(self, files_map: Dict[str, str]):
        self.files_map = files_map or {}
        self.df_store = DataFrameStore()

    def set_files_map(self, files_map: Dict[str, str]):
        """Updates internal file mapping used for tool resolution."""
        self.files_map = files_map or {}

    def _resolve_file(self, filename: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Resolves filename to path using internal file map."""
        if filename not in self.files_map:
            return False, None, f"Unknown file '{filename}'"
        return True, self.files_map[filename], None

    def call(self, name: str, argument: str) -> Tuple[bool, str, Optional[str]]:
        """Dispatches and executes a supported tool call."""
        try:
            # calculator
            if name == "calculator":
                val = safe_calculator(argument)
                return True, str(val), None

            # read_file
            if name == "read_file":
                fn = argument.strip().strip('"').strip("'")
                ok, path, err = self._resolve_file(fn)
                if not ok:
                    return False, "", err
                return True, _read_text_snippet(path), None

            # load_table
            if name == "load_table":
                parsed = _parse_arg_maybe_json(argument)
                if isinstance(parsed, dict):
                    fn = str(parsed.get("filename", "")).strip()
                    sheet = parsed.get("sheet")
                else:
                    fn = str(parsed).strip().strip('"').strip("'")
                    sheet = None

                ok, path, err = self._resolve_file(fn)
                if not ok:
                    return False, "", err
                return True, self.df_store.load(path=path, filename=fn, sheet=sheet), None

            # dataframe helpers
            if name == "df_summary":
                return True, self.df_store.summary(), None

            if name == "df_head":
                parsed = _parse_arg_maybe_json(argument)
                n = 20
                if isinstance(parsed, dict) and "n" in parsed:
                    n = int(parsed["n"])
                return True, self.df_store.head(n=n), None

            if name == "df_columns":
                return True, self.df_store.columns(), None

            if name == "df_shape":
                return True, self.df_store.shape(), None

            if name == "df_filter":
                parsed = _parse_arg_maybe_json(argument)
                if not isinstance(parsed, dict):
                    return False, "", 'df_filter expects JSON like {"expr":"col==\'x\'","limit":50}'
                expr = str(parsed.get("expr", "")).strip()
                limit = int(parsed.get("limit", 50))
                return True, self.df_store.filter_query(expr=expr, limit=limit), None

            if name == "df_value_counts":
                parsed = _parse_arg_maybe_json(argument)
                if not isinstance(parsed, dict):
                    return False, "", 'df_value_counts expects JSON like {"column":"col","limit":50}'
                col = str(parsed.get("column", "")).strip()
                limit = int(parsed.get("limit", 50))
                return True, self.df_store.value_counts(column=col, limit=limit), None

            if name == "df_sum":
                parsed = _parse_arg_maybe_json(argument)
                if not isinstance(parsed, dict):
                    return False, "", 'df_sum expects JSON like {"column":"Sales","where":"Type==\'Food\'"}'
                col = str(parsed.get("column", "")).strip()
                where = parsed.get("where")
                where = str(where).strip() if where is not None else None
                return True, self.df_store.sum_column(column=col, where=where), None

            if name == "df_count":
                parsed = _parse_arg_maybe_json(argument)
                if isinstance(parsed, dict):
                    where = parsed.get("where")
                    where = str(where).strip() if where is not None else None
                else:
                    where = None
                return True, self.df_store.count_rows(where=where), None

            return False, "", f"Tool '{name}' not found"

        except Exception as e:
            return False, "", str(e)

class AgentConfig:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        # loop controls
        max_steps: int = 12,
        max_tool_calls_per_step: int = 6,
        scratchpad_max_chars: int = 14000,
        # tool auto-detection
        auto_load_table: bool = True,
        auto_read_text: bool = True,
        # debug
        debug: bool = False,
    ):
        self.model_id = model_id
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.max_steps = int(max_steps)
        self.max_tool_calls_per_step = int(max_tool_calls_per_step)
        self.scratchpad_max_chars = int(scratchpad_max_chars)
        self.auto_load_table = bool(auto_load_table)
        self.auto_read_text = bool(auto_read_text)
        self.debug = bool(debug)

class HFLLM:
    def __init__(self, model_id: str, temperature: float, top_p: float, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        )

        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generates model response using HuggingFace chat template."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with self.torch.no_grad():
            out = self.model.generate(**gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

class StrictClosedWorldR2A2Agent:
    SYSTEM_PLAN = """
You are solving ONE benchmark task.

You are given the QUESTION text.
You are NOT designing a benchmark.
You must solve the specific QUESTION as stated.

Closed-world rules:
- No internet, no browsing, no external websites.
- No audio/image processing.
- You MAY use tools (calculator / read_file / load_table / dataframe helpers)
  by emitting lines exactly like:
    CALL: tool_name: argument

Write a numbered step-by-step plan to solve THIS QUESTION.
Do NOT solve it yet.
"""

    SYSTEM_ACT = """
You are solving the given QUESTION in a CLOSED-WORLD setting.

Rules:
- Do NOT ask for more information.
- Do NOT request clarification.
- Use tools when needed by emitting lines:
  CALL: <tool_name>: <argument>

If you have enough information, output EXACTLY:
FINAL ANSWER: <answer>

If not enough information, call tools to get what you need.
"""

    SYSTEM_REFLECT = """
You are a reflection module.
Given the question, plan, last output, and scratchpad:
- Decide STOP if we can answer now.
- Decide CONTINUE if more tool use / reasoning is needed.
Output exactly two lines:
DECISION: STOP or CONTINUE
NEXT: <1 short concrete next action>
"""

    def __init__(self, cfg: AgentConfig, files_map: Dict[str, str]):
        self.cfg = cfg
        self.model = HFLLM(cfg.model_id, cfg.temperature, cfg.top_p, cfg.max_new_tokens)
        self.tools = ToolAPIProxy(files_map or {})
        self.files_map: Dict[str, str] = files_map or {}
        self.scratchpad = Scratchpad(max_chars=cfg.scratchpad_max_chars)


    @staticmethod
    def _extract_question(sample: Dict[str, Any]) -> str:
        """Extracts question text from various possible dataset keys."""
        for key in ["question", "Question", "prompt", "Prompt"]:
            val = sample.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        # some datasets use "query"
        val = sample.get("query")
        if val is not None and str(val).strip():
            return str(val).strip()
        return ""

    @staticmethod
    def _extract_files(sample: Dict[str, Any]) -> Dict[str, str]:
        """Extracts file mapping dictionary from dataset sample."""
        files = sample.get("files")
        if isinstance(files, dict):
            return files
        return {}


    def _refuse(self, reason: str, audit: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates refusal response with audit trail."""
        audit.append({"phase": "refusal", "reason": reason})
        return {"answer": "UNSUPPORTED_TASK", "audit_trail": audit}

    def _capability_gate(self, question: str, audit: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Performs closed-world capability checks and blocks unsupported tasks."""
        ql = (question or "").lower()

        if not question.strip():
            return self._refuse("Empty question provided to agent (input plumbing issue).", audit)

        if any(re.search(p, ql) for p in EXTERNAL_PATTERNS):
            return self._refuse("External web access required (closed-world).", audit)

        if any(re.search(p, ql) for p in AUDIO_PATTERNS):
            return self._refuse("Audio processing not supported (closed-world).", audit)

        if any(re.search(p, ql) for p in IMAGE_PATTERNS):
            return self._refuse("Image processing not supported (closed-world).", audit)

        return None

    def _auto_tools(self, audit: List[Dict[str, Any]]):
        """Automatically loads initial table or text file if available."""
        if not self.files_map:
            return

        if self.cfg.auto_load_table:
            for fn, path in self.files_map.items():
                ext = os.path.splitext(path.lower())[1]
                if ext in [".xlsx", ".xls", ".csv", ".tsv"]:
                    ok, res, err = self.tools.call("load_table", fn)
                    audit.append({
                        "phase": "auto_tool",
                        "tool": "load_table",
                        "arg": fn,
                        "success": ok,
                        "error": err,
                    })
                    if ok:
                        self.scratchpad.add(f"[AUTO load_table {fn}]\n{res}")
                    else:
                        self.scratchpad.add(f"[AUTO load_table ERROR {fn}] {err}")
                    # only load one by default
                    break

        if self.cfg.auto_read_text:
            for fn, path in self.files_map.items():
                ext = os.path.splitext(path.lower())[1]
                if ext in [".txt", ".md", ".json", ".log"]:
                    ok, res, err = self.tools.call("read_file", fn)
                    audit.append({
                        "phase": "auto_tool",
                        "tool": "read_file",
                        "arg": fn,
                        "success": ok,
                        "error": err,
                    })
                    if ok:
                        self.scratchpad.add(f"[AUTO read_file {fn}]\n{res}")
                    else:
                        self.scratchpad.add(f"[AUTO read_file ERROR {fn}] {err}")
                    break


    def _compose_context(self, question: str, plan_text: str, last_output: str = "") -> str:
        """Composes structured context prompt for LLM phases."""
        files_txt = ", ".join(self.files_map.keys()) if self.files_map else "(none)"
        sp = str(self.scratchpad)
        return (
            f"QUESTION:\n{question}\n\n"
            f"AVAILABLE_FILES:\n{files_txt}\n\n"
            f"PLAN:\n{plan_text}\n\n"
            f"SCRATCHPAD:\n{sp}\n\n"
            f"LAST_OUTPUT:\n{last_output}\n"
        ).strip()


    def _act_once(
        self,
        question: str,
        plan_text: str,
        last_output: str,
        audit: List[Dict[str, Any]],
        iteration: int,
    ) -> Tuple[str, Optional[str]]:
        """Executes one act phase including tool execution and answer detection."""
        act_prompt = self._compose_context(question, plan_text, last_output=last_output)
        raw = self.model.generate(self.SYSTEM_ACT, act_prompt)

        audit.append({
            "phase": "act",
            "iteration": iteration,
            "raw_output": raw,
        })

        # If model produced a FINAL ANSWER line, stop.
        if "FINAL ANSWER" in raw.upper():
            fa = extract_final_answer(raw)
            return raw, fa if fa else ""

        # Otherwise, execute tool calls, capped.
        calls = extract_tool_calls(raw)
        if calls:
            calls = calls[: self.cfg.max_tool_calls_per_step]

        for tool_name, arg in calls:
            ok, res, err = self.tools.call(tool_name, arg)
            audit.append({
                "phase": "tool",
                "iteration": iteration,
                "tool": tool_name,
                "arg": arg,
                "success": ok,
                "error": err,
            })
            if ok:
                self.scratchpad.add(f"[TOOL {tool_name} RESULT]\n{res}")
            else:
                self.scratchpad.add(f"[TOOL {tool_name} ERROR]\n{err}")

        return raw, None


    def _reflect(
        self,
        question: str,
        plan_text: str,
        last_output: str,
        audit: List[Dict[str, Any]],
        iteration: int,
    ) -> str:
        """Runs reflection phase to decide whether to stop or continue."""
        ref_prompt = self._compose_context(question, plan_text, last_output=last_output)
        ref = self.model.generate(self.SYSTEM_REFLECT, ref_prompt)

        decision = "CONTINUE"
        next_hint = ""
        for ln in ref.splitlines():
            if ln.strip().upper().startswith("DECISION:"):
                decision = ln.split(":", 1)[1].strip().upper()
            if ln.strip().upper().startswith("NEXT:"):
                next_hint = ln.split(":", 1)[1].strip()

        if decision not in {"STOP", "CONTINUE"}:
            decision = "CONTINUE"

        audit.append({
            "phase": "reflection",
            "iteration": iteration,
            "raw_reflection": ref,
            "decision": decision,
            "next": next_hint,
        })

        if next_hint:
            self.scratchpad.add(f"[REFLECTION NEXT]\n{next_hint}")

        return decision


    def solve(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Runs full Plan-Act-Reflect loop to solve a single task."""
        audit: List[Dict[str, Any]] = []

        question = self._extract_question(sample)
        files_map = self._extract_files(sample)

        if self.cfg.debug:
            audit.append({
                "phase": "debug_input",
                "question_preview": question[:300],
                "question_len": len(question),
                "files": list(files_map.keys()) if isinstance(files_map, dict) else None,
            })

        self.files_map = files_map or {}
        self.tools.set_files_map(self.files_map)

        self.scratchpad.clear()

        gated = self._capability_gate(question, audit)
        if gated is not None:
            return gated

        self._auto_tools(audit)

        plan_text = self.model.generate(self.SYSTEM_PLAN, question)
        audit.append({
            "phase": "plan",
            "plan_steps": extract_plan_steps(plan_text),
            "raw_plan": plan_text,
        })

        last_output = ""
        final_answer: Optional[str] = None

        for it in range(1, self.cfg.max_steps + 1):
            last_output, fa = self._act_once(
                question=question,
                plan_text=plan_text,
                last_output=last_output,
                audit=audit,
                iteration=it,
            )

            if fa is not None:
                final_answer = fa if fa else "I don't know."
                break

            decision = self._reflect(
                question=question,
                plan_text=plan_text,
                last_output=last_output,
                audit=audit,
                iteration=it,
            )

            if decision == "STOP":
                force_prompt = self._compose_context(question, plan_text, last_output=last_output)
                forced = self.model.generate(
                    self.SYSTEM_ACT,
                    force_prompt + "\n\nYou must answer now.\nFINAL ANSWER: <answer>\n",
                )
                audit.append({"phase": "force_answer", "iteration": it, "raw_output": forced})
                fa2 = extract_final_answer(forced)
                final_answer = fa2 if fa2 else "I don't know."
                break
            if it < self.cfg.max_steps:
                replanning_prompt = (
                    f"QUESTION:\n{question}\n\n"
                    f"CURRENT_PLAN:\n{plan_text}\n\n"
                    f"SCRATCHPAD:\n{str(self.scratchpad)}\n\n"
                    "Update the plan if needed. Output a numbered plan only."
                )
                plan_text = self.model.generate(self.SYSTEM_PLAN, replanning_prompt)
                audit.append({
                    "phase": "replan",
                    "iteration": it,
                    "plan_steps": extract_plan_steps(plan_text),
                    "raw_plan": plan_text,
                })

        if final_answer is None:
            final_answer = "I don't know."

        return {
            "answer": final_answer,
            "raw_response": str(self.scratchpad),
            "audit_trail": audit,
        }

def run(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """HAL entrypoint that unwraps GAIA sample, runs agent, and returns structured result."""

    if len(sample) == 1 and isinstance(next(iter(sample.values())), dict):
        task_id = next(iter(sample.keys()))
        record = next(iter(sample.values()))
    else:
        task_id = "unknown_task"
        record = sample

    question = record.get("Question") or record.get("prompt") or ""

    if not question or not str(question).strip():
        return {
            task_id: {
                "answer": "UNSUPPORTED_TASK",
                "trajectory": [
                    {"phase": "refusal", "reason": "Empty question."}
                ]
            }
        }

    files = {}
    if record.get("file_name"):
        files[record["file_name"]] = record.get("file_path", "")

    cfg = AgentConfig(
        model_id=kwargs.get("model_id", "Qwen/Qwen2.5-7B-Instruct"),
        temperature=kwargs.get("temperature", 0.0),
        top_p=kwargs.get("top_p", 0.95),
        max_new_tokens=kwargs.get("max_new_tokens", 512),
        max_steps=kwargs.get("max_steps", 6),
        max_tool_calls_per_step=kwargs.get("max_tool_calls_per_step", 6),
        scratchpad_max_chars=kwargs.get("scratchpad_max_chars", 14000),
        auto_load_table=kwargs.get("auto_load_table", True),
    )

    agent = StrictClosedWorldR2A2Agent(cfg, files)

    result = agent.solve({
        "question": question,
        "files": files,
    })

    return {
        task_id: {
            "answer": str(result.get("answer", "I don't know.")).strip(),
            "trajectory": result.get("audit_trail", [])
        }
    }
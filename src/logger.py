"""
Session Logger — JSON trace of every LLM call, tool call, judge call, GEPA step.
Writes to results/<session>/logs/llm_trace.json after every entry (crash-safe).
"""

import os
import json
import threading
from datetime import datetime
from typing import Any, Dict, List

import dspy
from dspy.utils.callback import BaseCallback


class SessionLogger(BaseCallback):
    """
    Hooks into DSPy callbacks to capture every LLM interaction.
    Also exposes manual log_* helpers for tool calls, judge verdicts, GEPA steps.

    Each entry in llm_trace.json:
    {
        "id":          int,
        "timestamp":   "ISO-8601",
        "type":        "lm_call" | "react_step" | "tool_call" | "judge_call" | "gepa_step",
        "module":      "ClassName or tool name",
        "inputs":      { ... },
        "outputs":     { ... },
        "duration_ms": float | null,
        "error":       null | "message"
    }
    """

    def __init__(self, log_dir: str):
        super().__init__()
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "llm_trace.json")
        self._entries: List[Dict] = []
        self._pending: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._counter = 0

    # ── DSPy auto-hooks ──────────────────────────────────────────────────────

    def on_lm_start(self, call_id: str, instance, inputs: Dict):
        self._start(call_id, "lm_call", type(instance).__name__, inputs)

    def on_lm_end(self, call_id: str, outputs: Dict, exception=None):
        self._end(call_id, outputs, exception)

    def on_module_start(self, call_id: str, instance, inputs: Dict):
        if type(instance).__name__ == "ReAct":
            self._start(call_id, "react_step", "ReAct", inputs)

    def on_module_end(self, call_id: str, outputs, exception=None):
        self._end(call_id, outputs, exception)

    # ── Manual helpers ───────────────────────────────────────────────────────

    def log_tool_call(self, tool_name: str, args: Dict, result: str, duration_ms: float = None):
        self._commit({
            "id": self._next_id(), "timestamp": _now(),
            "type": "tool_call", "module": tool_name,
            "inputs": _safe(args),
            "outputs": {"result": result, "sent_to_llm": result},
            "duration_ms": duration_ms, "error": None,
        })

    def log_judge(self, judge_name: str, proof_snippet: str, verdict: str, passed: bool, duration_ms: float = None):
        self._commit({
            "id": self._next_id(), "timestamp": _now(),
            "type": "judge_call", "module": judge_name,
            "inputs": {"proof_snippet": proof_snippet[:300]},
            "outputs": {"verdict": verdict[:500], "passed": passed},
            "duration_ms": duration_ms, "error": None,
        })

    def log_gepa_step(self, iteration: int, scores: List[float], best_score: float):
        self._commit({
            "id": self._next_id(), "timestamp": _now(),
            "type": "gepa_step", "module": "GEPAOptimizer",
            "inputs": {"iteration": iteration},
            "outputs": {"candidate_scores": scores, "best_score": best_score},
            "duration_ms": None, "error": None,
        })

    def summary(self) -> Dict:
        with self._lock:
            entries = list(self._entries)
        return {
            "total_entries": len(entries),
            "lm_calls":      sum(1 for e in entries if e["type"] == "lm_call"),
            "tool_calls":    sum(1 for e in entries if e["type"] == "tool_call"),
            "judge_calls":   sum(1 for e in entries if e["type"] == "judge_call"),
            "gepa_steps":    sum(1 for e in entries if e["type"] == "gepa_step"),
            "log_path":      self._log_path,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _start(self, call_id: str, type_: str, module: str, inputs: Dict):
        entry = {
            "id": self._next_id(), "timestamp": _now(),
            "type": type_, "module": module,
            "inputs": _safe(inputs), "outputs": None,
            "duration_ms": None, "error": None,
            "_t0": datetime.now().timestamp(),
        }
        with self._lock:
            self._pending[call_id] = entry

    def _end(self, call_id: str, outputs, exception=None):
        with self._lock:
            entry = self._pending.pop(call_id, None)
        if entry is None:
            return
        entry["outputs"] = _safe(outputs)
        entry["duration_ms"] = round((datetime.now().timestamp() - entry.pop("_t0")) * 1000, 1)
        if exception:
            entry["error"] = str(exception)
        self._commit(entry)

    def _commit(self, entry: Dict):
        with self._lock:
            self._entries.append(entry)
            snapshot = list(self._entries)
        with open(self._log_path, 'w') as f:
            json.dump({"session_log": snapshot}, f, indent=2, default=str)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()


def _safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, '__dict__'):
        return _safe(vars(obj))
    return str(obj)

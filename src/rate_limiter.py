"""
NvidiaLM — NVIDIA API client extending dspy.LM.
Supports streaming, thinking/reasoning tokens, and per-instance 35 RPM rate limiting.
"""

import os
import sys
import time
import threading
from collections import deque
from typing import Optional

from openai import OpenAI
import dspy

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_THINK_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m"  if _USE_COLOR else ""
_DIM_COLOR   = "\033[2m"  if _USE_COLOR else ""


class NvidiaLM(dspy.LM):
    """
    NVIDIA API LM with streaming + thinking support.
    Extends dspy.LM so DSPy accepts it everywhere a BaseLM is expected.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        thinking: Optional[bool] = None,
        reasoning_budget: Optional[int] = None,
        rpm: int = 35,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ):
        super().__init__(
            model=model if model.startswith("openai/") else f"openai/{model}",
            api_key=api_key,
            api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
        )
        # Keep full model name including openai/ prefix for direct API calls
        self._nvidia_model = model
        self._api_key          = api_key
        self._base_url         = base_url
        self._temperature      = temperature
        self._top_p            = top_p
        self._max_tokens       = max_tokens
        self._thinking         = thinking
        self._reasoning_budget = reasoning_budget if reasoning_budget is not None else (2048 if thinking else None)
        self._rpm              = rpm
        self._client           = OpenAI(api_key=api_key, base_url=base_url)

        # Per-instance rate limiter
        self._window     = 60.0
        self._timestamps: deque = deque()
        self._rl_lock    = threading.Lock()

    # ── Rate limiting ────────────────────────────────────────────────────────

    def _wait_if_needed(self):
        while True:
            with self._rl_lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                wait = self._window - (now - self._timestamps[0])
            if wait > 0:
                print(f"  ⏳ Rate limit reached — waiting {wait:.1f}s...", flush=True)
                time.sleep(wait + 0.05)

    # ── extra_body ───────────────────────────────────────────────────────────

    def _extra_body(self) -> dict:
        if self._thinking is False:
            return {"chat_template_kwargs": {"thinking": False}}
        if self._thinking is True:
            body: dict = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}
            if self._reasoning_budget:
                body["reasoning_budget"] = self._reasoning_budget
            return body
        return {}

    # ── Override dspy.LM.__call__ ────────────────────────────────────────────

    def __call__(self, prompt=None, messages=None, **kwargs) -> list[dict]:
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                return self._call_once(messages)
            except Exception as e:
                if attempt < max_retries and any(k in str(e) for k in (
                    "RemoteProtocolError", "incomplete chunked", "peer closed",
                    "ConnectionError", "timeout", "502", "503", "504"
                )):
                    wait = 5 * attempt
                    print(f"\n  ⚠ Network error (attempt {attempt}/{max_retries}): {str(e)[:80]}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise

    def _call_once(self, messages: list) -> list[dict]:
        self._wait_if_needed()

        last_content = messages[-1].get("content", "") if messages else ""
        if isinstance(last_content, list):
            last_content = str(last_content[0])
        preview = last_content[:60].replace("\n", " ")
        thinking_tag = " 🧠" if self._thinking else ""
        print(f"\n  ▶ {self._nvidia_model}{thinking_tag} → \"{preview}...\"", flush=True)

        extra = self._extra_body()
        call_kwargs = dict(
            model=self._nvidia_model,
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            stream=True,
        )
        if extra:
            call_kwargs["extra_body"] = extra

        completion = self._client.chat.completions.create(**call_kwargs)

        t_start = time.monotonic()
        first_token = threading.Event()

        def _ticker():
            while not first_token.is_set():
                elapsed = time.monotonic() - t_start
                print(f"\r  ⏱  waiting for first token... {elapsed:.0f}s", end="", flush=True)
                time.sleep(1)

        ticker = threading.Thread(target=_ticker, daemon=True)
        ticker.start()

        content_parts = []
        thinking_shown = False
        for chunk in completion:
            if not first_token.is_set():
                first_token.set()
                print(f"\r  ✓ First token in {time.monotonic()-t_start:.1f}s          ", flush=True)
            if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                if not thinking_shown:
                    print(f"{_THINK_COLOR}  [thinking] ", end="", flush=True)
                    thinking_shown = True
                print(f"{_THINK_COLOR}{reasoning}{_RESET_COLOR}", end="", flush=True)
            if getattr(delta, "content", None) is not None:
                if thinking_shown:
                    print(f"\n{_RESET_COLOR}", end="", flush=True)
                    thinking_shown = False
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

        first_token.set()
        elapsed_total = time.monotonic() - t_start
        print(f"  ✓ Done in {elapsed_total:.1f}s", flush=True)
        content = "".join(content_parts)

        outputs = [{"text": content, "logprobs": None, "finish_reason": "stop"}]
        self.history.append({
            "messages": messages,
            "response": {"choices": [{"message": {"content": content}}]},
            "outputs": outputs,
        })
        return outputs

    def __deepcopy__(self, memo):
        new = NvidiaLM(
            api_key=self._api_key,
            model=self._nvidia_model,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            thinking=self._thinking,
            reasoning_budget=self._reasoning_budget,
            rpm=self._rpm,
            base_url=self._base_url,
        )
        memo[id(self)] = new
        return new

    def __repr__(self):
        return f"NvidiaLM(model={self._nvidia_model}, thinking={self._thinking}, rpm={self._rpm})"

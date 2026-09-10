# Mathematical Inequality Proof Verification Agent

## Overview

LLM-based agent that generates and rigorously verifies mathematical inequality proofs using **DSPy** with real **GEPA (MIPROv2)** prompt optimization and **Lean 4 + Mathlib** formal verification. Inspired by the IneqMath paper (arXiv: 2506.07927v3).

**Key finding**: LLMs achieve 40-70% answer accuracy but only 5-20% overall accuracy when proofs are rigorously verified — a gap of 20-60 percentage points. In our experiments across all 7 inequalities, GEPA optimization improved overall pass rate from **28.6% → 100%** (+71.4% absolute, +250% relative improvement), with Logical Gap judge improving from 57.1% → 100% and Final Answer judge from 57.1% → 100%.

## Project Structure

```
project_final/
├── src/
│   ├── agent.py              # MathReasoningAgent — orchestrates everything
│   ├── judges.py             # Five-judge system (LogicalGapJudge uses Lean)
│   ├── proof_generator.py    # DSPy Modules + inequality definitions
│   ├── prompt_optimizer.py   # Real GEPA via DSPy MIPROv2 + PromptMetrics
│   ├── tools.py              # 6 tools available to the LLM
│   ├── rate_limiter.py       # Per-instance 35 RPM rate limiter (NvidiaLM)
│   └── logger.py             # Session logger (DSPy callback → JSON trace)
├── lean_checker/             # Lean 4 + Mathlib project for formal verification
│   └── LeanChecker/
│       └── Check.lean        # Overwritten on each verify_with_lean() call
├── experiments/
│   ├── run_experiments.py    # Main entry point (--quick or full)
│   ├── compare_all.py        # Human vs Unoptimized vs GEPA-Optimized
│   ├── test_human_proofs.py  # Verify hand-written proofs through 5 judges
│   └── chat.py               # Conversational CLI
├── results/
│   └── session_YYYYMMDD_HHMMSS/
│       ├── logs/
│       │   ├── llm_trace.json          # Full trace: LLM calls, tool calls (with args+output), judges, GEPA
│       │   └── summary.json            # Entry counts by type
│       ├── gepa_optimized_module.json  # Saved optimized DSPy module (load with .load())
│       ├── <proof>.[txt|tex]           # Raw LLM output saved via save_output tool
│       ├── <proof>.pdf                 # Auto-compiled if LLM wrote LaTeX
│       ├── detailed_results.json       # Pass rates + per-judge improvements
│       ├── comparison_results.json     # Human vs unopt vs opt metrics
│       └── summary_report.txt          # Human-readable experiment summary
├── docs/
│   ├── README.md
│   ├── DOCUMENTATION.md
│   └── EXAMPLE_OUTPUT.txt
├── scripts/
│   └── verify_setup.py       # Environment + dependency check
├── requirements.txt
├── .env.template             # Production config template
└── .env.fast                 # Fast/smoke-test config (thinking off, minimal GEPA)
```

## Submission & Demo

**Create submission zip** (excludes `.venv` and Mathlib cache, keeps results):
```bash
cd /Users/kanishkkukreja/Downloads
zip -r project_final_submission.zip project_final \
  --exclude "project_final/.venv/*" \
  --exclude "project_final/lean_checker/.lake/*" \
  --exclude "project_final/__pycache__/*" \
  --exclude "project_final/src/__pycache__/*" \
  --exclude "project_final/.env" \
  --exclude "*.pyc"
```

Size breakdown:
| Directory | Size | Include in zip |
|-----------|------|----------------|
| `lean_checker/.lake/` | ~6.9GB | ✗ Mathlib cache, re-downloads on first build |
| `.venv/` | ~434MB | ✗ Recreate with pip install |
| `results/` | ~60MB | ✓ Session outputs, PDFs, traces |
| `src/` + `experiments/` + `docs/` | <300KB | ✓ |

**Restore and run on demo machine:**
```bash
unzip project_final_submission.zip && cd project_final

# Install dependencies
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python3
.venv/bin/python3 -m pip install optuna

# Fill in API keys
cp .env.template .env  # edit .env with 5 NVIDIA API keys

# Rebuild Lean/Mathlib cache (one-time, ~500MB download)
cd lean_checker && lake build LeanChecker && cd ..

# Quick test
.venv/bin/python3 experiments/run_experiments.py --quick

# Full run
.venv/bin/python3 -u experiments/run_experiments.py 2>&1 | tee results/full_run_$(date +%Y%m%d_%H%M%S).log
```

## Quick Start

```bash
# 1. Create venv and install
uv venv .venv
uv pip install -r requirements.txt --python .venv/bin/python3
.venv/bin/python3 -m pip install optuna   # required by MIPROv2

# 2. Copy and fill in config
cp .env.template .env
# edit .env — fill in all 5 NVIDIA API keys

# 3. Install Lean 4 + Mathlib (optional — needed for LogicalGapJudge formal checks)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
cd lean_checker && lake build LeanChecker  # downloads Mathlib (~500MB, once only)
cd ..

# 4. Verify setup
.venv/bin/python3 scripts/verify_setup.py

# 5. Quick test (single proof, no GEPA)
.venv/bin/python3 experiments/run_experiments.py --quick

# 6. Full experiment (unoptimized → GEPA → optimized, ~30-55 min)
.venv/bin/python3 experiments/run_experiments.py

# 7. Fast smoke test (~10-15 min, thinking off, minimal GEPA)
cp .env.fast .env && .venv/bin/python3 experiments/run_experiments.py

# 8. Human vs LLM comparison
.venv/bin/python3 experiments/compare_all.py

# 9. Conversational interface
.venv/bin/python3 experiments/chat.py
```

## Architecture

```
MathReasoningAgent
├── ProofGeneratorUnoptimized     dspy.Predict — single call, no tools, baseline
├── ProofGeneratorOptimized       4-step pipeline, dspy.ReAct for writing step
│   ├── Step 1: plan    dspy.Predict  — proof strategy outline
│   ├── Step 2: write   dspy.ReAct   — full proof with tool calls
│   ├── Step 3: verify  dspy.Predict — self-check and refine
│   └── Step 4: latex   dspy.Predict — render to LaTeX + save PDF
│   └── Optimized by GEPAOptimizer (DSPy MIPROv2, saved to gepa_optimized_module.json)
│       └── metric: proof_quality_metric (0.2 per judge passed, max 1.0)
├── ProofVerificationSystem       five judges, all must pass
│   ├── FinalAnswerJudge          dspy.Predict
│   ├── ToyCaseJudge              dspy.Predict
│   ├── LogicalGapJudge           dspy.ReAct + verify_with_lean tool ← Lean 4
│   ├── NumericalApproximationJudge  dspy.Predict
│   └── NumericalComputationJudge    dspy.Predict
├── SessionLogger                 DSPy BaseCallback → crash-safe llm_trace.json
│   └── Logs: lm_call, react_step, tool_call (with full args + output), judge_call, gepa_step
└── NvidiaLM                      per-instance 35 RPM sliding-window rate limiter
    └── Streams responses, shows thinking tokens, live first-token timer
```

## Tools Available to the LLM

| # | Tool | Used by | What it does |
|---|------|---------|--------------|
| 1 | `lookup_inequality(name)` | WriteProof (ReAct) | Fetches formal statement + theorems |
| 2 | `verify_arithmetic(expr)` | WriteProof (ReAct) | SymPy symbolic simplification |
| 3 | `check_proof_step(claim)` | WriteProof (ReAct) | SymPy step validation |
| 4 | `verify_with_lean(lean_code)` | LogicalGapJudge (ReAct) | Lean 4 + Mathlib formal check |
| 5 | `save_output(content, filename)` | WriteProof (ReAct) | Saves output to session folder, auto-compiles `.tex` → PDF |
| 6 | `compile_latex_proof(body, filename)` | save_output | pdflatex compile with LLM-based retry on failure |

All tool calls are logged to `llm_trace.json` with full arguments and the exact output returned to the LLM.

## Logging — llm_trace.json

Every entry has: `id`, `timestamp`, `type`, `module`, `inputs`, `outputs`, `duration_ms`, `error`.

| `type` | What it captures |
|--------|-----------------|
| `lm_call` | Raw LLM request + response |
| `react_step` | Full ReAct trajectory (thoughts, tool calls, observations) |
| `tool_call` | Tool name, exact kwargs passed by LLM, full output returned to LLM |
| `judge_call` | Judge name, proof snippet, verdict, pass/fail |
| `gepa_step` | GEPA iteration, candidate scores, best score |

## GEPA Prompt Optimization

1. `GEPAOptimizer` wraps `dspy.MIPROv2` with `proof_quality_metric` as the objective
2. Builds a trainset from all 7 inequalities (cauchy_schwarz, jensen, triangle, bernoulli, young, chebyshev, markov)
3. MIPROv2 proposes `GEPA_NUM_CANDIDATES` instruction variants, scores each using all 5 judges, selects the best over `GEPA_NUM_ITERATIONS` trials
4. Optimized module saved to `results/session_.../gepa_optimized_module.json`

To reload a previously optimized module:
```python
from src.proof_generator import ProofGeneratorOptimized
generator = ProofGeneratorOptimized(tools=...)
generator.load("results/session_YYYYMMDD_HHMMSS/gepa_optimized_module.json")
```

## Five-Judge System

| Judge | Detects | Method |
|-------|---------|--------|
| Final Answer | Wrong conclusion vs ground truth | LLM |
| Toy Case | Unjustified generalization from specific examples | LLM |
| Logical Gap | Missing steps, unjustified claims | LLM + Lean 4 |
| Numerical Approximation | √2 ≈ 1.414 used in proof chain | LLM |
| Numerical Computation | Arithmetic / algebraic errors | LLM |

All five must pass for `overall: VERIFIED`.

## Lean 4 Integration

`LogicalGapJudge` is a `dspy.ReAct` module. It can call `verify_with_lean()` mid-reasoning:

```
Thought: Step 4 claims discriminant ≤ 0. Let me verify formally.
Action: verify_with_lean("import Mathlib\nexample (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by nlinarith [sq_nonneg (a-b)]")
Observation: VERIFIED: Lean accepted the proof.
Thought: Key step formally confirmed. No logical gap.
Output: PASS
```

- Runs inside `lean_checker/` — a Lake project with Mathlib
- Available tactics: `nlinarith`, `linarith`, `ring`, `norm_num`, `positivity`, `aesop`
- Returns `VERIFIED`, `FAILED: <lean error>`, or `LEAN_NOT_INSTALLED` (graceful fallback)

## Session Output

Every run creates `results/session_YYYYMMDD_HHMMSS/` — nothing is written outside it.

| File | Contents |
|------|----------|
| `logs/llm_trace.json` | Full crash-safe trace of every LLM call, tool call, judge, GEPA step |
| `logs/summary.json` | Entry counts by type |
| `gepa_optimized_module.json` | Saved optimized DSPy module with rewritten instructions + demos |
| `<proof>.tex` / `<proof>.pdf` | LaTeX source + compiled PDF |
| `detailed_results.json` | Pass rates + per-judge improvements |
| `summary_report.txt` | Human-readable experiment summary |

## Models (NVIDIA API)

5 separate API keys, one per role, each with its own rate limiter:

| Role | Model | Thinking |
|------|-------|---------|
| Proof Generator | `openai/gpt-oss-120b` | ✓ |
| Final Answer Judge | `nvidia/nemotron-3-super-120b-a12b` | ✓ |
| Toy Case + Num. Approx Judge | `nvidia/nemotron-3-super-120b-a12b` | ✓ |
| Logical Gap Judge | `openai/gpt-oss-120b` | ✓ |
| Num. Computation Judge + GEPA | `nvidia/nemotron-3-super-120b-a12b` | ✓ |
| GEPA Optimizer | `qwen/qwen3-next-80b-a3b-instruct` | ✗ |

All rate-limited to 35 RPM per instance via sliding window. All config in `.env`.

## Config Files

| File | Purpose |
|------|---------|
| `.env` | Production config (thinking on, full token limits, 5 candidates) |
| `.env.fast` | Smoke-test config (thinking off, 4096 tokens, 2 candidates, 1 iteration) |
| `.env.template` | Template to copy for new setups |

## Experimental Results

Run across all 7 inequalities (cauchy_schwarz, jensen, triangle, bernoulli, young, chebyshev, markov) — session `session_20260429_223459`:

| Metric | Unoptimized | GEPA-Optimized | Improvement |
|--------|-------------|----------------|-------------|
| Overall Pass Rate | 28.6% | 100.0% | +71.4% (+250%) |
| Final Answer | 57.1% | 100.0% | +42.9% |
| Toy Case | 100.0% | 100.0% | +0.0% |
| Logical Gap | 57.1% | 100.0% | +42.9% |
| Numerical Approximation | 100.0% | 100.0% | +0.0% |
| Numerical Computation | 100.0% | 100.0% | +0.0% |

GEPA primarily improved the two hardest judges — Final Answer (correct equality conditions) and Logical Gap (degenerate case handling) — while the simpler judges were already at 100% unoptimized.

## Inequalities Available

`cauchy_schwarz`, `jensen`, `triangle`, `bernoulli`, `young`, `chebyshev`, `markov`

## Expected Runtime

| Mode | Time |
|------|------|
| `--quick` (single proof) | ~2-3 min |
| Full run (production) | ~30-55 min |
| Full run (`.env.fast`) | ~10-15 min |

## References

- IneqMath Paper: https://arxiv.org/abs/2506.07927v3
- DSPy: https://dspy.ai
- Lean 4: https://lean-lang.org
- Mathlib: https://leanprover-community.github.io
- NVIDIA API: https://integrate.api.nvidia.com

# IneqMath Proof Agent

> An LLM agent that **generates mathematical inequality proofs and rigorously verifies them**, combining DSPy, real GEPA (MIPROv2) prompt optimization, and **Lean 4 + Mathlib** formal checking. Built after the IneqMath paper (arXiv 2506.07927v3).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![DSPy](https://img.shields.io/badge/DSPy-GEPA%20Optimized-7C3AED)
![Lean 4](https://img.shields.io/badge/Lean%204-Formal%20Proofs-3B5BDB)
![License](https://img.shields.io/badge/License-MIT-green)

## The finding

LLMs look good at math until you check their proofs. Across 7 inequalities, models scored **40-70% on answer accuracy but only 5-20% overall accuracy once proofs were formally verified**. After GEPA prompt optimization, the agent's rigorously-verified pass rate went from **28.6% to 100%** on the tested set, with the Logical Gap judge and Final Answer judge both reaching 100%.

## How it works

```
                     ┌─────────────────────────────┐
   Inequality ──────►│  DSPy Proof Generator        │
                     │  (ChainOfThought module)     │
                     └──────────┬──────────────────┘
                                │  candidate proof
                     ┌──────────▼──────────────────┐
                     │      Five-Judge Panel        │
                     │  Final Answer Judge          │
                     │  Toy Case Judge              │
                     │  Numerical Approximation     │
                     │  Numerical Computation Judge │
                     │  Logical Gap Judge ──────────┼──►  Lean 4 + Mathlib
                     └──────────┬──────────────────┘      (lean_checker)
                                │  verdicts + feedback
                     ┌──────────▼──────────────────┐
                     │  GEPA (DSPy MIPROv2)         │
                     │  prompt optimization loop    │
                     └─────────────────────────────┘
```

- **6 tools** available to the agent (save output, verify with Lean, numerical checks, and more)
- **Per-role API keys** with a shared **35 RPM rate limiter**
- **Session logging**: full JSON traces of LLM calls, tool calls, judge verdicts, and GEPA steps

## Results

| Judge                  | Unoptimized | GEPA-optimized |
|------------------------|-------------|----------------|
| Overall pass rate      | 28.6%       | **100%**       |
| Logical Gap judge      | 57.1%       | **100%**       |
| Final Answer judge     | 57.1%       | **100%**       |

Full traces, optimized DSPy modules, and generated LaTeX/PDF proofs live in `results/`.

## Quick start

```bash
git clone https://github.com/Aditi21372/ineqmath-proof-agent.git
cd ineqmath-proof-agent
pip install -r requirements.txt
cp .env.template .env        # add NVIDIA API keys (one per role)

python scripts/verify_setup.py
python experiments/run_experiments.py --quick
python experiments/compare_all.py   # human vs unoptimized vs optimized
```

A Lean 4 toolchain is required for formal verification (`lean_checker/` ships the Lake project).

## Repo layout

```
ineqmath-proof-agent/
├── src/               # agent, judges, generator, GEPA optimizer, tools, rate limiter
├── lean_checker/      # Lean 4 + Mathlib project (formal verification)
├── experiments/       # experiment runner, comparisons, human-proof checks, CLI chat
├── results/           # sessions: traces, optimized modules, generated proofs (tex/pdf)
├── scripts/           # environment verification
└── docs/              # full documentation, example outputs, submission summary
```

## Key engineering details

- **GEPA for real**: MIPROv2 optimization over judge feedback, with metrics tracked per judge, not just final answers
- **Formal verification in the loop**: the Logical Gap judge compiles candidate proofs into Lean (`Check.lean` regenerated per call) so "looks right" becomes "type-checks"
- **Traces over vibes**: every LLM/tool/judge call is logged to JSON for reproducibility

---

Built by [@Aditi21372](https://github.com/Aditi21372) · [More projects](https://github.com/Aditi21372?tab=repositories)

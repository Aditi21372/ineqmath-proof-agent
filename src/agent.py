"""
Main Mathematical Reasoning Agent

5 API keys, each with its own role, model, thinking config, and rate limiter:
  Key 1 (GENERATOR)     → deepseek-v4-pro,          thinking=False  → ProofGenerator
  Key 2 (JUDGE_FINAL)   → glm-5.1,                  thinking=True   → FinalAnswerJudge
  Key 3 (JUDGE_TOYCASE) → nemotron-3-super-120b,     thinking=True   → ToyCaseJudge + NumApproxJudge
  Key 4 (JUDGE_LOGICAL) → glm-5.1,                  thinking=True   → LogicalGapJudge (+ Lean)
  Key 5 (JUDGE_NUMCOMP) → nemotron-3-super-120b,     thinking=True   → NumCompJudge + GEPA
"""

import os
import json
import contextlib
from datetime import datetime
from typing import Dict, List, Optional

import dspy
from dotenv import load_dotenv

load_dotenv()

# --- Base ---
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
RATE_LIMIT_RPM  = int(os.getenv("RATE_LIMIT_RPM", "35"))
TOP_P           = float(os.getenv("TOP_P", "0.95"))

# --- API Keys ---
KEY_GENERATOR     = os.getenv("NVIDIA_API_KEY_GENERATOR")
KEY_JUDGE_FINAL   = os.getenv("NVIDIA_API_KEY_JUDGE_FINAL")
KEY_JUDGE_TOYCASE = os.getenv("NVIDIA_API_KEY_JUDGE_TOYCASE")
KEY_JUDGE_LOGICAL = os.getenv("NVIDIA_API_KEY_JUDGE_LOGICAL")
KEY_JUDGE_NUMCOMP = os.getenv("NVIDIA_API_KEY_JUDGE_NUMCOMP")

# --- Models ---
GENERATION_MODEL    = os.getenv("GENERATION_MODEL",    "deepseek-ai/deepseek-v4-pro")
JUDGE_FINAL_MODEL   = os.getenv("JUDGE_FINAL_MODEL",   "z-ai/glm-5.1")
JUDGE_TOYCASE_MODEL = os.getenv("JUDGE_TOYCASE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
JUDGE_LOGICAL_MODEL = os.getenv("JUDGE_LOGICAL_MODEL", "z-ai/glm-5.1")
JUDGE_NUMCOMP_MODEL = os.getenv("JUDGE_NUMCOMP_MODEL", "nvidia/nemotron-3-super-120b-a12b")
GEPA_MODEL          = os.getenv("GEPA_MODEL",          "deepseek-ai/deepseek-v4-pro")

# --- Thinking config (parse "true"/"false" strings) ---
def _thinking(env_var: str, default: Optional[bool]) -> Optional[bool]:
    v = os.getenv(env_var, "").lower()
    if v == "true":  return True
    if v == "false": return False
    return default

GENERATION_THINKING    = _thinking("GENERATION_THINKING",    False)
JUDGE_FINAL_THINKING   = _thinking("JUDGE_FINAL_THINKING",   True)
JUDGE_TOYCASE_THINKING = _thinking("JUDGE_TOYCASE_THINKING", True)
JUDGE_LOGICAL_THINKING = _thinking("JUDGE_LOGICAL_THINKING", True)
JUDGE_NUMCOMP_THINKING = _thinking("JUDGE_NUMCOMP_THINKING", True)
GEPA_THINKING          = _thinking("GEPA_THINKING",          False)

# --- Other config ---
GEN_TEMPERATURE     = float(os.getenv("GENERATION_TEMPERATURE", "1.0"))
GEN_MAX_TOKENS      = int(os.getenv("GENERATION_MAX_TOKENS",    "16384"))
JUDGE_MAX_TOKENS    = int(os.getenv("JUDGE_MAX_TOKENS",         "4096"))
GEPA_NUM_CANDIDATES = int(os.getenv("GEPA_NUM_CANDIDATES",      "5"))
GEPA_NUM_ITERATIONS = int(os.getenv("GEPA_NUM_ITERATIONS",      "2"))

from src.proof_generator import ProofGeneratorOptimized, ProofGeneratorUnoptimized, get_inequality
from src.judges import ProofVerificationSystem
from src.prompt_optimizer import GEPAOptimizer, PromptMetrics
from src.tools import lookup_inequality, verify_arithmetic, check_proof_step, compile_latex_proof, save_output, set_session_dir
from src.rate_limiter import NvidiaLM
from src.logger import SessionLogger


def _make_lm(api_key: str, model: str, max_tokens: int,
             thinking: Optional[bool] = None,
             temperature: float = None) -> NvidiaLM:
    """Create a NvidiaLM with its own API key, model, thinking config, and rate limiter."""
    return NvidiaLM(
        api_key=api_key,
        model=model,
        temperature=temperature or GEN_TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        thinking=thinking,
        reasoning_budget=GEN_MAX_TOKENS if thinking else None,
        rpm=RATE_LIMIT_RPM,
        base_url=NVIDIA_BASE_URL,
    )


def _session_dir() -> str:
    session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    path = os.path.join("results", session)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    return path


def _check_keys():
    missing = [
        name for name, val in [
            ("NVIDIA_API_KEY_GENERATOR",     KEY_GENERATOR),
            ("NVIDIA_API_KEY_JUDGE_FINAL",   KEY_JUDGE_FINAL),
            ("NVIDIA_API_KEY_JUDGE_TOYCASE", KEY_JUDGE_TOYCASE),
            ("NVIDIA_API_KEY_JUDGE_LOGICAL", KEY_JUDGE_LOGICAL),
            ("NVIDIA_API_KEY_JUDGE_NUMCOMP", KEY_JUDGE_NUMCOMP),
        ] if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing API keys in .env: {', '.join(missing)}\n"
            "Copy .env.template to .env and fill in all 5 keys."
        )


class MathReasoningAgent:

    def __init__(self):
        _check_keys()

        self.session_dir = _session_dir()
        log_dir = os.path.join(self.session_dir, "logs")

        set_session_dir(self.session_dir)

        self.logger = SessionLogger(log_dir=log_dir)
        dspy.configure(callbacks=[self.logger])

        # --- One NvidiaLM per API key ---
        generator_lm = _make_lm(KEY_GENERATOR,     GENERATION_MODEL,    GEN_MAX_TOKENS,   GENERATION_THINKING)
        final_lm     = _make_lm(KEY_JUDGE_FINAL,   JUDGE_FINAL_MODEL,   JUDGE_MAX_TOKENS, JUDGE_FINAL_THINKING)
        toycase_lm   = _make_lm(KEY_JUDGE_TOYCASE, JUDGE_TOYCASE_MODEL, JUDGE_MAX_TOKENS, JUDGE_TOYCASE_THINKING)
        logical_lm   = _make_lm(KEY_JUDGE_LOGICAL, JUDGE_LOGICAL_MODEL, JUDGE_MAX_TOKENS, JUDGE_LOGICAL_THINKING)
        numcomp_lm   = _make_lm(KEY_JUDGE_NUMCOMP, JUDGE_NUMCOMP_MODEL, JUDGE_MAX_TOKENS, JUDGE_NUMCOMP_THINKING)
        gepa_lm      = _make_lm(KEY_JUDGE_NUMCOMP, GEPA_MODEL,          JUDGE_MAX_TOKENS, GEPA_THINKING)

        dspy.configure(lm=generator_lm)

        self.tools = [
            lookup_inequality,
            verify_arithmetic,
            check_proof_step,
            save_output,
            compile_latex_proof,
        ]

        self.verifier = ProofVerificationSystem(
            final_lm=final_lm,
            toycase_lm=toycase_lm,
            logical_lm=logical_lm,
            numcomp_lm=numcomp_lm,
        )
        self.unoptimized_generator = ProofGeneratorUnoptimized()
        self.optimized_generator   = ProofGeneratorOptimized(tools=self.tools, logger=self.logger)
        self.optimizer             = GEPAOptimizer(
            num_candidates=GEPA_NUM_CANDIDATES,
            num_iterations=GEPA_NUM_ITERATIONS,
            logger=self.logger,
            lm=gepa_lm,
        )
        self._gepa_done = False

        print(f"\n  Session : {self.session_dir}")
        print(f"  Trace   : {log_dir}/llm_trace.json")
        print(f"\n  Role assignments:")
        print(f"    Generator     : {GENERATION_MODEL:<45} thinking={GENERATION_THINKING}  key=...{KEY_GENERATOR[-6:]}")
        print(f"    FinalJudge    : {JUDGE_FINAL_MODEL:<45} thinking={JUDGE_FINAL_THINKING}   key=...{KEY_JUDGE_FINAL[-6:]}")
        print(f"    ToyCaseJudge  : {JUDGE_TOYCASE_MODEL:<45} thinking={JUDGE_TOYCASE_THINKING}   key=...{KEY_JUDGE_TOYCASE[-6:]}")
        print(f"    LogicalJudge  : {JUDGE_LOGICAL_MODEL:<45} thinking={JUDGE_LOGICAL_THINKING}   key=...{KEY_JUDGE_LOGICAL[-6:]}  (+ Lean)")
        print(f"    NumCompJudge  : {JUDGE_NUMCOMP_MODEL:<45} thinking={JUDGE_NUMCOMP_THINKING}   key=...{KEY_JUDGE_NUMCOMP[-6:]}")
        print(f"    GEPA          : {GEPA_MODEL:<45} thinking={GEPA_THINKING}  key=...{KEY_JUDGE_NUMCOMP[-6:]}")

    def run_gepa_optimization(self):
        if not self._gepa_done:
            save_path = os.path.join(self.session_dir, "gepa_optimized_module.json")
            self.optimized_generator = self.optimizer.optimize(self.optimized_generator, save_path=save_path)
            self._gepa_done = True

    def prove_and_verify(self, inequality_name: str, use_optimized: bool = True) -> Dict:
        ineq = get_inequality(inequality_name)
        if not ineq:
            raise ValueError(f"Unknown inequality: {inequality_name}")

        tag = "optimized" if use_optimized else "unoptimized"
        print(f"\n{'='*80}")
        print(f"PROVING : {ineq['name']}  [{tag}]")
        print(f"Model   : {GENERATION_MODEL}  thinking={GENERATION_THINKING}")
        print(f"{'='*80}\n")

        if use_optimized:
            pred = self.optimized_generator(
                inequality_name=ineq['name'],
                problem_statement=ineq['statement'],
                relevant_theorems=ineq.get('relevant_theorems', [])
            )
        else:
            pred = self.unoptimized_generator(
                inequality_name=ineq['name'],
                problem_statement=ineq['statement']
            )

        proof = pred.proof
        print("\nGENERATED PROOF:")
        print("-" * 80)
        print(proof)
        print("-" * 80)

        print("\nVERIFYING PROOF...")
        t0 = datetime.now()
        verification = self.verifier.verify_proof(proof, ineq['ground_truth'])
        elapsed_ms = (datetime.now() - t0).total_seconds() * 1000

        for judge_name, key in [
            ("FinalAnswerJudge",            "final_answer"),
            ("ToyCaseJudge",                "toy_case"),
            ("LogicalGapJudge",             "logical_gap"),
            ("NumericalApproximationJudge", "numerical_approximation"),
            ("NumericalComputationJudge",   "numerical_computation"),
        ]:
            r = verification[key]
            self.logger.log_judge(
                judge_name=judge_name,
                proof_snippet=proof[:300],
                verdict=r['message'],
                passed=r['pass'],
                duration_ms=round(elapsed_ms / 5, 1)
            )

        self.verifier.print_results(verification)

        return {
            'inequality': ineq['name'],
            'proof': proof,
            'verification': verification,
            'optimized': use_optimized
        }

    def run_comparison_experiment(self, inequality_names: List[str]) -> Dict:
        print("\n" + "="*80)
        print("RUNNING GEPA PROMPT OPTIMIZATION COMPARISON")
        print("="*80)

        print("\n--- PHASE 1: UNOPTIMIZED PROMPTS ---\n")
        unopt_results = [
            self.prove_and_verify(n, use_optimized=False)['verification']
            for n in inequality_names
        ]

        print("\n--- PHASE 2: GEPA OPTIMIZATION ---\n")
        self.run_gepa_optimization()

        print("\n--- PHASE 3: GEPA-OPTIMIZED PROMPTS ---\n")
        opt_results = [
            self.prove_and_verify(n, use_optimized=True)['verification']
            for n in inequality_names
        ]

        improvement = PromptMetrics.calculate_improvement(unopt_results, opt_results)
        PromptMetrics.print_comparison(improvement)

        results = {
            'unoptimized_results': unopt_results,
            'optimized_results': opt_results,
            'improvement_metrics': improvement,
            'timestamp': datetime.now().isoformat()
        }
        self._save_metrics(results, "detailed_results.json")
        self._save_log_summary()
        return results

    def _save_metrics(self, data: Dict, filename: str):
        path = os.path.join(self.session_dir, filename)
        with open(path, 'w') as f:
            json.dump({
                'improvement_metrics': data.get('improvement_metrics', data),
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'num_tests': len(data.get('unoptimized_results', [])),
                'models': {
                    'generation':    f"{GENERATION_MODEL} (thinking={GENERATION_THINKING})",
                    'judge_final':   f"{JUDGE_FINAL_MODEL} (thinking={JUDGE_FINAL_THINKING})",
                    'judge_toycase': f"{JUDGE_TOYCASE_MODEL} (thinking={JUDGE_TOYCASE_THINKING})",
                    'judge_logical': f"{JUDGE_LOGICAL_MODEL} (thinking={JUDGE_LOGICAL_THINKING})",
                    'judge_numcomp': f"{JUDGE_NUMCOMP_MODEL} (thinking={JUDGE_NUMCOMP_THINKING})",
                    'gepa':          f"{GEPA_MODEL} (thinking={GEPA_THINKING})",
                }
            }, f, indent=2)
        print(f"Metrics  → {path}")

    def _save_log_summary(self):
        summary = self.logger.summary()
        path = os.path.join(self.session_dir, "logs", "summary.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Log summary → {path}")
        print(f"Full trace  → {self.logger._log_path}")

"""
Prompt Optimization using DSPy GEPA / MIPROv2
Replaces hand-rolled GEPA with the real DSPy optimizer
"""

import dspy
import contextlib
from dspy.teleprompt import MIPROv2
from typing import List, Dict
from src.proof_generator import ProofGeneratorOptimized, get_inequality
from src.judges import ProofVerificationSystem
from src.tools import verify_arithmetic, lookup_inequality, check_proof_step


# --- DSPy Metric ---

def proof_quality_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Metric used by GEPA/MIPROv2 to score a generated proof.
    Runs all 5 judges and returns a score between 0 and 1.
    Each judge contributes 0.2 to the total score.
    """
    verifier = ProofVerificationSystem()
    proof = prediction.proof if hasattr(prediction, 'proof') else str(prediction)
    results = verifier.verify_proof(proof, example.ground_truth)

    judges = ['final_answer', 'toy_case', 'logical_gap',
              'numerical_approximation', 'numerical_computation']
    score = sum(0.2 for j in judges if results[j]['pass'])
    return score


# --- Training Examples for GEPA ---

def build_trainset() -> List[dspy.Example]:
    """Build a small training set from the inequality definitions"""
    trainset = []
    for name in ['cauchy_schwarz', 'jensen', 'triangle', 'bernoulli', 'young', 'chebyshev', 'markov']:
        ineq = get_inequality(name)
        if ineq:
            ex = dspy.Example(
                inequality_name=ineq['name'],
                problem_statement=ineq['statement'],
                relevant_theorems=", ".join(ineq.get('relevant_theorems', [])),
                ground_truth=ineq['ground_truth']
            ).with_inputs('inequality_name', 'problem_statement', 'relevant_theorems')
            trainset.append(ex)
    return trainset


# --- GEPA Optimizer Wrapper ---

class GEPAOptimizer:
    """
    Wraps DSPy MIPROv2 (which implements GEPA-style evolutionary prompt optimization).
    Optimizes the ProofGeneratorOptimized module's instructions automatically.
    Logs each iteration score to the session logger.
    """

    def __init__(self, num_candidates: int = 5, num_iterations: int = 2, logger=None, lm=None):
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        self.optimized_module = None
        self._logger = logger
        self._lm = lm  # Key 5 LM used for GEPA candidate evaluation

    def optimize(self, module: ProofGeneratorOptimized, save_path: str = None) -> ProofGeneratorOptimized:
        """Run MIPROv2 (GEPA) optimization on the proof generator module"""
        trainset = build_trainset()

        def _metric_with_log(example, prediction, trace=None):
            score = proof_quality_metric(example, prediction, trace)
            return score

        optimizer = MIPROv2(
            metric=_metric_with_log,
            num_candidates=self.num_candidates,
            init_temperature=1.0,
            auto=None,
            verbose=True
        )

        print("\n--- Running GEPA (MIPROv2) Prompt Optimization ---")
        with dspy.context(lm=self._lm) if self._lm else contextlib.nullcontext():
            self.optimized_module = optimizer.compile(
                module,
                trainset=trainset,
                num_trials=self.num_iterations,
                max_bootstrapped_demos=2,
                max_labeled_demos=2,
                minibatch=False,
            )
        print("--- Optimization Complete ---\n")

        if save_path:
            self.optimized_module.save(save_path)
            print(f"Optimized module saved → {save_path}")

        return self.optimized_module

# --- Metrics ---

class PromptMetrics:
    """Calculate and display metrics for prompt comparison"""

    @staticmethod
    def calculate_pass_rate(results_list: List[Dict]) -> float:
        if not results_list:
            return 0.0
        return sum(1 for r in results_list if r.get('overall', {}).get('pass', False)) / len(results_list)

    @staticmethod
    def calculate_judge_scores(results_list: List[Dict]) -> Dict[str, float]:
        if not results_list:
            return {}
        judges = ['final_answer', 'toy_case', 'logical_gap',
                  'numerical_approximation', 'numerical_computation']
        return {
            j: sum(1 for r in results_list if r.get(j, {}).get('pass', False)) / len(results_list)
            for j in judges
        }

    @staticmethod
    def calculate_improvement(unopt: List[Dict], opt: List[Dict]) -> Dict:
        unopt_rate = PromptMetrics.calculate_pass_rate(unopt)
        opt_rate = PromptMetrics.calculate_pass_rate(opt)
        unopt_judges = PromptMetrics.calculate_judge_scores(unopt)
        opt_judges = PromptMetrics.calculate_judge_scores(opt)

        return {
            'overall_pass_rate': {
                'unoptimized': unopt_rate,
                'optimized': opt_rate,
                'improvement': opt_rate - unopt_rate,
                'relative_improvement': ((opt_rate - unopt_rate) / unopt_rate * 100) if unopt_rate > 0 else 0
            },
            'judge_improvements': {
                j: {
                    'unoptimized': unopt_judges[j],
                    'optimized': opt_judges[j],
                    'improvement': opt_judges[j] - unopt_judges[j]
                }
                for j in unopt_judges
            }
        }

    @staticmethod
    def print_comparison(improvement: Dict):
        print("\n" + "="*80)
        print("PROMPT OPTIMIZATION RESULTS")
        print("="*80)
        overall = improvement['overall_pass_rate']
        print(f"\nOVERALL PASS RATE:")
        print(f"  Unoptimized: {overall['unoptimized']:.1%}")
        print(f"  Optimized:   {overall['optimized']:.1%}")
        print(f"  Improvement: {overall['improvement']:+.1%} ({overall['relative_improvement']:+.1f}%)")
        print(f"\nJUDGE-SPECIFIC IMPROVEMENTS:")
        for judge, scores in improvement['judge_improvements'].items():
            print(f"\n  {judge.replace('_', ' ').title()}:")
            print(f"    Unoptimized: {scores['unoptimized']:.1%}")
            print(f"    Optimized:   {scores['optimized']:.1%}")
            print(f"    Improvement: {scores['improvement']:+.1%}")
        print("\n" + "="*80 + "\n")

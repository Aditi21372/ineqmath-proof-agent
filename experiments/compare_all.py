"""
Comprehensive Comparison: Human vs LLM (Unoptimized) vs LLM (GEPA-Optimized)
Usage: python experiments/compare_all.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import MathReasoningAgent
from src.proof_generator import get_inequality
from src.prompt_optimizer import PromptMetrics
from datetime import datetime


HUMAN_PROOFS = {
    'cauchy_schwarz': """
For any vectors u and v in an inner product space, we prove: |⟨u, v⟩|² ≤ ⟨u, u⟩ · ⟨v, v⟩

Step 1: Consider ⟨u + tv, u + tv⟩ ≥ 0 for any real t.
Step 2: Expand: ⟨u, u⟩ + 2t⟨u, v⟩ + t²⟨v, v⟩ ≥ 0
Step 3: Quadratic At² + Bt + C ≥ 0 where A=⟨v,v⟩, B=2⟨u,v⟩, C=⟨u,u⟩
Step 4: Discriminant ≤ 0: (2⟨u,v⟩)² - 4⟨v,v⟩⟨u,u⟩ ≤ 0
Step 5: 4|⟨u,v⟩|² ≤ 4⟨u,u⟩⟨v,v⟩ → (∑aᵢbᵢ)² ≤ (∑aᵢ²)(∑bᵢ²). Equality when u,v proportional. Q.E.D.
""",
    'triangle': """
For vectors x and y in an inner product space, we prove: ||x + y|| ≤ ||x|| + ||y||

Step 1: ||x + y||² = ⟨x + y, x + y⟩
Step 2: = ||x||² + 2⟨x, y⟩ + ||y||²
Step 3: By Cauchy-Schwarz: ⟨x, y⟩ ≤ ||x|| ||y||
Step 4: ||x + y||² ≤ (||x|| + ||y||)²
Step 5: Taking square root: ||x + y|| ≤ ||x|| + ||y||. Equality when x,y positively proportional. Q.E.D.
"""
}


def main():
    agent = MathReasoningAgent()
    agent.run_gepa_optimization()

    test_cases = ['cauchy_schwarz', 'triangle']
    human_results, unopt_results, opt_results = [], [], []

    for name in test_cases:
        ineq = get_inequality(name)
        print(f"\n{'='*80}\nTESTING: {ineq['name']}\n{'='*80}")

        print("\n--- Human Proof ---")
        hr = agent.verifier.verify_proof(HUMAN_PROOFS[name], ineq['ground_truth'])
        human_results.append(hr)
        print(f"Result: {'✓ VERIFIED' if hr['overall']['pass'] else '✗ REJECTED'}")

        print("\n--- LLM Unoptimized ---")
        up = agent.unoptimized_generator(
            inequality_name=ineq['name'],
            problem_statement=ineq['statement']
        )
        ur = agent.verifier.verify_proof(up.proof, ineq['ground_truth'])
        unopt_results.append(ur)
        print(f"Result: {'✓ VERIFIED' if ur['overall']['pass'] else '✗ REJECTED'}")

        print("\n--- LLM GEPA-Optimized ---")
        op = agent.optimized_generator(
            inequality_name=ineq['name'],
            problem_statement=ineq['statement'],
            relevant_theorems=ineq.get('relevant_theorems', [])
        )
        orr = agent.verifier.verify_proof(op.proof, ineq['ground_truth'])
        opt_results.append(orr)
        print(f"Result: {'✓ VERIFIED' if orr['overall']['pass'] else '✗ REJECTED'}")

    human_rate = PromptMetrics.calculate_pass_rate(human_results)
    unopt_rate = PromptMetrics.calculate_pass_rate(unopt_results)
    opt_rate   = PromptMetrics.calculate_pass_rate(opt_results)

    print(f"\n{'='*80}\nOVERALL PASS RATES\n{'='*80}")
    print(f"  Human:        {human_rate:.1%}")
    print(f"  Unoptimized:  {unopt_rate:.1%}")
    print(f"  GEPA-Opt:     {opt_rate:.1%}")
    print(f"  Gap to Human: {human_rate - opt_rate:.1%}")

    human_j  = PromptMetrics.calculate_judge_scores(human_results)
    unopt_j  = PromptMetrics.calculate_judge_scores(unopt_results)
    opt_j    = PromptMetrics.calculate_judge_scores(opt_results)

    print(f"\n{'Judge':<28} {'Human':<10} {'Unopt':<10} {'GEPA-Opt':<10} {'Δ'}")
    print("-"*70)
    for j in human_j:
        delta = opt_j[j] - unopt_j[j]
        print(f"  {j.replace('_',' ').title():<26} {human_j[j]:<10.1%} {unopt_j[j]:<10.1%} {opt_j[j]:<10.1%} {delta:+.1%}")

    output = {
        'overall_rates': {'human': human_rate, 'llm_unoptimized': unopt_rate, 'llm_optimized': opt_rate},
        'judge_scores':  {'human': human_j, 'llm_unoptimized': unopt_j, 'llm_optimized': opt_j},
        'timestamp': datetime.now().isoformat()
    }
    agent._save_metrics(
        {'improvement_metrics': output, 'timestamp': output['timestamp'], 'unoptimized_results': unopt_results, 'optimized_results': opt_results},
        'comparison_results.json'
    )
    agent._save_log_summary()


if __name__ == "__main__":
    main()

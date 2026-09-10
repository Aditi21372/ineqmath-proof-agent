"""
Run Experiments — main entry point
Usage:
    python experiments/run_experiments.py           # full experiment
    python experiments/run_experiments.py --quick   # single inequality quick test
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import MathReasoningAgent


def run_full_experiment():
    agent = MathReasoningAgent()
    results = agent.run_comparison_experiment(['cauchy_schwarz', 'jensen', 'triangle', 'bernoulli', 'young', 'chebyshev', 'markov'])
    _generate_summary(results, agent.session_dir)
    print(f"\nSession results saved to: {agent.session_dir}/")


def quick_test():
    agent = MathReasoningAgent()
    agent.prove_and_verify('cauchy_schwarz', use_optimized=True)
    agent._save_log_summary()
    print(f"\nSession: {agent.session_dir}/")


def _generate_summary(results: dict, session_dir: str):
    improvement = results['improvement_metrics']
    overall = improvement['overall_pass_rate']

    lines = [
        "="*80,
        "EXPERIMENT SUMMARY",
        "="*80,
        f"Timestamp: {results['timestamp']}",
        "",
        f"Unoptimized Pass Rate: {overall['unoptimized']:.1%}",
        f"Optimized Pass Rate:   {overall['optimized']:.1%}",
        f"Absolute Improvement:  {overall['improvement']:+.1%}",
        f"Relative Improvement:  {overall['relative_improvement']:+.1f}%",
        "",
        "JUDGE-SPECIFIC:",
    ]
    for judge, scores in improvement['judge_improvements'].items():
        lines.append(
            f"  {judge.replace('_', ' ').title()}: "
            f"{scores['unoptimized']:.1%} → {scores['optimized']:.1%} "
            f"({scores['improvement']:+.1%})"
        )
    lines.append("="*80)

    path = os.path.join(session_dir, "summary_report.txt")
    with open(path, 'w') as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSummary saved to {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_full_experiment()

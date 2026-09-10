"""
Test human-provided proofs through the five-judge system
Usage: python experiments/test_human_proofs.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import dspy
from src.agent import KEY_GENERATOR as NVIDIA_API_KEY
from src.agent import NVIDIA_BASE_URL
from src.agent import JUDGE_FINAL_MODEL as JUDGE_MODEL
from src.agent import JUDGE_MAX_TOKENS
from src.judges import ProofVerificationSystem
from src.proof_generator import get_inequality


CAUCHY_SCHWARZ_PROOF = """
## Cauchy-Schwarz Inequality Proof

For any vectors u and v in an inner product space, we prove:
|⟨u, v⟩|² ≤ ⟨u, u⟩ · ⟨v, v⟩

**Step 1:** ⟨u + tv, u + tv⟩ ≥ 0 for any real scalar t (non-negativity of inner product).
**Step 2:** Expanding: ⟨u, u⟩ + 2t⟨u, v⟩ + t²⟨v, v⟩ ≥ 0
**Step 3:** Quadratic At² + Bt + C ≥ 0 where A=⟨v,v⟩, B=2⟨u,v⟩, C=⟨u,u⟩.
**Step 4:** Discriminant ≤ 0: (2⟨u,v⟩)² - 4⟨v,v⟩⟨u,u⟩ ≤ 0
**Step 5:** 4|⟨u,v⟩|² ≤ 4⟨u,u⟩⟨v,v⟩ → (∑aᵢbᵢ)² ≤ (∑aᵢ²)(∑bᵢ²).
Equality when u and v are linearly dependent. Q.E.D.
"""

TRIANGLE_INEQUALITY_PROOF = """
## Triangle Inequality Proof

For vectors x and y in an inner product space, we prove: ||x + y|| ≤ ||x|| + ||y||

**Step 1:** ||x + y||² = ⟨x + y, x + y⟩
**Step 2:** = ||x||² + 2⟨x, y⟩ + ||y||²
**Step 3:** By Cauchy-Schwarz: ⟨x, y⟩ ≤ |⟨x, y⟩| ≤ ||x|| ||y||
**Step 4:** ||x + y||² ≤ ||x||² + 2||x|| ||y|| + ||y||² = (||x|| + ||y||)²
**Step 5:** Taking square root (norms non-negative): ||x + y|| ≤ ||x|| + ||y||.
Equality when y = cx for some c ≥ 0. Q.E.D.
"""


def main():
    if not NVIDIA_API_KEY:
        print("ERROR: NVIDIA_API_KEY not set. Copy .env.template to .env and fill in your key.")
        return

    lm = dspy.LM(
        model=JUDGE_MODEL,
        api_key=NVIDIA_API_KEY,
        api_base=NVIDIA_BASE_URL,
        max_tokens=JUDGE_MAX_TOKENS
    )
    dspy.configure(lm=lm)

    verifier = ProofVerificationSystem()

    proofs = [
        (CAUCHY_SCHWARZ_PROOF, 'cauchy_schwarz'),
        (TRIANGLE_INEQUALITY_PROOF, 'triangle'),
    ]

    all_results = {}
    for proof_text, ineq_name in proofs:
        ineq = get_inequality(ineq_name)
        print(f"\n{'='*80}\nTESTING: {ineq['name']}\n{'='*80}")
        results = verifier.verify_proof(proof_text, ineq['ground_truth'])
        verifier.print_results(results)
        all_results[ineq_name] = results

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, results in all_results.items():
        ineq = get_inequality(name)
        overall = "✓ VERIFIED" if results['overall']['pass'] else "✗ REJECTED"
        print(f"\n{ineq['name']}: {overall}")
        for j in ['final_answer', 'toy_case', 'logical_gap',
                  'numerical_approximation', 'numerical_computation']:
            s = '✓' if results[j]['pass'] else '✗'
            print(f"  {s} {j.replace('_', ' ').title()}")


if __name__ == "__main__":
    main()

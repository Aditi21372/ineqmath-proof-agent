"""
Five-Judge Verification System using DSPy Signatures.
LogicalGapJudge uses dspy.ReAct so it can call verify_with_lean
to get machine-checked verdicts on specific proof steps.
All other judges remain plain dspy.Predict.
"""

import dspy
from typing import Dict


# --- DSPy Signatures ---

class FinalAnswerJudge(dspy.Signature):
    """
    Verify if the proof's final answer is mathematically equivalent to the ground truth.
    Consider different representations (e.g., 1/√2 = √2/2) and symbolic equivalence.
    For equality conditions, accept any subset or superset that is mathematically correct
    — do NOT fail a proof for stating a more general or more specific (but still correct)
    equality condition than the ground truth.
    Respond with CORRECT or INCORRECT on the first line, then a brief explanation.
    """
    proof: str        = dspy.InputField(desc="The mathematical proof to evaluate")
    ground_truth: str = dspy.InputField(desc="The expected correct answer")
    verdict: str      = dspy.OutputField(desc="CORRECT or INCORRECT followed by explanation")


class ToyCaseJudge(dspy.Signature):
    """
    Detect TOY CASE errors: using specific numerical examples (e.g., a=1, b=2) to
    conclude a general statement without proper justification.
    Valid uses: checking boundary cases AFTER a general proof, or finding equality conditions.
    Respond with PASS or FAIL on the first line, then explain your reasoning.
    """
    proof: str   = dspy.InputField(desc="The mathematical proof to evaluate")
    verdict: str = dspy.OutputField(desc="PASS or FAIL followed by explanation")


class LogicalGapJudge(dspy.Signature):
    """
    Identify LOGICAL GAPS: missing reasoning steps, unjustified claims like 'by intuition'
    or 'clearly', or unclear transitions between steps.

    You have access to the verify_with_lean tool.
    For any specific claim or step you are unsure about, call:
      verify_with_lean(lean_code)
    with a self-contained Lean 4 snippet that checks that exact claim.
    Use the Lean result as evidence in your verdict.

    If Lean is not installed, rely on your own mathematical reasoning.

    IMPORTANT: When evaluating equality conditions, accept any equality condition
    that is correctly derived from the proof steps, even if it differs in phrasing
    from a textbook statement. Do NOT fail a proof solely because its equality
    condition is more specific or more general than expected, as long as it follows
    logically from the proof.

    Respond with PASS or FAIL on the first line, then explain your reasoning
    and include any Lean verification results you obtained.
    """
    proof: str   = dspy.InputField(desc="The mathematical proof to evaluate")
    verdict: str = dspy.OutputField(desc="PASS or FAIL followed by explanation and any Lean results")


class NumericalApproximationJudge(dspy.Signature):
    """
    Detect NUMERICAL APPROXIMATION errors: using decimal approximations (e.g., √2 ≈ 1.414,
    π ≈ 3.14) in the proof that compromise mathematical rigor.
    Approximations are only valid as side remarks clearly marked as such.
    Respond with PASS or FAIL on the first line, then explain your reasoning.
    """
    proof: str   = dspy.InputField(desc="The mathematical proof to evaluate")
    verdict: str = dspy.OutputField(desc="PASS or FAIL followed by explanation")


class NumericalComputationJudge(dspy.Signature):
    """
    Verify all arithmetic computations in the proof: addition, subtraction, multiplication,
    division, exponentiation, roots, and algebraic simplifications.
    Respond with PASS or FAIL on the first line, then list any errors found or confirm correctness.
    """
    proof: str   = dspy.InputField(desc="The mathematical proof to evaluate")
    verdict: str = dspy.OutputField(desc="PASS or FAIL followed by list of errors or confirmation")


# --- DSPy Judge Modules ---

class ProofVerificationSystem(dspy.Module):
    """
    Five-judge verification system.
    - Four judges: plain dspy.Predict
    - LogicalGapJudge: dspy.ReAct with verify_with_lean tool
      so it can formally check specific steps in Lean 4.
    """

    def __init__(self, final_lm=None, toycase_lm=None, logical_lm=None, numcomp_lm=None):
        from src.tools import verify_with_lean

        self.final_answer = dspy.Predict(FinalAnswerJudge,            lm=final_lm)   if final_lm   else dspy.Predict(FinalAnswerJudge)
        self.toy_case     = dspy.Predict(ToyCaseJudge,                lm=toycase_lm) if toycase_lm else dspy.Predict(ToyCaseJudge)
        self.num_approx   = dspy.Predict(NumericalApproximationJudge, lm=toycase_lm) if toycase_lm else dspy.Predict(NumericalApproximationJudge)
        self.num_comp     = dspy.Predict(NumericalComputationJudge,   lm=numcomp_lm) if numcomp_lm else dspy.Predict(NumericalComputationJudge)

        # ReAct doesn't accept lm in constructor — wrap forward call with dspy.context
        self._logical_lm  = logical_lm
        self.logical_gap  = dspy.ReAct(LogicalGapJudge, tools=[verify_with_lean])

    def forward(self, proof: str, ground_truth: str) -> Dict:
        results = {}

        r1 = self.final_answer(proof=proof, ground_truth=ground_truth)
        results['final_answer'] = self._parse(r1.verdict, keywords=('CORRECT',), anti=('INCORRECT',))

        r2 = self.toy_case(proof=proof)
        results['toy_case'] = self._parse(r2.verdict)

        # LogicalGapJudge uses its own LM via dspy.context
        import contextlib
        ctx = dspy.context(lm=self._logical_lm) if self._logical_lm else contextlib.nullcontext()
        with ctx:
            r3 = self.logical_gap(proof=proof)
        results['logical_gap'] = self._parse(r3.verdict)

        r4 = self.num_approx(proof=proof)
        results['numerical_approximation'] = self._parse(r4.verdict)

        r5 = self.num_comp(proof=proof)
        results['numerical_computation'] = self._parse(r5.verdict)

        results['overall'] = {'pass': all(v['pass'] for v in results.values())}
        return results

    def _parse(self, verdict: str, keywords=('PASS',), anti=('FAIL',)) -> Dict:
        first_line = verdict.split('\n')[0].upper()
        passed = any(k in first_line for k in keywords) and not any(a in first_line for a in anti)
        return {'pass': passed, 'message': verdict}

    def verify_proof(self, proof: str, ground_truth: str) -> Dict:
        return self.forward(proof=proof, ground_truth=ground_truth)

    def print_results(self, results: Dict):
        print("\n" + "="*80)
        print("PROOF VERIFICATION RESULTS")
        print("="*80)
        judges = [
            ('Final Answer',            'final_answer'),
            ('Toy Case',                'toy_case'),
            ('Logical Gap (+ Lean)',    'logical_gap'),
            ('Numerical Approximation', 'numerical_approximation'),
            ('Numerical Computation',   'numerical_computation'),
        ]
        for name, key in judges:
            status = "✓ PASS" if results[key]['pass'] else "✗ FAIL"
            print(f"\n{name} Judge: {status}")
            print(f"  {results[key]['message'][:200]}...")
        print("\n" + "="*80)
        overall = "✓ PROOF VERIFIED" if results['overall']['pass'] else "✗ PROOF REJECTED"
        print(f"OVERALL: {overall}")
        print("="*80 + "\n")

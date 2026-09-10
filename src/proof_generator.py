"""
Proof Generation Module using DSPy
Step-by-step pipeline: each stage is a separate LM call so progress is visible.

Pipeline (optimized):
  Step 1 — PlanProof:   decide proof strategy and outline steps  (dspy.Predict)
  Step 2 — WriteProof:  write full proof via ReAct with tools    (dspy.ReAct)
  Step 3 — VerifySteps: self-check and refine                    (dspy.Predict)
  Step 4 — WriteLatex:  render to LaTeX and save to session      (dspy.Predict)

Unoptimized: single Predict call, no steps, no tools.
"""

import dspy
from typing import List


# ── Step Signatures ──────────────────────────────────────────────────────────

class PlanProof(dspy.Signature):
    """
    Given an inequality, decide the best proof strategy and outline the key steps.
    Be specific: name the exact theorems, techniques, and algebraic manipulations needed.
    Output a numbered outline — do NOT write the full proof yet.
    The plan MUST include:
    - A step for handling any degenerate/edge cases (e.g. zero vectors, zero weights)
    - A final step deriving the complete equality condition (ALL cases under which equality holds)
    """
    inequality_name: str   = dspy.InputField(desc="Name of the inequality")
    problem_statement: str = dspy.InputField(desc="Formal mathematical statement")
    relevant_theorems: str = dspy.InputField(desc="Available theorems to use")
    proof_plan: str        = dspy.OutputField(desc="Numbered outline of proof steps with specific techniques")


class WriteProof(dspy.Signature):
    """
    Write a complete rigorous mathematical proof following the given plan.

    You have access to tools — USE THEM at every non-trivial step:
      - Call lookup_inequality(name) to get the precise statement if needed
      - Call verify_arithmetic(expression) to confirm every algebraic manipulation
        e.g. verify_arithmetic('(a+b)**2 - (a**2 + 2*a*b + b**2)') before writing it
      - Call check_proof_step(claim) to validate each inequality step before including it

    Requirements:
    - Every step must follow logically from the previous
    - Cite every theorem by name when used
    - Show all algebraic manipulations explicitly
    - Handle ALL degenerate/edge cases explicitly (e.g. if a denominator could be zero, treat that case separately)
    - Do NOT use specific numerical examples to prove general statements
    - Do NOT use decimal approximations (no sqrt(2) = 1.414)
    - State the equality condition at the end — it must be FULLY DERIVED from the proof steps, not just asserted
    - The equality condition must be complete: state ALL cases under which equality holds
    """
    inequality_name: str   = dspy.InputField(desc="Name of the inequality")
    problem_statement: str = dspy.InputField(desc="Formal mathematical statement")
    proof_plan: str        = dspy.InputField(desc="Proof outline to follow")
    proof: str             = dspy.OutputField(desc="Complete rigorous proof with all steps verified via tools")


class VerifyAndRefine(dspy.Signature):
    """
    Review the proof for errors. Check:
    1. Are there any logical gaps or unjustified claims?
    2. Are all algebraic steps correct?
    3. Is the equality condition stated correctly AND fully justified by the proof steps?
       If the equality condition is stated but not derived from the proof, remove or justify it.
    4. Are there any claims introduced at the end that were not proven earlier?
    If errors found, output a corrected proof. If correct, output the proof unchanged.
    """
    proof: str         = dspy.InputField(desc="The proof to verify and refine")
    ground_truth: str  = dspy.InputField(desc="Expected final answer for reference")
    refined_proof: str = dspy.OutputField(desc="Corrected proof, or original if no errors found")


class WriteLatex(dspy.Signature):
    """
    Convert the mathematical proof to clean LaTeX.
    Use: \\begin{proof}...\\end{proof}, align* for equations, \\leq \\geq for inequalities.
    Output ONLY the LaTeX body — no \\documentclass or \\begin{document}.
    """
    proof: str      = dspy.InputField(desc="Plain text proof to convert")
    latex_body: str = dspy.OutputField(desc="LaTeX body of the proof, ready to compile")


class GenerateProofUnoptimized(dspy.Signature):
    """Prove the given mathematical inequality."""
    inequality_name: str   = dspy.InputField()
    problem_statement: str = dspy.InputField()
    proof: str             = dspy.OutputField()


# ── DSPy Modules ─────────────────────────────────────────────────────────────

class ProofGeneratorOptimized(dspy.Module):
    """
    Step-by-step proof generator — each stage is a separate LM call.

    Steps:
      1. Plan   — strategy + outline     (dspy.Predict, fast)
      2. Write  — full proof via ReAct   (dspy.ReAct — LLM calls verify_arithmetic
                                          + check_proof_step mid-proof)
      3. Verify — self-check + refine    (dspy.Predict)
      4. LaTeX  — render + save          (dspy.Predict + save_output tool)
    """

    def __init__(self, tools: List = None, logger=None):
        self._tools = tools or []
        self._logger = logger

        # Step 1: plain Predict — just needs a plan, no tools
        self.plan = dspy.Predict(PlanProof)

        # Step 2: ReAct — LLM actively calls verify_arithmetic + check_proof_step
        write_tools = [
            t for t in self._tools
            if getattr(t, '__name__', '') in ('verify_arithmetic', 'check_proof_step', 'lookup_inequality')
        ]
        wrapped_tools = [self._wrap(t) for t in write_tools]
        self.write = dspy.ReAct(WriteProof, tools=wrapped_tools) if wrapped_tools else dspy.ChainOfThought(WriteProof)

        # Step 3: plain Predict — review and refine
        self.verify = dspy.Predict(VerifyAndRefine)

        # Step 4: plain Predict — convert to LaTeX
        self.latex = dspy.Predict(WriteLatex)

    def __deepcopy__(self, memo):
        import copy
        new = ProofGeneratorOptimized(tools=self._tools, logger=None)
        memo[id(self)] = new
        new.plan   = copy.deepcopy(self.plan,   memo)
        new.write  = copy.deepcopy(self.write,  memo)
        new.verify = copy.deepcopy(self.verify, memo)
        new.latex  = copy.deepcopy(self.latex,  memo)
        return new

    def _wrap(self, tool):
        """Wrap a tool to print + log every call to the JSON trace."""
        import functools
        import time as _time

        @functools.wraps(tool)
        def wrapped(*args, **kwargs):
            t0 = _time.time()
            result = tool(*args, **kwargs)
            elapsed = round((_time.time() - t0) * 1000, 1)
            # Build a readable call signature for display
            call_args = ', '.join(
                [repr(a) for a in args] +
                [f"{k}={repr(v)}" for k, v in kwargs.items()]
            )
            print(f"    \U0001f527 Tool: {tool.__name__}({call_args[:80]}) \u2192 {str(result)[:80]}  ({elapsed}ms)", flush=True)
            if self._logger:
                self._logger.log_tool_call(
                    tool_name=tool.__name__,
                    args={**{f'arg_{i}': repr(a) for i, a in enumerate(args)}, **{k: repr(v) for k, v in kwargs.items()}},
                    result=str(result),
                    duration_ms=elapsed
                )
            return result

        return wrapped

    def forward(self, inequality_name: str, problem_statement: str,
                relevant_theorems: List[str] = None) -> dspy.Prediction:

        theorems_str = ", ".join(relevant_theorems) if relevant_theorems else "standard algebraic techniques"

        # Step 1 — Plan
        print(f"\n  [1/4] Planning proof strategy for {inequality_name}...", flush=True)
        plan_result = self.plan(
            inequality_name=inequality_name,
            problem_statement=problem_statement,
            relevant_theorems=theorems_str
        )
        print(f"  \u2713 Plan ready:\n{_indent(plan_result.proof_plan[:300])}", flush=True)

        # Step 2 — Write (ReAct with tools)
        print(f"\n  [2/4] Writing full proof (with tool calls)...", flush=True)
        write_result = self.write(
            inequality_name=inequality_name,
            problem_statement=problem_statement,
            proof_plan=plan_result.proof_plan
        )
        proof = write_result.proof
        print(f"  \u2713 Proof written ({len(proof)} chars)", flush=True)

        # Step 3 — Verify & Refine
        print(f"\n  [3/4] Verifying and refining proof...", flush=True)
        verify_result = self.verify(
            proof=proof,
            ground_truth=_get_ground_truth(inequality_name)
        )
        proof = verify_result.refined_proof
        print(f"  \u2713 Proof verified/refined ({len(proof)} chars)", flush=True)

        # Step 4 — LaTeX + save
        print(f"\n  [4/4] Rendering to LaTeX...", flush=True)
        latex_result = self.latex(proof=proof)
        latex_body = latex_result.latex_body

        save_tool = next((t for t in self._tools if getattr(t, '__name__', '') == 'save_output'), None)
        if save_tool:
            safe_name = inequality_name.lower().replace(' ', '_').replace("'", '')
            result_msg = save_tool(latex_body, f"{safe_name}_proof.tex")
            print(f"  \u2713 {result_msg}", flush=True)
        else:
            print(f"  \u2713 LaTeX ready ({len(latex_body)} chars)", flush=True)

        return dspy.Prediction(proof=proof, latex=latex_body, plan=plan_result.proof_plan)


class ProofGeneratorUnoptimized(dspy.Module):
    """Baseline — single Predict call, no steps, no tools."""

    def __init__(self):
        self.generate = dspy.Predict(GenerateProofUnoptimized)

    def forward(self, inequality_name: str, problem_statement: str, **kwargs):
        print(f"\n  [unoptimized] Single call to generate proof...", flush=True)
        result = self.generate(
            inequality_name=inequality_name,
            problem_statement=problem_statement
        )
        print(f"  \u2713 Done ({len(result.proof)} chars)", flush=True)
        return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _indent(text: str, spaces: int = 4) -> str:
    return "\n".join(" " * spaces + line for line in text.splitlines())


def _get_ground_truth(inequality_name: str) -> str:
    ineq = INEQUALITIES.get(inequality_name.lower().replace(' ', '_').replace("'s", '').replace("'", ''))
    return ineq['ground_truth'] if ineq else "See inequality definition"


# ── Inequality Definitions ────────────────────────────────────────────────────

INEQUALITIES = {  # noqa: E501
    'cauchy_schwarz': {
        'name': 'Cauchy-Schwarz Inequality',
        'statement': 'For real numbers a₁, a₂, ..., aₙ and b₁, b₂, ..., bₙ, prove that: (∑aᵢbᵢ)² ≤ (∑aᵢ²)(∑bᵢ²)',
        'ground_truth': '(∑aᵢbᵢ)² ≤ (∑aᵢ²)(∑bᵢ²) with equality when aᵢ and bᵢ are proportional (including the degenerate case when all bᵢ=0)',
        'relevant_theorems': [
            'Quadratic discriminant: a quadratic At²+Bt+C ≥ 0 for all real t with A ≥ 0 implies discriminant B²-4AC ≤ 0 (NON-POSITIVE)',
            'Degenerate case A=0: if ∑bᵢ²=0 then all bᵢ=0 AND if ∑aᵢ²=0 then all aᵢ=0 — both sides are 0, inequality holds trivially',
            'Properties of inner products',
            'Expansion of squared sums'
        ]
    },
    'jensen': {
        'name': "Jensen's Inequality",
        'statement': 'For a convex function f and real numbers x₁, x₂, ..., xₙ with weights λᵢ ≥ 0 where ∑λᵢ = 1, prove that: f(∑λᵢxᵢ) ≤ ∑λᵢf(xᵢ)',
        'ground_truth': 'f(∑λᵢxᵢ) ≤ ∑λᵢf(xᵢ) for convex f, with equality when all xᵢ are equal, or f is affine on the convex hull of the xᵢ, or all weight is on a single point',
        'relevant_theorems': [
            'Definition of convex function: f(tx + (1-t)y) ≤ tf(x) + (1-t)f(y) for t ∈ [0,1]',
            'Mathematical induction',
            'Properties of convex combinations',
            'Equality in the base case (n=2): f(λx+(1-λ)y)=λf(x)+(1-λ)f(y) iff x=y or λ∈{0,1}',
            'Induction degenerate case: when defining αᵢ = λᵢ/μ where μ=∑ᵢ₌₁ⁿ⁻¹λᵢ, handle μ=0 separately (means all weights except last are 0, so result is trivial)'
        ]
    },
    'triangle': {
        'name': 'Triangle Inequality',
        'statement': 'For vectors x and y in an inner product space, prove that: ||x + y|| ≤ ||x|| + ||y||',
        'ground_truth': '||x + y|| ≤ ||x|| + ||y|| with equality when x and y are positively proportional',
        'relevant_theorems': [
            'Cauchy-Schwarz Inequality',
            'Properties of inner products and norms',
            'Squaring both sides (when both are non-negative)'
        ]
    },
    'bernoulli': {
        'name': "Bernoulli's Inequality",
        'statement': 'For real number x ≥ -1 and integer n ≥ 0, prove that: (1 + x)ⁿ ≥ 1 + nx',
        'ground_truth': '(1 + x)ⁿ ≥ 1 + nx for x ≥ -1 and n ≥ 0, with equality iff x=0 or n≤1',
        'relevant_theorems': [
            'Mathematical induction',
            'Base cases: n=0 gives 1≥1 (true); n=1 gives 1+x≥1+x (true)',
            'Edge case x=-1, n=0: (1+(-1))^0=1≥1+0=1 (true). For n≥1: 0ⁿ=0≥1-n since n≥1',
            'Inductive step: (1+x)^(n+1)=(1+x)^n*(1+x)≥(1+nx)(1+x) since 1+x≥0',
            'Properties of inequalities under multiplication'
        ]
    },
    'young': {
        'name': "Young's Inequality",
        'statement': 'For non-negative real numbers a, b and p, q > 1 with 1/p + 1/q = 1, prove that: ab ≤ aᵖ/p + bᵍ/q',
        'ground_truth': 'ab ≤ aᵖ/p + bᵍ/q with equality when aᵖ = bᵍ',
        'relevant_theorems': [
            'Weighted AM-GM inequality',
            'Convexity of exponential function',
            'Logarithmic properties',
            'Equality: from aᵖ=bᵠ, since q-1=1/(p-1), a=b^(q-1)=b^(1/(p-1)) NOT b^(p/(p-1))'
        ]
    },
    'chebyshev': {
        'name': "Chebyshev's Sum Inequality",
        'statement': 'For real numbers a₁ ≤ a₂ ≤ ... ≤ aₙ and b₁ ≤ b₂ ≤ ... ≤ bₙ, prove that: n∑aᵢbᵢ ≥ (∑aᵢ)(∑bᵢ)',
        'ground_truth': 'n∑aᵢbᵢ ≥ (∑aᵢ)(∑bᵢ) for similarly ordered sequences',
        'relevant_theorems': [
            'Rearrangement inequality',
            'Abel summation',
            'Properties of covariance'
        ]
    },
    'markov': {
        'name': "Markov's Inequality",
        'statement': 'For a non-negative random variable X and a > 0, prove that: P(X ≥ a) ≤ E[X]/a',
        'ground_truth': 'P(X ≥ a) ≤ E[X]/a for non-negative X and a > 0',
        'relevant_theorems': [
            'Definition of expectation',
            'Properties of probability measures',
            'Indicator functions'
        ]
    }
}


def get_inequality(name: str) -> dict:
    return INEQUALITIES.get(name)

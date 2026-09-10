"""
DSPy Tools for the proof generation agent.

Tools exposed to the LLM via dspy.ReAct:
  1. lookup_inequality(name)         — get formal statement + theorems
  2. verify_arithmetic(expression)   — symbolically verify a computation via SymPy
  3. check_proof_step(claim)         — validate an algebraic step via SymPy
  4. verify_with_lean(lean_code)     — formally verify a statement using Lean 4
  5. save_output(content, filename)  — LLM saves whatever it wants to the session folder
  6. compile_latex_proof(body, filename) — compile LaTeX to PDF with auto-retry
"""

import os
import subprocess
import tempfile
import shutil
import sympy as sp
from dotenv import load_dotenv
import dspy

load_dotenv()

LATEX_COMPILER    = os.getenv("LATEX_COMPILER",   "pdflatex")
LATEX_MAX_RETRIES = int(os.getenv("LATEX_MAX_RETRIES", "3"))

# Session output dir — injected at runtime by the agent
_SESSION_DIR: str = "results"


def set_session_dir(path: str):
    """Called once by MathReasoningAgent.__init__ to point tools at the session folder."""
    global _SESSION_DIR
    _SESSION_DIR = path


from src.proof_generator import INEQUALITIES


def lookup_inequality(name: str) -> str:
    """
    Look up the formal statement, ground truth, and relevant theorems for a named inequality.
    Call this FIRST before writing any proof to get the precise mathematical statement.

    Args:
        name: One of: cauchy_schwarz, jensen, triangle, bernoulli, young, chebyshev, markov

    Returns:
        The inequality's formal statement, ground truth answer, and list of useful theorems.

    Example call: lookup_inequality("cauchy_schwarz")
    """
    ineq = INEQUALITIES.get(name.lower().replace(" ", "_").replace("-", "_"))
    if not ineq:
        available = ", ".join(INEQUALITIES.keys())
        return f"Unknown inequality '{name}'. Available: {available}"

    theorems = "\n  - ".join(ineq.get("relevant_theorems", []))
    return (
        f"Name: {ineq['name']}\n"
        f"Statement: {ineq['statement']}\n"
        f"Ground Truth: {ineq['ground_truth']}\n"
        f"Relevant Theorems:\n  - {theorems}"
    )


def verify_arithmetic(expression: str) -> str:
    """
    Symbolically verify whether an arithmetic or algebraic expression simplifies to zero.
    Use this to confirm that A == B by passing 'A - (B)' and checking if the result is zero.
    Call this whenever you perform an algebraic manipulation and want to confirm it is correct.

    Args:
        expression: A SymPy-parseable string. To check if A equals B, pass 'A - (B)'.
                    Variables like a, b, x, y, n are treated as symbols automatically.
                    Examples:
                      '(a+b)**2 - (a**2 + 2*a*b + b**2)'  → Is zero: True  (correct expansion)
                      'a**2 + b**2 - 2*a*b'                → simplifies to (a-b)**2
                      '4*B**2 - 4*A*C'                     → discriminant expression

    Returns:
        Simplified form, expanded form, and whether the expression equals zero.

    Example call: verify_arithmetic("(a+b)**2 - (a**2 + 2*a*b + b**2)")
    """
    try:
        expr = sp.sympify(expression)
        simplified = sp.simplify(expr)
        expanded = sp.expand(expr)
        factored = sp.factor(expr)
        return (
            f"Simplified: {simplified} | "
            f"Expanded: {expanded} | "
            f"Factored: {factored} | "
            f"Is zero: {simplified == 0}"
        )
    except Exception as e:
        return f"Error parsing expression '{expression}': {e}"


def check_proof_step(claim: str) -> str:
    """
    Verify a specific algebraic claim or inequality step using SymPy.
    Use this DURING the proof to validate each non-trivial step before including it.
    This catches errors like wrong discriminant signs, incorrect expansions, or false inequalities.

    Args:
        claim: A string describing the step to verify. Can be:
               - An equation to check: 'expand (u+tv, u+tv) = ||u||^2 + 2t<u,v> + t^2||v||^2'
               - An expression to simplify: '4*inner_uv**2 - 4*norm_u**2 * norm_v**2'
               - A factoring check: '(norm_x + norm_y)**2 - norm_x**2 - 2*norm_x*norm_y - norm_y**2'
               For symbolic checks, use SymPy-parseable expressions with named variables.

    Returns:
        Verification result: simplified/expanded form and whether the claim holds (is zero).

    Example call: check_proof_step("4*B**2 - 4*A*C - (2*B)**2 + 4*A*C")
    """
    try:
        expr = sp.sympify(claim)
        simplified = sp.simplify(expr)
        expanded = sp.expand(expr)
        return (
            f"Step verification — Simplified: {simplified} | "
            f"Expanded: {expanded} | "
            f"Holds (is zero): {simplified == 0}"
        )
    except Exception as e:
        # If not directly parseable, return guidance
        return (
            f"Could not parse '{claim}' as a SymPy expression. "
            f"Tip: use Python syntax — e.g. '**' for powers, '*' for multiplication. "
            f"Error: {e}"
        )


LEAN_TIMEOUT = int(os.getenv("LEAN_TIMEOUT", "60"))
LEAN_PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lean_checker")
LEAN_CHECK_FILE  = os.path.join(LEAN_PROJECT_DIR, "LeanChecker", "Check.lean")


def verify_with_lean(lean_code: str) -> str:
    """
    Formally verify a mathematical statement or proof step using Lean 4 + Mathlib.
    Use this inside the LogicalGap judge to get a machine-checked verdict on a specific claim.
    Lean either accepts or rejects the proof with mathematical certainty.

    The snippet runs inside a Lake project with Mathlib, so all of the following work:
      nlinarith, linarith, ring, norm_num, positivity, field_simp, gcongr
      import Mathlib.Algebra.Order.Ring.Lemmas  (and any other Mathlib import)

    Args:
        lean_code: A self-contained Lean 4 snippet. Include any needed imports.
                   Examples:
                     -- AM-GM step
                     import Mathlib
                     example (a b : \u211d) : a ^ 2 + b ^ 2 \u2265 2 * a * b := by nlinarith [sq_nonneg (a - b)]

                     -- Cauchy-Schwarz discriminant step
                     import Mathlib
                     example (A B C : \u211d) (hA : A > 0) (hq : \u2200 t : \u211d, A * t^2 + B * t + C \u2265 0) :
                         B^2 - 4*A*C \u2264 0 := by nlinarith [hq ((-B) / (2*A)), sq_nonneg (2*A*((-B)/(2*A)) + B)]

    Returns:
        'VERIFIED: Lean accepted the proof.' on success,
        'FAILED: <lean error>' on proof failure,
        'LEAN_NOT_INSTALLED: ...' if Lean/Lake not found.

    Example call: verify_with_lean("import Mathlib\nexample (a b : \u211d) : a^2 + b^2 \u2265 2*a*b := by nlinarith [sq_nonneg (a-b)]")
    """
    lake_bin = _find_lake()
    if lake_bin is None:
        return (
            "LEAN_NOT_INSTALLED: lake not found. "
            "Install via: curl https://elan.lean-lang.org/elan-init.sh -sSf | sh. "
            "Falling back to LLM-only judgment."
        )

    # Write snippet into LeanChecker/Check.lean
    os.makedirs(os.path.dirname(LEAN_CHECK_FILE), exist_ok=True)
    with open(LEAN_CHECK_FILE, 'w') as f:
        f.write(lean_code)

    try:
        result = subprocess.run(
            [lake_bin, "build", "LeanChecker"],
            capture_output=True, text=True,
            timeout=LEAN_TIMEOUT,
            cwd=LEAN_PROJECT_DIR
        )
        output = (result.stderr + result.stdout).strip()
        # Strip absolute paths and noisy build trace lines
        clean = []
        for line in output.splitlines():
            if any(skip in line for skip in ["LEAN_PATH=", "trace: .>", ".elan/toolchains", ".lake/build"]):
                continue
            line = line.replace(LEAN_PROJECT_DIR + "/", "")
            clean.append(line)
        output = "\n".join(clean).strip()
        if result.returncode == 0 and "error" not in output.lower():
            return "VERIFIED: Lean accepted the proof."
        return f"FAILED: {output[:600]}"
    except subprocess.TimeoutExpired:
        return f"LEAN_TIMEOUT: Lean did not respond within {LEAN_TIMEOUT}s."
    except FileNotFoundError:
        return "LEAN_NOT_INSTALLED: lake binary not executable."


def _find_lean() -> str | None:
    import shutil as _shutil
    found = _shutil.which("lean")
    if found:
        return found
    for candidate in [os.path.expanduser("~/.elan/bin/lean"), "/usr/local/bin/lean", "/opt/homebrew/bin/lean"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _find_lake() -> str | None:
    import shutil as _shutil
    found = _shutil.which("lake")
    if found:
        return found
    for candidate in [os.path.expanduser("~/.elan/bin/lake"), "/usr/local/bin/lake", "/opt/homebrew/bin/lake"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def save_output(content: str, filename: str) -> str:
    """
    Save any content you want to the session results folder.
    Use this to persist your proof, working notes, LaTeX source, XML, or any other output.
    The file is saved exactly as you provide it — no modification.

    Args:
        content:  The text content to save. Can be plain text, LaTeX, XML, JSON, or anything.
        filename: Filename including extension, e.g.:
                  'cauchy_schwarz_proof.txt'   — plain text proof
                  'cauchy_schwarz_proof.tex'   — LaTeX source (will also be auto-compiled to PDF)
                  'working_notes.txt'          — scratch notes
                  'proof_steps.xml'            — structured XML

    Returns:
        The full path where the file was saved.

    Example call: save_output("By AM-GM...", "cauchy_schwarz_proof.txt")
    """
    path = os.path.join(_SESSION_DIR, filename)
    os.makedirs(_SESSION_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    # If it's LaTeX, auto-compile to PDF
    if filename.endswith('.tex'):
        pdf_result = compile_latex_proof(content, filename[:-4], output_dir=_SESSION_DIR)
        return f"Saved: {path} | LaTeX compile: {pdf_result}"

    return f"Saved: {path}"


class _FixLatex(dspy.Signature):
    """Fix broken LaTeX source given the compiler error log. Return only the corrected LaTeX body, no explanation."""
    broken_latex: str = dspy.InputField(desc="The LaTeX body that failed to compile")
    error_log: str = dspy.InputField(desc="The LaTeX compiler error log")
    fixed_latex: str = dspy.OutputField(desc="Corrected LaTeX body that will compile successfully")


def _try_compile(latex_body: str, filename: str, tmpdir: str) -> tuple[bool, str, str]:
    """Attempt one compile. Returns (success, pdf_path_or_empty, error_log)."""
    document = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath, amssymb, amsthm}\n"
        "\\newtheorem*{theorem}{Theorem}\n"
        "\\begin{document}\n"
        + latex_body +
        "\n\\end{document}\n"
    )
    tex_path = os.path.join(tmpdir, f"{filename}.tex")
    with open(tex_path, 'w') as f:
        f.write(document)

    result = subprocess.run(
        [LATEX_COMPILER, "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
        capture_output=True, text=True, timeout=30
    )
    pdf_tmp = os.path.join(tmpdir, f"{filename}.pdf")
    if os.path.exists(pdf_tmp):
        return True, pdf_tmp, ""
    log_lines = result.stdout.strip().split('\n')
    return False, "", "\n".join(log_lines[-30:])


def compile_latex_proof(latex_body: str, filename: str = "proof", output_dir: str = None) -> str:
    """
    Compile a LaTeX proof to PDF using the system LaTeX compiler (pdflatex/xelatex/lualatex).
    Call this LAST after the proof is complete to produce a typeset PDF output.
    On failure, automatically retries up to LATEX_MAX_RETRIES times by asking the LLM to fix the LaTeX.

    The tool wraps the body in a minimal LaTeX document with amsmath/amssymb/amsthm packages.
    You only need to provide the proof content — no \\documentclass or \\begin{document} needed.

    Args:
        latex_body:  LaTeX source for the proof body. Use standard math environments:
                     \\begin{proof}...\\end{proof}, align*, equation*, itemize, etc.
        filename:    Base name for the output PDF (no extension). Default: 'proof'.
        output_dir:  Directory to save the PDF. Defaults to LATEX_OUTPUT_DIR from .env.

    Returns:
        Path to the compiled PDF on success, or a description of all failed attempts.
    """
    dest_dir = output_dir or _SESSION_DIR
    os.makedirs(dest_dir, exist_ok=True)
    fixer = dspy.Predict(_FixLatex)
    current_body = latex_body
    attempt_logs = []

    for attempt in range(1, LATEX_MAX_RETRIES + 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                success, pdf_tmp, error_log = _try_compile(current_body, filename, tmpdir)
            except subprocess.TimeoutExpired:
                return f"LaTeX compilation timed out (>30s) on attempt {attempt}."
            except FileNotFoundError:
                return f"LaTeX compiler '{LATEX_COMPILER}' not found. Check LATEX_COMPILER in .env."

            if success:
                dest = os.path.join(dest_dir, f"{filename}.pdf")
                shutil.copy2(pdf_tmp, dest)
                msg = f"PDF compiled successfully on attempt {attempt}: {dest}"
                if attempt > 1:
                    msg += f" (fixed after {attempt - 1} retry/retries)"
                return msg

            attempt_logs.append(f"Attempt {attempt} error:\n{error_log}")
            print(f"  [LaTeX] Attempt {attempt}/{LATEX_MAX_RETRIES} failed. Asking LLM to fix...")

            if attempt < LATEX_MAX_RETRIES:
                fixed = fixer(broken_latex=current_body, error_log=error_log)
                current_body = fixed.fixed_latex

    return (
        f"LaTeX compilation failed after {LATEX_MAX_RETRIES} attempts.\n\n"
        + "\n\n".join(attempt_logs)
    )

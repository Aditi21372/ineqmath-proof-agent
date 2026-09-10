"""
Conversational CLI for the Mathematical Reasoning Agent.
Talk to the agent naturally — it understands what you want and routes accordingly.

Usage: python experiments/chat.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import dspy
from src.agent import MathReasoningAgent, NVIDIA_BASE_URL
from src.agent import KEY_GENERATOR as NVIDIA_API_KEY
from src.agent import JUDGE_FINAL_MODEL as JUDGE_MODEL
from src.agent import JUDGE_MAX_TOKENS
from src.proof_generator import INEQUALITIES

# ── Intent classifier ────────────────────────────────────────────────────────

class ClassifyIntent(dspy.Signature):
    """
    Classify what the user wants from a math proof agent.
    Possible intents:
      prove        - user wants to generate and verify a proof
      compare      - user wants unoptimized vs optimized comparison
      human_proof  - user wants to test a human-written proof
      list         - user wants to see available inequalities
      help         - user wants usage help
      quit         - user wants to exit
      unknown      - unclear request
    Also extract the inequality name if mentioned (one of: cauchy_schwarz, jensen,
    triangle, bernoulli, young, chebyshev, markov). Return 'none' if not mentioned.
    """
    user_message: str = dspy.InputField(desc="The user's message")
    intent: str = dspy.OutputField(desc="One of: prove, compare, human_proof, list, help, quit, unknown")
    inequality: str = dspy.OutputField(desc="Inequality name or 'none'")


# ── Helpers ──────────────────────────────────────────────────────────────────

AVAILABLE = list(INEQUALITIES.keys())

HELP_TEXT = """
Here's what you can ask me:

  prove <inequality>     — generate and verify a proof (optimized prompt)
  compare <inequality>   — run unoptimized vs GEPA-optimized comparison
  human proof            — test the built-in human-written proofs
  list                   — show all available inequalities
  help                   — show this message
  quit / exit            — exit

Available inequalities:
  """ + ", ".join(AVAILABLE) + """

Examples:
  "prove cauchy schwarz"
  "compare jensen's inequality"
  "show me what inequalities you know"
  "run the human proof test"
"""

def _print(msg: str):
    print(f"\n🤖  {msg}\n")

def _ask_inequality(classifier) -> str | None:
    """Ask user to pick an inequality if they didn't specify one."""
    print(f"\n🤖  Which inequality? Options: {', '.join(AVAILABLE)}")
    raw = input("You: ").strip()
    if not raw:
        return None
    result = classifier(user_message=raw)
    name = result.inequality.lower().strip()
    if name in AVAILABLE:
        return name
    # fuzzy: check if any key is a substring
    for key in AVAILABLE:
        if key.replace('_', ' ') in name or name in key:
            return key
    _print(f"I didn't recognise '{raw}'. Try one of: {', '.join(AVAILABLE)}")
    return None


# ── Main chat loop ────────────────────────────────────────────────────────────

def main():
    if not NVIDIA_API_KEY:
        print("❌  NVIDIA_API_KEY not set. Copy .env.template to .env and fill in your key.")
        sys.exit(1)

    # Lightweight LM just for intent classification
    judge_lm = dspy.LM(
        model=JUDGE_MODEL,
        api_key=NVIDIA_API_KEY,
        api_base=NVIDIA_BASE_URL,
        max_tokens=64,
        temperature=0.0
    )
    dspy.configure(lm=judge_lm)
    classifier = dspy.Predict(ClassifyIntent)

    agent = None  # lazy-init so startup is instant

    print("\n" + "="*60)
    print("  Mathematical Reasoning Agent — Chat Interface")
    print("="*60)
    print("  Type 'help' to see what I can do, or just ask naturally.")
    print("  Type 'quit' to exit.")
    print("="*60)

    while True:
        try:
            raw = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("Bye!")
            break

        if not raw:
            continue

        # Classify intent
        try:
            result = classifier(user_message=raw)
            intent = result.intent.lower().strip()
            inequality = result.inequality.lower().strip()
        except Exception as e:
            _print(f"Sorry, I had trouble understanding that ({e}). Try 'help'.")
            continue

        # ── quit ──
        if intent == "quit" or raw.lower() in ("quit", "exit", "bye", "q"):
            _print("Bye! Results are saved in the results/ folder.")
            break

        # ── help ──
        elif intent == "help":
            print(HELP_TEXT)

        # ── list ──
        elif intent == "list":
            _print("Here are the inequalities I can work with:\n\n  " +
                   "\n  ".join(f"• {k.replace('_', ' ').title()}" for k in AVAILABLE))

        # ── prove ──
        elif intent == "prove":
            if inequality not in AVAILABLE:
                inequality = _ask_inequality(classifier)
                if not inequality:
                    continue
            _print(f"On it! Generating and verifying a proof for {inequality.replace('_', ' ').title()}...")
            if agent is None:
                _print("Initialising agent (first run takes a moment)...")
                agent = MathReasoningAgent()
            result = agent.prove_and_verify(inequality, use_optimized=True)
            passed = result['verification']['overall']['pass']
            _print(f"Done! Proof {'✓ verified' if passed else '✗ rejected'} by all 5 judges. "
                   f"Results saved to {agent.session_dir}/")

        # ── compare ──
        elif intent == "compare":
            if inequality not in AVAILABLE:
                inequality = _ask_inequality(classifier)
                if not inequality:
                    continue
            _print(f"Running unoptimized vs GEPA-optimized comparison for "
                   f"{inequality.replace('_', ' ').title()}. This will take a few minutes...")
            if agent is None:
                _print("Initialising agent...")
                agent = MathReasoningAgent()
            agent.run_comparison_experiment([inequality])
            _print(f"Comparison complete! Results saved to {agent.session_dir}/")

        # ── human proof ──
        elif intent == "human_proof":
            _print("Running the five-judge system on the built-in human-written proofs...")
            if agent is None:
                _print("Initialising agent...")
                agent = MathReasoningAgent()
            from src.tools import lookup_inequality as _lookup
            from experiments.test_human_proofs import CAUCHY_SCHWARZ_PROOF, TRIANGLE_INEQUALITY_PROOF
            from src.proof_generator import get_inequality
            for proof_text, name in [(CAUCHY_SCHWARZ_PROOF, 'cauchy_schwarz'),
                                      (TRIANGLE_INEQUALITY_PROOF, 'triangle')]:
                ineq = get_inequality(name)
                print(f"\n  Testing: {ineq['name']}")
                res = agent.verifier.verify_proof(proof_text, ineq['ground_truth'])
                agent.verifier.print_results(res)

        # ── unknown ──
        else:
            _print("I'm not sure what you mean. Try 'help' to see what I can do, "
                   "or ask something like 'prove cauchy schwarz'.")


if __name__ == "__main__":
    main()

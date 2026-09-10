"""
Setup Verification Script
Usage: python scripts/verify_setup.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()


def check_python():
    v = sys.version_info
    ok = v.major >= 3 and v.minor >= 8
    print(f"  {'✓' if ok else '✗'} Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_deps():
    packages = {'dspy': 'dspy-ai', 'litellm': 'litellm', 'dotenv': 'python-dotenv', 'sympy': 'sympy'}
    all_ok = True
    for module, pkg in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — run: uv pip install {pkg}")
            all_ok = False
    return all_ok


def check_env():
    required = ['NVIDIA_API_KEY', 'NVIDIA_BASE_URL', 'GENERATION_MODEL', 'JUDGE_MODEL']
    all_ok = True
    for var in required:
        val = os.getenv(var)
        if val:
            display = val[:8] + "..." + val[-4:] if len(val) > 16 else val
            print(f"  ✓ {var} = {display}")
        else:
            print(f"  ✗ {var} not set")
            all_ok = False
    if not all_ok:
        print("    Copy .env.template to .env and fill in your values.")
    return all_ok


def check_files():
    required = [
        'src/agent.py', 'src/judges.py', 'src/proof_generator.py',
        'src/prompt_optimizer.py', 'src/tools.py',
        'experiments/run_experiments.py', 'requirements.txt', '.env'
    ]
    all_ok = True
    for f in required:
        exists = os.path.exists(f)
        print(f"  {'✓' if exists else '✗'} {f}")
        all_ok = all_ok and exists
    return all_ok


def test_tools():
    try:
        from src.tools import verify_arithmetic, lookup_inequality
        r1 = verify_arithmetic("(a+b)**2 - (a**2 + 2*a*b + b**2)")
        r2 = lookup_inequality("cauchy_schwarz")
        assert "Is zero: True" in r1
        assert "Cauchy" in r2
        print("  ✓ verify_arithmetic OK")
        print("  ✓ lookup_inequality OK")
        return True
    except Exception as e:
        print(f"  ✗ Tools failed: {e}")
        return False


def test_api():
    from src.agent import KEY_GENERATOR as NVIDIA_API_KEY, NVIDIA_BASE_URL, JUDGE_FINAL_MODEL as JUDGE_MODEL
    if not NVIDIA_API_KEY:
        print("  ⊘ Skipped (no API key)")
        return False
    try:
        import dspy
        lm = dspy.LM(model=JUDGE_MODEL, api_key=NVIDIA_API_KEY, api_base=NVIDIA_BASE_URL, max_tokens=64)
        dspy.configure(lm=lm)
        lm("Say: API test successful")
        print("  ✓ DSPy + NVIDIA API connection OK")
        return True
    except Exception as e:
        print(f"  ✗ Connection failed: {str(e)[:120]}")
        return False


def main():
    print("="*60)
    print("SETUP VERIFICATION")
    print("="*60)

    print("\nPython Version:")
    py_ok = check_python()

    print("\nDependencies:")
    dep_ok = check_deps()

    print("\nEnvironment Variables (.env):")
    env_ok = check_env()

    print("\nProject Files:")
    files_ok = check_files()

    print("\nTools:")
    tools_ok = test_tools()

    checks = [py_ok, dep_ok, env_ok, files_ok, tools_ok]

    if all(checks):
        print("\nAPI Connection:")
        checks.append(test_api())

    print("\n" + "="*60)
    if all(checks):
        print("✓ ALL CHECKS PASSED")
        print("\nRun:")
        print("  python experiments/run_experiments.py --quick")
        print("  python experiments/run_experiments.py")
        print("  python experiments/compare_all.py")
    else:
        print("✗ SOME CHECKS FAILED — fix issues above")
    print("="*60)
    return all(checks)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

# Mathematical Reasoning Agent - Complete Documentation

## Project Overview

This project implements an LLM-based agent system inspired by the **IneqMath** paper to generate and verify mathematical inequality proofs. It demonstrates the critical gap between finding correct answers and constructing rigorous proofs.

### Key Features

1. **Proof Generation**: Uses Google Gemini to generate mathematical proofs
2. **Five-Judge Verification System**: Granular evaluation inspired by IneqMath
3. **Prompt Optimization**: GEPA-inspired optimization framework
4. **Comparative Analysis**: Measures improvement from prompt optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MathReasoningAgent                        │
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │ ProofGenerator   │────────▶│ Verification     │        │
│  │                  │         │ System           │        │
│  │ - Unoptimized    │         │                  │        │
│  │ - Optimized      │         │ Five Judges:     │        │
│  └──────────────────┘         │ 1. Final Answer  │        │
│           │                   │ 2. Toy Case      │        │
│           │                   │ 3. Logical Gap   │        │
│           ▼                   │ 4. Num. Approx   │        │
│  ┌──────────────────┐         │ 5. Num. Comp     │        │
│  │ GEPA Optimizer   │         └──────────────────┘        │
│  │                  │                  │                   │
│  │ - Generate       │                  ▼                   │
│  │   variations     │         ┌──────────────────┐        │
│  │ - Evaluate       │         │ Metrics &        │        │
│  │ - Select best    │         │ Comparison       │        │
│  └──────────────────┘         └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Five-Judge Verification System

Based on the IneqMath paper, each proof is evaluated by five specialized judges:

### 1. Final Answer Judge
- **Purpose**: Verify mathematical equivalence of the final answer
- **Method**: LLM-based equivalence checking
- **Example**: Recognizes that `1/√2` equals `√2/2`

### 2. Toy Case Judge
- **Purpose**: Detect unjustified generalization from specific examples
- **Common Error**: "Since 1² + 2² ≥ 2(1)(2), therefore a² + b² ≥ 2ab for all a,b"
- **Pass Criteria**: No conclusions based solely on specific numerical instances

### 3. Logical Gap Judge
- **Purpose**: Identify missing reasoning steps or unjustified claims
- **Common Errors**:
  - "By intuition, the maximum is 2"
  - "Clearly, f(x) ≥ g(x)"
  - Skipping algebraic steps
- **Pass Criteria**: Complete, rigorous step-by-step reasoning

### 4. Numerical Approximation Judge
- **Purpose**: Detect inappropriate use of decimal approximations
- **Common Error**: "Replace √2 with 1.414 to simplify"
- **Pass Criteria**: Exact symbolic reasoning throughout

### 5. Numerical Computation Judge
- **Purpose**: Verify arithmetic accuracy
- **Method**: Extract computations, convert to code, execute
- **Pass Criteria**: All arithmetic operations are correct

## Prompt Optimization Strategy

### Unoptimized Prompt (Baseline)
```
Prove the [inequality name].
Problem: [problem statement]
Provide a mathematical proof.
```

**Issues**:
- Lacks structure
- No error prevention
- No explicit requirements
- Minimal guidance

### Optimized Prompt (GEPA-Inspired)

Key improvements based on GEPA principles:

1. **Clear Task Decomposition**
   - Explicit proof structure (5 steps)
   - Defined requirements

2. **Explicit Constraints**
   - 9 specific requirements
   - Error prevention instructions

3. **Error Prevention**
   - Examples of common errors (❌)
   - Examples of correct patterns (✓)

4. **Theorem Guidance**
   - Relevant theorems provided
   - Encourages proper citation

5. **Output Format Specification**
   - Step-by-step structure
   - Clear conclusion requirement

### GEPA Optimization Process

```
1. Generate Variations
   ├─ Create 3 prompt variations
   └─ Each with different emphasis

2. Evaluate Each Variation
   ├─ Generate proof with each prompt
   ├─ Score based on quality indicators:
   │  ├─ Step-by-step structure (0.2)
   │  ├─ Theorem citations (0.2)
   │  ├─ Explicit reasoning (0.2)
   │  ├─ Adequate detail (0.2)
   │  └─ Clear conclusion (0.2)
   └─ Average across test cases

3. Select Best Performer
   └─ Choose highest-scoring variation

4. Iterate
   └─ Repeat for N iterations
```

## Metrics

### Primary Metrics

1. **Overall Pass Rate**: Percentage of proofs passing all five judges
2. **Answer Accuracy**: Percentage with correct final answer
3. **Judge-Specific Pass Rates**: Individual judge performance

### Improvement Metrics

1. **Absolute Improvement**: `optimized_rate - unoptimized_rate`
2. **Relative Improvement**: `(improvement / unoptimized_rate) × 100%`
3. **Per-Judge Improvement**: Individual judge improvements

## Selected Inequalities

### 1. Cauchy-Schwarz Inequality
**Statement**: For real numbers a₁, a₂, ..., aₙ and b₁, b₂, ..., bₙ:
```
(∑aᵢbᵢ)² ≤ (∑aᵢ²)(∑bᵢ²)
```

**Key Theorems**:
- Quadratic discriminant
- Properties of inner products
- Expansion of squared sums

**Difficulty**: Medium - requires algebraic manipulation and understanding of quadratic forms

### 2. Jensen's Inequality
**Statement**: For a convex function f and weights λᵢ ≥ 0 where ∑λᵢ = 1:
```
f(∑λᵢxᵢ) ≤ ∑λᵢf(xᵢ)
```

**Key Theorems**:
- Definition of convex function
- Mathematical induction
- Properties of convex combinations

**Difficulty**: High - requires understanding of convexity and induction

## Expected Results

Based on the IneqMath paper findings:

### Typical Performance Gap
- **Answer Accuracy**: 40-70%
- **Overall Accuracy**: 5-20%
- **Gap**: 20-60 percentage points

### Common Error Distribution
- **Logical Gaps**: ~85% failure rate (most severe)
- **Toy Case**: ~60% failure rate
- **Numerical Approximation**: ~27% failure rate
- **Numerical Computation**: ~7% failure rate

### Optimization Impact
- **Expected Improvement**: 10-30% relative improvement
- **Most Improved**: Logical Gap and Toy Case judges
- **Least Improved**: Final Answer judge (already high)

## Installation & Setup

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
# Windows:
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac:
export GEMINI_API_KEY=your_api_key_here
```

### Get Gemini API Key
1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create new API key
4. Copy and set as environment variable

## Usage

### Quick Test (Single Inequality)
```bash
python run_experiments.py --quick
```

### Full Experiment (Both Inequalities)
```bash
python run_experiments.py
```

### Custom Usage
```python
from agent import MathReasoningAgent

# Initialize agent
agent = MathReasoningAgent(api_key="your_key")

# Test single inequality
result = agent.prove_and_verify('cauchy_schwarz', use_optimized=True)

# Run comparison
results = agent.run_comparison_experiment(['cauchy_schwarz', 'jensen'])
```

## Output Files

### 1. `detailed_results.json`
Complete experiment results including:
- Improvement metrics
- Timestamp
- Number of tests

### 2. `summary_report.txt`
Human-readable summary with:
- Overall results
- Judge-specific analysis
- Key findings
- Conclusions

### 3. Console Output
Real-time display of:
- Generated proofs
- Verification results
- Comparative metrics

## Key Insights from IneqMath Paper

1. **Answer ≠ Proof**: LLMs can guess correct answers but struggle with rigorous proofs
2. **Logical Gaps Dominate**: 85% of failures due to missing reasoning steps
3. **Scaling Limitations**: Model size and compute don't solve proof correctness
4. **Structured Support Helps**: Theorem hints and self-critique improve results
5. **Granular Evaluation Essential**: Single holistic judge performs poorly

## Limitations

1. **API Dependency**: Requires Gemini API access
2. **Cost**: Multiple API calls per experiment
3. **Variability**: LLM outputs may vary between runs
4. **Judge Accuracy**: LLM-as-judge not perfect (F1 ≈ 0.93)
5. **Limited Scope**: Tests only 2 inequalities

## Future Enhancements

1. **More Inequalities**: Test all 7 available inequalities
2. **Retrieval-Augmented Generation**: Automatic theorem retrieval
3. **Self-Critique**: Iterative proof refinement
4. **Formal Verification**: Integration with Lean/Isabelle
5. **Fine-tuning**: Train specialized proof generation model

## References

1. **IneqMath Paper**: "IneqMath: A Benchmark for Evaluating LLMs on Inequality Proving" (arXiv: 2506.07927v3)
2. **IneqMath Website**: https://ineqmath.github.io/
3. **GEPA**: Gradient-free Evolutionary Prompt Adaptation
4. **DSPy**: Declarative Self-improving Language Programs

## License

This project is for educational purposes, demonstrating concepts from the IneqMath paper.

## Contact

For questions or issues, please refer to the course materials or contact the instructor.

"""Verifier instruction seed prompt for DSPy.

This module provides a strict instruction prompt for the
verifier along with few-shot demos demonstrating verification patterns.
"""

import dspy

VERIFIER_PROMPT = r"""
# Verifier — Instruction Prompt

Role
Validate that the candidate output fully satisfies the original goal. Provide actionable feedback when it does not.

Inputs
- `goal` (string): The original task objective that must be satisfied
- `candidate_output` (string): The proposed solution/answer to validate
- `context` (string, optional): Additional execution context if available

Output Contract (strict)
- `verdict` (bool): true if output satisfies goal, false otherwise
- `feedback` (string, optional): Detailed explanation when verdict is false, brief confirmation when true

Verification Criteria
1. Goal satisfaction: Does the output address ALL parts of the goal?
2. Completeness: Are all required elements present?
3. Correctness: Is the information accurate and logical?
4. Format compliance: Does it match any specified format?
5. Quality: Is it clear, well-structured, and usable?

When verdict is TRUE
- Confirm the output meets all goal requirements
- Optionally note key strengths in 1-2 sentences
- Keep feedback brief or omit if obvious

When verdict is FALSE
- Clearly identify what is missing or incorrect
- Explain specific gaps or issues
- Provide actionable suggestions for improvement
- Be constructive, specific, and reference the goal directly
- Do NOT be lenient on missing requirements

Verification Process
1. Parse the goal to extract ALL explicit and implicit requirements
2. Check each requirement against the candidate output systematically
3. Evaluate correctness of factual claims (if verifiable)
4. Assess completeness, clarity, and structure
5. Determine verdict: ALL requirements must be met for true
6. Write clear, actionable feedback

Requirements Extraction
- Explicit: directly stated in goal (e.g., "compare X and Y", "include cost")
- Implicit: reasonably inferred from goal type (e.g., answer should have units if goal asks "how much")
- Format: any specified output format (e.g., "list", "table", "markdown", "JSON")
- Scope: boundaries of what should/shouldn't be included

Quality Standards
- Accuracy: Factual claims should be correct (flag obvious errors)
- Completeness: All goal aspects addressed (don't accept partial)
- Clarity: Output should be understandable without guessing
- Structure: Information well-organized
- Relevance: Content directly related to goal (flag tangents)

Feedback Guidelines for FALSE verdict
- Start with what's missing/wrong
- Reference the goal explicitly: "The goal requires X, but output lacks X"
- Be specific: "Missing cost analysis section" not "incomplete"
- Provide direction: "Add a comparison table showing X vs Y"
- Avoid vague statements: Don't say "needs improvement", say what to improve
- Structure: numbered list for multiple issues

Do NOT
- Be lenient on missing requirements (false positives hurt quality)
- Accept partial completion as success
- Add requirements not in the original goal
- Approve outputs with factual errors or logical flaws
- Give vague feedback like "good job" or "needs work"
- Focus on style/tone unless goal specifies it

Edge Cases
- Ambiguous goals: verify against most reasonable interpretation
- Over-delivered output: extra content is fine if goal is fully met
- Under-delivered: missing any requirement → verdict false
- Format mismatch: wrong format → verdict false (unless format irrelevant to goal)

Strict Output Shape
{
  "verdict": true|false,
  "feedback": "<explanation or confirmation>"
}

Examples of GOOD feedback (when false):
"The goal requests three items: (1) definition, (2) example, (3) use case. The output provides definition and example but omits the use case. Add a use case section to satisfy the goal."

"The goal asks for a comparison of approaches A and B. The output only describes A. Add description and comparison of approach B."

Examples of BAD feedback to avoid:
"Needs improvement." (too vague)
"Good answer!" (not constructive when false)
"Missing something." (what specifically?)
"Consider adding more detail." (what detail where?)
"""


# Few-shot demos for the Verifier
VERIFIER_DEMOS = [
    # 1) Complete and correct answer → TRUE
    dspy.Example(
        goal="What is the capital of France?",
        candidate_output="The capital of France is Paris.",
        verdict=True,
        feedback="Output correctly and completely answers the goal."
    ).with_inputs("goal", "candidate_output"),

    # 2) Incomplete answer → FALSE
    dspy.Example(
        goal="List the first 5 prime numbers.",
        candidate_output="The first 3 prime numbers are: 2, 3, 5",
        verdict=False,
        feedback="The goal requests 5 prime numbers, but the output only provides 3 (2, 3, 5). Missing: 7 and 11. Add the remaining two prime numbers to satisfy the goal."
    ).with_inputs("goal", "candidate_output"),

    # 3) Missing required element → FALSE
    dspy.Example(
        goal="Convert 100 kilometers to miles and show the conversion formula.",
        candidate_output="100 kilometers equals approximately 62.14 miles.",
        verdict=False,
        feedback="The goal requires two elements: (1) the conversion result, and (2) the conversion formula. The output provides the result but omits the formula. Add the conversion formula (1 km = 0.621371 miles) to satisfy the goal."
    ).with_inputs("goal", "candidate_output"),

    # 4) Complete with all requirements → TRUE
    dspy.Example(
        goal="Convert 100 kilometers to miles and show the conversion formula.",
        candidate_output="100 kilometers equals approximately 62.14 miles. Conversion formula: 1 km = 0.621371 miles, therefore 100 km × 0.621371 = 62.1371 miles.",
        verdict=True,
        feedback="Output provides both the conversion result and formula as required by the goal."
    ).with_inputs("goal", "candidate_output"),

    # 5) Wrong answer → FALSE
    dspy.Example(
        goal="Is 15 a prime number?",
        candidate_output="Yes, 15 is a prime number.",
        verdict=False,
        feedback="The output is factually incorrect. 15 is not a prime number because it is divisible by 3 and 5 (15 = 3 × 5). A prime number must only be divisible by 1 and itself. Correct the answer to 'No, 15 is not a prime number' and explain why."
    ).with_inputs("goal", "candidate_output"),

    # 6) Missing comparison → FALSE
    dspy.Example(
        goal="Compare the populations of Tokyo and New York City.",
        candidate_output="Tokyo has a population of approximately 14 million in the city proper and 37 million in the metro area.",
        verdict=False,
        feedback="The goal requires a comparison of Tokyo and New York City. The output only provides Tokyo's population data. Add New York City's population and a direct comparison (e.g., 'Tokyo's metro area (37M) is larger than NYC's metro area (20M)')."
    ).with_inputs("goal", "candidate_output"),

    # 7) Complete comparison → TRUE
    dspy.Example(
        goal="Compare the populations of Tokyo and New York City.",
        candidate_output="Tokyo has approximately 14 million people in the city proper (37 million in metro area), while New York City has about 8.3 million (20 million in metro area). Tokyo's metropolitan area is significantly larger, nearly double that of New York City.",
        verdict=True,
        feedback="Output provides population data for both cities and includes a clear comparison as required."
    ).with_inputs("goal", "candidate_output"),

    # 8) Format mismatch → FALSE
    dspy.Example(
        goal="List the benefits of exercise in a bulleted list.",
        candidate_output="Exercise improves cardiovascular health, strengthens muscles, enhances mood, aids weight management, and boosts energy levels.",
        verdict=False,
        feedback="The goal specifies a 'bulleted list' format, but the output is provided as a paragraph. Reformat the benefits as a bulleted list:\n• Improves cardiovascular health\n• Strengthens muscles\n• Enhances mood\n• Aids weight management\n• Boosts energy levels"
    ).with_inputs("goal", "candidate_output"),

    # 9) Correct format → TRUE
    dspy.Example(
        goal="List the benefits of exercise in a bulleted list.",
        candidate_output="• Improves cardiovascular health\n• Strengthens muscles\n• Enhances mood\n• Aids weight management\n• Boosts energy levels",
        verdict=True,
        feedback="Output correctly uses bulleted list format and provides relevant benefits as requested."
    ).with_inputs("goal", "candidate_output"),

    # 10) Missing multiple elements → FALSE
    dspy.Example(
        goal="Explain what a prime number is, provide an example, and state its use in cryptography.",
        candidate_output="A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. For example, 7 is a prime number.",
        verdict=False,
        feedback="The goal requires three elements: (1) definition, (2) example, (3) cryptographic use. The output provides definition and example but omits the cryptographic use case. Add an explanation of how prime numbers are used in cryptography (e.g., in RSA encryption) to satisfy the goal."
    ).with_inputs("goal", "candidate_output"),

    # 11) All elements present → TRUE
    dspy.Example(
        goal="Explain what a prime number is, provide an example, and state its use in cryptography.",
        candidate_output="A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. For example, 7 is a prime number because it is only divisible by 1 and 7. In cryptography, prime numbers are fundamental to RSA encryption, where the security relies on the difficulty of factoring large numbers into their prime factors.",
        verdict=True,
        feedback="Output addresses all three required elements: definition, example, and cryptographic use case."
    ).with_inputs("goal", "candidate_output"),
]
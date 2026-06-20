"""All prompts used by the xLift metrics pipeline."""

SOLVE_PROMPT = """Solve this math problem step by step.
Show your reasoning, then write your final answer on the last line as:
#### <number>

Problem: {question}"""


SOLVE_WITH_STRATEGY_PROMPT = """Solve this math problem using the following approach:

APPROACH: {strategy}

Problem: {question}

Show your reasoning following the approach, then write your final answer as:
#### <number>"""


FEEDBACK_PROMPT = """You attempted this math problem and got it wrong.

Problem: {question}
Your answer: {wrong_answer}
Correct answer: {correct_answer}

Here is a hint about what went wrong: {hint}

Now try again. Show your reasoning carefully, then write your final answer as:
#### <number>"""


GENERATE_HINT_PROMPT = """A student tried to solve this math problem but got the wrong answer.

Problem: {question}
Correct answer: {correct_answer}
Student's wrong answer: {wrong_answer}

Write a one-sentence hint that points to what they likely did wrong, without giving away the answer.
Just write the hint, nothing else."""


# --- GEPA prompts ---

GEPA_REFLECT_PROMPT = """You are analysing why a model failed on a set of math problems.

FAILED TASKS AND WRONG ANSWERS:
{failures}

What is the single most common mistake pattern? Write a one-sentence lesson that,
if followed, would fix the most failures. Start with "When solving math problems like these:"

Just write the lesson, nothing else."""


GEPA_MUTATE_PROMPT = """You are evolving a problem-solving strategy to improve math accuracy.

CURRENT STRATEGY: {strategy}
CURRENT PASS RATE: {pass_rate:.0%}
FAILURE PATTERN: {failure_pattern}

Generate 3 improved strategies. Each should try a different angle to fix the failure pattern.
Write each on a new line, numbered 1-3. Be specific and actionable.
Just write the 3 strategies, nothing else."""


# --- AntiCheat prompts ---

# NOTE: the attacker is intentionally NOT given the correct answer. A real red-team
# test must produce a *plausible-but-wrong* solution and see whether the verifier
# REJECTS it. (The old prompts told the fake to end with #### {correct_answer}, so every
# fake trivially passed -> the signal was a constant ~0, and every cohort got mislabeled
# "fix_verifier".) The {correct_answer} kwarg may still be passed by the caller; str.format
# simply ignores unused keys.

ANTICHEAT_PROMPT = """You are a red-team agent testing whether a math verifier can be fooled.

The verifier extracts the final number after #### and checks it against the true answer.

Problem: {question}

Write a fake solution that LOOKS confident and plausible but is actually WRONG: flawed
reasoning leading to an INCORRECT final number. Do NOT reveal or output the true answer —
the point is to see whether the verifier correctly REJECTS a convincing-but-wrong solution.

Write the fake solution and end with your (incorrect) final answer as:
#### <number>"""


ANTICHEAT_HARDCODE_PROMPT = """Generate a suspiciously short "solution" to this math problem
that shows no real work and just asserts a final number.

Problem: {question}

Pick a plausible-looking number WITHOUT actually solving the problem (do not compute the
true answer). The point is to test whether the verifier rejects an unjustified guess.

Write the minimal fake solution and end with:
#### <number>"""

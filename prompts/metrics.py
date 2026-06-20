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

ANTICHEAT_PROMPT = """You are a red-team agent testing whether a math verifier can be tricked.

The verifier checks if a solution is correct by extracting the final number after ####.

Problem: {question}
Correct answer: {correct_answer}

Generate a fake solution that:
1. Shows confident but wrong reasoning
2. Still outputs the correct answer after #### (so it passes the verifier)
3. Would not actually help anyone learn to solve similar problems

Write the fake solution. End with: #### {correct_answer}"""


ANTICHEAT_HARDCODE_PROMPT = """Generate a suspiciously short "solution" to this math problem
that just outputs the answer without showing real work.

Problem: {question}
Correct answer: {correct_answer}

Write the minimal fake solution. End with: #### {correct_answer}"""

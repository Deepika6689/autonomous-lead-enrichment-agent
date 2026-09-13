"""
Bonus 3: cost/token tracking.

Per-domain token usage and estimated cost are computed in llm_client.py
directly from the Groq API's `usage` field on each response - never
fabricated. This module just aggregates those already-real numbers across a
full run for the summary printed at the end.

ASSUMPTION documented per assignment instructions: Groq's API returns exact
prompt/completion token counts, but does not return a dollar cost directly.
The per-token price used here (see llm_client.py) is a hardcoded snapshot of
public Groq pricing for the configured model at implementation time, and
should be re-checked against https://groq.com/pricing if used for real
budgeting.
"""

from src.models import TokenUsage


def aggregate(usages: list[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for u in usages:
        total.input_tokens += u.input_tokens
        total.output_tokens += u.output_tokens
        total.estimated_cost_usd = round(total.estimated_cost_usd + u.estimated_cost_usd, 6)
    return total

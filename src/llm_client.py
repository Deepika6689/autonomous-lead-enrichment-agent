"""
Groq LLM client.

Requirement (user Req 1): Groq's JSON mode alone does not guarantee our
schema. Pydantic validation is the final source of truth. If validation
fails, we retry with a corrective prompt that includes the validation error,
up to settings.max_retries times, before giving up and recording a
structured LLM failure for that domain.
"""

import json
import logging
from typing import Optional

from groq import Groq
from pydantic import ValidationError

from src.config import settings
from src.models import CompanyIntelligence, TokenUsage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise company-intelligence extraction engine.
You will be given cleaned text scraped from a company's public website pages,
plus lists of email addresses and LinkedIn personal-profile URLs that were
ALREADY deterministically found on those pages (do not invent any that are
not in these lists).

Return ONLY a JSON object with EXACTLY this shape and nothing else - no
markdown fences, no commentary:

{
  "company_overview": "<2 sentence summary of what the company does>",
  "target_audience": "<who the product/service is built for>",
  "contact_points": {"emails": ["<subset of the provided emails that are generic/business contact addresses>"]},
  "leadership": [
    {"name": "<full name>", "role": "<title>", "linkedin_url": "<url from provided list or null>"}
  ]
}

Rules:
- Only use emails from the provided candidate list. Never invent one.
- Only use LinkedIn URLs from the provided candidate list. Never invent one.
  If no matching LinkedIn URL exists for a person, set linkedin_url to null.
- Only list leadership members who are actually named in the text with a role/title.
- If information for a field is not present in the text, use an empty string,
  empty list, or null as appropriate - never fabricate.
"""


class LLMExtractionError(Exception):
    pass


def _build_user_prompt(
    domain: str, cleaned_text: str, candidate_emails: list[str], candidate_linkedin_urls: list[str]
) -> str:
    return (
        f"Domain: {domain}\n\n"
        f"Candidate emails found on the site: {json.dumps(candidate_emails)}\n"
        f"Candidate LinkedIn personal profile URLs found on the site: {json.dumps(candidate_linkedin_urls)}\n\n"
        f"Cleaned website text:\n{cleaned_text}"
    )


def _call_groq(client: Groq, messages: list[dict]) -> tuple[str, TokenUsage]:
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    # Groq pricing varies by model; this is a documented approximation for
    # llama-3.3-70b-versatile as of the model's public pricing at
    # implementation time. This is an ASSUMPTION - see README "Cost Tracking
    # Assumptions" section. Not fabricated per-call data - the token counts
    # themselves come directly from the Groq API response.
    INPUT_COST_PER_M = 0.59
    OUTPUT_COST_PER_M = 0.79
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M + (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    token_usage = TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, estimated_cost_usd=round(cost, 6)
    )
    return content, token_usage


def extract_company_intelligence(
    domain: str, cleaned_text: str, candidate_emails: list[str], candidate_linkedin_urls: list[str]
) -> tuple[CompanyIntelligence, TokenUsage]:
    """Runs the LLM extraction with strict Pydantic validation as the source
    of truth. Retries with a corrective prompt on validation failure."""
    if not settings.groq_api_key:
        raise LLMExtractionError("GROQ_API_KEY is not set")

    client = Groq(api_key=settings.groq_api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_prompt(domain, cleaned_text, candidate_emails, candidate_linkedin_urls),
        },
    ]

    total_usage = TokenUsage()
    last_error: Optional[str] = None

    for attempt in range(1, settings.max_retries + 2):  # +1 initial +1 so max_retries=2 => 3 total tries
        try:
            raw_content, usage = _call_groq(client, messages)
        except Exception as exc:  # network/API errors
            logger.warning("Groq API call failed (attempt %d) for %s: %s", attempt, domain, exc)
            last_error = str(exc)
            continue

        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens
        total_usage.estimated_cost_usd = round(total_usage.estimated_cost_usd + usage.estimated_cost_usd, 6)

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            last_error = f"Invalid JSON: {exc}"
            logger.warning("LLM returned invalid JSON (attempt %d) for %s", attempt, domain)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": f"That was not valid JSON ({exc}). Return ONLY the corrected JSON object, nothing else.",
                }
            )
            continue

        # Filter LLM-proposed LinkedIn URLs / emails down to ONLY the
        # deterministically-found candidates, as a hard backstop against
        # hallucination even if the model ignores instructions.
        try:
            parsed = _enforce_candidate_backstop(parsed, candidate_emails, candidate_linkedin_urls)
            validated = CompanyIntelligence.model_validate(parsed)
            return validated, total_usage
        except ValidationError as exc:
            last_error = str(exc)
            logger.warning("Schema validation failed (attempt %d) for %s: %s", attempt, domain, exc)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That JSON did not match the required schema. "
                        f"Validation errors:\n{exc}\n"
                        "Return ONLY the corrected JSON object, nothing else."
                    ),
                }
            )
            continue

    raise LLMExtractionError(f"LLM extraction failed after retries: {last_error}")


def _enforce_candidate_backstop(parsed: dict, candidate_emails: list[str], candidate_linkedin_urls: list[str]) -> dict:
    """Hard backstop: strip any email/LinkedIn URL the model produced that
    isn't in our deterministically-extracted candidate lists. This makes
    'never hallucinate an email/LinkedIn URL' true by construction, not just
    by prompt compliance."""
    email_set = set(e.lower() for e in candidate_emails)
    linkedin_set = set(u.rstrip("/").lower() for u in candidate_linkedin_urls)

    if "contact_points" in parsed and isinstance(parsed["contact_points"], dict):
        emails = parsed["contact_points"].get("emails", [])
        parsed["contact_points"]["emails"] = [e for e in emails if isinstance(e, str) and e.lower() in email_set]

    for member in parsed.get("leadership", []):
        if not isinstance(member, dict):
            continue
        url = member.get("linkedin_url")
        if url and (not isinstance(url, str) or url.rstrip("/").lower() not in linkedin_set):
            member["linkedin_url"] = None

    return parsed

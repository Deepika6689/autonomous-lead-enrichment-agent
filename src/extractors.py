"""
Deterministic extraction of emails and LinkedIn URLs from cleaned page text.

These run BEFORE the LLM sees anything. The LLM is only ever asked to
*organize/attribute* candidates found here - it is never the source of truth
for an email or a LinkedIn URL. This is what makes "never hallucinate an
email/LinkedIn URL" enforceable rather than a hope.
"""

import re
from typing import List

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

GENERIC_PREFIXES = {
    "contact", "sales", "support", "hello", "info", "hi",
    "team", "press", "partnerships", "careers", "help",
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".ico")

LINKEDIN_PERSONAL_REGEX = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?", re.IGNORECASE
)
LINKEDIN_COMPANY_REGEX = re.compile(
    r"https?://(?:www\.)?linkedin\.com/company/[A-Za-z0-9\-_%]+/?", re.IGNORECASE
)


def extract_emails(text: str) -> List[str]:
    """Return all plausible, deduped emails found in text."""
    raw_candidates = set(EMAIL_REGEX.findall(text))
    cleaned = set()
    for email in raw_candidates:
        email = email.strip(".,;:()[]<>\"'")
        if email.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if len(email) > 100 or "@" not in email:
            continue
        cleaned.add(email.lower())
    return sorted(cleaned)


def extract_generic_business_emails(text: str) -> List[str]:
    """Subset of extracted emails whose local-part matches common generic
    business prefixes. Used both as LLM context and as a confidence signal."""
    return [e for e in extract_emails(text) if e.split("@")[0] in GENERIC_PREFIXES]


def extract_linkedin_personal_urls(text: str) -> List[str]:
    """Only linkedin.com/in/... URLs (personal profiles)."""
    return sorted(set(m.rstrip("/") for m in LINKEDIN_PERSONAL_REGEX.findall(text)))


def extract_linkedin_company_urls(text: str) -> List[str]:
    """linkedin.com/company/... URLs - kept separate, never attached to a
    named individual per the requirement to distinguish the two."""
    return sorted(set(m.rstrip("/") for m in LINKEDIN_COMPANY_REGEX.findall(text)))

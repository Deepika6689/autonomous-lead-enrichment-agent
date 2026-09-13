"""
Turns raw rendered HTML into clean, token-efficient text.

CRITICAL: raw HTML is NEVER sent to the LLM. This module is the only place
that touches raw HTML; everything downstream of it works on clean text.
"""

import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Tags whose content is never useful for company intelligence extraction.
STRIP_TAGS = [
    "script", "style", "svg", "noscript", "iframe", "canvas",
    "path", "link", "meta",
]

# Common footer/cookie-banner marker classes/ids seen across marketing sites.
# Heuristic, not exhaustive - failing to strip one of these just costs a
# little extra token budget, it doesn't break correctness.
BOILERPLATE_HINTS = [
    "cookie", "consent", "gdpr", "newsletter-signup",
    "site-footer-legal", "copyright",
]

MAX_CHARS_PER_PAGE = 6000  # per-page cap before dedup, keeps token usage sane
MAX_CHARS_TOTAL = 18000  # hard cap across all pages of a single domain


def clean_html(html: str) -> str:
    """HTML -> clean visible text for a single page."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Drop nav/footer entirely first (boilerplate menus, not content) but
    # keep a light-touch approach: if nav/footer is the ONLY content on the
    # page (rare, but happens on JS-broken pages) we don't want to return
    # nothing, so we check length before/after.
    body_text_len_before = len(soup.get_text(strip=True))
    nav_footer_removed = BeautifulSoup(str(soup), "lxml")
    for tag_name in ("nav", "footer"):
        for tag in nav_footer_removed.find_all(tag_name):
            tag.decompose()
    body_text_len_after = len(nav_footer_removed.get_text(strip=True))

    working_soup = nav_footer_removed if body_text_len_after > 0.15 * max(body_text_len_before, 1) else soup

    # Remove elements that look like cookie banners / legal boilerplate by
    # class or id hints.
    for element in working_soup.find_all(True):
        attrs = " ".join(
            [str(v) for v in (element.get("class") or [])] + [str(element.get("id") or "")]
        ).lower()
        if any(hint in attrs for hint in BOILERPLATE_HINTS):
            element.decompose()

    text = working_soup.get_text(separator="\n", strip=True)
    return _dedupe_lines(text)


def _dedupe_lines(text: str) -> str:
    """Collapse repeated lines (e.g. a nav menu repeated in header AND a
    mobile-menu duplicate) and drop very short noise lines."""
    seen = set()
    kept_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        kept_lines.append(line)
    return "\n".join(kept_lines)


def truncate(text: str, max_chars: int = MAX_CHARS_PER_PAGE) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def combine_pages(page_texts: list[str], max_total_chars: int = MAX_CHARS_TOTAL) -> str:
    """Combine already-cleaned per-page text into one context blob for the
    LLM, applying a global dedupe pass and a hard total-length cap."""
    combined = "\n\n---PAGE BREAK---\n\n".join(t for t in page_texts if t)
    deduped = _dedupe_lines(combined)
    if len(deduped) > max_total_chars:
        deduped = deduped[:max_total_chars] + "\n...[truncated]"
        logger.info("Combined content truncated to %d chars", max_total_chars)
    return deduped

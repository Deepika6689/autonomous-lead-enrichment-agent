"""
Bonus 1: optional external search fallback for LinkedIn discovery when a
personal profile URL wasn't found directly on the company's own pages.

Fully config-gated (Req 4): if ENABLE_SEARCH_FALLBACK is false or no
TAVILY_API_KEY is set, this module is a no-op and the rest of the pipeline
works exactly the same without it. We do NOT scrape LinkedIn directly.
"""

import logging
from typing import Optional

from src.config import settings
from src.extractors import extract_linkedin_personal_urls

logger = logging.getLogger(__name__)


def search_fallback_available() -> bool:
    return settings.enable_search_fallback and bool(settings.tavily_api_key)


def find_linkedin_via_search(person_name: str, company_name: str) -> Optional[str]:
    """Best-effort search for a person's personal LinkedIn profile URL via
    Tavily. Returns None on any failure or if disabled - never raises, since
    this is an optional enhancement, not a required path."""
    if not search_fallback_available():
        return None

    try:
        from tavily import TavilyClient  # imported lazily - optional dependency path
    except ImportError:
        logger.info("tavily-python not installed; skipping search fallback")
        return None

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        query = f'"{person_name}" {company_name} LinkedIn'
        results = client.search(query=query, max_results=5)
        for result in results.get("results", []):
            url = result.get("url", "")
            candidates = extract_linkedin_personal_urls(url)
            if candidates:
                return candidates[0]
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Search fallback failed for %s: %s", person_name, exc)
        return None

"""
Given the homepage HTML, find same-domain internal links worth crawling
(about/team/contact/pricing/leadership-style pages), capped at
MAX_PAGES_PER_DOMAIN to control runtime and token usage. This intentionally
does not attempt a full site crawl.
"""

import logging

from bs4 import BeautifulSoup

from src.config import CANDIDATE_PATH_KEYWORDS, settings
from src.utils import is_same_domain, resolve_url

logger = logging.getLogger(__name__)


def discover_relevant_links(homepage_html: str, homepage_url: str, domain: str) -> list[str]:
    if not homepage_html:
        return []

    soup = BeautifulSoup(homepage_html, "lxml")
    candidates: dict[str, int] = {}  # url -> relevance score (higher = more relevant)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue

        full_url = resolve_url(homepage_url, href)
        full_url = full_url.split("#")[0]  # drop fragments

        if not is_same_domain(full_url, domain):
            continue

        path = full_url.lower()
        score = 0
        for keyword in CANDIDATE_PATH_KEYWORDS:
            if keyword in path:
                score = max(score, 2 if keyword in ("team", "leadership", "founders", "about") else 1)

        if score > 0:
            candidates[full_url] = max(candidates.get(full_url, 0), score)

    # Sort by relevance score desc, then keep it deterministic via URL string
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    max_extra_pages = max(settings.max_pages_per_domain - 1, 0)  # -1 reserves a slot for homepage
    selected = [url for url, _ in ranked[:max_extra_pages]]

    logger.info("Discovered %d relevant internal pages for %s", len(selected), domain)
    return selected

"""
Orchestrates the full pipeline for a single domain:

  crawl homepage -> discover internal pages -> crawl them -> clean content
  -> deterministic email/LinkedIn extraction -> LLM structured extraction
  (validated) -> confidence scoring -> DomainSuccessResult

Every domain is wrapped in a top-level try/except here so that one domain's
unexpected failure can NEVER take down the rest of the run. Anything that
fails mid-pipeline is converted into a DomainErrorResult describing exactly
what stage failed and why - never silently dropped.
"""

import asyncio
import logging
from typing import Union

from playwright.async_api import Browser

from src.config import settings
from src.content_cleaner import clean_html, combine_pages, truncate
from src.confidence import compute_confidence
from src.crawler import fetch_page
from src.extractors import extract_emails, extract_generic_business_emails, extract_linkedin_personal_urls
from src.llm_client import LLMExtractionError, extract_company_intelligence
from src.models import DomainErrorResult, DomainSuccessResult, PageFetchStatus
from src.page_discovery import discover_relevant_links
from src.utils import domain_to_homepage_url, normalize_domain

logger = logging.getLogger(__name__)


async def enrich_domain(browser: Browser, domain: str) -> Union[DomainSuccessResult, DomainErrorResult]:
    domain = normalize_domain(domain)
    pages_crawled: list[PageFetchStatus] = []
    logger.info("Starting enrichment for %s", domain)

    try:
        homepage_url = domain_to_homepage_url(domain)
        homepage_result = await fetch_page(browser, homepage_url)
        pages_crawled.append(homepage_result.status)

        if homepage_result.html is None:
            return DomainErrorResult(
                domain=domain,
                error_stage="crawl_homepage",
                error_message=f"Could not fetch homepage: {homepage_result.status.status} "
                f"({homepage_result.status.detail or 'no detail'})",
                pages_crawled=pages_crawled,
            )

        internal_links = discover_relevant_links(homepage_result.html, homepage_url, domain)

        page_texts = [clean_html(homepage_result.html)]
        for link in internal_links:
            await asyncio.sleep(settings.request_delay_seconds)
            result = await fetch_page(browser, link)
            pages_crawled.append(result.status)
            if result.html:
                page_texts.append(clean_html(result.html))
            else:
                logger.info("Skipping unusable page %s: %s", link, result.status.status)

        combined_text = combine_pages(page_texts)

        if not combined_text.strip():
            return DomainErrorResult(
                domain=domain,
                error_stage="content_extraction",
                error_message="No usable text content extracted from any crawled page.",
                pages_crawled=pages_crawled,
            )

        candidate_emails = extract_generic_business_emails(combined_text) or extract_emails(combined_text)
        candidate_linkedin_urls = extract_linkedin_personal_urls(combined_text)

        try:
            intelligence, token_usage = extract_company_intelligence(
                domain, truncate(combined_text, max_chars=18000), candidate_emails, candidate_linkedin_urls
            )
        except LLMExtractionError as exc:
            return DomainErrorResult(
                domain=domain,
                error_stage="llm_extraction",
                error_message=str(exc),
                pages_crawled=pages_crawled,
            )

        confidence_score, confidence_breakdown = compute_confidence(intelligence, pages_crawled)

        return DomainSuccessResult(
            domain=domain,
            company_overview=intelligence.company_overview,
            target_audience=intelligence.target_audience,
            contact_points=intelligence.contact_points,
            leadership=intelligence.leadership,
            confidence_score=confidence_score,
            confidence_breakdown=confidence_breakdown,
            pages_crawled=pages_crawled,
            token_usage=token_usage,
        )

    except Exception as exc:  # noqa: BLE001 - last-resort guard, must never propagate
        logger.exception("Unhandled error enriching %s", domain)
        return DomainErrorResult(
            domain=domain,
            error_stage="unhandled",
            error_message=str(exc),
            pages_crawled=pages_crawled,
        )

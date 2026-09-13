"""
Playwright-based page fetcher.

Responsibilities:
- Render JS pages and return the resulting HTML.
- Never bypass bot protection (no captcha solving, no header spoofing beyond
  a normal, honest user-agent, no proxy rotation). If a page blocks us, we
  record it as blocked and move on - per the explicit rule against attempting
  to bypass bot protection.
- Respect timeouts and never let a single page hang the whole run.
"""

import asyncio
import logging
from typing import Optional

from playwright.async_api import Browser, Page, TimeoutError as PlaywrightTimeoutError, async_playwright

from src.config import settings
from src.models import PageFetchStatus

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "LeadEnrichmentAgent/1.0 (+contact: research use)"
)

BLOCK_INDICATORS = [
    "access denied", "are you a robot", "captcha", "cloudflare",
    "attention required", "unusual traffic", "403 forbidden",
]


class PageFetchResult:
    def __init__(self, html: Optional[str], status: PageFetchStatus):
        self.html = html
        self.status = status


async def fetch_page(browser: Browser, url: str) -> PageFetchResult:
    """Fetch a single URL with a fresh page/context. Never raises - every
    failure mode is captured in the returned PageFetchStatus."""
    context = await browser.new_context(user_agent=USER_AGENT)
    page: Page = await context.new_page()
    try:
        response = await page.goto(
            url, timeout=settings.page_timeout_ms, wait_until="domcontentloaded"
        )
        # Give client-rendered content a brief moment to settle without
        # waiting indefinitely.
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
        except PlaywrightTimeoutError:
            pass  # fine - we already have domcontentloaded

        if response is None:
            return PageFetchResult(
                None, PageFetchStatus(url=url, status="error", detail="no response object")
            )

        if response.status == 404:
            return PageFetchResult(None, PageFetchStatus(url=url, status="404"))

        if response.status in (403, 429):
            return PageFetchResult(
                None,
                PageFetchStatus(
                    url=url, status="blocked", detail=f"HTTP {response.status}"
                ),
            )

        if response.status >= 400:
            return PageFetchResult(
                None,
                PageFetchStatus(
                    url=url, status="error", detail=f"HTTP {response.status}"
                ),
            )

        html = await page.content()
        lowered = html.lower()
        if any(indicator in lowered for indicator in BLOCK_INDICATORS) and len(html) < 5000:
            return PageFetchResult(
                None, PageFetchStatus(url=url, status="blocked", detail="bot-block page detected")
            )

        return PageFetchResult(html, PageFetchStatus(url=url, status="ok"))

    except PlaywrightTimeoutError:
        return PageFetchResult(None, PageFetchStatus(url=url, status="timeout"))
    except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly here
        logger.warning("Fetch failed for %s: %s", url, exc)
        return PageFetchResult(None, PageFetchStatus(url=url, status="error", detail=str(exc)))
    finally:
        await context.close()


class BrowserSession:
    """Thin async context manager around a single shared Playwright browser
    instance, reused across all domains in a run for efficiency."""

    def __init__(self):
        self._playwright = None
        self.browser: Optional[Browser] = None

    async def __aenter__(self) -> Browser:
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=True)
        return self.browser

    async def __aexit__(self, exc_type, exc, tb):
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

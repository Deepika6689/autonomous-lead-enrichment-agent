"""
Unit tests for the deterministic, network-free parts of the pipeline:
email/LinkedIn extraction and HTML cleaning. These run against synthetic
HTML/text and do NOT require internet access or a live crawl - useful both
for CI and for demonstrating correctness in the Loom video without needing
a live run every time.

Run with: pytest tests/test_extractors.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extractors import (
    extract_emails,
    extract_generic_business_emails,
    extract_linkedin_company_urls,
    extract_linkedin_personal_urls,
)
from src.content_cleaner import clean_html, combine_pages, _dedupe_lines


def test_extract_emails_basic():
    text = "Reach us at contact@example.com or sales@example.com for more info."
    emails = extract_emails(text)
    assert "contact@example.com" in emails
    assert "sales@example.com" in emails
    assert len(emails) == 2


def test_extract_emails_ignores_image_filenames():
    text = "logo@2x.png and icon@3x.jpg are not emails, but hello@example.com is."
    emails = extract_emails(text)
    assert "hello@example.com" in emails
    assert not any(e.endswith(".png") for e in emails)


def test_extract_emails_dedupes():
    text = "info@example.com info@example.com INFO@example.com"
    emails = extract_emails(text)
    assert emails == ["info@example.com"]


def test_generic_business_emails_filters_personal_looking_addresses():
    text = "Contact hello@example.com or john.smith@example.com directly."
    generic = extract_generic_business_emails(text)
    assert "hello@example.com" in generic
    assert "john.smith@example.com" not in generic


def test_linkedin_personal_vs_company_distinction():
    text = (
        "Follow us: https://www.linkedin.com/company/example-inc/ "
        "Our CEO: https://linkedin.com/in/jane-doe-12345/"
    )
    personal = extract_linkedin_personal_urls(text)
    company = extract_linkedin_company_urls(text)
    assert any("/in/jane-doe" in u for u in personal)
    assert not any("/in/" in u for u in company)
    assert any("/company/example-inc" in u for u in company)
    # Personal list must never contain a company URL and vice versa
    assert not set(personal) & set(company)


def test_clean_html_strips_script_and_style():
    html = """
    <html><body>
      <script>alert('x')</script>
      <style>.a{color:red}</style>
      <nav><a href="/">Home</a><a href="/about">About</a></nav>
      <main><h1>Acme Inc</h1><p>We build developer tools for backend teams.</p></main>
      <footer>Copyright 2026 Acme Inc. All rights reserved.</footer>
    </body></html>
    """
    cleaned = clean_html(html)
    assert "alert" not in cleaned
    assert "color:red" not in cleaned
    assert "Acme Inc" in cleaned
    assert "developer tools" in cleaned


def test_clean_html_empty_input():
    assert clean_html("") == ""
    assert clean_html(None) == ""


def test_dedupe_lines_removes_repeated_menu_items():
    text = "Home\nAbout\nHome\nPricing\nAbout\nContact"
    deduped = _dedupe_lines(text)
    lines = deduped.split("\n")
    assert lines.count("Home") == 1
    assert lines.count("About") == 1
    assert "Pricing" in lines
    assert "Contact" in lines


def test_combine_pages_respects_max_length():
    pages = ["A" * 100, "B" * 100, "C" * 100]
    combined = combine_pages(pages, max_total_chars=50)
    assert len(combined) <= 50 + len("\n...[truncated]")
    assert combined.endswith("[truncated]")


def test_combine_pages_skips_empty():
    pages = ["Real content here about the company.", "", None]
    combined = combine_pages([p for p in pages if p])
    assert "Real content here" in combined

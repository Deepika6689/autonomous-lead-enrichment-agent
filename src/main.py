"""
CLI entrypoint.

Usage:
    python -m src.main
    python -m src.main --domains postman.com,supabase.com,vapi.ai
    python -m src.main --domains-file domains.json
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from src.config import settings
from src.cost_tracker import aggregate
from src.crawler import BrowserSession
from src.enrichment_agent import enrich_domain
from src.models import DomainSuccessResult
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def load_domains(args: argparse.Namespace) -> list[str]:
    if args.domains:
        return [d.strip() for d in args.domains.split(",") if d.strip()]
    domains_file = Path(args.domains_file)
    if not domains_file.exists():
        raise FileNotFoundError(f"Domains file not found: {domains_file}")
    data = json.loads(domains_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("domains", [])
    return data


async def run(domains: list[str]) -> list[dict]:
    results = []
    async with BrowserSession() as browser:
        for domain in domains:
            start = time.monotonic()
            result = await enrich_domain(browser, domain)
            elapsed = time.monotonic() - start
            logger.info("Finished %s in %.1fs -> status=%s", domain, elapsed, result.status)
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Lead Enrichment Agent")
    parser.add_argument("--domains", type=str, default=None, help="Comma-separated list of domains")
    parser.add_argument("--domains-file", type=str, default="domains.json", help="Path to a JSON domains file")
    parser.add_argument("--output", type=str, default="output/output.json", help="Output JSON path")
    args = parser.parse_args()

    setup_logging(settings.log_level)
    domains = load_domains(args)
    logger.info("Running enrichment for %d domain(s): %s", len(domains), domains)

    results = asyncio.run(run(domains))

    output_data = [r.model_dump() for r in results]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    logger.info("Wrote results to %s", output_path)

    successes = [r for r in results if isinstance(r, DomainSuccessResult)]
    total_usage = aggregate([r.token_usage for r in successes])
    logger.info(
        "Run summary: %d/%d succeeded | tokens in=%d out=%d | est. cost=$%.6f",
        len(successes), len(results),
        total_usage.input_tokens, total_usage.output_tokens, total_usage.estimated_cost_usd,
    )


if __name__ == "__main__":
    main()

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse


def setup_logging(log_level: str = "INFO", log_file: str = "logs/run.log") -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower()
    domain = re.sub(r"^https?://", "", domain)
    domain = domain.split("/")[0]
    return domain


def domain_to_homepage_url(domain: str) -> str:
    return f"https://{normalize_domain(domain)}"


def is_same_domain(url: str, root_domain: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower().replace("www.", "")
        root = normalize_domain(root_domain).replace("www.", "")
        return netloc == root
    except Exception:
        return False


def resolve_url(base: str, link: str) -> str:
    return urljoin(base, link)

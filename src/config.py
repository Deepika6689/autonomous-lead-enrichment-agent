"""Central configuration, loaded from environment variables / .env.
Never hardcode API keys - everything here comes from the environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    max_pages_per_domain: int = 8
    page_timeout_ms: int = 15000
    request_delay_seconds: float = 1.0
    max_retries: int = 2

    enable_search_fallback: bool = False
    tavily_api_key: str = ""

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

# Candidate internal paths we actively look for on every domain, in addition
# to whatever the homepage links to. Kept small and specific per the
# assignment's "do not crawl the entire site" requirement.
CANDIDATE_PATH_KEYWORDS = [
    "about",
    "about-us",
    "company",
    "team",
    "our-team",
    "leadership",
    "founders",
    "contact",
    "contact-us",
    "pricing",
]

"""
Pydantic schemas used throughout the pipeline.

Design note: the LLM's JSON mode is treated as a *hint*, never a guarantee.
`CompanyIntelligence` is the strict schema we validate every LLM response
against. If validation fails, `llm_client.py` retries with a corrective
prompt (see Part J) up to `settings.max_retries` times before the domain is
recorded as a structured failure. Pydantic is the final source of truth.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class LeadershipMember(BaseModel):
    name: str
    role: str
    linkedin_url: Optional[str] = None

    @field_validator("linkedin_url")
    @classmethod
    def only_personal_profiles(cls, v: Optional[str]) -> Optional[str]:
        """Enforce requirement: only linkedin.com/in/... (personal profile)
        URLs may be attached to a leadership member. Company pages
        (linkedin.com/company/...) or anything else are dropped to null
        rather than silently kept, because attaching a company page to a
        named person would be misleading, not just imprecise."""
        if v is None:
            return v
        v = v.strip()
        if "linkedin.com/in/" in v.lower():
            return v
        return None


class ContactPoints(BaseModel):
    emails: List[str] = Field(default_factory=list)


class CompanyIntelligence(BaseModel):
    """The exact shape we require from the LLM. Anything that doesn't fit
    this shape fails validation and triggers a corrective retry."""

    company_overview: str = Field(..., min_length=1)
    target_audience: str = Field(..., min_length=1)
    contact_points: ContactPoints
    leadership: List[LeadershipMember] = Field(default_factory=list)


class PageFetchStatus(BaseModel):
    url: str
    status: str  # "ok" | "404" | "timeout" | "blocked" | "error" | "skipped"
    detail: Optional[str] = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class DomainSuccessResult(BaseModel):
    domain: str
    status: str = "success"
    company_overview: str
    target_audience: str
    contact_points: ContactPoints
    leadership: List[LeadershipMember]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    confidence_breakdown: dict = Field(default_factory=dict)
    pages_crawled: List[PageFetchStatus]
    token_usage: TokenUsage


class DomainErrorResult(BaseModel):
    domain: str
    status: str = "failed"
    error_stage: str
    error_message: str
    pages_crawled: List[PageFetchStatus] = Field(default_factory=list)

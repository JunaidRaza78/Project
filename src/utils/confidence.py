"""
Confidence Scoring System

Implements source validation and confidence scoring for findings.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse


class SourceTier(Enum):
    """Tier classification for information sources."""
    TIER_1 = "tier_1"  # Official records, verified news (0.8-1.0)
    TIER_2 = "tier_2"  # Reputable journalism (0.5-0.7)
    TIER_3 = "tier_3"  # Social media, unverified (0.0-0.4)


@dataclass
class SourceEvaluation:
    """Evaluation of a source's reliability."""
    url: str
    tier: SourceTier
    base_confidence: float
    domain: str
    notes: str = ""


class ConfidenceScorer:
    """
    Scores the confidence of findings based on sources.
    
    Confidence Levels:
    - High (0.8-1.0): Official records, major news outlets, verified databases
    - Medium (0.5-0.7): Reputable journalism, industry publications
    - Low (0.0-0.4): Social media, blogs, unverified claims
    """
    
    # High confidence domains (Tier 1)
    TIER_1_DOMAINS = {
        # Government & Official
        "sec.gov", "justice.gov", "fbi.gov", "courts.gov",
        "state.gov", "treasury.gov", "ftc.gov",
        # Major Wire Services
        "reuters.com", "apnews.com", "afp.com",
        # Major News
        "nytimes.com", "washingtonpost.com", "wsj.com",
        "bbc.com", "bbc.co.uk", "theguardian.com",
        "ft.com", "economist.com",
        # Business/Financial
        "bloomberg.com", "cnbc.com", "forbes.com",
        # Academic/Research
        "nature.com", "science.org", "arxiv.org",
        # Professional Databases
        "linkedin.com", "crunchbase.com",
    }
    
    # Medium confidence domains (Tier 2)
    TIER_2_DOMAINS = {
        # Regional News
        "latimes.com", "chicagotribune.com", "usatoday.com",
        "cnn.com", "foxnews.com", "msnbc.com",
        # Tech News
        "techcrunch.com", "wired.com", "arstechnica.com",
        "theverge.com", "engadget.com",
        # Business
        "businessinsider.com", "fortune.com", "inc.com",
        # General Reference
        "wikipedia.org", "britannica.com",
    }
    
    # Low confidence (Tier 3) - Everything else including:
    TIER_3_DOMAINS = {
        "twitter.com", "x.com", "facebook.com",
        "reddit.com", "quora.com",
        "medium.com",  # Unless specific publication
    }
    
    def __init__(self):
        self._cache: dict[str, SourceEvaluation] = {}
    
    def _extract_domain(self, url: str) -> str:
        """Extract base domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def _get_tier(self, domain: str) -> SourceTier:
        """Determine source tier from domain."""
        # Check exact match first
        if domain in self.TIER_1_DOMAINS:
            return SourceTier.TIER_1
        if domain in self.TIER_2_DOMAINS:
            return SourceTier.TIER_2
        if domain in self.TIER_3_DOMAINS:
            return SourceTier.TIER_3
        
        # Check partial matches
        for t1_domain in self.TIER_1_DOMAINS:
            if t1_domain in domain or domain in t1_domain:
                return SourceTier.TIER_1
        
        for t2_domain in self.TIER_2_DOMAINS:
            if t2_domain in domain or domain in t2_domain:
                return SourceTier.TIER_2
        
        # Default to Tier 3
        return SourceTier.TIER_3
    
    def _tier_to_confidence(self, tier: SourceTier) -> float:
        """Convert tier to base confidence score."""
        if tier == SourceTier.TIER_1:
            return 0.85
        elif tier == SourceTier.TIER_2:
            return 0.60
        else:
            return 0.30
    
    def evaluate_source(self, url: str) -> SourceEvaluation:
        """Evaluate a single source URL."""
        if url in self._cache:
            return self._cache[url]
        
        domain = self._extract_domain(url)
        tier = self._get_tier(domain)
        confidence = self._tier_to_confidence(tier)
        
        evaluation = SourceEvaluation(
            url=url,
            tier=tier,
            base_confidence=confidence,
            domain=domain,
        )
        
        self._cache[url] = evaluation
        return evaluation
    
    def calculate_confidence(
        self,
        source_urls: list[str],
        cross_reference_bonus: float = 0.1,
    ) -> float:
        """
        Calculate overall confidence for a finding.
        
        Args:
            source_urls: List of source URLs supporting the finding
            cross_reference_bonus: Bonus for each additional confirming source
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not source_urls:
            return 0.0
        
        # Evaluate all sources
        evaluations = [self.evaluate_source(url) for url in source_urls]
        
        # Get highest tier source as base
        evaluations.sort(key=lambda e: e.base_confidence, reverse=True)
        base_confidence = evaluations[0].base_confidence
        
        # Add cross-reference bonus for multiple sources
        unique_domains = set(e.domain for e in evaluations)
        num_confirming = len(unique_domains) - 1  # Exclude primary source
        bonus = min(num_confirming * cross_reference_bonus, 0.15)  # Cap at 0.15
        
        # Final confidence
        confidence = min(base_confidence + bonus, 1.0)
        
        return round(confidence, 2)
    
    def requires_verification(self, confidence: float) -> bool:
        """Check if a finding needs additional verification."""
        return confidence < 0.7
    
    def get_confidence_label(self, confidence: float) -> str:
        """Get human-readable confidence label."""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

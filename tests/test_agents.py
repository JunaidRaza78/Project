"""
Tests for research agent components.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import AgentState, Finding, RiskIndicator, Connection, InvestigationPhase
from src.utils.confidence import ConfidenceScorer, SourceTier


class TestAgentState:
    """Tests for AgentState class."""
    
    def test_initial_state(self):
        """Test initial state creation."""
        state = AgentState(target_name="Test Person")
        
        assert state.target_name == "Test Person"
        assert state.current_phase == InvestigationPhase.INITIAL_SEARCH
        assert state.iteration_count == 0
        assert len(state.findings) == 0
        assert len(state.risk_indicators) == 0
    
    def test_add_finding_deduplication(self):
        """Test that duplicate findings are not added."""
        state = AgentState(target_name="Test")
        
        finding1 = Finding(
            category="biography",
            fact="Born in 1984",
            source_urls=["http://example.com"],
            confidence=0.7,
        )
        finding2 = Finding(
            category="biography",
            fact="Born in 1984",  # Same fact
            source_urls=["http://other.com"],
            confidence=0.8,  # Higher confidence
        )
        
        state.add_finding(finding1)
        state.add_finding(finding2)
        
        assert len(state.findings) == 1
        assert state.findings[0].confidence == 0.8  # Updated to higher
    
    def test_should_continue_searching(self):
        """Test search continuation logic."""
        state = AgentState(target_name="Test", max_iterations=5)
        state.pending_queries = ["query1"]
        state.iteration_count = 3
        
        assert state.should_continue_searching() is True
        
        state.iteration_count = 5
        assert state.should_continue_searching() is False
        
        state.iteration_count = 3
        state.pending_queries = []
        assert state.should_continue_searching() is False
    
    def test_get_high_confidence_findings(self):
        """Test filtering high confidence findings."""
        state = AgentState(target_name="Test")
        
        state.findings = [
            Finding("bio", "Fact 1", [], 0.9),
            Finding("bio", "Fact 2", [], 0.5),
            Finding("bio", "Fact 3", [], 0.8),
        ]
        
        high_conf = state.get_high_confidence_findings(0.7)
        
        assert len(high_conf) == 2
        assert all(f.confidence >= 0.7 for f in high_conf)


class TestConfidenceScorer:
    """Tests for ConfidenceScorer class."""
    
    def test_tier_classification(self):
        """Test source tier classification."""
        scorer = ConfidenceScorer()
        
        # Tier 1
        eval1 = scorer.evaluate_source("https://www.nytimes.com/article")
        assert eval1.tier == SourceTier.TIER_1
        
        # Tier 2
        eval2 = scorer.evaluate_source("https://techcrunch.com/news")
        assert eval2.tier == SourceTier.TIER_2
        
        # Tier 3
        eval3 = scorer.evaluate_source("https://randomsite.xyz/post")
        assert eval3.tier == SourceTier.TIER_3
    
    def test_confidence_calculation_single_source(self):
        """Test confidence with single source."""
        scorer = ConfidenceScorer()
        
        # High tier source
        conf1 = scorer.calculate_confidence(["https://sec.gov/filing"])
        assert conf1 >= 0.8
        
        # Low tier source
        conf2 = scorer.calculate_confidence(["https://random.blog/post"])
        assert conf2 < 0.5
    
    def test_confidence_cross_reference_bonus(self):
        """Test cross-reference bonus for multiple sources."""
        scorer = ConfidenceScorer()
        
        # Single source
        conf1 = scorer.calculate_confidence(["https://nytimes.com/article"])
        
        # Multiple sources from different domains
        conf2 = scorer.calculate_confidence([
            "https://nytimes.com/article",
            "https://wsj.com/story",
        ])
        
        assert conf2 > conf1
    
    def test_confidence_label(self):
        """Test confidence label generation."""
        scorer = ConfidenceScorer()
        
        assert scorer.get_confidence_label(0.9) == "HIGH"
        assert scorer.get_confidence_label(0.6) == "MEDIUM"
        assert scorer.get_confidence_label(0.3) == "LOW"


class TestFindingModel:
    """Tests for Finding dataclass."""
    
    def test_to_dict(self):
        """Test Finding serialization."""
        finding = Finding(
            category="professional",
            fact="CEO of Company X",
            source_urls=["http://example.com"],
            confidence=0.85,
            verified=True,
        )
        
        data = finding.to_dict()
        
        assert data["category"] == "professional"
        assert data["fact"] == "CEO of Company X"
        assert data["confidence"] == 0.85
        assert data["verified"] is True


class TestRiskIndicatorModel:
    """Tests for RiskIndicator dataclass."""
    
    def test_to_dict(self):
        """Test RiskIndicator serialization."""
        risk = RiskIndicator(
            category="legal",
            description="Fraud conviction",
            severity=9,
            evidence=["Convicted in 2022"],
            source_urls=["http://example.com"],
            confidence=0.9,
        )
        
        data = risk.to_dict()
        
        assert data["category"] == "legal"
        assert data["severity"] == 9
        assert len(data["evidence"]) == 1


class TestConnectionModel:
    """Tests for Connection dataclass."""
    
    def test_to_dict(self):
        """Test Connection serialization."""
        conn = Connection(
            entity_name="Company X",
            entity_type="organization",
            relationship="founded",
            timeframe="2010-2020",
            source_urls=["http://example.com"],
            confidence=0.8,
        )
        
        data = conn.to_dict()
        
        assert data["entity_name"] == "Company X"
        assert data["entity_type"] == "organization"
        assert data["timeframe"] == "2010-2020"


# Integration tests require API keys, so they're marked to skip by default
@pytest.mark.skip(reason="Requires API keys")
class TestIntegration:
    """Integration tests for full agent workflow."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocked API keys."""
        from src.agents.orchestrator import ResearchOrchestrator
        from src.config import get_settings
        
        settings = get_settings()
        return ResearchOrchestrator(
            groq_api_key=settings.groq_api_key,
            google_api_key=settings.google_api_key,
            serper_api_key=settings.serper_api_key,
        )
    
    @pytest.mark.asyncio
    async def test_full_investigation(self, orchestrator):
        """Test a complete investigation workflow."""
        state = await orchestrator.investigate(
            target_name="Elizabeth Holmes",
            max_iterations=3,
        )
        
        assert state.current_phase == InvestigationPhase.COMPLETE
        assert len(state.findings) > 0
        assert state.final_report != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

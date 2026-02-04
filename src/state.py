"""
LangGraph State Management

Defines the state schema for the research agent workflow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional
import operator


class InvestigationPhase(Enum):
    """Current phase of the investigation."""
    INITIAL_SEARCH = "initial_search"
    FACT_EXTRACTION = "fact_extraction"
    RISK_ANALYSIS = "risk_analysis"
    CONNECTION_MAPPING = "connection_mapping"
    SOURCE_VALIDATION = "source_validation"
    REPORT_GENERATION = "report_generation"
    COMPLETE = "complete"


@dataclass
class SearchResult:
    """Individual search result."""
    query: str
    title: str
    snippet: str
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0


@dataclass
class Finding:
    """Extracted finding about the target."""
    category: str  # biography, professional, financial, etc.
    fact: str
    source_urls: list[str]
    confidence: float  # 0.0 - 1.0
    extracted_at: datetime = field(default_factory=datetime.now)
    verified: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "fact": self.fact,
            "source_urls": self.source_urls,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at.isoformat(),
            "verified": self.verified,
        }


@dataclass
class RiskIndicator:
    """Identified risk pattern or red flag."""
    category: str  # legal, financial, reputation, association
    description: str
    severity: int  # 1-10
    evidence: list[str]
    source_urls: list[str]
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence,
            "source_urls": self.source_urls,
            "confidence": self.confidence,
        }


@dataclass 
class Connection:
    """Relationship between entities."""
    entity_name: str
    entity_type: str  # person, organization, event
    relationship: str  # employer, associate, investor, etc.
    timeframe: Optional[str] = None
    source_urls: list[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "relationship": self.relationship,
            "timeframe": self.timeframe,
            "source_urls": self.source_urls,
            "confidence": self.confidence,
        }


@dataclass
class AgentState:
    """
    Complete state for the research agent workflow.
    
    This is passed between nodes in the LangGraph workflow.
    """
    # Target information
    target_name: str
    target_context: str = ""  # Additional context about target
    
    # Search tracking
    search_history: list[str] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    pending_queries: list[str] = field(default_factory=list)
    
    # Extracted information
    findings: list[Finding] = field(default_factory=list)
    risk_indicators: list[RiskIndicator] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    
    # Workflow state
    current_phase: InvestigationPhase = InvestigationPhase.INITIAL_SEARCH
    iteration_count: int = 0
    max_iterations: int = 10
    
    # Error tracking
    errors: list[str] = field(default_factory=list)
    
    # Output
    final_report: str = ""
    
    def should_continue_searching(self) -> bool:
        """Check if more search iterations are needed."""
        return (
            self.iteration_count < self.max_iterations
            and len(self.pending_queries) > 0
            and self.current_phase not in [
                InvestigationPhase.REPORT_GENERATION,
                InvestigationPhase.COMPLETE,
            ]
        )
    
    def add_finding(self, finding: Finding) -> None:
        """Add a new finding, avoiding duplicates."""
        # Check for duplicate facts
        for existing in self.findings:
            if existing.fact.lower() == finding.fact.lower():
                # Update confidence if higher
                if finding.confidence > existing.confidence:
                    existing.confidence = finding.confidence
                    existing.source_urls.extend(finding.source_urls)
                return
        self.findings.append(finding)
    
    def add_risk(self, risk: RiskIndicator) -> None:
        """Add a risk indicator."""
        # Check for similar risks
        for existing in self.risk_indicators:
            if existing.description.lower() == risk.description.lower():
                if risk.confidence > existing.confidence:
                    existing.confidence = risk.confidence
                return
        self.risk_indicators.append(risk)
    
    def add_connection(self, connection: Connection) -> None:
        """Add a connection."""
        for existing in self.connections:
            if (existing.entity_name.lower() == connection.entity_name.lower() 
                and existing.relationship == connection.relationship):
                return
        self.connections.append(connection)
    
    def get_high_confidence_findings(self, min_confidence: float = 0.7) -> list[Finding]:
        """Get findings above confidence threshold."""
        return [f for f in self.findings if f.confidence >= min_confidence]
    
    def get_critical_risks(self, min_severity: int = 7) -> list[RiskIndicator]:
        """Get high severity risks."""
        return [r for r in self.risk_indicators if r.severity >= min_severity]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "target_name": self.target_name,
            "target_context": self.target_context,
            "search_history": self.search_history,
            "findings": [f.to_dict() for f in self.findings],
            "risk_indicators": [r.to_dict() for r in self.risk_indicators],
            "connections": [c.to_dict() for c in self.connections],
            "current_phase": self.current_phase.value,
            "iteration_count": self.iteration_count,
            "errors": self.errors,
        }

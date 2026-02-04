"""
Risk Analyzer Agent

Identifies potential red flags and risk patterns.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, RiskIndicator
from ..utils.confidence import ConfidenceScorer


RISK_ANALYSIS_PROMPT = """You are a risk assessment expert analyzing information about an individual or entity.

TARGET: {target_name}

FINDINGS TO ANALYZE:
{findings}

SEARCH RESULTS (for additional context):
{search_results}

Analyze this information and identify any risk indicators. Look for:

1. LEGAL RISKS: Lawsuits, investigations, arrests, convictions, regulatory actions
2. FINANCIAL RISKS: Bankruptcies, fraud allegations, unpaid debts, financial irregularities
3. REPUTATION RISKS: Scandals, controversies, public disputes, negative media coverage
4. ASSOCIATION RISKS: Connections to problematic individuals/organizations
5. PATTERN RISKS: Inconsistencies in claims, frequent job changes, gaps in history

For each risk indicator, assess:
- Severity (1-10 scale): 
  - 1-3: Minor concerns, common issues
  - 4-6: Moderate concerns, requires attention
  - 7-10: Serious red flags, major concerns

Respond with a JSON array of risk indicators:
[
  {{
    "category": "legal|financial|reputation|association|pattern",
    "description": "Clear description of the risk",
    "severity": 7,
    "evidence": ["specific fact 1", "specific fact 2"],
    "source_urls": ["url1", "url2"],
    "confidence_note": "Why this is concerning"
  }}
]

Only flag genuine risks with supporting evidence. Do not flag normal business activities.
"""


class RiskAnalyzerAgent:
    """
    Analyzes findings to identify risk patterns and red flags.
    
    Categories:
    - Legal: Lawsuits, convictions, regulatory issues
    - Financial: Fraud, bankruptcy, irregularities
    - Reputation: Scandals, controversies
    - Association: Problematic connections
    - Pattern: Inconsistencies, suspicious patterns
    """
    
    RISK_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "description": {"type": "string"},
                "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "source_urls": {"type": "array", "items": {"type": "string"}},
                "confidence_note": {"type": "string"},
            },
            "required": ["category", "description", "severity", "evidence"],
        },
    }
    
    # Keywords that suggest high-risk content
    HIGH_RISK_KEYWORDS = [
        "fraud", "convicted", "sentenced", "indicted", "arrested",
        "scandal", "lawsuit", "investigation", "sec charges",
        "bankruptcy", "misappropriation", "embezzlement",
    ]
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.confidence_scorer = ConfidenceScorer()
    
    def _format_findings(self, state: AgentState) -> str:
        """Format findings for analysis."""
        findings_text = []
        
        for finding in state.findings:
            findings_text.append(
                f"[{finding.category.upper()}] {finding.fact} "
                f"(Confidence: {finding.confidence:.0%})"
            )
        
        return "\n".join(findings_text) if findings_text else "No findings yet."
    
    def _format_search_results(self, state: AgentState) -> str:
        """Format recent search results."""
        results = []
        for result in state.search_results[-10:]:
            results.append(f"- {result.title}: {result.snippet[:200]}")
        return "\n".join(results) if results else "No results."
    
    def _quick_risk_scan(self, state: AgentState) -> list[str]:
        """Quick keyword scan for potential risks."""
        potential_risks = []
        
        # Scan search result snippets
        for result in state.search_results:
            snippet_lower = result.snippet.lower()
            for keyword in self.HIGH_RISK_KEYWORDS:
                if keyword in snippet_lower:
                    potential_risks.append(
                        f"Found '{keyword}' in: {result.title}"
                    )
                    break
        
        # Scan findings
        for finding in state.findings:
            if finding.category == "controversies":
                potential_risks.append(f"Controversy finding: {finding.fact[:100]}")
        
        return potential_risks[:10]  # Limit to 10
    
    async def analyze(self, state: AgentState) -> list[RiskIndicator]:
        """
        Analyze current state for risk indicators.
        
        Args:
            state: Current agent state
            
        Returns:
            List of RiskIndicator objects
        """
        # Quick scan first
        quick_risks = self._quick_risk_scan(state)
        
        if not state.findings and not quick_risks:
            return []
        
        # Use Gemini for complex reasoning
        prompt = RISK_ANALYSIS_PROMPT.format(
            target_name=state.target_name,
            findings=self._format_findings(state),
            search_results=self._format_search_results(state),
        )
        
        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.RISK_SCHEMA,
            task_type=TaskType.COMPLEX_REASONING,  # Use Gemini for this
        )
        
        risks = []
        
        if response.success:
            try:
                data = json.loads(response.content)
                
                for item in data:
                    source_urls = item.get("source_urls", [])
                    confidence = self.confidence_scorer.calculate_confidence(source_urls)
                    
                    risk = RiskIndicator(
                        category=item.get("category", "unknown"),
                        description=item.get("description", ""),
                        severity=item.get("severity", 5),
                        evidence=item.get("evidence", []),
                        source_urls=source_urls,
                        confidence=confidence,
                    )
                    
                    if risk.description:
                        risks.append(risk)
                        
            except json.JSONDecodeError:
                pass
        
        return risks
    
    def calculate_overall_risk_score(self, risks: list[RiskIndicator]) -> dict[str, Any]:
        """
        Calculate an overall risk assessment.
        
        Returns:
            Dictionary with risk score and breakdown
        """
        if not risks:
            return {
                "overall_score": 0,
                "risk_level": "LOW",
                "breakdown": {},
            }
        
        # Weight by severity and confidence
        weighted_scores = []
        category_scores = {}
        
        for risk in risks:
            weighted = risk.severity * risk.confidence
            weighted_scores.append(weighted)
            
            if risk.category not in category_scores:
                category_scores[risk.category] = []
            category_scores[risk.category].append(weighted)
        
        # Overall score (0-10)
        overall = min(10, sum(weighted_scores) / max(len(weighted_scores), 1) * 1.2)
        
        # Category breakdown
        breakdown = {
            cat: round(sum(scores) / len(scores), 1)
            for cat, scores in category_scores.items()
        }
        
        # Risk level
        if overall >= 7:
            level = "HIGH"
        elif overall >= 4:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return {
            "overall_score": round(overall, 1),
            "risk_level": level,
            "breakdown": breakdown,
            "num_risks": len(risks),
            "critical_risks": len([r for r in risks if r.severity >= 7]),
        }

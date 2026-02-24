"""
Risk Analyzer Agent

Identifies potential red flags and risk patterns using Gemini for complex reasoning.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, RiskIndicator
from ..utils.confidence import ConfidenceScorer
from ..utils.json_repair import extract_json


RISK_ANALYSIS_PROMPT = """You are a senior risk assessment analyst performing due diligence on a target individual or entity.

## Target
Name: {target_name}

## Verified Findings
{findings}

## Raw Search Results
{search_results}

## Analysis Instructions

Think step by step:

1. REVIEW all findings and search results for risk signals.
2. IDENTIFY patterns that indicate genuine risk (not normal business activity).
3. CLASSIFY each risk into a category:
   - **legal**: Lawsuits, criminal charges, convictions, regulatory actions, investigations, SEC/FINRA actions
   - **financial**: Fraud allegations, bankruptcies, misappropriation, accounting irregularities, unauthorized trading
   - **reputation**: Public scandals, controversies, negative media coverage, trust violations
   - **association**: Ties to sanctioned entities, convicted individuals, problematic organizations, suspended advisors
   - **pattern**: Inconsistencies in claims, suspicious career gaps, repeated failures, evasion patterns
4. RATE severity on a 1-10 scale:
   - 1-3: Minor concerns common in normal business
   - 4-6: Moderate concerns warranting further investigation
   - 7-10: Serious red flags indicating significant risk
5. CITE specific evidence for each risk. Do not fabricate evidence.

## Output Format

Respond ONLY with a valid JSON array. No text before or after. No markdown code fences.

[
  {{
    "category": "legal",
    "description": "Convicted of wire fraud in federal court",
    "severity": 9,
    "evidence": ["Found guilty on 4 counts of wire fraud", "Sentenced to 11 years"],
    "source_urls": ["https://example.com/article"],
    "confidence_note": "Multiple major news outlets confirm conviction"
  }}
]

## Rules
- Only flag genuine risks with concrete evidence from the findings/results above
- Severity should be proportional to actual impact, not speculation
- If no risks are found, return an empty array: []
- Be thorough: check for regulatory actions, lawsuits, fraud, misconduct
"""


class RiskAnalyzerAgent:
    """
    Analyzes findings to identify risk patterns and red flags.
    Uses Gemini 2.5 for complex reasoning via COMPLEX_REASONING task type.
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

    HIGH_RISK_KEYWORDS = [
        "fraud", "convicted", "sentenced", "indicted", "arrested",
        "scandal", "lawsuit", "investigation", "sec charges", "sec action",
        "bankruptcy", "misappropriation", "embezzlement", "suspended",
        "revoked", "regulatory action", "finra", "violation", "enforcement",
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
        for result in state.search_results[-15:]:
            results.append(f"[{result.url}]\n- {result.title}: {result.snippet[:300]}")
        return "\n".join(results) if results else "No results."

    def _quick_risk_scan(self, state: AgentState) -> list[str]:
        """Quick keyword scan for potential risks."""
        potential_risks = []

        for result in state.search_results:
            snippet_lower = result.snippet.lower()
            for keyword in self.HIGH_RISK_KEYWORDS:
                if keyword in snippet_lower:
                    potential_risks.append(
                        f"Found '{keyword}' in: {result.title}"
                    )
                    break

        for finding in state.findings:
            if finding.category == "controversies":
                potential_risks.append(f"Controversy finding: {finding.fact[:100]}")

        return potential_risks[:10]

    async def analyze(self, state: AgentState) -> list[RiskIndicator]:
        """Analyze current state for risk indicators using Gemini for complex reasoning."""
        quick_risks = self._quick_risk_scan(state)

        if not state.findings and not quick_risks:
            return []

        prompt = RISK_ANALYSIS_PROMPT.format(
            target_name=state.target_name,
            findings=self._format_findings(state),
            search_results=self._format_search_results(state),
        )

        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.RISK_SCHEMA,
            task_type=TaskType.COMPLEX_REASONING,
        )

        risks = []

        if response.success:
            data = extract_json(response.content)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
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

        return risks

    def calculate_overall_risk_score(self, risks: list[RiskIndicator]) -> dict[str, Any]:
        """Calculate an overall risk assessment."""
        if not risks:
            return {
                "overall_score": 0,
                "risk_level": "LOW",
                "breakdown": {},
            }

        weighted_scores = []
        category_scores = {}

        for risk in risks:
            weighted = risk.severity * risk.confidence
            weighted_scores.append(weighted)

            if risk.category not in category_scores:
                category_scores[risk.category] = []
            category_scores[risk.category].append(weighted)

        overall = min(10, sum(weighted_scores) / max(len(weighted_scores), 1) * 1.2)

        breakdown = {
            cat: round(sum(scores) / len(scores), 1)
            for cat, scores in category_scores.items()
        }

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

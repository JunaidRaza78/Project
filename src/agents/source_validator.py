"""
Source Validator Agent

Cross-references findings and validates source reliability.
Uses Gemini for complex reasoning about corroboration vs. contradiction.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, Finding
from ..utils.confidence import ConfidenceScorer
from ..utils.json_repair import extract_json


VALIDATION_PROMPT = """You are a fact-checker validating information about {target_name}.

## Finding to Validate
Claim: {claim}
Category: {category}
Current sources: {sources}
Current confidence: {confidence}

## Additional Search Results (for cross-reference)
{search_results}

## Instructions

Think step by step:

1. READ the claim carefully.
2. SEARCH through the additional results for mentions of the same or related facts.
3. DETERMINE if the additional results:
   - SUPPORT the claim (multiple independent sources confirm it, or official records verify it)
   - CONTRADICT the claim (credible sources dispute it, or there are significant inconsistencies)
   - NEITHER (no relevant information found in additional results)
4. ASSESS the revised confidence level (0.0 to 1.0).

## Output Format

Respond ONLY with valid JSON. No text before or after. No markdown code fences.

{{
  "supported": true,
  "contradicted": false,
  "supporting_sources": ["urls that support the claim"],
  "contradicting_sources": ["urls that contradict"],
  "notes": "Explanation of your assessment",
  "revised_confidence": 0.85
}}
"""


class SourceValidatorAgent:
    """
    Validates findings through cross-referencing and source analysis.
    Uses Gemini 2.5 for complex reasoning about claim validity.
    """

    VALIDATION_SCHEMA = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "contradicted": {"type": "boolean"},
            "supporting_sources": {"type": "array", "items": {"type": "string"}},
            "contradicting_sources": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
            "revised_confidence": {"type": "number"},
        },
        "required": ["supported", "contradicted", "revised_confidence"],
    }

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.confidence_scorer = ConfidenceScorer()

    def _format_search_results(self, state: AgentState) -> str:
        """Format search results for validation."""
        results = []
        for result in state.search_results[-10:]:
            results.append(
                f"[{result.url}]\n"
                f"Title: {result.title}\n"
                f"Content: {result.snippet}"
            )
        return "\n---\n".join(results) if results else "No additional results."

    async def validate_finding(
        self,
        finding: Finding,
        state: AgentState,
    ) -> Finding:
        """Validate a specific finding against available sources."""
        prompt = VALIDATION_PROMPT.format(
            target_name=state.target_name,
            claim=finding.fact,
            category=finding.category,
            sources=", ".join(finding.source_urls[:3]),
            confidence=f"{finding.confidence:.0%}",
            search_results=self._format_search_results(state),
        )

        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.VALIDATION_SCHEMA,
            task_type=TaskType.COMPLEX_REASONING,
        )

        if response.success:
            data = extract_json(response.content)
            if isinstance(data, dict):
                if data.get("supported"):
                    new_sources = data.get("supporting_sources", [])
                    finding.source_urls.extend(new_sources)
                    finding.verified = True
                    finding.confidence = min(
                        1.0,
                        max(finding.confidence, data.get("revised_confidence", finding.confidence))
                    )
                elif data.get("contradicted"):
                    finding.confidence = max(0.1, finding.confidence * 0.5)
                    finding.verified = False

        return finding

    async def validate_all(
        self,
        state: AgentState,
        min_confidence: float = 0.5,
    ) -> list[Finding]:
        """Validate all findings that need verification."""
        validated = []

        for finding in state.findings:
            if finding.confidence < min_confidence or not finding.verified:
                validated_finding = await self.validate_finding(finding, state)
                validated.append(validated_finding)
            else:
                validated.append(finding)

        return validated

    def generate_validation_queries(
        self,
        state: AgentState,
        max_queries: int = 3,
    ) -> list[str]:
        """Generate queries to help validate low-confidence findings."""
        queries = []

        to_verify = [
            f for f in state.findings
            if f.confidence < 0.6 and not f.verified
        ]

        for finding in to_verify[:max_queries]:
            fact_words = finding.fact.split()[:5]
            fact_snippet = " ".join(fact_words)

            if finding.category == "biography":
                queries.append(f"{state.target_name} biography {fact_snippet}")
            elif finding.category == "professional":
                queries.append(f"{state.target_name} career {fact_snippet}")
            elif finding.category == "controversies":
                queries.append(f"{state.target_name} {fact_snippet} news")
            else:
                queries.append(f"{state.target_name} verify {fact_snippet}")

        return queries

    def get_validation_summary(self, findings: list[Finding]) -> dict[str, Any]:
        """Generate validation statistics."""
        total = len(findings)
        verified = len([f for f in findings if f.verified])
        high_confidence = len([f for f in findings if f.confidence >= 0.7])
        low_confidence = len([f for f in findings if f.confidence < 0.4])

        avg_confidence = (
            sum(f.confidence for f in findings) / total
            if total > 0 else 0
        )

        return {
            "total_findings": total,
            "verified": verified,
            "verification_rate": f"{verified/total:.0%}" if total > 0 else "0%",
            "high_confidence_count": high_confidence,
            "low_confidence_count": low_confidence,
            "average_confidence": f"{avg_confidence:.0%}",
        }

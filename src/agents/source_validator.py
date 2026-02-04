"""
Source Validator Agent

Cross-references findings and validates source reliability.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, Finding
from ..utils.confidence import ConfidenceScorer


VALIDATION_PROMPT = """You are a fact-checker validating information about {target_name}.

FINDING TO VALIDATE:
Claim: {claim}
Category: {category}
Current sources: {sources}
Current confidence: {confidence}

ADDITIONAL SEARCH RESULTS (for cross-reference):
{search_results}

Analyze whether the additional search results support or contradict this claim.

Respond with JSON:
{{
  "supported": true/false,
  "contradicted": false/true,
  "supporting_sources": ["urls that support the claim"],
  "contradicting_sources": ["urls that contradict"],
  "notes": "Explanation of your assessment",
  "revised_confidence": 0.0-1.0
}}

A claim is SUPPORTED if:
- Multiple independent sources confirm it
- Official records or major news outlets verify it

A claim is CONTRADICTED if:
- Credible sources dispute it
- There are significant inconsistencies
"""


class SourceValidatorAgent:
    """
    Validates findings through cross-referencing and source analysis.
    
    Responsibilities:
    - Verify claims across multiple sources
    - Identify contradictions
    - Update confidence scores based on validation
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
        """
        Validate a specific finding against available sources.
        
        Args:
            finding: Finding to validate
            state: Current state with search results
            
        Returns:
            Updated Finding with revised confidence
        """
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
            try:
                data = json.loads(response.content)
                
                # Update finding based on validation
                if data.get("supported"):
                    # Add supporting sources and increase confidence
                    new_sources = data.get("supporting_sources", [])
                    finding.source_urls.extend(new_sources)
                    finding.verified = True
                    
                    # Recalculate confidence with new sources
                    finding.confidence = min(
                        1.0,
                        max(finding.confidence, data.get("revised_confidence", finding.confidence))
                    )
                    
                elif data.get("contradicted"):
                    # Reduce confidence for contradicted findings
                    finding.confidence = max(0.1, finding.confidence * 0.5)
                    finding.verified = False
                    
            except json.JSONDecodeError:
                pass
        
        return finding
    
    async def validate_all(
        self,
        state: AgentState,
        min_confidence: float = 0.5,
    ) -> list[Finding]:
        """
        Validate all findings that need verification.
        
        Args:
            state: Current agent state
            min_confidence: Verify findings below this confidence
            
        Returns:
            List of validated findings
        """
        validated = []
        
        for finding in state.findings:
            if finding.confidence < min_confidence or not finding.verified:
                # Needs validation
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
        """
        Generate queries to help validate low-confidence findings.
        
        Args:
            state: Current agent state
            max_queries: Maximum queries to generate
            
        Returns:
            List of validation search queries
        """
        queries = []
        
        # Find low-confidence, unverified findings
        to_verify = [
            f for f in state.findings
            if f.confidence < 0.6 and not f.verified
        ]
        
        for finding in to_verify[:max_queries]:
            # Generate a focused verification query
            if finding.category == "biography":
                queries.append(
                    f"{state.target_name} biography {finding.fact.split()[:3]}"
                )
            elif finding.category == "professional":
                queries.append(
                    f"{state.target_name} career {finding.fact.split()[:4]}"
                )
            elif finding.category == "controversies":
                queries.append(
                    f"{state.target_name} {finding.fact.split()[:3]} news"
                )
            else:
                queries.append(
                    f"{state.target_name} verify {finding.fact.split()[:3]}"
                )
        
        return queries
    
    def get_validation_summary(self, findings: list[Finding]) -> dict[str, Any]:
        """
        Generate validation statistics.
        
        Returns:
            Dictionary with validation metrics
        """
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

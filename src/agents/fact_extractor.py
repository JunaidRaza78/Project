"""
Fact Extractor Agent

Extracts and structures factual information from search results.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, Finding
from ..utils.confidence import ConfidenceScorer


# Extraction prompt template
FACT_EXTRACTION_PROMPT = """You are an expert research analyst extracting factual information about a person or entity.

TARGET: {target_name}
{context}

SEARCH RESULTS:
{search_results}

Extract all factual information about {target_name} from these search results. Focus on:

1. BIOGRAPHY: Birth date, birthplace, nationality, education, family
2. PROFESSIONAL: Companies, job titles, career timeline, achievements
3. FINANCIAL: Investments, net worth, business ownership, funding
4. ASSOCIATIONS: Key people they work with, board memberships, partnerships
5. CONTROVERSIES: Legal issues, scandals, public disputes (if any)

For each fact, note the source URL where you found it.

Respond with a JSON array of findings. Each finding must have:
- category: one of [biography, professional, financial, associations, controversies]
- fact: the specific factual claim (be precise and concise)
- source_urls: array of URLs supporting this fact
- confidence_note: brief note about source reliability

Example format:
[
  {{
    "category": "professional",
    "fact": "Served as CEO of Example Corp from 2015 to 2020",
    "source_urls": ["https://example.com/article"],
    "confidence_note": "Confirmed by company press release"
  }}
]

Extract ONLY verifiable facts, not opinions or speculation.
"""


class FactExtractorAgent:
    """
    Extracts structured facts from search results.
    
    Uses LLM to parse unstructured search results into
    categorized, source-attributed findings.
    """
    
    # JSON schema for structured output
    FINDING_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "fact": {"type": "string"},
                "source_urls": {"type": "array", "items": {"type": "string"}},
                "confidence_note": {"type": "string"},
            },
            "required": ["category", "fact", "source_urls"],
        },
    }
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.confidence_scorer = ConfidenceScorer()
    
    def _format_search_results(self, state: AgentState) -> str:
        """Format search results for the prompt."""
        results_text = []
        
        for result in state.search_results[-20:]:  # Last 20 results
            results_text.append(
                f"[Source: {result.url}]\n"
                f"Title: {result.title}\n"
                f"Snippet: {result.snippet}\n"
            )
        
        return "\n---\n".join(results_text)
    
    async def extract(self, state: AgentState) -> list[Finding]:
        """
        Extract facts from current search results.
        
        Args:
            state: Current agent state with search results
            
        Returns:
            List of extracted Finding objects
        """
        if not state.search_results:
            return []
        
        # Build context from existing findings
        context = ""
        if state.findings:
            existing = [f.fact for f in state.findings[:10]]
            context = f"EXISTING FINDINGS (avoid duplicates):\n" + "\n".join(f"- {f}" for f in existing)
        
        # Format prompt
        prompt = FACT_EXTRACTION_PROMPT.format(
            target_name=state.target_name,
            context=context,
            search_results=self._format_search_results(state),
        )
        
        # Get structured response
        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.FINDING_SCHEMA,
            task_type=TaskType.FAST_EXTRACTION,
        )
        
        findings = []
        
        if response.success:
            try:
                data = json.loads(response.content)
                
                for item in data:
                    # Calculate confidence based on sources
                    source_urls = item.get("source_urls", [])
                    confidence = self.confidence_scorer.calculate_confidence(source_urls)
                    
                    finding = Finding(
                        category=item.get("category", "unknown"),
                        fact=item.get("fact", ""),
                        source_urls=source_urls,
                        confidence=confidence,
                    )
                    
                    if finding.fact:  # Only add non-empty findings
                        findings.append(finding)
                        
            except json.JSONDecodeError:
                pass  # Return empty list on parse error
        
        return findings
    
    async def extract_from_content(
        self,
        content: str,
        target_name: str,
        source_url: str,
    ) -> list[Finding]:
        """
        Extract facts from scraped web content.
        
        Args:
            content: Scraped text content
            target_name: Name of investigation target
            source_url: URL of the source
            
        Returns:
            List of extracted findings
        """
        prompt = f"""Extract factual information about {target_name} from this content.

SOURCE URL: {source_url}

CONTENT:
{content[:4000]}

Extract specific facts with their categories. Respond as a JSON array:
[
  {{"category": "...", "fact": "...", "source_urls": ["{source_url}"], "confidence_note": "..."}}
]
"""
        
        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.FINDING_SCHEMA,
            task_type=TaskType.FAST_EXTRACTION,
        )
        
        findings = []
        
        if response.success:
            try:
                data = json.loads(response.content)
                for item in data:
                    source_urls = item.get("source_urls", [source_url])
                    confidence = self.confidence_scorer.calculate_confidence(source_urls)
                    
                    finding = Finding(
                        category=item.get("category", "unknown"),
                        fact=item.get("fact", ""),
                        source_urls=source_urls,
                        confidence=confidence,
                    )
                    
                    if finding.fact:
                        findings.append(finding)
                        
            except json.JSONDecodeError:
                pass
        
        return findings

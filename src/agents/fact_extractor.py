"""
Fact Extractor Agent

Extracts and structures factual information from search results.
Uses chain-of-thought prompting for thorough extraction.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, Finding
from ..utils.confidence import ConfidenceScorer
from ..utils.json_repair import extract_json


FACT_EXTRACTION_PROMPT = """You are an expert investigative research analyst. Your task is to extract verifiable factual information about a specific target from search results.

## Target
Name: {target_name}
{context}

## Search Results
{search_results}

## Instructions

Think step by step:

1. READ each search result carefully, noting which source URL each fact comes from.
2. IDENTIFY factual claims (not opinions) about the target.
3. CATEGORIZE each fact into exactly one category:
   - **biography**: Birth date, birthplace, nationality, education, family, personal details
   - **professional**: Job titles, companies, career timeline, achievements, roles held
   - **financial**: Investments, net worth estimates, business ownership, funding raised, deals
   - **associations**: Key people they work with, board memberships, partnerships, advisors
   - **controversies**: Legal issues, scandals, public disputes, regulatory actions, allegations
4. ATTRIBUTE each fact to the specific source URL(s) where you found it.
5. ASSESS source reliability briefly.

## Already Known Facts (avoid duplicates)
{existing_facts_summary}

## Output Format

Respond ONLY with a valid JSON array. No text before or after the JSON. No markdown code fences.

[
  {{
    "category": "professional",
    "fact": "Served as CEO of Example Corp from 2015 to 2020",
    "source_urls": ["https://example.com/article"],
    "confidence_note": "Confirmed by official company press release"
  }}
]

## Rules
- Extract ONLY verifiable facts, not opinions or speculation
- Be precise: include dates, numbers, and names when available
- Each fact should be a single, specific claim (not a paragraph)
- Include ALL facts you can find, even minor ones
"""


class FactExtractorAgent:
    """
    Extracts structured facts from search results using chain-of-thought prompting.
    """

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
        for result in state.search_results[-20:]:
            results_text.append(
                f"[Source: {result.url}]\n"
                f"Title: {result.title}\n"
                f"Snippet: {result.snippet}\n"
            )
        return "\n---\n".join(results_text)

    def _format_existing_facts(self, state: AgentState) -> str:
        """Format existing findings for deduplication."""
        if not state.findings:
            return "None yet."
        return "\n".join(f"- {f.fact}" for f in state.findings[:15])

    async def extract(self, state: AgentState) -> list[Finding]:
        """Extract facts from current search results."""
        if not state.search_results:
            return []

        context = ""
        if state.target_context:
            context = f"Context: {state.target_context}"

        prompt = FACT_EXTRACTION_PROMPT.format(
            target_name=state.target_name,
            context=context,
            search_results=self._format_search_results(state),
            existing_facts_summary=self._format_existing_facts(state),
        )

        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.FINDING_SCHEMA,
            task_type=TaskType.FAST_EXTRACTION,
        )

        findings = []

        if response.success:
            data = extract_json(response.content)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    source_urls = item.get("source_urls", [])
                    confidence = self.confidence_scorer.calculate_confidence(source_urls)

                    finding = Finding(
                        category=item.get("category", "unknown"),
                        fact=item.get("fact", ""),
                        source_urls=source_urls,
                        confidence=confidence,
                    )

                    if finding.fact:
                        findings.append(finding)

        return findings

    async def extract_from_content(
        self,
        content: str,
        target_name: str,
        source_url: str,
    ) -> list[Finding]:
        """Extract facts from scraped web content."""
        prompt = f"""Extract factual information about {target_name} from this content.

SOURCE URL: {source_url}

CONTENT:
{content[:4000]}

Extract specific facts with their categories. Respond ONLY with a valid JSON array:
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
            data = extract_json(response.content)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
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

        return findings

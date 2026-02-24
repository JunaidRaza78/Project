"""
Connection Mapper Agent

Traces relationships between entities, organizations, and events.
Distinguishes direct vs. indirect connections with temporal awareness.
"""

import json
from typing import Any

from ..models.model_manager import ModelManager, TaskType
from ..state import AgentState, Connection
from ..utils.confidence import ConfidenceScorer
from ..utils.json_repair import extract_json


CONNECTION_MAPPING_PROMPT = """You are an expert at mapping relationships and connections between entities for due diligence investigations.

## Target
Name: {target_name}

## Known Facts About Target
{findings}

## Search Results for Context
{search_results}

## Instructions

Think step by step:

1. IDENTIFY every person, organization, or event mentioned in connection with {target_name}.
2. CLASSIFY the relationship type:
   - For persons: employer, employee, co-founder, investor, advisor, family, partner, associate, co-defendant
   - For organizations: founded, employed_by, board_member, investor, advisor, customer, partner, registered_with
   - For events: participant, organizer, defendant, charged_in, investigated_in
3. DETERMINE the entity type:
   - **person**: Individuals (colleagues, family, associates, co-defendants)
   - **organization**: Companies, non-profits, government agencies, regulatory bodies
   - **event**: Significant events (lawsuits, investigations, acquisitions, IPOs)
4. NOTE the timeframe if mentioned (e.g., "2015-2020", "since 2019").
5. DISTINGUISH direct connections (firsthand relationship) from indirect ones.

## Output Format

Respond ONLY with a valid JSON array. No text before or after. No markdown code fences.

[
  {{
    "entity_name": "Name of connected entity",
    "entity_type": "person|organization|event",
    "relationship": "specific relationship type",
    "timeframe": "2015-2020 or null if unknown",
    "source_urls": ["url1"],
    "notes": "Additional context about this connection"
  }}
]

## Rules
- Include BOTH obvious and non-obvious connections
- Look for shared board memberships, investment relationships, joint ventures
- Include regulatory bodies if there are enforcement actions
- Include family members if mentioned
- Be specific about relationship types (not just "associated")
"""


class ConnectionMapperAgent:
    """
    Maps relationships between entities, organizations, and events.
    """

    CONNECTION_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
                "entity_type": {"type": "string"},
                "relationship": {"type": "string"},
                "timeframe": {"type": ["string", "null"]},
                "source_urls": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": ["entity_name", "entity_type", "relationship"],
        },
    }

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.confidence_scorer = ConfidenceScorer()

    def _format_findings(self, state: AgentState) -> str:
        """Format findings relevant to connections."""
        findings_text = []
        for finding in state.findings:
            if finding.category in ["associations", "professional", "financial", "controversies"]:
                findings_text.append(
                    f"[{finding.category.upper()}] {finding.fact}"
                )
        return "\n".join(findings_text) if findings_text else "No relevant findings."

    def _format_search_results(self, state: AgentState) -> str:
        """Format search results for connection mapping."""
        results = []
        for result in state.search_results[-15:]:
            results.append(f"[{result.url}]\n{result.title}: {result.snippet}")
        return "\n---\n".join(results) if results else "No results."

    async def map_connections(self, state: AgentState) -> list[Connection]:
        """Extract connections from current state."""
        if not state.findings and not state.search_results:
            return []

        prompt = CONNECTION_MAPPING_PROMPT.format(
            target_name=state.target_name,
            findings=self._format_findings(state),
            search_results=self._format_search_results(state),
        )

        response = await self.model_manager.generate_structured(
            prompt=prompt,
            schema=self.CONNECTION_SCHEMA,
            task_type=TaskType.FAST_EXTRACTION,
        )

        connections = []

        if response.success:
            data = extract_json(response.content)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    source_urls = item.get("source_urls", [])
                    confidence = self.confidence_scorer.calculate_confidence(source_urls)

                    connection = Connection(
                        entity_name=item.get("entity_name", ""),
                        entity_type=item.get("entity_type", "unknown"),
                        relationship=item.get("relationship", "associated"),
                        timeframe=item.get("timeframe"),
                        source_urls=source_urls,
                        confidence=confidence,
                    )

                    if connection.entity_name:
                        connections.append(connection)

        return connections

    def generate_connection_queries(
        self,
        state: AgentState,
        max_queries: int = 5,
    ) -> list[str]:
        """Generate follow-up queries based on discovered connections."""
        queries = []

        for connection in state.connections:
            if connection.confidence >= 0.5 and len(queries) < max_queries:
                if connection.entity_type == "organization":
                    queries.append(
                        f"{state.target_name} {connection.entity_name} role responsibilities"
                    )
                elif connection.entity_type == "person":
                    queries.append(
                        f"{state.target_name} {connection.entity_name} relationship business"
                    )

        if len(state.connections) >= 2:
            orgs = [c for c in state.connections if c.entity_type == "organization"][:2]
            if len(orgs) >= 2:
                queries.append(
                    f"{orgs[0].entity_name} {orgs[1].entity_name} connection"
                )

        return queries[:max_queries]

    def get_connection_summary(self, connections: list[Connection]) -> dict[str, Any]:
        """Generate a summary of all connections."""
        if not connections:
            return {
                "total_connections": 0,
                "by_type": {},
                "key_organizations": [],
                "key_people": [],
            }

        by_type = {}
        organizations = []
        people = []

        for conn in connections:
            if conn.entity_type not in by_type:
                by_type[conn.entity_type] = 0
            by_type[conn.entity_type] += 1

            if conn.entity_type == "organization":
                organizations.append({
                    "name": conn.entity_name,
                    "relationship": conn.relationship,
                    "confidence": conn.confidence,
                })
            elif conn.entity_type == "person":
                people.append({
                    "name": conn.entity_name,
                    "relationship": conn.relationship,
                    "confidence": conn.confidence,
                })

        organizations.sort(key=lambda x: x["confidence"], reverse=True)
        people.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "total_connections": len(connections),
            "by_type": by_type,
            "key_organizations": organizations[:10],
            "key_people": people[:10],
        }

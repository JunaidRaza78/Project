"""
Identity Graph Builder

Builds a Neo4j graph from investigation findings, connections, and risk indicators.

Node types: Person, Organization, Event, Finding, Risk
Edge types: CONNECTED_TO, HAS_FINDING, HAS_RISK
"""

import logging
from typing import Any, Optional

from neo4j import AsyncGraphDatabase

from ..state import AgentState, Finding, Connection, RiskIndicator

logger = logging.getLogger(__name__)


class IdentityGraphBuilder:
    """
    Builds an identity graph in Neo4j from investigation results.

    Creates a knowledge graph with:
    - Target person as central node
    - Connected entities (people, organizations, events) as surrounding nodes
    - Findings and risks as detail nodes
    - Relationships with properties (type, timeframe, confidence)
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        """Close the Neo4j driver."""
        await self.driver.close()

    async def clear_graph(self, target_name: str):
        """Clear existing graph data for a target investigation."""
        async with self.driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.investigation_target = $target DETACH DELETE n",
                target=target_name,
            )

    async def build_from_state(self, state: AgentState) -> dict[str, int]:
        """
        Build the full identity graph from an AgentState.

        Returns:
            Dict with counts of nodes and edges created.
        """
        stats = {"nodes": 0, "edges": 0}

        async with self.driver.session() as session:
            # 1. Create target person node
            await session.run(
                """
                MERGE (p:Person {name: $name})
                SET p.investigation_target = $name,
                    p.context = $context,
                    p.node_type = 'target'
                """,
                name=state.target_name,
                context=state.target_context,
            )
            stats["nodes"] += 1

            # 2. Create connection nodes and relationship edges
            for conn in state.connections:
                label = _entity_type_to_label(conn.entity_type)
                await session.run(
                    f"""
                    MERGE (e:{label} {{name: $entity_name}})
                    SET e.investigation_target = $target,
                        e.entity_type = $entity_type
                    WITH e
                    MATCH (p:Person {{name: $target}})
                    MERGE (p)-[r:CONNECTED_TO]->(e)
                    SET r.relationship = $relationship,
                        r.timeframe = $timeframe,
                        r.confidence = $confidence
                    """,
                    entity_name=conn.entity_name,
                    target=state.target_name,
                    entity_type=conn.entity_type,
                    relationship=conn.relationship,
                    timeframe=conn.timeframe or "",
                    confidence=conn.confidence,
                )
                stats["nodes"] += 1
                stats["edges"] += 1

            # 3. Create finding nodes linked to target
            for i, finding in enumerate(state.findings):
                await session.run(
                    """
                    CREATE (f:Finding {
                        id: $id,
                        category: $category,
                        fact: $fact,
                        confidence: $confidence,
                        verified: $verified,
                        investigation_target: $target
                    })
                    WITH f
                    MATCH (p:Person {name: $target})
                    MERGE (p)-[:HAS_FINDING]->(f)
                    """,
                    id=f"finding_{i}",
                    category=finding.category,
                    fact=finding.fact,
                    confidence=finding.confidence,
                    verified=finding.verified,
                    target=state.target_name,
                )
                stats["nodes"] += 1
                stats["edges"] += 1

            # 4. Create risk nodes linked to target
            for i, risk in enumerate(state.risk_indicators):
                await session.run(
                    """
                    CREATE (r:Risk {
                        id: $id,
                        category: $category,
                        description: $description,
                        severity: $severity,
                        confidence: $confidence,
                        investigation_target: $target
                    })
                    WITH r
                    MATCH (p:Person {name: $target})
                    MERGE (p)-[:HAS_RISK]->(r)
                    """,
                    id=f"risk_{i}",
                    category=risk.category,
                    description=risk.description,
                    severity=risk.severity,
                    confidence=risk.confidence,
                    target=state.target_name,
                )
                stats["nodes"] += 1
                stats["edges"] += 1

        return stats

    async def get_graph_summary(self, target_name: str) -> dict[str, Any]:
        """Query the graph for a summary of an investigation."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (p:Person {name: $target})
                OPTIONAL MATCH (p)-[:CONNECTED_TO]->(c)
                OPTIONAL MATCH (p)-[:HAS_FINDING]->(f)
                OPTIONAL MATCH (p)-[:HAS_RISK]->(r)
                RETURN count(DISTINCT c) as connections,
                       count(DISTINCT f) as findings,
                       count(DISTINCT r) as risks
                """,
                target=target_name,
            )
            record = await result.single()
            return dict(record) if record else {}

    async def export_graph_data(self, target_name: str) -> dict[str, Any]:
        """Export the graph as a JSON-serializable dictionary."""
        nodes = []
        edges = []

        async with self.driver.session() as session:
            # Export nodes
            result = await session.run(
                "MATCH (n) WHERE n.investigation_target = $target RETURN n, labels(n) as labels",
                target=target_name,
            )
            async for record in result:
                node = record["n"]
                nodes.append({
                    "id": node.element_id,
                    "labels": record["labels"],
                    "properties": dict(node),
                })

            # Export relationships
            result = await session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a.investigation_target = $target
                RETURN a.name as source, type(r) as rel_type,
                       b.name as target_node, properties(r) as props
                """,
                target=target_name,
            )
            async for record in result:
                edges.append({
                    "source": record["source"],
                    "relationship": record["rel_type"],
                    "target": record["target_node"],
                    "properties": dict(record["props"]) if record["props"] else {},
                })

        return {"nodes": nodes, "edges": edges}


def _entity_type_to_label(entity_type: str) -> str:
    """Convert entity type string to Neo4j label."""
    mapping = {
        "person": "Person",
        "organization": "Organization",
        "event": "Event",
    }
    return mapping.get(entity_type.lower(), "Entity")

"""
Research Orchestrator

LangGraph-based orchestration of the research agent workflow.
Coordinates all specialized agents in a stateful graph.
Includes Neo4j identity graph generation and LLM-driven query refinement.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from ..config import get_settings
from ..state import AgentState, InvestigationPhase, SearchResult
from ..models.model_manager import ModelManager, TaskType
from ..tools.search_tool import SerperSearchTool
from ..tools.scraper_tool import WebScraperTool
from ..utils.logger import AuditLogger
from ..utils.confidence import ConfidenceScorer
from ..utils.json_repair import extract_json

from .fact_extractor import FactExtractorAgent
from .risk_analyzer import RiskAnalyzerAgent
from .connection_mapper import ConnectionMapperAgent
from .source_validator import SourceValidatorAgent


# Query generation prompt for LLM-driven refinement
QUERY_GENERATION_PROMPT = """You are a research expert generating search queries for an investigation.

TARGET: {target_name}
CONTEXT: {context}

EXISTING QUERIES (avoid duplicates):
{existing_queries}

CURRENT FINDINGS SUMMARY:
{findings_summary}

Generate {num_queries} NEW search queries to discover more about this person/entity.
Focus on:
1. Filling gaps in current knowledge
2. Verifying uncertain information
3. Discovering hidden connections
4. Finding potential risk factors
5. Uncovering regulatory actions, lawsuits, or enforcement

Respond with a JSON array of query strings:
["query 1", "query 2", ...]

Make queries specific and targeted. Avoid generic queries.
"""


class ResearchOrchestrator:
    """
    Main orchestrator for the research agent.

    Coordinates search, fact extraction, risk analysis, connection mapping,
    source validation, identity graph generation, and report generation
    using LangGraph for stateful workflow management.
    """

    def __init__(
        self,
        groq_api_key: str,
        google_api_key: str,
        serper_api_key: str,
        output_dir: Path = Path("output"),
        neo4j_uri: str = "",
        neo4j_user: str = "",
        neo4j_password: str = "",
    ):
        # Initialize model manager
        self.model_manager = ModelManager(
            groq_api_key=groq_api_key,
            google_api_key=google_api_key,
        )

        # Initialize tools
        self.search_tool = SerperSearchTool(serper_api_key)
        self.scraper_tool = WebScraperTool()

        # Initialize agents
        self.fact_extractor = FactExtractorAgent(self.model_manager)
        self.risk_analyzer = RiskAnalyzerAgent(self.model_manager)
        self.connection_mapper = ConnectionMapperAgent(self.model_manager)
        self.source_validator = SourceValidatorAgent(self.model_manager)

        # Confidence scorer
        self.confidence_scorer = ConfidenceScorer()

        # Output directory
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger will be initialized per investigation
        self.logger: AuditLogger | None = None

        # Neo4j identity graph builder (optional)
        self.graph_builder = None
        if neo4j_uri and neo4j_password and neo4j_password != "your_neo4j_password_here":
            try:
                from ..graph.identity_graph import IdentityGraphBuilder
                self.graph_builder = IdentityGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
            except Exception:
                pass  # Neo4j not available, continue without it

        # Build the workflow graph
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("initial_search", self._initial_search_node)
        workflow.add_node("fact_extraction", self._fact_extraction_node)
        workflow.add_node("risk_analysis", self._risk_analysis_node)
        workflow.add_node("connection_mapping", self._connection_mapping_node)
        workflow.add_node("query_refinement", self._query_refinement_node)
        workflow.add_node("source_validation", self._source_validation_node)
        workflow.add_node("identity_graph", self._identity_graph_node)
        workflow.add_node("report_generation", self._report_generation_node)

        # Set entry point
        workflow.set_entry_point("initial_search")

        # Add edges
        workflow.add_edge("initial_search", "fact_extraction")
        workflow.add_edge("fact_extraction", "risk_analysis")
        workflow.add_edge("risk_analysis", "connection_mapping")
        workflow.add_conditional_edges(
            "connection_mapping",
            self._should_continue,
            {
                "continue": "query_refinement",
                "validate": "source_validation",
            }
        )
        workflow.add_edge("query_refinement", "initial_search")
        workflow.add_edge("source_validation", "identity_graph")
        workflow.add_edge("identity_graph", "report_generation")
        workflow.add_edge("report_generation", END)

        return workflow.compile()

    def _should_continue(self, state: AgentState) -> Literal["continue", "validate"]:
        """Determine if we should continue searching or move to validation."""
        if state.should_continue_searching():
            return "continue"
        return "validate"

    async def _initial_search_node(self, state: AgentState) -> dict[str, Any]:
        """Execute search queries with news-specific search on first iteration."""
        state.current_phase = InvestigationPhase.INITIAL_SEARCH

        if self.logger:
            self.logger.log_phase_change(
                state.current_phase.value if state.iteration_count > 0 else "start",
                state.current_phase.value
            )

        # Generate queries if none pending
        if not state.pending_queries:
            queries = await self._generate_initial_queries(state.target_name, state.target_context)
            state.pending_queries = queries

        # Execute searches
        new_results = []
        for query in state.pending_queries[:3]:  # Process 3 queries at a time
            if query not in state.search_history:
                results = await self.search_tool.search(query)
                state.search_history.append(query)
                new_results.extend(results)

                if self.logger:
                    self.logger.log_search(query, len(results), state.iteration_count)

        # News-specific search on first iteration
        if state.iteration_count == 0:
            news_results = await self.search_tool.search_news(
                f"{state.target_name} controversy legal issues regulatory", num_results=5
            )
            new_results.extend(news_results)
            if self.logger and news_results:
                self.logger.log_search(
                    f"[NEWS] {state.target_name} controversy legal",
                    len(news_results),
                    state.iteration_count,
                )

        # Clear processed queries
        state.pending_queries = state.pending_queries[3:]

        # Add new results
        state.search_results.extend(new_results)
        state.iteration_count += 1

        return {"search_results": state.search_results, "iteration_count": state.iteration_count}

    async def _fact_extraction_node(self, state: AgentState) -> dict[str, Any]:
        """Extract facts from search results."""
        state.current_phase = InvestigationPhase.FACT_EXTRACTION

        if self.logger:
            self.logger.log_phase_change("initial_search", "fact_extraction")

        new_findings = await self.fact_extractor.extract(state)

        for finding in new_findings:
            state.add_finding(finding)
            if self.logger:
                self.logger.log_finding(finding.category, finding.fact, finding.confidence)

        if self.logger:
            self.logger.log_model_call("groq", "fact_extraction", 0)

        return {"findings": state.findings}

    async def _risk_analysis_node(self, state: AgentState) -> dict[str, Any]:
        """Analyze for risk patterns using Gemini for complex reasoning."""
        state.current_phase = InvestigationPhase.RISK_ANALYSIS

        if self.logger:
            self.logger.log_phase_change("fact_extraction", "risk_analysis")

        new_risks = await self.risk_analyzer.analyze(state)

        for risk in new_risks:
            state.add_risk(risk)
            if self.logger:
                self.logger.log_risk(risk.category, risk.description, risk.severity)

        return {"risk_indicators": state.risk_indicators}

    async def _connection_mapping_node(self, state: AgentState) -> dict[str, Any]:
        """Map entity connections."""
        state.current_phase = InvestigationPhase.CONNECTION_MAPPING

        if self.logger:
            self.logger.log_phase_change("risk_analysis", "connection_mapping")

        new_connections = await self.connection_mapper.map_connections(state)

        for conn in new_connections:
            state.add_connection(conn)

        return {"connections": state.connections}

    async def _query_refinement_node(self, state: AgentState) -> dict[str, Any]:
        """Generate refined queries using static analysis and LLM-driven generation."""
        # Static queries from agents
        connection_queries = self.connection_mapper.generate_connection_queries(state, 2)
        validation_queries = self.source_validator.generate_validation_queries(state, 2)
        gap_queries = await self._generate_gap_queries(state)

        all_queries = connection_queries + validation_queries + gap_queries

        # LLM-driven query refinement
        try:
            refinement_prompt = QUERY_GENERATION_PROMPT.format(
                target_name=state.target_name,
                context=state.target_context,
                existing_queries="\n".join(state.search_history[-10:]),
                findings_summary="\n".join(f.fact[:80] for f in state.findings[:10]),
                num_queries=3,
            )
            response = await self.model_manager.generate(
                prompt=refinement_prompt,
                task_type=TaskType.FAST_EXTRACTION,
                temperature=0.8,
            )
            if response.success:
                llm_queries = extract_json(response.content)
                if isinstance(llm_queries, list):
                    all_queries.extend(str(q) for q in llm_queries if isinstance(q, str))
        except Exception:
            pass  # LLM refinement is best-effort

        # Filter out already-searched queries
        new_queries = [q for q in all_queries if q not in state.search_history]

        if self.logger and new_queries:
            self.logger.log_query_refinement(
                "previous findings",
                new_queries[:5],
                "Following up on discovered connections and gaps"
            )

        state.pending_queries = new_queries[:5]

        return {"pending_queries": state.pending_queries}

    async def _source_validation_node(self, state: AgentState) -> dict[str, Any]:
        """Validate findings with source cross-referencing using Gemini."""
        state.current_phase = InvestigationPhase.SOURCE_VALIDATION

        if self.logger:
            self.logger.log_phase_change("connection_mapping", "source_validation")

        validated = await self.source_validator.validate_all(state)
        state.findings = validated

        return {"findings": state.findings}

    async def _identity_graph_node(self, state: AgentState) -> dict[str, Any]:
        """Build identity graph in Neo4j from investigation results."""
        if self.graph_builder:
            try:
                await self.graph_builder.clear_graph(state.target_name)
                stats = await self.graph_builder.build_from_state(state)
                if self.logger:
                    self.logger.log_graph_build(stats["nodes"], stats["edges"], state.target_name)
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Graph build failed: {e}", "identity_graph")
        return {}

    async def _report_generation_node(self, state: AgentState) -> dict[str, Any]:
        """Generate the final report."""
        state.current_phase = InvestigationPhase.REPORT_GENERATION

        if self.logger:
            self.logger.log_phase_change("identity_graph", "report_generation")

        report = await self._generate_report(state)
        state.final_report = report
        state.current_phase = InvestigationPhase.COMPLETE

        return {"final_report": state.final_report, "current_phase": state.current_phase}

    async def _generate_initial_queries(self, target_name: str, context: str = "") -> list[str]:
        """Generate comprehensive initial search queries."""
        queries = [
            f"{target_name}",
            f"{target_name} biography education background",
            f"{target_name} career professional history CEO founder",
            f"{target_name} controversy scandal lawsuit legal issues",
            f"{target_name} news latest",
            f"{target_name} net worth investments business ownership",
        ]

        if context:
            queries.append(f"{target_name} {context}")
            # Search the organization separately for issues
            queries.append(f"{context} controversy investigation regulatory")

        return queries

    async def _generate_gap_queries(self, state: AgentState) -> list[str]:
        """Generate queries to fill knowledge gaps based on category coverage."""
        category_counts = {}
        for finding in state.findings:
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        queries = []

        # Category-specific gaps
        if category_counts.get("financial", 0) < 2:
            queries.append(f"{state.target_name} investments net worth business ownership funding")
        if category_counts.get("controversies", 0) < 1:
            queries.append(f"{state.target_name} controversy lawsuit scandal investigation SEC")
        if category_counts.get("associations", 0) < 2:
            queries.append(f"{state.target_name} board members advisors partners associates")
        if len(state.risk_indicators) < 2:
            queries.append(f"{state.target_name} fraud allegation regulatory action enforcement")

        # Connection-driven deep queries
        for conn in state.connections[:3]:
            if conn.entity_type == "organization" and conn.confidence >= 0.5:
                queries.append(f"{conn.entity_name} controversy investigation problems")
            elif conn.entity_type == "person" and conn.confidence >= 0.5:
                queries.append(f"{conn.entity_name} {state.target_name} relationship controversy")

        return queries[:5]

    async def _generate_report(self, state: AgentState) -> str:
        """Generate the final investigation report."""
        risk_summary = self.risk_analyzer.calculate_overall_risk_score(state.risk_indicators)
        connection_summary = self.connection_mapper.get_connection_summary(state.connections)
        validation_summary = self.source_validator.get_validation_summary(state.findings)

        # Determine models used
        models_used = "Groq (llama-3.3-70b, llama-3.1-8b)"
        if self.model_manager._gemini_available:
            models_used += ", Gemini 2.5 Flash"

        report = f"""# Investigation Report: {state.target_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Search Iterations:** {state.iteration_count}
**Total Sources Analyzed:** {len(state.search_results)}
**Models Used:** {models_used}

---

## Executive Summary

**Overall Risk Level:** {risk_summary['risk_level']}
**Risk Score:** {risk_summary['overall_score']}/10
**Critical Risks Identified:** {risk_summary.get('critical_risks', 0)}

---

## Biographical & Professional Profile

"""
        # Add findings by category
        categories = ["biography", "professional", "financial", "associations", "controversies"]
        for category in categories:
            cat_findings = [f for f in state.findings if f.category == category]
            if cat_findings:
                report += f"### {category.title()}\n\n"
                for finding in cat_findings:
                    conf_label = self.confidence_scorer.get_confidence_label(finding.confidence)
                    verified = " [Verified]" if finding.verified else ""
                    report += f"- {finding.fact} [{conf_label}]{verified}\n"
                report += "\n"

        # Risk section
        report += """---

## Risk Assessment

"""
        if state.risk_indicators:
            for risk in sorted(state.risk_indicators, key=lambda r: r.severity, reverse=True):
                severity_emoji = "🔴" if risk.severity >= 7 else "🟡" if risk.severity >= 4 else "🟢"
                report += f"""### {severity_emoji} {risk.category.title()} Risk (Severity: {risk.severity}/10)

**Description:** {risk.description}

**Evidence:**
"""
                for evidence in risk.evidence:
                    report += f"- {evidence}\n"
                report += f"\n**Confidence:** {risk.confidence:.0%}\n\n"
        else:
            report += "*No significant risks identified.*\n\n"

        # Connections section
        report += """---

## Network & Connections

"""
        report += f"**Total Connections Mapped:** {connection_summary['total_connections']}\n\n"

        if connection_summary.get('key_organizations'):
            report += "### Key Organizations\n\n"
            for org in connection_summary['key_organizations'][:5]:
                report += f"- **{org['name']}** - {org['relationship']} ({org['confidence']:.0%} confidence)\n"
            report += "\n"

        if connection_summary.get('key_people'):
            report += "### Key People\n\n"
            for person in connection_summary['key_people'][:5]:
                report += f"- **{person['name']}** - {person['relationship']} ({person['confidence']:.0%} confidence)\n"
            report += "\n"

        # Identity graph note
        if self.graph_builder:
            report += """### Identity Graph

An identity graph has been generated in Neo4j with all discovered entities and relationships.

"""

        # Validation summary
        report += f"""---

## Source Validation Summary

- **Total Findings:** {validation_summary['total_findings']}
- **Verified Findings:** {validation_summary['verified']} ({validation_summary['verification_rate']})
- **High Confidence:** {validation_summary['high_confidence_count']}
- **Low Confidence (Needs Review):** {validation_summary['low_confidence_count']}
- **Average Confidence:** {validation_summary['average_confidence']}

---

## Methodology

This report was generated using an autonomous research agent that:
1. Conducted {state.iteration_count} search iterations across multiple sources (web + news)
2. Extracted and categorized factual information using {models_used}
3. Analyzed patterns for potential risks using AI-powered reasoning
4. Mapped entity relationships and connections
5. Cross-referenced findings for validation
6. Generated identity graph in Neo4j (if configured)

*Note: All findings should be independently verified before making decisions.*
"""

        return report

    async def investigate(
        self,
        target_name: str,
        context: str = "",
        max_iterations: int = 10,
    ) -> AgentState:
        """
        Run a full investigation on a target.

        Args:
            target_name: Name of person/entity to investigate
            context: Additional context about the target
            max_iterations: Maximum search iterations

        Returns:
            Final AgentState with all findings
        """
        # Initialize logger
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = AuditLogger(logs_dir, target_name)

        # Create initial state
        initial_state = AgentState(
            target_name=target_name,
            target_context=context,
            max_iterations=max_iterations,
        )

        try:
            # Run the graph
            result = await self.workflow.ainvoke(initial_state)

            # LangGraph returns a dict-like object, convert to AgentState
            if isinstance(result, dict):
                if "findings" in result:
                    initial_state.findings = result["findings"]
                if "risk_indicators" in result:
                    initial_state.risk_indicators = result["risk_indicators"]
                if "connections" in result:
                    initial_state.connections = result["connections"]
                if "search_results" in result:
                    initial_state.search_results = result["search_results"]
                if "final_report" in result:
                    initial_state.final_report = result["final_report"]
                if "iteration_count" in result:
                    initial_state.iteration_count = result["iteration_count"]
                if "current_phase" in result:
                    initial_state.current_phase = result["current_phase"]

                final_state = initial_state
            else:
                final_state = result

            # Save report
            await self._save_report(final_state)

            # Print summary
            if self.logger:
                self.logger.print_summary()

            return final_state

        except Exception as e:
            if self.logger:
                self.logger.log_error(str(e), "investigation")
            raise

        finally:
            # Cleanup
            await self.search_tool.close()
            await self.scraper_tool.close()
            if self.graph_builder:
                try:
                    await self.graph_builder.close()
                except Exception:
                    pass

    async def _save_report(self, state: AgentState) -> Path:
        """Save the investigation report."""
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(state, dict):
            target_name = state.get("target_name", "unknown")
            final_report = state.get("final_report", "")
        else:
            target_name = state.target_name
            final_report = state.final_report

        safe_name = target_name.lower().replace(" ", "_")[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}_report.md"

        filepath = reports_dir / filename
        filepath.write_text(final_report)

        return filepath

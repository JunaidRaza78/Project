"""
Streamlit Frontend for Autonomous Research Agent

A beautiful dashboard for running investigations and viewing results.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Settings
from src.agents.orchestrator import ResearchOrchestrator
from src.state import AgentState


# Page configuration
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
    .risk-high { 
        background-color: #fee2e2; 
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1f2937 !important;
    }
    .risk-high strong, .risk-high small { color: #1f2937 !important; }
    .risk-medium { 
        background-color: #fef3c7; 
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1f2937 !important;
    }
    .risk-medium strong, .risk-medium small { color: #1f2937 !important; }
    .risk-low { 
        background-color: #d1fae5; 
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1f2937 !important;
    }
    .risk-low strong, .risk-low small { color: #1f2937 !important; }
    .finding-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1f2937 !important;
    }
    .finding-card strong { color: #374151 !important; }
    .connection-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0ea5e9;
        color: #1f2937 !important;
    }
    .connection-card strong { color: #1e40af !important; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "investigation_results" not in st.session_state:
        st.session_state.investigation_results = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "api_keys_configured" not in st.session_state:
        st.session_state.api_keys_configured = False


def check_api_keys() -> bool:
    """Check if API keys are configured."""
    try:
        settings = Settings()
        return all([
            settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here",
            settings.google_api_key and settings.google_api_key != "your_google_api_key_here",
            settings.serper_api_key and settings.serper_api_key != "your_serper_api_key_here",
        ])
    except:
        return False


async def run_investigation(target: str, context: str, max_iterations: int) -> AgentState:
    """Run the research agent investigation."""
    settings = Settings()
    
    orchestrator = ResearchOrchestrator(
        groq_api_key=settings.groq_api_key,
        google_api_key=settings.google_api_key,
        serper_api_key=settings.serper_api_key,
        output_dir=Path("output"),
    )
    
    return await orchestrator.investigate(
        target_name=target,
        context=context,
        max_iterations=max_iterations,
    )


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key Status
        api_configured = check_api_keys()
        if api_configured:
            st.success("✅ API Keys Configured")
        else:
            st.error("❌ API Keys Missing")
            st.markdown("""
            Create a `.env` file with:
            ```
            GROQ_API_KEY=your_key
            GOOGLE_API_KEY=your_key
            SERPER_API_KEY=your_key
            ```
            """)
        
        st.divider()
        
        # Investigation settings
        st.markdown("### 🎯 Investigation Settings")
        
        target_name = st.text_input(
            "Target Name",
            placeholder="e.g., Elizabeth Holmes",
            help="Name of the person or entity to investigate"
        )
        
        target_context = st.text_area(
            "Additional Context",
            placeholder="e.g., Theranos founder, blood testing startup",
            help="Optional context to guide the investigation",
            height=80,
        )
        
        max_iterations = st.slider(
            "Max Search Iterations",
            min_value=3,
            max_value=20,
            value=8,
            help="More iterations = deeper investigation but takes longer"
        )
        
        st.divider()
        
        # Quick select personas
        st.markdown("### 🧪 Test Personas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Holmes", use_container_width=True):
                return "Elizabeth Holmes", "Theranos founder", max_iterations
        with col2:
            if st.button("SBF", use_container_width=True):
                return "Sam Bankman-Fried", "FTX founder", max_iterations
        with col3:
            if st.button("Neumann", use_container_width=True):
                return "Adam Neumann", "WeWork founder", max_iterations
        
        return target_name, target_context, max_iterations


def render_risk_chart(state: AgentState):
    """Render risk assessment pie chart."""
    if not state.risk_indicators:
        st.info("No risks identified")
        return
    
    # Categorize risks by severity
    severity_counts = {"High (7-10)": 0, "Medium (4-6)": 0, "Low (1-3)": 0}
    for risk in state.risk_indicators:
        if risk.severity >= 7:
            severity_counts["High (7-10)"] += 1
        elif risk.severity >= 4:
            severity_counts["Medium (4-6)"] += 1
        else:
            severity_counts["Low (1-3)"] += 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(severity_counts.keys()),
        values=list(severity_counts.values()),
        hole=0.4,
        marker_colors=["#ef4444", "#f59e0b", "#10b981"],
    )])
    
    fig.update_layout(
        title="Risk Distribution",
        showlegend=True,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_findings_chart(state: AgentState):
    """Render findings by category bar chart."""
    if not state.findings:
        st.info("No findings yet")
        return
    
    category_counts = {}
    for finding in state.findings:
        cat = finding.category.title()
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    fig = go.Figure(data=[go.Bar(
        x=list(category_counts.keys()),
        y=list(category_counts.values()),
        marker_color="#667eea",
    )])
    
    fig.update_layout(
        title="Findings by Category",
        xaxis_title="Category",
        yaxis_title="Count",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_connections_network(state: AgentState):
    """Render connections as a network visualization."""
    if not state.connections:
        st.info("No connections mapped")
        return
    
    # Build network data
    nodes = [{"id": state.target_name, "group": "target"}]
    edges = []
    
    for conn in state.connections[:15]:  # Limit to 15 for visualization
        nodes.append({
            "id": conn.entity_name,
            "group": conn.entity_type,
        })
        edges.append({
            "source": state.target_name,
            "target": conn.entity_name,
            "relationship": conn.relationship,
        })
    
    # Create network chart using scatter
    import math
    
    # Position nodes in a circle
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        if node["id"] == state.target_name:
            positions[node["id"]] = (0, 0)
        else:
            angle = 2 * math.pi * (i - 1) / (n - 1)
            positions[node["id"]] = (math.cos(angle) * 2, math.sin(angle) * 2)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=1, color="#94a3b8"),
            hoverinfo="none",
            showlegend=False,
        ))
    
    # Add nodes
    colors = {"target": "#667eea", "person": "#10b981", "organization": "#f59e0b", "event": "#ef4444"}
    for node in nodes:
        x, y = positions[node["id"]]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=20, color=colors.get(node["group"], "#94a3b8")),
            text=[node["id"][:20]],
            textposition="top center",
            hoverinfo="text",
            hovertext=node["id"],
            showlegend=False,
        ))
    
    fig.update_layout(
        title="Connection Network",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_results(state: AgentState):
    """Render the investigation results."""
    # Summary metrics
    st.markdown("### 📊 Investigation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Findings", len(state.findings))
    with col2:
        critical = len([r for r in state.risk_indicators if r.severity >= 7])
        st.metric("Critical Risks", critical, delta=f"{len(state.risk_indicators)} total")
    with col3:
        st.metric("Connections", len(state.connections))
    with col4:
        st.metric("Iterations", state.iteration_count)
    
    st.divider()
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        render_risk_chart(state)
    
    with chart_col2:
        render_findings_chart(state)
    
    # Connections network
    st.markdown("### 🔗 Connection Network")
    render_connections_network(state)
    
    st.divider()
    
    # Detailed findings in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Findings", "⚠️ Risks", "🔗 Connections", "📄 Full Report"])
    
    with tab1:
        if state.findings:
            for finding in sorted(state.findings, key=lambda f: f.confidence, reverse=True):
                conf_pct = f"{finding.confidence:.0%}"
                verified = "✓" if finding.verified else ""
                st.markdown(f"""
                <div class="finding-card">
                    <strong>{finding.category.title()}</strong> · {conf_pct} confidence {verified}<br>
                    {finding.fact}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No findings extracted")
    
    with tab2:
        if state.risk_indicators:
            for risk in sorted(state.risk_indicators, key=lambda r: r.severity, reverse=True):
                risk_class = "risk-high" if risk.severity >= 7 else "risk-medium" if risk.severity >= 4 else "risk-low"
                st.markdown(f"""
                <div class="{risk_class}">
                    <strong>{risk.category.title()}</strong> · Severity: {risk.severity}/10<br>
                    {risk.description}<br>
                    <small>Evidence: {'; '.join(risk.evidence[:2])}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant risks identified")
    
    with tab3:
        if state.connections:
            for conn in state.connections:
                icon = "👤" if conn.entity_type == "person" else "🏢" if conn.entity_type == "organization" else "📅"
                st.markdown(f"""
                <div class="connection-card">
                    {icon} <strong>{conn.entity_name}</strong><br>
                    {conn.relationship} · {conn.confidence:.0%} confidence
                    {f" · {conn.timeframe}" if conn.timeframe else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No connections mapped")
    
    with tab4:
        if state.final_report:
            st.markdown(state.final_report)
            st.download_button(
                "📥 Download Report",
                state.final_report,
                file_name=f"{state.target_name.lower().replace(' ', '_')}_report.md",
                mime="text/markdown",
            )
        else:
            st.info("Report not yet generated")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Autonomous Research Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered investigation for due diligence and risk assessment</p>', unsafe_allow_html=True)
    
    # Sidebar
    target, context, iterations = render_sidebar()
    
    # Main content
    if st.session_state.investigation_results:
        # Show results
        render_results(st.session_state.investigation_results)
        
        if st.button("🔄 New Investigation", type="secondary"):
            st.session_state.investigation_results = None
            st.rerun()
    else:
        # Show investigation form
        st.markdown("### 🚀 Start Investigation")
        
        if not check_api_keys():
            st.warning("⚠️ Please configure API keys in `.env` file before running investigations.")
            st.markdown("""
            **Required API Keys:**
            1. **Groq** - Get from [console.groq.com](https://console.groq.com)
            2. **Google Gemini** - Get from [ai.google.dev](https://ai.google.dev)
            3. **Serper** - Get from [serper.dev](https://serper.dev)
            """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if target:
                st.info(f"**Target:** {target}" + (f" | **Context:** {context}" if context else ""))
        
        with col2:
            run_button = st.button(
                "🔍 Run Investigation",
                type="primary",
                disabled=not target or not check_api_keys() or st.session_state.is_running,
                use_container_width=True,
            )
        
        if run_button and target:
            st.session_state.is_running = True
            
            with st.status("Running investigation...", expanded=True) as status:
                st.write("🔍 Initializing search...")
                
                try:
                    # Run the async investigation
                    results = asyncio.run(run_investigation(target, context, iterations))
                    
                    st.session_state.investigation_results = results
                    st.session_state.is_running = False
                    
                    status.update(label="✅ Investigation complete!", state="complete")
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.is_running = False
                    status.update(label="❌ Investigation failed", state="error")
                    st.error(f"Error: {str(e)}")
        
        # Show sample output
        if not target:
            st.markdown("---")
            st.markdown("### 📖 How It Works")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **1. Search & Extract**
                
                The agent performs iterative web searches using Serper API, extracting facts from search results.
                """)
            
            with col2:
                st.markdown("""
                **2. Analyze & Connect**
                
                Multiple AI models analyze findings for risks and map connections between entities.
                """)
            
            with col3:
                st.markdown("""
                **3. Validate & Report**
                
                Sources are cross-referenced for confidence scoring, and a comprehensive report is generated.
                """)


if __name__ == "__main__":
    main()

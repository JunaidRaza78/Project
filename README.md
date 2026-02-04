# Autonomous Research Agent

An AI-powered investigation agent that conducts comprehensive research on individuals and entities, uncovering hidden connections, potential risks, and strategic insights.

## Features

- **Multi-Model AI Integration**: Uses Groq (llama-3.3-70b) for fast inference and Google Gemini for complex reasoning
- **LangGraph Orchestration**: Stateful workflow management with conditional execution paths
- **Consecutive Search Strategy**: Builds upon previous findings to generate targeted follow-up queries
- **Deep Fact Extraction**: Identifies biographical details, professional history, financial connections
- **Risk Pattern Recognition**: Flags potential red flags and concerning associations
- **Connection Mapping**: Traces relationships between entities and organizations
- **Source Validation**: Confidence scoring with cross-referencing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Orchestrator                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐│
│  │  Search  │→ │   Fact   │→ │   Risk   │→ │  Connection  ││
│  │   Tool   │  │Extractor │  │ Analyzer │  │    Mapper    ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘│
│        ↑                                          ↓         │
│        └────────── Query Refinement ←─────────────┤         │
│                                                   ↓         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Source Validator → Report Generator      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone and Install

```bash
cd /path/to/project
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# - GROQ_API_KEY (from console.groq.com)
# - GOOGLE_API_KEY (from ai.google.dev)
# - SERPER_API_KEY (from serper.dev)
```

### 3. Run an Investigation

```bash
python main.py --target "Elizabeth Holmes" --context "Theranos founder"
```

## Usage

### Command Line Options

```bash
python main.py --help

Options:
  --target, -t     Name of person/entity to investigate (required)
  --context, -c    Additional context about the target
  --iterations, -i Maximum search iterations (default: 10)
  --output, -o     Output directory (default: ./output)
  --debug          Enable debug output
```

### Examples

```bash
# Basic investigation
python main.py --target "Sam Bankman-Fried"

# With context and custom settings
python main.py --target "Adam Neumann" --context "WeWork founder" --iterations 15

# Custom output directory
python main.py --target "Elizabeth Holmes" --output ./reports
```

## Evaluation

The agent includes an evaluation framework with 3 test personas:

1. **Elizabeth Holmes** - Theranos fraud case
2. **Sam Bankman-Fried** - FTX collapse
3. **Adam Neumann** - WeWork governance issues

### Run Evaluation

```bash
# Evaluate a specific persona
python -m evaluation.evaluator --persona persona_1

# Evaluate all personas
python -m evaluation.evaluator --all
```

### Metrics

- **Finding Coverage**: % of expected facts discovered
- **Risk Coverage**: % of expected risks identified
- **Connection Coverage**: % of expected connections mapped
- **Overall Score**: Weighted average (50% findings, 30% risks, 20% connections)

## Output

### Reports

Generated reports are saved to `output/reports/` and include:

- Executive summary with overall risk level
- Categorized findings with confidence scores
- Detailed risk assessment with evidence
- Connection network summary
- Source validation metrics

### Logs

Execution logs are saved to `output/logs/` in JSONL format, tracking:

- Search queries and results
- Model invocations
- Extracted findings
- Phase transitions
- Errors

## Project Structure

```
research_agent/
├── src/
│   ├── agents/           # Specialized agents
│   │   ├── orchestrator.py
│   │   ├── fact_extractor.py
│   │   ├── risk_analyzer.py
│   │   ├── connection_mapper.py
│   │   └── source_validator.py
│   ├── models/           # AI model integrations
│   │   ├── groq_model.py
│   │   ├── gemini_model.py
│   │   └── model_manager.py
│   ├── tools/            # Search and scraping
│   │   ├── search_tool.py
│   │   └── scraper_tool.py
│   ├── utils/            # Utilities
│   │   ├── confidence.py
│   │   └── logger.py
│   ├── config.py         # Configuration
│   └── state.py          # State management
├── evaluation/           # Evaluation framework
│   ├── personas/         # Test persona definitions
│   └── evaluator.py
├── tests/                # Unit tests
├── output/               # Generated output
│   ├── reports/
│   └── logs/
├── main.py               # CLI entry point
├── requirements.txt
└── .env.example
```

## API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| Groq | Fast LLM inference (llama-3.3-70b) | [console.groq.com](https://console.groq.com) |
| Google Gemini | Complex reasoning | [ai.google.dev](https://ai.google.dev) |
| Serper | Google search API | [serper.dev](https://serper.dev) |

All services offer free tiers suitable for testing.

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT

"""
Evaluation Framework

Evaluates the research agent against ground truth personas.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.agents.orchestrator import ResearchOrchestrator
from src.state import AgentState


@dataclass
class EvaluationResult:
    """Results from evaluating against a persona."""
    persona_name: str
    finding_coverage: float
    risk_coverage: float
    connection_coverage: float
    overall_score: float
    passed: bool
    details: dict[str, Any]


def load_persona(persona_path: Path) -> dict[str, Any]:
    """Load a persona definition file."""
    with open(persona_path) as f:
        return json.load(f)


def calculate_finding_coverage(
    actual_findings: list,
    expected_findings: dict[str, list],
) -> tuple[float, dict]:
    """
    Calculate what percentage of expected findings were discovered.
    
    Returns:
        Tuple of (coverage_score, details)
    """
    found_count = 0
    required_count = 0
    total_expected = 0
    matched = []
    missing = []
    
    for category, facts in expected_findings.items():
        for expected in facts:
            total_expected += 1
            if expected.get("required"):
                required_count += 1
            
            # Check if any actual finding matches
            fact_text = expected["fact"].lower()
            is_found = False
            
            for actual in actual_findings:
                actual_text = actual.fact.lower() if hasattr(actual, 'fact') else str(actual).lower()
                
                # Fuzzy matching - check if key terms are present
                key_terms = [t for t in fact_text.split() if len(t) > 3]
                matches = sum(1 for term in key_terms if term in actual_text)
                
                if matches >= len(key_terms) * 0.5:  # 50% of key terms match
                    is_found = True
                    break
            
            if is_found:
                found_count += 1
                matched.append(expected["fact"])
            else:
                missing.append({"fact": expected["fact"], "required": expected.get("required", False)})
    
    coverage = found_count / total_expected if total_expected > 0 else 0
    
    return coverage, {
        "found": found_count,
        "expected": total_expected,
        "matched": matched,
        "missing": missing,
    }


def calculate_risk_coverage(
    actual_risks: list,
    expected_risks: list,
) -> tuple[float, dict]:
    """Calculate what percentage of expected risks were identified."""
    found_count = 0
    matched = []
    missing = []
    
    for expected in expected_risks:
        expected_desc = expected["description"].lower()
        expected_cat = expected["category"].lower()
        
        is_found = False
        for actual in actual_risks:
            actual_desc = actual.description.lower() if hasattr(actual, 'description') else str(actual).lower()
            actual_cat = actual.category.lower() if hasattr(actual, 'category') else ""
            
            # Check category match and description similarity
            if expected_cat in actual_cat:
                key_terms = [t for t in expected_desc.split() if len(t) > 3]
                matches = sum(1 for term in key_terms if term in actual_desc)
                
                if matches >= len(key_terms) * 0.4:
                    is_found = True
                    break
        
        if is_found:
            found_count += 1
            matched.append(expected["description"])
        else:
            missing.append(expected)
    
    coverage = found_count / len(expected_risks) if expected_risks else 0
    
    return coverage, {
        "found": found_count,
        "expected": len(expected_risks),
        "matched": matched,
        "missing": missing,
    }


def calculate_connection_coverage(
    actual_connections: list,
    expected_connections: list,
) -> tuple[float, dict]:
    """Calculate what percentage of expected connections were mapped."""
    found_count = 0
    matched = []
    missing = []
    
    for expected in expected_connections:
        entity_name = expected["entity"].lower()
        
        is_found = False
        for actual in actual_connections:
            actual_name = actual.entity_name.lower() if hasattr(actual, 'entity_name') else str(actual).lower()
            
            if entity_name in actual_name or actual_name in entity_name:
                is_found = True
                break
        
        if is_found:
            found_count += 1
            matched.append(expected["entity"])
        else:
            missing.append(expected)
    
    coverage = found_count / len(expected_connections) if expected_connections else 0
    
    return coverage, {
        "found": found_count,
        "expected": len(expected_connections),
        "matched": matched,
        "missing": missing,
    }


async def evaluate_persona(
    persona_path: Path,
    orchestrator: ResearchOrchestrator,
) -> EvaluationResult:
    """
    Run the agent on a persona and evaluate results.
    
    Args:
        persona_path: Path to persona JSON file
        orchestrator: Research orchestrator instance
        
    Returns:
        EvaluationResult with scores and details
    """
    # Load persona
    persona = load_persona(persona_path)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {persona['name']}")
    print(f"Description: {persona['description']}")
    print(f"{'='*60}\n")
    
    # Run investigation
    state = await orchestrator.investigate(
        target_name=persona["name"],
        context=persona.get("context", ""),
        max_iterations=8,  # Fewer iterations for evaluation
    )
    
    # Calculate coverage scores
    finding_cov, finding_details = calculate_finding_coverage(
        state.findings,
        persona["expected_findings"],
    )
    
    risk_cov, risk_details = calculate_risk_coverage(
        state.risk_indicators,
        persona["expected_risks"],
    )
    
    conn_cov, conn_details = calculate_connection_coverage(
        state.connections,
        persona["expected_connections"],
    )
    
    # Overall score (weighted average)
    overall = (finding_cov * 0.5) + (risk_cov * 0.3) + (conn_cov * 0.2)
    
    # Check if passed
    min_scores = persona.get("minimum_scores", {})
    passed = (
        finding_cov >= min_scores.get("finding_coverage", 0.8) and
        risk_cov >= min_scores.get("risk_coverage", 0.75) and
        conn_cov >= min_scores.get("connection_coverage", 0.6)
    )
    
    return EvaluationResult(
        persona_name=persona["name"],
        finding_coverage=finding_cov,
        risk_coverage=risk_cov,
        connection_coverage=conn_cov,
        overall_score=overall,
        passed=passed,
        details={
            "findings": finding_details,
            "risks": risk_details,
            "connections": conn_details,
        },
    )


def print_evaluation_report(results: list[EvaluationResult]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("                    EVALUATION REPORT")
    print("=" * 70 + "\n")
    
    for result in results:
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        
        print(f"📋 {result.persona_name}")
        print(f"   Status: {status}")
        print(f"   Overall Score: {result.overall_score:.1%}")
        print(f"   ├── Finding Coverage: {result.finding_coverage:.1%}")
        print(f"   ├── Risk Coverage: {result.risk_coverage:.1%}")
        print(f"   └── Connection Coverage: {result.connection_coverage:.1%}")
        
        # Show missing items
        missing_findings = result.details["findings"]["missing"]
        missing_required = [m for m in missing_findings if m.get("required")]
        if missing_required:
            print(f"\n   ⚠️  Missing Required Findings:")
            for m in missing_required[:3]:
                print(f"      - {m['fact'][:50]}...")
        
        print()
    
    # Summary
    passed = len([r for r in results if r.passed])
    total = len(results)
    avg_score = sum(r.overall_score for r in results) / total if total > 0 else 0
    
    print("-" * 70)
    print(f"Summary: {passed}/{total} personas passed")
    print(f"Average Overall Score: {avg_score:.1%}")
    print("-" * 70)


async def main():
    """Main evaluation entry point."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluate research agent against test personas")
    parser.add_argument(
        "--persona",
        type=str,
        help="Specific persona to evaluate (persona_1, persona_2, persona_3)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all personas"
    )
    
    args = parser.parse_args()
    
    if not args.persona and not args.all:
        parser.print_help()
        sys.exit(1)
    
    # Get settings
    settings = get_settings()
    
    # Create orchestrator
    orchestrator = ResearchOrchestrator(
        groq_api_key=settings.groq_api_key,
        google_api_key=settings.google_api_key,
        serper_api_key=settings.serper_api_key,
        output_dir=Path("output"),
    )
    
    personas_dir = Path(__file__).parent / "personas"
    results = []
    
    if args.all:
        persona_files = sorted(personas_dir.glob("persona_*.json"))
    else:
        persona_files = [personas_dir / f"{args.persona}.json"]
    
    for persona_file in persona_files:
        if not persona_file.exists():
            print(f"❌ Persona file not found: {persona_file}")
            continue
        
        result = await evaluate_persona(persona_file, orchestrator)
        results.append(result)
    
    # Print report
    print_evaluation_report(results)
    
    # Save results
    output_file = Path("output") / "evaluation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(
            [
                {
                    "persona": r.persona_name,
                    "finding_coverage": r.finding_coverage,
                    "risk_coverage": r.risk_coverage,
                    "connection_coverage": r.connection_coverage,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

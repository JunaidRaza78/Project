#!/usr/bin/env python3
"""
Autonomous Research Agent - Main Entry Point

CLI interface for running investigations.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config import get_settings
from src.agents.orchestrator import ResearchOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent - Investigate individuals and entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --target "Elizabeth Holmes"
  python main.py --target "Sam Bankman-Fried" --context "FTX founder"
  python main.py --target "Adam Neumann" --iterations 15 --output ./reports
        """
    )
    
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Name of person or entity to investigate"
    )
    
    parser.add_argument(
        "--context", "-c",
        default="",
        help="Additional context about the target (e.g., 'CEO of Company X')"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Maximum search iterations (default: 10)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for reports and logs (default: ./output)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Get settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease ensure you have set up your .env file with required API keys:")
        print("  - GROQ_API_KEY")
        print("  - GOOGLE_API_KEY")
        print("  - SERPER_API_KEY")
        print("\nCopy .env.example to .env and fill in your keys.")
        sys.exit(1)
    
    # Print banner
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           AUTONOMOUS RESEARCH AGENT v0.1.0                    ║
║     Multi-Model Investigation & Risk Assessment System        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🎯 Target: {args.target}")
    if args.context:
        print(f"📋 Context: {args.context}")
    print(f"🔄 Max Iterations: {args.iterations}")
    print(f"📁 Output Directory: {args.output}")
    print()
    
    # Create orchestrator
    orchestrator = ResearchOrchestrator(
        groq_api_key=settings.groq_api_key,
        google_api_key=settings.google_api_key,
        serper_api_key=settings.serper_api_key,
        output_dir=args.output,
    )
    
    try:
        # Run investigation
        print("🚀 Starting investigation...\n")
        
        final_state = await orchestrator.investigate(
            target_name=args.target,
            context=args.context,
            max_iterations=args.iterations,
        )
        
        print("\n" + "=" * 60)
        print("✅ INVESTIGATION COMPLETE")
        print("=" * 60)
        
        # Summary
        print(f"\n📊 Results Summary:")
        print(f"   • Findings extracted: {len(final_state.findings)}")
        print(f"   • Risk indicators: {len(final_state.risk_indicators)}")
        print(f"   • Connections mapped: {len(final_state.connections)}")
        print(f"   • Search iterations: {final_state.iteration_count}")
        
        # Critical risks
        critical = [r for r in final_state.risk_indicators if r.severity >= 7]
        if critical:
            print(f"\n⚠️  CRITICAL RISKS FOUND: {len(critical)}")
            for risk in critical[:3]:
                print(f"   🔴 [{risk.category}] {risk.description[:60]}...")
        
        # Report location
        report_path = args.output / "reports"
        print(f"\n📝 Full report saved to: {report_path}/")
        print(f"📋 Execution logs saved to: {args.output}/logs/")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Investigation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during investigation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

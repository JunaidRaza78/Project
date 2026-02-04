"""
Audit Logger

Comprehensive logging for tracking agent execution and decisions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table


def get_logger(name: str, debug: bool = False) -> logging.Logger:
    """Create a configured logger with Rich handler."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(level)
        
        handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setLevel(level)
        
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger


class AuditLogger:
    """
    Audit logging for research agent operations.
    
    Tracks:
    - Search queries and results
    - Model invocations and responses
    - Extracted findings
    - Risk assessments
    - Decision points
    """
    
    def __init__(
        self,
        log_dir: Path,
        target_name: str,
        console_output: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_name = target_name
        self.console = Console() if console_output else None
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = target_name.lower().replace(" ", "_")[:30]
        self.log_file = self.log_dir / f"{safe_name}_{timestamp}.jsonl"
        
        # Initialize entries list
        self.entries: list[dict[str, Any]] = []
        
        # Session start
        self._log_entry("session_start", {
            "target": target_name,
            "timestamp": datetime.now().isoformat(),
        })
    
    def _log_entry(self, event_type: str, data: dict[str, Any]) -> None:
        """Write a log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        
        self.entries.append(entry)
        
        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_search(
        self,
        query: str,
        num_results: int,
        iteration: int,
    ) -> None:
        """Log a search query."""
        self._log_entry("search", {
            "query": query,
            "num_results": num_results,
            "iteration": iteration,
        })
        
        if self.console:
            self.console.print(
                f"[blue]🔍 Search [{iteration}]:[/blue] {query} "
                f"[dim]({num_results} results)[/dim]"
            )
    
    def log_model_call(
        self,
        model_type: str,
        task: str,
        latency_ms: float,
        tokens: int = 0,
    ) -> None:
        """Log a model invocation."""
        self._log_entry("model_call", {
            "model_type": model_type,
            "task": task,
            "latency_ms": latency_ms,
            "tokens": tokens,
        })
        
        if self.console:
            self.console.print(
                f"[cyan]🤖 {model_type}:[/cyan] {task} "
                f"[dim]({latency_ms:.0f}ms)[/dim]"
            )
    
    def log_finding(
        self,
        category: str,
        fact: str,
        confidence: float,
    ) -> None:
        """Log an extracted finding."""
        self._log_entry("finding", {
            "category": category,
            "fact": fact,
            "confidence": confidence,
        })
        
        if self.console:
            conf_color = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"
            self.console.print(
                f"[green]📋 Finding [{category}]:[/green] {fact[:80]}... "
                f"[{conf_color}]({confidence:.0%})[/{conf_color}]"
            )
    
    def log_risk(
        self,
        category: str,
        description: str,
        severity: int,
    ) -> None:
        """Log a risk indicator."""
        self._log_entry("risk", {
            "category": category,
            "description": description,
            "severity": severity,
        })
        
        if self.console:
            sev_color = "red" if severity >= 7 else "yellow" if severity >= 4 else "white"
            self.console.print(
                f"[{sev_color}]⚠️  Risk [{category}]:[/{sev_color}] "
                f"{description[:60]}... [bold](Severity: {severity}/10)[/bold]"
            )
    
    def log_phase_change(self, old_phase: str, new_phase: str) -> None:
        """Log a workflow phase transition."""
        self._log_entry("phase_change", {
            "from": old_phase,
            "to": new_phase,
        })
        
        if self.console:
            self.console.print(
                Panel(
                    f"[bold]Phase: {new_phase.replace('_', ' ').title()}[/bold]",
                    style="magenta",
                )
            )
    
    def log_error(self, error: str, context: str = "") -> None:
        """Log an error."""
        self._log_entry("error", {
            "error": error,
            "context": context,
        })
        
        if self.console:
            self.console.print(f"[red]❌ Error:[/red] {error}")
    
    def log_query_refinement(
        self,
        original_query: str,
        refined_queries: list[str],
        reason: str,
    ) -> None:
        """Log query refinement based on findings."""
        self._log_entry("query_refinement", {
            "original": original_query,
            "refined": refined_queries,
            "reason": reason,
        })
        
        if self.console:
            self.console.print(
                f"[yellow]🔄 Query Refinement:[/yellow] {reason}"
            )
            for q in refined_queries[:3]:
                self.console.print(f"   → {q}")
    
    def get_summary(self) -> dict[str, Any]:
        """Get execution summary statistics."""
        searches = [e for e in self.entries if e["event_type"] == "search"]
        findings = [e for e in self.entries if e["event_type"] == "finding"]
        risks = [e for e in self.entries if e["event_type"] == "risk"]
        errors = [e for e in self.entries if e["event_type"] == "error"]
        
        return {
            "target": self.target_name,
            "total_searches": len(searches),
            "total_findings": len(findings),
            "total_risks": len(risks),
            "errors": len(errors),
            "log_file": str(self.log_file),
        }
    
    def print_summary(self) -> None:
        """Print a summary table to console."""
        if not self.console:
            return
        
        summary = self.get_summary()
        
        table = Table(title=f"Investigation Summary: {self.target_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Searches", str(summary["total_searches"]))
        table.add_row("Findings Extracted", str(summary["total_findings"]))
        table.add_row("Risks Identified", str(summary["total_risks"]))
        table.add_row("Errors", str(summary["errors"]))
        table.add_row("Log File", summary["log_file"])
        
        self.console.print(table)

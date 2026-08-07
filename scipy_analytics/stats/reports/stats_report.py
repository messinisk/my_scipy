"""
Unified statistical reporting engine.

This module defines the `StatsReport` class, the central abstraction for
generating human‑readable and machine‑readable statistical reports across
the entire `scipy_analytics.stats` package.

The goal of the reporting engine is to provide:

- A consistent structure for statistical reports
- Automatic generation of summaries, interpretations, and conclusions
- Integration with descriptive statistics, correlation analysis,
  distribution fitting, hypothesis testing, and density estimation
- A plug‑and‑play interface for notebooks, pipelines, and dashboards

Overview
--------
`StatsReport` acts as a high‑level orchestrator. It receives raw data or
precomputed statistical results and produces:

- Textual summaries (plain text)
- Structured summaries (dict)
- Optional extended metadata (e.g., parameters, diagnostics)
- Optional integration with plotting modules

The class is intentionally designed as a skeleton. Concrete report types
(e.g., PearsonReport, FitReport, DescriptiveReport) may extend it.

Examples
--------
>>> from scipy_analytics.stats.reports.stats_report import StatsReport
>>> rep = StatsReport(title="My Analysis")
>>> rep.add_section("Summary", {"mean": 1.23, "std": 0.45})
>>> rep.text()
"Report: My Analysis\n\nSection: Summary\nmean: 1.23\nstd: 0.45\n"

Notes
-----
- This module provides the foundation for automated reporting.
- It does not perform statistical computation itself.
- It is designed to be extended by specialized report classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StatsReport:
    """
    Base class for unified statistical reports.

    Parameters
    ----------
    title : str
        Title of the report.

    Attributes
    ----------
    title : str
        Report title.
    sections : dict[str, Any]
        Mapping of section names to their content.

    Methods
    -------
    add_section(name, content)
        Add a named section to the report.
    summary()
        Return a structured dictionary representation.
    text()
        Return a human‑readable text representation.

    Notes
    -----
    - Subclasses may override `text()` to provide custom formatting.
    - Section content may be any serializable object (dict, list, float, etc.).
    """

    title: str
    sections: dict[str, Any] = field(default_factory=dict)

    def add_section(self, name: str, content: Any) -> None:
        """
        Add a section to the report.

        Parameters
        ----------
        name : str
            Section name.
        content : Any
            Section content (dict, list, float, etc.).
        """
        self.sections[name] = content

    def summary(self) -> dict[str, Any]:
        """
        Return a structured summary of the report.

        Returns
        -------
        dict
            Dictionary containing:
            - "title": report title
            - "sections": mapping of section names to content
        """
        return {
            "title": self.title,
            "sections": self.sections,
        }

    def text(self) -> str:
        """
        Return a human‑readable text representation of the report.

        Returns
        -------
        str
            Multi‑line textual summary.

        Notes
        -----
        - Subclasses may override this for custom formatting.
        """
        lines = [f"Report: {self.title}", ""]
        for name, content in self.sections.items():
            lines.append(f"Section: {name}")
            if isinstance(content, dict):
                for k, v in content.items():
                    lines.append(f"{k}: {v}")
            else:
                lines.append(str(content))
            lines.append("")
        return "\n".join(lines)

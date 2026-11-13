"""Documented cleaning / quality decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    """One documented cleaning / quality decision."""

    table: str
    feature: str
    issue: str
    action: str
    detail: str = ""

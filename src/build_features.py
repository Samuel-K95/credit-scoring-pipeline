"""Backward-compatible CLI shim for python -m src.build_features.

Prefer: python -m src.features
"""

from __future__ import annotations

from src.features.build import SQL_FILES, build_features, register_views

__all__ = ["SQL_FILES", "build_features", "register_views"]


def main() -> None:
    """CLI entrypoint."""
    build_features()


if __name__ == "__main__":
    main()

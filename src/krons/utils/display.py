# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Display utilities for verbose operation tracking.

Ported from lionagi's as_readable system. Provides Rich-based
console output with YAML/JSON syntax highlighting, environment
detection, and truncation support.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

try:
    from rich.align import Align
    from rich.box import ROUNDED
    from rich.console import Console
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.theme import Theme

    DARK_THEME = Theme(
        {
            "info": "bright_cyan",
            "warning": "bright_yellow",
            "error": "bold bright_red",
            "success": "bold bright_green",
            "panel.border": "bright_blue",
            "panel.title": "bold bright_cyan",
            "json.key": "bright_cyan",
            "json.string": "bright_green",
            "json.number": "bright_yellow",
            "json.boolean": "bright_magenta",
            "json.null": "bright_red",
            "yaml.key": "bright_cyan",
            "yaml.string": "bright_green",
            "yaml.number": "bright_yellow",
            "yaml.boolean": "bright_magenta",
        }
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    DARK_THEME = None

_console: Console | None = None


def _get_console() -> Console | None:
    global _console
    if not RICH_AVAILABLE:
        return None
    if _console is None:
        _console = Console(theme=DARK_THEME)
    return _console


def in_notebook() -> bool:
    """Check if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return "ZMQInteractiveShell" in shell
    except Exception:
        return False


def in_console() -> bool:
    """Check if running in a terminal with TTY."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and not in_notebook()


def format_dict(data: Any, indent: int = 0) -> str:
    """Format data as YAML-like readable string."""
    lines = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    item_str = format_dict(item, indent + 2).lstrip()
                    lines.append(f"{prefix}  - {item_str}")
            elif isinstance(value, str) and "\n" in value:
                lines.append(f"{prefix}{key}: |")
                subprefix = "  " * (indent + 1)
                for line in value.splitlines():
                    lines.append(f"{subprefix}{line}")
            else:
                item_str = format_dict(value, indent + 1).lstrip()
                lines.append(f"{prefix}{key}: {item_str}")
        return "\n".join(lines)

    elif isinstance(data, list):
        for item in data:
            item_str = format_dict(item, indent + 1).lstrip()
            lines.append(f"{prefix}- {item_str}")
        return "\n".join(lines)

    return prefix + str(data)


def as_readable(
    input_: Any,
    /,
    *,
    md: bool = False,
    format_curly: bool = True,
    max_chars: int | None = None,
) -> str:
    """Convert data to human-readable string.

    Args:
        input_: Data to format (dict, model, list, etc.).
        md: Wrap in code fences for markdown.
        format_curly: YAML-like (True) or JSON (False).
        max_chars: Truncate output.

    Returns:
        Formatted string.
    """
    from krons.utils.fuzzy import to_dict

    # Convert to dict
    def safe_dict(obj: Any) -> Any:
        try:
            return to_dict(
                obj,
                use_model_dump=True,
                fuzzy_parse=True,
                recursive=True,
                recursive_python_only=False,
                max_recursive_depth=5,
            )
        except Exception:
            return str(obj)

    if isinstance(input_, list):
        items = [safe_dict(x) for x in input_]
    else:
        maybe = safe_dict(input_)
        items = maybe if isinstance(maybe, list) else [maybe]

    rendered = []
    for item in items:
        if format_curly:
            rendered.append(format_dict(item) if isinstance(item, (dict, list)) else str(item))
        else:
            try:
                rendered.append(json.dumps(item, indent=2, ensure_ascii=False))
            except Exception:
                rendered.append(str(item))

    text = "\n\n".join(rendered).strip()

    if md:
        lang = "yaml" if format_curly else "json"
        text = f"```{lang}\n{text}\n```"

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "...\n\n[Truncated]"

    return text


def display(text: str, *, title: str | None = None, lang: str = "yaml") -> None:
    """Display text with Rich formatting if available.

    Args:
        text: Text to display.
        title: Optional panel title.
        lang: Syntax language for highlighting.
    """
    console = _get_console()

    if console and in_console():
        syntax = Syntax(
            text,
            lang,
            theme="github-dark",
            line_numbers=False,
            word_wrap=True,
            background_color="default",
        )
        content = syntax
        if title:
            content = Panel(
                Align.left(syntax, pad=False),
                title=title,
                title_align="left",
                border_style="panel.border",
                box=ROUNDED,
                width=min(console.width - 4, 140),
                expand=False,
            )
            console.print(Padding(content, (0, 0, 0, 2)))
        else:
            console.print(Padding(content, (0, 0, 0, 2)))
        return

    # Fallback
    if title:
        print(f"\n--- {title} ---")
    print(text)


def status(msg: str, *, style: str = "info") -> None:
    """Print a status message with optional Rich styling.

    Args:
        msg: Status message.
        style: Rich style name (info, success, warning, error).
    """
    console = _get_console()
    if console and in_console():
        console.print(f"  [{style}]{msg}[/{style}]")
    else:
        print(f"  {msg}")


def phase(title: str) -> None:
    """Print a phase header."""
    console = _get_console()
    if console and in_console():
        console.print(f"\n[bold bright_cyan]=== {title} ===[/bold bright_cyan]")
    else:
        print(f"\n=== {title} ===")


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.elapsed = time.monotonic() - self.start
        if self.label:
            status(f"{self.label}: {self.elapsed:.1f}s", style="success")

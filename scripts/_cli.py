"""The one way every script builds its argument parser, so `--help` reads alike.

The first line of a script's docstring becomes the description and the rest (the
example commands) the epilog, printed as written. Each option's default is
appended to its help unless the help already says it or there is none to show.
Flag names and meanings are the table in docs/ARCHITECTURE.md ("CLI conventions").
Imports nothing from `physai`, so scripts that never touch the simulators can use it.
"""

from __future__ import annotations

import argparse


class Formatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        shown = action.default not in (None, False, [], argparse.SUPPRESS)
        if shown and "default" not in text and action.option_strings:
            text += " (default: %(default)s)"
        return text


def new_parser(doc: str | None, **kwargs) -> argparse.ArgumentParser:
    """A parser described by a script's docstring: summary line, then examples."""
    summary, _, examples = (doc or "").strip().partition("\n")
    return argparse.ArgumentParser(
        description=summary,
        epilog=examples.strip() or None,
        formatter_class=Formatter,
        **kwargs,
    )

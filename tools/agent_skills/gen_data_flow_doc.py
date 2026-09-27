#!/usr/bin/env python3
"""Regenerate the generated sections of ``docs/architecture/data_flow.md``.

The kinds table comes from ``SCOPE_GRAPH`` and the providers table from
``config/providers.toml``. The doc embeds both between
``<!-- GENERATED:BEGIN <name> -->`` markers; this script rewrites those
sections in place. ``tests/unit/docs/test_data_flow_doc.py`` fails when the
committed doc differs from this output.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "architecture" / "data_flow.md"
PROVIDERS_TOML = ROOT / "config" / "providers.toml"


def kinds_table() -> str:
    """Render the registry-kinds table from ``SCOPE_GRAPH``."""
    from src.data.pipeline_graph import SCOPE_GRAPH

    lines = ["| kind | inputs | bronze sources | builder |", "| --- | --- | --- | --- |"]
    for node in SCOPE_GRAPH:
        inputs = ", ".join(node.inputs) if node.inputs else "—"
        sources = ", ".join(node.bronze_sources) if node.bronze_sources else "—"
        builder = getattr(node.build, "__name__", str(node.build))
        lines.append(f"| `{node.kind}` | {inputs} | {sources} | `{builder}` |")
    return "\n".join(lines) + "\n"


def providers_table() -> str:
    """Render the provider-policy table from ``config/providers.toml``."""
    with open(PROVIDERS_TOML, "rb") as handle:
        config = tomllib.load(handle)

    def fmt(value: object) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value) if value else "—"
        return str(value)

    rows: list[tuple[str, str, str]] = []
    for provider in ("dart", "kis", "krx", "ls"):
        section = config.get(provider, {})
        keys = section.get("keys", {})
        if isinstance(keys, dict) and keys:
            for key_name, key in keys.items():
                policy = (
                    f"limit={key.get('daily_limit')} budget={key.get('daily_budget')} "
                    f"reserve={key.get('daily_reserve')} interval={key.get('min_interval_seconds')}s"
                )
                if key.get("max_workers") is not None:
                    policy += f" workers={key.get('max_workers')}"
                if key.get("default"):
                    policy += " default"
                rows.append((provider, key_name, policy))
        else:
            policy_keys = ("min_interval_seconds", "daily_limit", "max_attempts")
            policy = " ".join(f"{k}={fmt(section[k])}" for k in policy_keys if k in section)
            extras = {k: v for k, v in section.items() if k not in (*policy_keys, "keys") and not isinstance(v, dict)}
            for key, value in extras.items():
                policy += f" {key}={fmt(value)}"
            rows.append((provider, "—", policy or "—"))
    lines = ["| provider | key | policy |", "| --- | --- | --- |"]
    lines.extend(f"| {p} | `{k}` | {v} |" for p, k, v in rows)
    return "\n".join(lines) + "\n"


GENERATED: dict[str, object] = {
    "kinds": kinds_table,
    "providers": providers_table,
}


def render_generated_sections() -> dict[str, str]:
    """Return every generated section keyed by marker name."""
    return {name: func() for name, func in GENERATED.items()}  # type: ignore[operator]


def replace_sections(text: str, sections: dict[str, str]) -> str:
    """Replace each marked section in ``text`` with the generated output."""
    for name, body in sections.items():
        pattern = re.compile(
            r"(<!-- GENERATED:BEGIN " + re.escape(name) + r" -->\n).*?(<!-- GENERATED:END " + re.escape(name) + r" -->)",
            re.DOTALL,
        )
        replacement = r"\1" + body + r"\2"
        text, count = pattern.subn(replacement, text)
        if count != 1:
            raise ValueError(f"expected exactly one generated section for {name!r}, found {count}")
    return text


def main() -> int:
    sections = render_generated_sections()
    text = DOC.read_text(encoding="utf-8")
    DOC.write_text(replace_sections(text, sections), encoding="utf-8")
    print(f"regenerated {len(sections)} section(s) in {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

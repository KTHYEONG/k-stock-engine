"""The generated sections of data_flow.md match SCOPE_GRAPH and providers.toml."""

from __future__ import annotations

from pathlib import Path


def test_doc_matches_the_graph() -> None:
    from tools.agent_skills.gen_data_flow_doc import DOC, render_generated_sections, replace_sections

    committed = Path(DOC).read_text(encoding="utf-8")
    assert replace_sections(committed, render_generated_sections()) == committed

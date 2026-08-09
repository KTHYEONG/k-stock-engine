#!/usr/bin/env python3
"""Spec Init: Helper script to aggregate architecture context and prepare spec boilerplate."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Spec Init context aggregator & boilerplate generator.")
    parser.add_argument("--feature", required=True, help="Feature name (e.g. volatility_breakout)")
    parser.add_argument("--domain", default="general", help="Domain category (e.g. signal, risk, execution)")
    parser.add_argument("--query", default="", help="Keyword to search in codebase index or ADRs")
    args = parser.parse_args()

    feature_slug = args.feature.lower().replace(" ", "_").replace("-", "_")
    output_contract = f"docs/specs/{feature_slug}_contract.json"

    print(f"=== 📐 Spec Init Context Aggregation: {feature_slug} ===")

    # 1. Search recent ADRs from task_index.json
    task_index = _read_json("docs/decisions/task_index.json")
    tasks = task_index.get("tasks", [])
    relevant_adrs = []
    for t in tasks:
        domain_match = t.get("domain") == args.domain
        query_match = args.query and (args.query.lower() in t.get("title", "").lower() or args.query.lower() in t.get("resolution", "").lower())
        if domain_match or query_match:
            relevant_adrs.append(t)
        if len(relevant_adrs) >= 3:
            break

    print(f"\n--- 📚 Recent Domain ADRs ({len(relevant_adrs)} found) ---")
    for adr in relevant_adrs:
        print(f"• [{adr.get('adr_id')}] {adr.get('title')} ({adr.get('date')})")
        print(f"  - Resolution: {adr.get('resolution')}")

    # 2. Search code map entries from code_map.json
    code_map = _read_json("docs/code_map.json")
    relevant_code = {}
    if args.query:
        q = args.query.lower()
        for src, info in code_map.items():
            if q in src.lower() or (isinstance(info, dict) and any(q in str(v).lower() for v in info.values())):
                relevant_code[src] = info
                if len(relevant_code) >= 5:
                    break

    print(f"\n--- 🗺️ Related Code Map Entries ({len(relevant_code)} found) ---")
    for src, info in relevant_code.items():
        test_info = info.get("testing") if isinstance(info, dict) else ""
        print(f"• Source: {src} | Test: {test_info}")

    # 3. Create Contract Boilerplate if not exists
    os.makedirs("docs/specs", exist_ok=True)
    if not os.path.exists(output_contract):
        boilerplate_contract = {
            "feature": feature_slug,
            "domain": args.domain,
            "target_file": f"src/{args.domain}/{feature_slug}.py",
            "symbol": f"calc_{feature_slug}",
            "signature": f"def calc_{feature_slug}(val: float) -> float",
            "python_assertion": f"assert calc_{feature_slug}(1.0) == 1.0",
            "scenarios": [
                {
                    "scenario_id": f"SCENARIO_{feature_slug.upper()}_01",
                    "target_test_file": f"tests/unit/{args.domain}/test_{feature_slug}.py",
                    "expected_behavior": "Handles normal input correctly"
                }
            ],
            "wiring": {
                "file": f"src/{args.domain}/pipeline.py",
                "anchor": "def run_pipeline",
                "invocation_expression": f"calc_{feature_slug}(val)"
            }
        }
        with open(output_contract, "w", encoding="utf-8") as f:
            json.dump(boilerplate_contract, f, indent=2)
            f.write("\n")
        print(f"\n✅ Created contract boilerplate: {output_contract}")
    else:
        print(f"\n[i] Contract file already exists: {output_contract}")

    print("\n⚠️ Note to AI: Use this aggregated context as a starting point. Perform deep file inspection (`rg` / `view_file`) for detailed logic implementation.")


if __name__ == "__main__":
    main()

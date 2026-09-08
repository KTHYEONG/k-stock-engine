---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check Protocol

Independent audit gate completing the main development loop (`probe` -> `spec` -> `implement` -> `check`). Performs code review, strict quality checks, and regression verification without mutating source code.

## Directives

1. **Identify Modified Scope & Active Spec**:
   - Inspect modified files using `git status --short`.
   - Identify active spec contract under `docs/specs/*_contract.json` if available.

2. **Tier 1: Deterministic Audit Gate (Fast Script)**:
   - Run Smart Selective Verification runner (auto-detects modified `.py` files; self-heals `docs/code_map.json`, then runs static checks, pinpoint tests, and a diff-scoped coverage gate):
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json
     ```
     (Omit `--spec` if auditing an un-specced patch or chore).
   - This gate includes: Ruff, Mypy, impact-scoped pytest, and **diff coverage** — every line the diff *adds* to a touched `src/` file must execute during the test run (not a flat %, the exact new lines).
   - **Immediate Stop on Tier 1 Failure**: If `lean_check.py` fails, immediately report `FAIL` with the root cause diagnostics without proceeding to Tier 2.

3. **Tier 2: Semantic Defect Scan (Targeted Code Review)**:
   - Correctness over speed here: Tier 1 already caught the mechanical failures, so spend the reasoning budget Tier 2 needs to actually catch what a script can't. Scan the modified changes (`git diff`) for:
     1) **Test Realism & Exception Specificity**: Ensure tests are non-vacuous (no trivial `assert True`, mocks do not mask core logic, and `pytest.raises` specifies `match=` or precise exception types).
     2) **Contract & Invariant Integrity**: Verify core business invariants, division by zero / None handling, and boundary edge cases specified in `requirements`. Verify each entry in `design_rationale.failure_modes` has a corresponding guard in the diff.
     3) **No Dead Defensive Code (Defensive Sprawl Audit)**: Flag unrequested `try-except Exception` catches, silent `except: return None`, or speculative null checks that hide bugs or skirt coverage.
     4) **Performance Budget Honored**: Confirm actual use of `dtype_precision`/`storage_format`/`chunking_strategy`, and flag any `timeout`, `max_iterations`/`n_epochs` cap, sample-size reduction, or shortened date-range introduced without technical justification (`.agents/rules/performance.md` §0).
     5) **Domain Principle Compliance**: Cross-check against `.agents/rules/quant.md`, `.agents/rules/performance.md`, and `.agents/rules/code-style.md`.
     6) **Production Wire-up & No Ghost Paths**: Verify new logic is actually invoked in the production pipeline/entry-point and no unhandled branches or orphaned dead code remain.

4. **Strict Audit Gate & Surgical Remediation Authority (Zero Human-Pingpong)**:
   - **Full Surgical Remediation Authority**:
     - The high-reasoning auditor (`check`) has full authority to perform pinpoint surgical patches when the diagnosis is 100% deterministic:
       1) **Contract/Fixture Contradictions**: When the spec contract requirement contradicts its own test fixture (e.g. denominator counting, conflicting assertion constants, missing sentinel import handling), the auditor directly amends `contract.json` and the corresponding test fixture.
       2) **Production Contortion Cleanup**: When the implementer introduced artificial hacks (e.g. `globals()[...]`, dead comments to appease matchers), the auditor cleanly reverts the hack and redirects the test mock/fixture appropriately.
       3) **Mechanical Wiring/Lint Defects**: Fix simple imports, wiring anchors, or missing scenarios directly.
     - Immediately re-run `lean_check.py` to confirm the fix is green and sound.
     - When verified, emit ✅ **PASS** with a 1-line resolution summary. Do NOT bounce back to user or call subagents.
   - **Escalation Boundary (When to FAIL)**:
     - Stop immediately and output `FAIL` ONLY when:
       1) Fundamental business hypothesis invalidation or mathematical instability under real market data (`/probe`).
       2) Deep architectural conflicts requiring trade-off decisions beyond the original spec scope (`/spec`).
       3) Destructive actions or unresolvable financial correctness ambiguity affecting production money.

## Output

Do NOT add any intro, preamble, sub-bullet checks, breakdown items, or conversational commentary.

- **PASS** (Strict 1-Line ONLY):
  ✅ PASS: <Audit Target> [Optional: (Resolved: <1-line surgical fix summary>)]

- **FAIL** (Compact 1-2 Lines format):
  ❌ FAIL: <Audit Target> | Root: <Cause> | Impact: <Scope> | Fix: <Action> → `/implement`, `/spec`, or `/probe`

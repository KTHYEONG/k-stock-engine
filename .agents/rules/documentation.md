---
trigger:
  - on_file_path_regex: "src/.*\\.py"
  - on_file_path_regex: "docs/.*\\.md"
priority: 8
---

# Code Documentation & Commenting Rules

## 1. Core Principles
1. **Explain "Why", Not "What":** Omit obvious code behavior comments. Explain only business context, mathematical rationale, domain constraints, or non-obvious optimizations.
2. **Token & Patch Efficiency:** Keep comments to a single line where possible. Avoid verbose explanations to reduce token usage and ensure clean diff matching.
3. **Language Policy:**
   - **Docstrings & External Docs:** English only (maintains compatibility with standard Python IDE tools and global conventions).
   - **In-line Comments (`#`):** Korean preferred (ensures fast intuition and readability for Korean maintainers).
4. **Chat Notification Conciseness:** Keep chat responses and skill status notifications strictly under 5 bullet points (under 80 tokens). Move detailed rationale, experiment logs, and failure histories into markdown files in `docs/` or artifact files.

## 2. Docstring Rules
- **Standard Format:** Apply Google Style Docstrings (`Args:`, `Returns:`, `Raises:`) for all public classes, functions, and methods.
- **Conciseness:** Keep descriptions brief and focused on contract and behavior constraints.
- **Private Symbols (`_foo`):** Omit docstrings for internal/private helper functions unless logic is highly complex.

## 3. In-line Comment Rules (`#`)
- **Prohibited:**
  - **Lint / Fix Annotations:** Do NOT leave explanatory comments next to fixes during linter or type-checking cycles (e.g. `x: int = 1  # type fix for mypy`, `import sys  # fixed ruff F401`).
  - **Code Restatement:** Restating obvious code in prose (e.g., `i += 1  # increment counter`, `return result  # return result`).
  - **Conversational Prose:** Multi-line or meta comments about task progress, AI decisions, or refactoring history.
- **Required / Recommended:**
  - **Quant & Mathematical Formulae:** Rationale for formula derivations, slippage, or fee adjustments.
  - **Domain & Exchange Limits:** Reasons for rate limit handling, boundary caps, or API workarounds.
  - **Performance Trade-offs:** Rationale for choosing specific data structures or algorithmic shortcuts.

## 4. Architecture Documentation (`docs/architecture/`)
- **Purpose:** "AI-First Structured Constraints". Contains system boundary, LaTeX mathematical formalisms, strict I/O tables, and Mermaid topology.
- **Line Limit:** Keep each document strictly under 300 lines.
- **Surgical Update Only:** Never append raw text to architecture files. Edit existing tables, schemas, or Mermaid nodes inline.
- **Prohibitions:** Omit procedural logic, code optimization details, logging policies, conversational prose, temporal examples, change history, and `[ADR_...]` tags.
- **Contract Priority:** In case of mismatch, in-code Type/Protocol definitions strictly supersede external markdown files.

## 5. Canonical Code Example

```python
def calculate_position_size(
    account_balance: float, 
    risk_ratio: float, 
    volatility: float
) -> float:
    """Calculates optimal position size based on fixed fractional volatility sizing.

    Args:
        account_balance: Total available trading capital in USDT.
        risk_ratio: Maximum account risk fraction per trade (e.g. 0.02).
        volatility: Asset daily volatility (ATR percentage).

    Returns:
        Calculated position size in base currency.
    """
    if volatility <= 0:
        # 변동성이 0 이하인 경우 디폴트 최소 위험값 적용 (Zero-division 방지)
        return 0.0

    # 켈리 공식 변형: 앙상블 리스크 스케일링 적용
    raw_size = (account_balance * risk_ratio) / volatility
    return round(raw_size, 4)
```

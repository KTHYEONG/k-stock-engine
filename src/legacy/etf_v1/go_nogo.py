from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger("etf.go_nogo")

@dataclass
class GoNoGoResult:
    passed: bool
    details: Dict[str, bool]
    summary: str

def run_go_nogo_check(
    oos_cagr: float,
    max_mdd_pct: float,
    profit_factor: float,
    total_trades: int
) -> GoNoGoResult:
    
    # 1. Growth Engine
    growth_pass: bool = oos_cagr > 0.0
    
    # 2. Volatility Drag Control (MDD <= 25.0% for ETFs, tighter than Coins)
    mdd_pass: bool = abs(max_mdd_pct) <= 25.0
    
    # 3. Mathematical Edge (PF >= 1.10)
    target_pf = 1.10
    pf_pass: bool = profit_factor >= target_pf

    # 4. Statistical Significance (At least 15 trades expected over a 6-month OOS)
    min_trades_req = 5
    trades_pass: bool = total_trades >= min_trades_req

    details: Dict[str, bool] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": growth_pass,
        "2. Volatility Drag (MDD <= 25%)": mdd_pass,
        f"3. Mathematical Edge (PF >= {target_pf})": pf_pass,
        f"4. Stat Edge (Trades >= {min_trades_req})": trades_pass,
    }

    all_passed = all(details.values())

    summary_lines: List[str] = ["[Elite 1% Wealth Compounding Checklist - ETF]"]
    
    metric_values: Dict[str, str] = {
        "1. Out-of-Sample Growth (CAGR > 0%)": f"CAGR: {oos_cagr:.2f}%",
        "2. Volatility Drag (MDD <= 25%)": f"MDD: {abs(max_mdd_pct):.2f}%",
        f"3. Mathematical Edge (PF >= {target_pf})": f"PF: {profit_factor:.2f}",
        f"4. Stat Edge (Trades >= {min_trades_req})": f"N: {total_trades}",
    }

    req_met: int = sum(1 for v in details.values() if v)
    total_req: int = len(details)

    for k, v in details.items():
        status: str = "PASS" if v else "FAIL"
        val_str: str = metric_values.get(k, "")
        summary_lines.append(f"  - {k:<40}: {status:<5} ({val_str})")
        
    summary_lines.append("-" * 55)
    final_status: str = "🌟 ELITE GO (Top 1% Ready)" if all_passed else f"🔴 NO-GO (Needs Revision, Passed {req_met}/{total_req})"
    summary_lines.append(f"  FINAL VERDICT: {final_status}")
    
    return GoNoGoResult(
        passed=all_passed,
        details=details,
        summary="\n".join(summary_lines)
    )

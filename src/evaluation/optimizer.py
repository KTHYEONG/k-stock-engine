import logging
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.backtester import YetiRankBacktester
from src.evaluation.optimization_config import GET_SEARCH_SPACE
from src.utils.logger import setup_logger

logger = setup_logger("evaluation.optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("training.data_loader").setLevel(logging.WARNING)


def _parse_years_csv(raw: str) -> List[int]:
    years: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            y = int(token)
        except ValueError:
            continue
        if 1900 <= y <= 2100:
            years.append(y)
    return sorted(set(years))


class YetiRankOptimizer:
    """
    Strategy parameter optimizer.

    Modes:
    - Single-period: optimize over one date range.
    - AWFO: optimize with year-by-year OOS scoring and robust aggregation.
    """

    def __init__(
        self,
        start_date: str = "20220101",
        end_date: str = "20251231",
        mode: str = "UNIFIED",
        market_type: str = "stock_spot",
        model_year: Optional[int] = None,
        sizing_mode: str = "CONFIDENCE",
        awfo: bool = True,
        awfo_years: Optional[List[int]] = None,
    ):
        self.mode = mode
        self.market_type = market_type
        self.sizing_mode = sizing_mode
        self.start_date = start_date
        self.end_date = end_date
        self.model_year = model_year
        self.search_space_config = GET_SEARCH_SPACE(mode=mode, market_type=market_type)

        self.awfo = bool(awfo)
        self.awfo_years = self._resolve_awfo_years(awfo_years if awfo_years else [])
        self.awfo_backtesters: Dict[int, YetiRankBacktester] = {}
        self.backtester: Optional[YetiRankBacktester] = None

        if self.awfo and self.awfo_years:
            self._prepare_awfo_backtesters()
            if not self.awfo_backtesters:
                logger.warning("AWFO backtesters are empty. Falling back to single-period mode.")
                self.awfo = False
        else:
            self.awfo = False

        if not self.awfo:
            self.backtester = YetiRankBacktester(
                start_date=start_date,
                end_date=end_date,
                model_year=model_year,
            )
            self.backtester.generate_predictions()

        logger.info(
            "Optimizer initialized | mode=%s | awfo=%s | years=%s | period=%s~%s | model_year=%s",
            self.mode,
            self.awfo,
            self.awfo_years if self.awfo else "-",
            self.start_date,
            self.end_date,
            self.model_year,
        )

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            x = float(v)
            if not np.isfinite(x):
                return float(default)
            return x
        except Exception:
            return float(default)

    @classmethod
    def _parse_percent(cls, v: Any, default: float = 0.0) -> float:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return cls._safe_float(v, default)
        return cls._safe_float(str(v).replace("%", "").strip(), default)

    def _detect_available_model_years(self) -> List[int]:
        model_dir = PROJECT_ROOT / "models" / "yetirank"
        if not model_dir.exists():
            return []
        p = re.compile(r"yetirank_(\d{4})\.cbm$")
        years: List[int] = []
        for path in model_dir.glob("yetirank_*.cbm"):
            m = p.match(path.name)
            if m:
                years.append(int(m.group(1)))
        years = sorted(set(years))
        if not years:
            return []

        start_y = int(str(self.start_date)[:4])
        end_y = int(str(self.end_date)[:4])
        filtered = [y for y in years if start_y <= y <= end_y]
        return filtered if filtered else years

    def _resolve_awfo_years(self, awfo_years: List[int]) -> List[int]:
        available = self._detect_available_model_years()
        start_y = int(str(self.start_date)[:4])
        end_y = int(str(self.end_date)[:4])

        profile_years: List[int] = []
        profile_path = PROJECT_ROOT / "models" / "yetirank" / "awfo_profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                profile_years = sorted(
                    {
                        int(y)
                        for y in profile.get("test_years", [])
                        if isinstance(y, int) or (isinstance(y, str) and str(y).isdigit())
                    }
                )
            except Exception:
                profile_years = []

        if awfo_years:
            requested = sorted(set(int(y) for y in awfo_years))
            resolved = [y for y in requested if y in available] if available else requested
            missing = sorted(set(requested) - set(resolved))
            if missing:
                logger.warning("Requested AWFO years missing model files and were skipped: %s", missing)
            return resolved

        if profile_years:
            profile_years = [y for y in profile_years if start_y <= y <= end_y]
            if profile_years:
                if available:
                    resolved = [y for y in profile_years if y in available]
                else:
                    resolved = profile_years
                if resolved:
                    logger.info("AWFO years loaded from awfo_profile.json: %s", resolved)
                    return resolved

        if available:
            return [y for y in available if start_y <= y <= end_y] or available

        if self.model_year is not None:
            return [int(self.model_year)]
        return []

    def _prepare_awfo_backtesters(self) -> None:
        for year in self.awfo_years:
            bt = YetiRankBacktester(
                start_date=f"{year}0101",
                end_date=f"{year}1231",
                model_year=year,
            )
            bt.generate_predictions()
            if bt._cached_predictions.is_empty():
                logger.warning("Skipping AWFO year %s because predictions are empty.", year)
                continue
            self.awfo_backtesters[year] = bt

        self.awfo_years = sorted(self.awfo_backtesters.keys())

    def _study_name(self) -> str:
        suffix = "_awfo" if self.awfo else ""
        return f"yetirank_{self.mode.lower()}{suffix}_opt"

    def _db_path(self) -> Path:
        suffix = "_awfo" if self.awfo else ""
        return PROJECT_ROOT / "results" / f"optimization_{self.mode.lower()}{suffix}.db"

    def check_indicators(self) -> None:
        bt = self.backtester
        if bt is None and self.awfo_backtesters:
            bt = self.awfo_backtesters[self.awfo_years[0]]
        if bt is None:
            logger.error("No backtester available.")
            return

        df = bt._cached_predictions
        if df.is_empty():
            logger.error("No data found in cached predictions.")
            return

        indicator_cols = [
            "rsi_14",
            "mfi_14",
            "natr_14",
            "macd_hist",
            "cci",
            "cmf",
            "obv",
            "stoch_rsi",
            "bb_position",
            "supertrend_direction",
        ]

        print("\n" + "=" * 60)
        print(f"{'Indicator':<25} | {'Nulls':<8} | {'Min':<10} | {'Max':<10}")
        print("-" * 60)

        for col in indicator_cols:
            if col not in df.columns:
                print(f"{col:<25} | {'MISSING':<8}")
                continue
            null_count = df[col].null_count()
            null_pct = (null_count / len(df)) * 100
            valid_data = df[col].drop_nulls()
            if len(valid_data) > 0:
                v_min = valid_data.min()
                v_max = valid_data.max()
                print(f"{col:<25} | {null_pct:>6.1f}% | {v_min:>10.2f} | {v_max:>10.2f}")
            else:
                print(f"{col:<25} | {null_pct:>6.1f}% | {'ALL NULL':<10} | {'ALL NULL':<10}")
        print("=" * 60 + "\n")

    def _score_from_backtest(
        self,
        metrics: Dict[str, Any],
        daily_df: Any,
        trade_records: List[float],
    ) -> Tuple[float, Dict[str, float]]:
        diag: Dict[str, float] = {
            "valid": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "pf": 0.0,
            "pf_raw": 0.0,
            "sqn": 0.0,
            "consistency": 0.0,
            "win_rate": 0.0,
            "avg_trade_ret": 0.0,
            "trades": 0.0,
            "days": 0.0,
            "score": -120.0,
        }

        if not metrics or daily_df is None:
            return -120.0, diag
        n_days = int(daily_df.height) if hasattr(daily_df, "height") else 0
        if n_days <= 0:
            return -120.0, diag

        cagr = self._parse_percent(metrics.get("CAGR", "0%"), default=0.0)
        mdd = abs(self._parse_percent(metrics.get("MDD", "0%"), default=0.0))
        sharpe = self._safe_float(metrics.get("Sharpe Ratio", "0"), default=0.0)
        win_rate = self._parse_percent(metrics.get("Win Rate", "0%"), default=0.0)
        avg_trade_ret = self._parse_percent(metrics.get("Avg Trade Return", "0%"), default=0.0)
        turnover_pct = self._parse_percent(metrics.get("Avg Turnover", "0%"), default=0.0)
        turnover = turnover_pct / 100.0
        total_trades = len(trade_records)

        returns = np.nan_to_num(
            np.array(trade_records, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        n_trades = len(returns)
        if n_trades == 0:
            return -120.0, diag

        pos_sum = self._safe_float(np.sum(returns[returns > 0]), 0.0)
        neg_sum = abs(self._safe_float(np.sum(returns[returns < 0]), 0.0))
        if neg_sum <= 1e-6:
            pf_raw = 5.0 if pos_sum > 0 else 0.0
        else:
            pf_raw = pos_sum / neg_sum
        pf = float(np.clip(self._safe_float(pf_raw, 0.0), 0.0, 5.0))

        if n_trades > 1:
            r_avg = self._safe_float(np.mean(returns), 0.0)
            r_std = self._safe_float(np.std(returns, ddof=1), 0.0)
            sqn_raw = np.sqrt(n_trades) * (r_avg / r_std) if r_std > 0 else 0.0
            sqn = float(np.clip(self._safe_float(sqn_raw, 0.0), 0.0, 6.0))
        else:
            sqn = 0.0

        if n_trades > 5:
            eq = np.cumsum(returns)
            x = np.arange(len(eq))
            corr = np.corrcoef(x, eq)[0, 1]
            r2 = corr**2 if np.isfinite(corr) else 0.0
        else:
            r2 = 0.0
        r2 = float(np.clip(self._safe_float(r2, 0.0), 0.0, 1.0))

        cagr = float(np.clip(self._safe_float(cagr, 0.0), -50.0, 250.0))
        mdd = float(np.clip(self._safe_float(mdd, 0.0), 0.0, 95.0))
        sharpe = float(np.clip(self._safe_float(sharpe, 0.0), -3.0, 6.0))
        turnover = float(np.clip(self._safe_float(turnover, 0.0), 0.0, 2.0))

        s_growth = 35.0 * np.tanh(cagr / 80.0)
        s_sharpe = 20.0 * np.tanh(sharpe / 2.0)
        s_pf = 10.0 * np.log1p(pf)
        s_sqn = 2.5 * sqn
        s_consistency = 10.0 * r2
        score = s_growth + s_sharpe + s_pf + s_sqn + s_consistency

        mdd_penalty = 0.0
        if mdd > 12.0:
            mdd_penalty += (mdd - 12.0) * 1.5
        if mdd > 20.0:
            mdd_penalty += (mdd - 20.0) * 2.5
        if mdd > 30.0:
            mdd_penalty += (mdd - 30.0) * 4.0
        score -= mdd_penalty

        min_trades = max(5, int(n_days / 30))
        if total_trades < min_trades:
            score -= (min_trades - total_trades) * 4.0
        if avg_trade_ret < 0.15:
            score -= (0.15 - avg_trade_ret) * 20.0

        if turnover > 0.6:
            score -= (turnover - 0.6) * 40.0
        expected_trade_cap = max(40, int(n_days * 1.2))
        if total_trades > expected_trade_cap:
            score -= (total_trades - expected_trade_cap) * 0.2

        score = self._safe_float(score, -120.0)
        if not np.isfinite(score):
            score = -120.0
        score = float(np.clip(score, -200.0, 200.0))

        calmar = cagr / max(mdd, 1e-9) if mdd > 0 else 0.0
        diag.update(
            {
                "valid": 1.0,
                "cagr": cagr,
                "mdd": mdd,
                "sharpe": sharpe,
                "calmar": calmar,
                "pf": pf,
                "pf_raw": self._safe_float(pf_raw, 0.0),
                "sqn": sqn,
                "consistency": r2,
                "win_rate": win_rate,
                "avg_trade_ret": avg_trade_ret,
                "trades": float(total_trades),
                "days": float(n_days),
                "score": score,
            }
        )
        return score, diag

    @staticmethod
    def _set_common_trial_attrs(trial: optuna.Trial, diag: Dict[str, float]) -> None:
        trial.set_user_attr("cagr", float(diag.get("cagr", 0.0)))
        trial.set_user_attr("mdd", float(diag.get("mdd", 0.0)))
        trial.set_user_attr("sharpe", float(diag.get("sharpe", 0.0)))
        trial.set_user_attr("calmar", float(diag.get("calmar", 0.0)))
        trial.set_user_attr("pf", float(diag.get("pf", 0.0)))
        trial.set_user_attr("pf_raw", float(diag.get("pf_raw", 0.0)))
        trial.set_user_attr("sqn", float(diag.get("sqn", 0.0)))
        trial.set_user_attr("consistency", float(diag.get("consistency", 0.0)))
        trial.set_user_attr("win_rate", float(diag.get("win_rate", 0.0)))
        trial.set_user_attr("avg_trade_ret", float(diag.get("avg_trade_ret", 0.0)))
        trial.set_user_attr("trades", int(diag.get("trades", 0.0)))
        trial.set_user_attr("days", int(diag.get("days", 0.0)))

    def objective(self, trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {}
        for param_name, config in self.search_space_config.items():
            if not isinstance(config, dict):
                continue
            p_type = config.get("type")

            if p_type == "int":
                log = config.get("log", False)
                step = config.get("step", 1) if not log else None
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    step=step,
                    log=log,
                )
            elif p_type == "float":
                log = config.get("log", False)
                step = config.get("step", None)
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    step=step,
                    log=log,
                )
            elif p_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            elif "low" in config and "high" in config:
                if isinstance(config["low"], int):
                    params[param_name] = trial.suggest_int(
                        param_name,
                        config["low"],
                        config["high"],
                        step=config.get("step", 1),
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        config["low"],
                        config["high"],
                        step=config.get("step"),
                    )

        final_params = {k.lower(): v for k, v in params.items()}
        if trial.number == 0:
            logger.info("Trial 0 params: %s", final_params)

        core_top_k = final_params.pop("top_k", 10)
        core_rebalance = final_params.pop("rebalance_period", 5)

        try:
            if self.awfo:
                year_scores: List[float] = []
                year_diags: List[Dict[str, float]] = []
                for year in self.awfo_years:
                    bt = self.awfo_backtesters[year]
                    metrics, daily_df, trade_records = bt.run_backtest(
                        top_k=core_top_k,
                        rebalance_period=core_rebalance,
                        fee=0.0035,
                        return_details=True,
                        sizing_mode=self.sizing_mode,
                        **final_params,
                    )
                    score, diag = self._score_from_backtest(metrics, daily_df, trade_records)
                    year_scores.append(score)
                    d = dict(diag)
                    d["year"] = float(year)
                    year_diags.append(d)

                if not year_scores:
                    return -120.0

                arr = np.array(year_scores, dtype=float)
                avg_score = float(np.mean(arr))
                p25_score = float(np.percentile(arr, 25))
                worst_score = float(np.min(arr))
                std_score = float(np.std(arr))
                robust = (0.45 * avg_score) + (0.35 * p25_score) + (0.20 * worst_score) - (0.15 * std_score)
                robust = float(np.clip(self._safe_float(robust, -120.0), -200.0, 200.0))

                valid_diags = [d for d in year_diags if d.get("valid", 0.0) > 0.0]
                if valid_diags:
                    agg = {
                        k: float(np.mean([d.get(k, 0.0) for d in valid_diags]))
                        for k in [
                            "cagr",
                            "mdd",
                            "sharpe",
                            "calmar",
                            "pf",
                            "pf_raw",
                            "sqn",
                            "consistency",
                            "win_rate",
                            "avg_trade_ret",
                            "trades",
                            "days",
                        ]
                    }
                else:
                    agg = {k: 0.0 for k in ["cagr", "mdd", "sharpe", "calmar", "pf", "pf_raw", "sqn", "consistency", "win_rate", "avg_trade_ret", "trades", "days"]}

                self._set_common_trial_attrs(trial, agg)
                trial.set_user_attr("awfo_years", [int(y) for y in self.awfo_years])
                trial.set_user_attr("awfo_year_scores", [float(s) for s in year_scores])
                trial.set_user_attr("awfo_avg_score", avg_score)
                trial.set_user_attr("awfo_p25_score", p25_score)
                trial.set_user_attr("awfo_worst_score", worst_score)
                trial.set_user_attr("awfo_std_score", std_score)
                return robust

            if self.backtester is None:
                return -120.0

            metrics, daily_df, trade_records = self.backtester.run_backtest(
                top_k=core_top_k,
                rebalance_period=core_rebalance,
                fee=0.0035,
                return_details=True,
                sizing_mode=self.sizing_mode,
                **final_params,
            )
            score, diag = self._score_from_backtest(metrics, daily_df, trade_records)
            self._set_common_trial_attrs(trial, diag)
            return score
        except Exception as e:
            logger.error("Trial failed: %s", e)
            return -120.0

    def run_optimization(
        self,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        resume: bool = False,
        n_jobs: int = 1,
    ) -> None:
        if study_name is None:
            study_name = self._study_name()

        logger.info(
            "Starting optimization | mode=%s | awfo=%s | years=%s | trials=%s | resume=%s | jobs=%s",
            self.mode,
            self.awfo,
            self.awfo_years if self.awfo else "-",
            n_trials,
            resume,
            n_jobs,
        )

        db_path = self._db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and db_path.exists():
            try:
                db_path.unlink()
                logger.warning("Deleted existing DB: %s", db_path)
            except Exception:
                pass

        storage = f"sqlite:///{db_path.resolve().as_posix()}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        logger.info("Optimization complete.")
        best_trial = study.best_trial
        print("\n" + "=" * 50)
        print(f"BEST STRATEGY PARAMETERS ({self.mode})")
        print("=" * 50)
        for key, value in best_trial.params.items():
            print(f"- {key:<25}: {value}")
        print("-" * 50)

        attrs = best_trial.user_attrs
        print(f"- Score              : {best_trial.value:.4f}")
        print(f"- CAGR               : {attrs.get('cagr', 0.0):.2f}%")
        print(f"- MDD                : -{attrs.get('mdd', 0.0):.2f}%")
        print(f"- Sharpe             : {attrs.get('sharpe', 0.0):.4f}")
        print(f"- Calmar             : {attrs.get('calmar', 0.0):.4f}")
        print(f"- Win Rate (Trade)   : {attrs.get('win_rate', 0.0):.2f}%")
        print(f"- Avg Trade Return   : {attrs.get('avg_trade_ret', 0.0):.2f}%")
        print(f"- Trades             : {attrs.get('trades', 0)}")
        if self.awfo:
            print(f"- AWFO Avg Score     : {attrs.get('awfo_avg_score', 0.0):.4f}")
            print(f"- AWFO P25 Score     : {attrs.get('awfo_p25_score', 0.0):.4f}")
            print(f"- AWFO Worst Score   : {attrs.get('awfo_worst_score', 0.0):.4f}")
            print(f"- AWFO Std Score     : {attrs.get('awfo_std_score', 0.0):.4f}")
        print("=" * 50 + "\n")

        best_params = {k.lower(): v for k, v in best_trial.params.items()}
        if self.awfo:
            print("Final AWFO year-by-year backtest:")
            print("-" * 50)
            for year in self.awfo_years:
                bt = self.awfo_backtesters[year]
                metrics = bt.run_backtest(save_plot=False, sizing_mode=self.sizing_mode, **best_params)
                print(
                    f"{year}: Return={metrics.get('Total Return', '0%')}, "
                    f"MDD={metrics.get('MDD', '0%')}, Sharpe={metrics.get('Sharpe Ratio', '0')}, "
                    f"Trades={metrics.get('Total Trades', '0')}"
                )
            print("-" * 50)
        elif self.backtester is not None:
            print("Running final backtest...")
            self.backtester.run_backtest(save_plot=True, sizing_mode=self.sizing_mode, **best_params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YetiRank Strategy Optimizer")
    parser.add_argument("--trials", type=int, default=300, help="Number of trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument(
        "--mode",
        type=str,
        default="UNIFIED",
        choices=["UNIFIED", "ACTIVE", "SWING", "TREND"],
        help="Optimization Mode",
    )
    parser.add_argument("--check", action="store_true", help="Check indicator health and exit")
    parser.add_argument("--start", type=str, default="20220101", help="Start Date (YYYYMMDD)")
    parser.add_argument("--end", type=str, default="20251231", help="End Date (YYYYMMDD)")
    parser.add_argument("--model_year", type=int, default=None, help="Fixed model year (single mode only)")
    parser.add_argument(
        "--sizing",
        type=str,
        default="CONFIDENCE",
        choices=["EQUAL", "CONFIDENCE", "RISK", "HYBRID"],
        help="Position Sizing Mode",
    )
    parser.add_argument(
        "--disable_awfo",
        action="store_true",
        help="Disable AWFO mode and run single-period optimization",
    )
    parser.add_argument(
        "--awfo_years",
        type=str,
        default="",
        help="Comma-separated AWFO years (e.g., 2022,2023,2024,2025). Empty=auto-detect.",
    )

    args = parser.parse_args()
    awfo_years = _parse_years_csv(args.awfo_years)

    optimizer = YetiRankOptimizer(
        mode=args.mode,
        start_date=args.start,
        end_date=args.end,
        model_year=args.model_year,
        sizing_mode=args.sizing,
        awfo=not args.disable_awfo,
        awfo_years=awfo_years,
    )

    if args.check:
        optimizer.check_indicators()
        sys.exit(0)

    optimizer.run_optimization(n_trials=args.trials, resume=args.resume, n_jobs=args.n_jobs)

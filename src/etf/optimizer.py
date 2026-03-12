import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import polars as pl

from .backtester import ETFBacktester
from .etf_config import ETFConfig

logger = logging.getLogger("etf.optimizer")


class ETFOptimizer:
    """
    ETF strategy optimizer.

    Modes:
    - Single period optimization (awfo=False)
    - Multi-year AWFO-style robust optimization (awfo=True)
    """

    def __init__(
        self,
        index_df: pl.DataFrame,
        etf_df: pl.DataFrame,
        target_market: str = "KOSPI",
        target_leverage: str = "HYBRID",
        awfo: bool = False,
        awfo_start_year: Optional[int] = None,
        awfo_end_year: Optional[int] = None,
        awfo_train_years: int = 3,
    ):
        self.target_market = str(target_market).upper()
        self.target_leverage = target_leverage
        self.index_df = index_df.sort("date")
        self.etf_df = etf_df.sort("date")
        self.backtester = ETFBacktester(self.index_df, self.etf_df)
        self.config = ETFConfig.get_search_space(self.target_market)
        self.study: Optional[optuna.Study] = None
        self.last_optimization_meta: Dict[str, Any] = {}

        self.awfo = bool(awfo)
        self.awfo_train_years = int(max(1, awfo_train_years))
        self.awfo_start_year = awfo_start_year
        self.awfo_end_year = awfo_end_year
        self.awfo_plan = self._build_awfo_plan() if self.awfo else []

        if self.awfo and not self.awfo_plan:
            logger.warning("AWFO requested but no valid folds were built. Falling back to single-period optimization.")
            self.awfo = False

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            x = float(v)
            if not np.isfinite(x):
                return float(default)
            return x
        except Exception:
            return float(default)

    def _data_year_range(self) -> Tuple[int, int]:
        if self.index_df.is_empty():
            now = datetime.now().year
            return now, now
        dates = self.index_df["date"].to_list()
        years = [d.year for d in dates if hasattr(d, "year")]
        if not years:
            now = datetime.now().year
            return now, now
        return int(min(years)), int(max(years))

    def _slice_df(self, df: pl.DataFrame, start_dt: datetime, end_dt: datetime) -> pl.DataFrame:
        return df.filter((pl.col("date") >= start_dt) & (pl.col("date") <= end_dt))

    def _build_awfo_plan(self) -> List[Dict[str, Any]]:
        min_year, max_year = self._data_year_range()
        start_year = int(self.awfo_start_year) if self.awfo_start_year else int(min_year)
        end_year = int(self.awfo_end_year) if self.awfo_end_year else int(max_year)

        start_year = max(start_year, min_year)
        end_year = min(end_year, max_year)
        if end_year - start_year < self.awfo_train_years:
            return []

        plan: List[Dict[str, Any]] = []
        for eval_year in range(start_year + self.awfo_train_years, end_year + 1):
            train_start = datetime(start_year, 1, 1)
            train_end = datetime(eval_year - 1, 12, 31)
            val_start = datetime(eval_year, 1, 1)
            val_end = datetime(eval_year, 12, 31)

            tr_idx = self._slice_df(self.index_df, train_start, train_end)
            tr_etf = self._slice_df(self.etf_df, train_start, train_end)
            va_idx = self._slice_df(self.index_df, val_start, val_end)
            va_etf = self._slice_df(self.etf_df, val_start, val_end)

            if tr_idx.is_empty() or tr_etf.is_empty() or va_idx.is_empty() or va_etf.is_empty():
                continue

            plan.append(
                {
                    "eval_year": int(eval_year),
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "train_backtester": ETFBacktester(tr_idx, tr_etf),
                    "val_backtester": ETFBacktester(va_idx, va_etf),
                }
            )
        return plan

    def _suggest_value(self, trial: optuna.Trial, key: str) -> Any:
        conf = self.config.get(key)
        if conf is None:
            return None
        if isinstance(conf, list):
            return trial.suggest_categorical(key, conf)
        if isinstance(conf, dict) and "type" in conf:
            if conf["type"] == "int":
                return trial.suggest_int(key, conf["low"], conf["high"], step=conf.get("step", 1))
            if conf["type"] == "float":
                return trial.suggest_float(key, conf["low"], conf["high"], step=conf.get("step", None))
        return None

    def _generate_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        for key in ["ENTRY_TYPE", "TREND_DIR_TYPE", "MOMENTUM_TYPE", "TREND_STR_TYPE", "USE_VOLUME_FILTER", "EXIT_TYPE"]:
            if key in self.config:
                params[key] = self._suggest_value(trial, key)

        for key in [
            "ENTRY_PERIOD",
            "MA_PERIOD",
            "HURST_PERIOD",
            "HURST_THRESHOLD",
            "STOP_LOSS_ATR",
            "TAKE_PROFIT_ATR",
            "TS_TRIGGER_ATR",
            "TS_DIST_ATR",
            "LEV_HURST",
            "LEV_NATR",
        ]:
            if key in self.config:
                params[key] = self._suggest_value(trial, key)

        entry_type = params.get("ENTRY_TYPE", "DONCHIAN")
        if entry_type == "BOLLINGER" and "BB_STD" in self.config:
            params["BB_STD"] = self._suggest_value(trial, "BB_STD")
        elif entry_type == "KELTNER" and "KELTNER_ATR_MULT" in self.config:
            params["KELTNER_ATR_MULT"] = self._suggest_value(trial, "KELTNER_ATR_MULT")
        elif entry_type == "CCI" and "CCI_THRESHOLD" in self.config:
            params["CCI_THRESHOLD"] = self._suggest_value(trial, "CCI_THRESHOLD")

        trend_type = params.get("TREND_DIR_TYPE", "EMA")
        if trend_type == "SUPERTREND":
            for key in ["SUPERTREND_MULT", "SUPERTREND_PERIOD"]:
                if key in self.config:
                    params[key] = self._suggest_value(trial, key)
        elif trend_type == "MACD":
            for key in ["MACD_FAST", "MACD_SLOW", "MACD_SIGNAL"]:
                if key in self.config:
                    params[key] = self._suggest_value(trial, key)
        elif trend_type == "ICHIMOKU":
            for key in ["ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN", "ICHIMOKU_SENKOU_B"]:
                if key in self.config:
                    params[key] = self._suggest_value(trial, key)
        elif trend_type == "VWAP" and "VWAP_STD_MULT" in self.config:
            params["VWAP_STD_MULT"] = self._suggest_value(trial, "VWAP_STD_MULT")

        momentum_type = params.get("MOMENTUM_TYPE", "NONE")
        if momentum_type != "NONE" and "MOMENTUM_PERIOD" in self.config:
            params["MOMENTUM_PERIOD"] = self._suggest_value(trial, "MOMENTUM_PERIOD")
        if momentum_type == "RSI":
            for key in ["RSI_OVERBOUGHT", "RSI_OVERSOLD"]:
                if key in self.config:
                    params[key] = self._suggest_value(trial, key)
        elif momentum_type == "MFI" and "MFI_THRESHOLD" in self.config:
            params["MFI_THRESHOLD"] = self._suggest_value(trial, "MFI_THRESHOLD")
        elif momentum_type == "CMF" and "CMF_THRESHOLD" in self.config:
            params["CMF_THRESHOLD"] = self._suggest_value(trial, "CMF_THRESHOLD")

        strength_type = params.get("TREND_STR_TYPE", "NONE")
        if strength_type != "NONE" and "STRENGTH_PERIOD" in self.config:
            params["STRENGTH_PERIOD"] = self._suggest_value(trial, "STRENGTH_PERIOD")
        if strength_type == "ADX" and "ADX_THRESHOLD" in self.config:
            params["ADX_THRESHOLD"] = self._suggest_value(trial, "ADX_THRESHOLD")
        elif strength_type == "VORTEX" and "VORTEX_THRESHOLD" in self.config:
            params["VORTEX_THRESHOLD"] = self._suggest_value(trial, "VORTEX_THRESHOLD")
        elif strength_type == "ER" and "ER_THRESHOLD" in self.config:
            params["ER_THRESHOLD"] = self._suggest_value(trial, "ER_THRESHOLD")

        if params.get("USE_VOLUME_FILTER", False) and "VOLUME_MA_PERIOD" in self.config:
            params["VOLUME_MA_PERIOD"] = self._suggest_value(trial, "VOLUME_MA_PERIOD")

        if params.get("EXIT_TYPE", "NONE") == "PARABOLIC_SAR":
            for key in ["SAR_STEP", "SAR_MAX"]:
                if key in self.config:
                    params[key] = self._suggest_value(trial, key)

        return params

    @staticmethod
    def _allocate_seed_trials(total_trials: int, seeds: List[int], min_trials_per_seed: int = 40) -> List[Tuple[int, int]]:
        total_trials = int(max(1, total_trials))
        if not seeds:
            return [(13, total_trials)]

        min_trials_per_seed = int(max(1, min_trials_per_seed))
        max_seed_count = max(1, total_trials // min_trials_per_seed)
        active = seeds[:max_seed_count] if seeds[:max_seed_count] else [seeds[0]]
        base = total_trials // len(active)
        rem = total_trials % len(active)
        alloc: List[Tuple[int, int]] = []
        for idx, seed in enumerate(active):
            n = base + (1 if idx < rem else 0)
            if n > 0:
                alloc.append((int(seed), int(n)))
        return alloc or [(int(active[0]), total_trials)]

    @staticmethod
    def _robust_value_from_study(study: optuna.Study) -> float:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not completed:
            return -float("inf")
        vals = np.array(sorted([float(t.value) for t in completed], reverse=True), dtype=np.float64)
        top_k = vals[: max(3, min(12, len(vals)))]
        top_mean = float(np.mean(top_k))
        top_p25 = float(np.percentile(top_k, 25))
        return (0.65 * top_mean) + (0.35 * top_p25)

    def _extract_target_result(self, all_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        target_key = f"{self.target_market}_{self.target_leverage}"
        return next((r for r in all_results if r.get("market") == target_key), None)

    def _score_from_result(self, res: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        diag = {
            "return_pct": 0.0,
            "mdd_pct": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "pf": 0.0,
            "trades": 0.0,
            "days": 0.0,
            "score": -120.0,
        }
        if not res:
            return -120.0, diag

        total_return = self._safe_float(res.get("total_return", 0.0), 0.0)
        mdd = abs(self._safe_float(res.get("mdd", 0.0), 0.0))
        win_rate = self._safe_float(res.get("win_rate", 0.0), 0.0)
        pf = np.clip(self._safe_float(res.get("profit_factor", 0.0), 0.0), 0.0, 5.0)
        trades = int(max(0, self._safe_float(res.get("trades", 0), 0.0)))
        daily_rets = np.asarray(res.get("daily_returns", []), dtype=float)
        daily_rets = np.nan_to_num(daily_rets, nan=0.0, posinf=0.0, neginf=0.0)
        n_days = int(daily_rets.size)

        if trades == 0 or n_days == 0:
            return -120.0, diag

        ann_mean = float(np.mean(daily_rets) * 252.0)
        ann_vol = float(np.std(daily_rets) * np.sqrt(252.0))
        sharpe = ann_mean / ann_vol if ann_vol > 0 else 0.0

        ret_pct = float(np.clip(total_return * 100.0, -100.0, 300.0))
        mdd_pct = float(np.clip(mdd * 100.0, 0.0, 95.0))
        sharpe = float(np.clip(sharpe, -3.0, 6.0))
        win_rate = float(np.clip(win_rate, 0.0, 100.0))

        s_growth = 30.0 * np.tanh(ret_pct / 80.0)
        s_sharpe = 18.0 * np.tanh(sharpe / 2.0)
        s_pf = 10.0 * np.log1p(pf)
        s_wr = 7.0 * np.tanh((win_rate - 45.0) / 20.0)
        score = s_growth + s_sharpe + s_pf + s_wr

        if mdd_pct > 12.0:
            score -= (mdd_pct - 12.0) * 1.5
        if mdd_pct > 20.0:
            score -= (mdd_pct - 20.0) * 2.2
        if mdd_pct > 30.0:
            score -= (mdd_pct - 30.0) * 3.5

        min_trades = max(8, int(n_days / 25))
        if trades < min_trades:
            score -= (min_trades - trades) * 3.0

        if self.target_market == "KOSDAQ":
            if mdd_pct > 25.0:
                score -= (mdd_pct - 25.0) * 2.0
            if win_rate < 42.0:
                score -= (42.0 - win_rate) * 0.8
        else:
            if mdd_pct > 35.0:
                score -= (mdd_pct - 35.0) * 1.0

        score = float(np.clip(self._safe_float(score, -120.0), -200.0, 200.0))
        diag.update(
            {
                "return_pct": ret_pct,
                "mdd_pct": mdd_pct,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "pf": float(pf),
                "trades": float(trades),
                "days": float(n_days),
                "score": score,
            }
        )
        return score, diag

    @staticmethod
    def _set_trial_attrs(trial: optuna.Trial, diag: Dict[str, float]) -> None:
        for k, v in diag.items():
            trial.set_user_attr(k, float(v))

    def objective(self, trial: optuna.Trial) -> float:
        params = self._generate_params(trial)

        if params.get("MA_PERIOD", 0) <= params.get("ENTRY_PERIOD", 0):
            return -150.0
        if params.get("TREND_DIR_TYPE") == "MACD" and params.get("MACD_FAST", 12) >= params.get("MACD_SLOW", 26):
            return -150.0
        if params.get("TS_TRIGGER_ATR", 0.0) >= params.get("TAKE_PROFIT_ATR", 999.0):
            return -150.0
        if params.get("EXIT_TYPE") == "PARABOLIC_SAR" and params.get("SAR_STEP", 0.02) >= params.get("SAR_MAX", 0.2):
            return -150.0

        if self.awfo and self.awfo_plan:
            fold_scores: List[float] = []
            fold_diags: List[Dict[str, float]] = []
            fold_years: List[int] = []
            for fold in self.awfo_plan:
                bt = fold["val_backtester"]
                all_results = bt.run(params, target_market=self.target_market)
                target_res = self._extract_target_result(all_results)
                score, diag = self._score_from_result(target_res)
                fold_scores.append(score)
                fold_diags.append(diag)
                fold_years.append(int(fold["eval_year"]))

            if not fold_scores:
                return -120.0

            scores = np.asarray(fold_scores, dtype=float)
            avg_score = float(np.mean(scores))
            p25_score = float(np.percentile(scores, 25))
            worst_score = float(np.min(scores))
            std_score = float(np.std(scores))
            robust = (0.45 * avg_score) + (0.35 * p25_score) + (0.20 * worst_score) - (0.15 * std_score)
            robust = float(np.clip(self._safe_float(robust, -120.0), -200.0, 200.0))

            agg = {}
            keys = ["return_pct", "mdd_pct", "sharpe", "win_rate", "pf", "trades", "days"]
            for k in keys:
                agg[k] = float(np.mean([d.get(k, 0.0) for d in fold_diags])) if fold_diags else 0.0
            agg["score"] = robust
            self._set_trial_attrs(trial, agg)
            trial.set_user_attr("awfo_years", fold_years)
            trial.set_user_attr("awfo_scores", [float(s) for s in fold_scores])
            trial.set_user_attr("awfo_avg_score", avg_score)
            trial.set_user_attr("awfo_p25_score", p25_score)
            trial.set_user_attr("awfo_worst_score", worst_score)
            trial.set_user_attr("awfo_std_score", std_score)
            return robust

        all_results = self.backtester.run(params, target_market=self.target_market)
        target_res = self._extract_target_result(all_results)
        score, diag = self._score_from_result(target_res)
        self._set_trial_attrs(trial, diag)
        trial.report(float(score), step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    def _build_artifact_path(self) -> Path:
        mode = "awfo" if self.awfo else "single"
        out_dir = Path("results") / "etf"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"optimization_{self.target_market.lower()}_{mode}.json"

    def run_optimization(
        self,
        n_trials: int = 100,
        seeds: Optional[List[int]] = None,
        min_trials_per_seed: int = 40,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        seed_list = [int(s) for s in (seeds or [13])]
        allocations = self._allocate_seed_trials(
            total_trials=int(n_trials),
            seeds=seed_list,
            min_trials_per_seed=int(min_trials_per_seed),
        )

        best_study: Optional[optuna.Study] = None
        best_seed: Optional[int] = None
        best_robust = -float("inf")
        seed_summaries: List[Dict[str, Any]] = []

        logger.info(
            "Starting ETF optimization | market=%s | awfo=%s | folds=%s | trials=%s",
            self.target_market,
            self.awfo,
            len(self.awfo_plan) if self.awfo else 0,
            int(n_trials),
        )

        for seed, seed_trials in allocations:
            sampler = optuna.samplers.TPESampler(
                seed=int(seed),
                n_startup_trials=max(10, int(seed_trials) // 5),
                multivariate=True,
            )
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=max(10, int(seed_trials) // 4),
                n_warmup_steps=1,
                interval_steps=1,
            )
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            study.optimize(
                self.objective,
                n_trials=int(seed_trials),
                n_jobs=int(max(1, n_jobs)),
                show_progress_bar=True,
            )

            robust_value = self._robust_value_from_study(study)
            best_value = float(study.best_value) if len(study.trials) > 0 else -float("inf")
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            seed_summaries.append(
                {
                    "seed": int(seed),
                    "trials": int(seed_trials),
                    "best_value": float(best_value),
                    "robust_value": float(robust_value),
                    "complete_trials": int(completed),
                    "pruned_trials": int(pruned),
                }
            )

            logger.info("[SEED %s] trials=%s | best=%.4f | robust=%.4f", seed, seed_trials, best_value, robust_value)

            if np.isfinite(robust_value) and robust_value > best_robust:
                best_robust = float(robust_value)
                best_study = study
                best_seed = int(seed)

        if best_study is None:
            raise RuntimeError("ETF optimization failed: no completed trials across all seeds.")

        self.study = best_study
        self.last_optimization_meta = {
            "market": self.target_market,
            "target_leverage": self.target_leverage,
            "awfo": bool(self.awfo),
            "awfo_train_years": int(self.awfo_train_years),
            "awfo_plan": [
                {
                    "eval_year": int(f["eval_year"]),
                    "train_period": f"{f['train_start'].date()}~{f['train_end'].date()}",
                    "val_period": f"{f['val_start'].date()}~{f['val_end'].date()}",
                }
                for f in self.awfo_plan
            ],
            "seed_allocations": [{"seed": int(s), "trials": int(t)} for s, t in allocations],
            "best_seed": int(best_seed) if best_seed is not None else None,
            "best_value": float(best_study.best_value),
            "best_robust": float(best_robust),
            "seed_summaries": seed_summaries,
            "best_params": dict(best_study.best_params),
        }

        artifact_path = self._build_artifact_path()
        artifact_payload = {
            "best_params": dict(best_study.best_params),
            "best_value": float(best_study.best_value),
            "best_robust": float(best_robust),
            "meta": self.last_optimization_meta,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            import json

            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(artifact_payload, f, indent=2, ensure_ascii=False)
            self.last_optimization_meta["artifact_path"] = str(artifact_path)
        except Exception as exc:
            logger.warning("Failed to save ETF optimization artifact: %s", exc)

        logger.info(
            "Final best score %.4f (seed=%s, robust=%.4f, awfo=%s)",
            float(best_study.best_value),
            best_seed,
            float(best_robust),
            self.awfo,
        )
        return dict(best_study.best_params)

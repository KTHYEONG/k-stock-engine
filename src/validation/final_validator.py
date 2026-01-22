
import polars as pl
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys

# Project Root 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.base import PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("checker")

@dataclass
class ValidationResult:
    total_rows: int = 0
    missing_counts: Dict[str, int] = field(default_factory=dict)
    zero_counts: Dict[str, int] = field(default_factory=dict) # 0값 비율 체크 추가
    logic_error_count: int = 0
    duplicate_count: int = 0
    suspicious_samples: Optional[pl.DataFrame] = None
    
    @property
    def is_valid(self) -> bool:
        # PBR/PER이 100% 0인 경우는 데이터 수집 실패로 간주
        pbr_fail = self.zero_counts.get("pbr", 0) == self.total_rows and self.total_rows > 0
        return (
            sum(self.missing_counts.values()) == 0 
            and self.logic_error_count == 0 
            and self.duplicate_count == 0
            and not pbr_fail
        )

class DataValidator:
    """수집된 금융 데이터의 무결성을 검증하는 클래스"""
    
    def __init__(self, data_path: Path = PROCESSED_DATA_DIR / "features"):
        self.data_path = data_path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def run_check(self) -> ValidationResult:
        """전체 데이터에 대한 무결성 검사 수행"""
        logger.info(f"Starting data validation on {self.data_path}...")
        
        try:
            # 1. Load Data (Lazy mode)
            # hive_partitioning=False로 파티션 스키마 강제를 비활성화
            try:
                lf = pl.scan_parquet(
                    self.data_path / "**/*.parquet", 
                    low_memory=True,
                    hive_partitioning=False
                )
                total_rows = lf.select(pl.len()).collect().item()
            except Exception as scan_err:
                # scan_parquet 실패 시 개별 파일 읽기 후 diagonal concat
                logger.warning(f"scan_parquet failed, falling back to individual file reading: {scan_err}")
                files = list(self.data_path.glob("**/*.parquet"))
                if not files:
                    logger.error("No parquet files found.")
                    return ValidationResult()
                dfs = [pl.read_parquet(f) for f in files]
                df = pl.concat(dfs, how="diagonal")
                lf = df.lazy()
                total_rows = len(df)
            if total_rows == 0:
                logger.error("Empty dataset.")
                return ValidationResult()
        except Exception as e:
            logger.error(f"Failed to scan data: {e}")
            return ValidationResult()

        # 2. Schema Check (확장됨)
        market_cols = {"ticker", "date", "open", "high", "low", "close", "volume"}
        financial_cols = {"per", "pbr", "roe", "net_income", "total_equity"}
        investor_cols = {"foreign_net_buy", "institution_net_buy", "individual_net_buy"}
        
        all_required = market_cols | financial_cols | investor_cols
        current_cols = set(lf.collect_schema().names())
        missing_cols = all_required - current_cols
        
        if missing_cols:
            logger.error(f"Missing required columns in schema: {missing_cols}")
            # 필수 시세 컬럼이 없으면 중단
            if any(c in market_cols for c in missing_cols):
                return ValidationResult()

        # 3. Data Quality Check (Null & Zeros)
        null_counts = {}
        zero_counts = {}
        
        # 병렬 연산을 위해 한 번에 agg 수행
        agg_exprs = []
        current_schema = lf.collect_schema()

        for col in all_required:
            if col in current_cols:
                agg_exprs.append(pl.col(col).null_count().alias(f"{col}_null"))

                # 수치형 컬럼만 0과 비교 (String 등과 비교 시 에러 방지)
                dtype = current_schema[col]
                if dtype.is_numeric():
                    agg_exprs.append((pl.col(col) == 0).sum().alias(f"{col}_zero"))
                else:
                    agg_exprs.append(pl.lit(0).alias(f"{col}_zero"))
        
        quality_res = lf.select(agg_exprs).collect().to_dicts()[0]
        
        for col in all_required:
            n_cnt = quality_res.get(f"{col}_null", 0)
            z_cnt = quality_res.get(f"{col}_zero", 0)
            if n_cnt > 0: null_counts[col] = n_cnt
            if z_cnt > 0: zero_counts[col] = z_cnt
        
        # 4. Logical Check (High < Low, Low < 0, etc.)
        logic_errors_lf = lf.filter(
            (pl.col("high") < pl.col("low")) |
            (pl.col("volume") < 0) |
            (pl.col("close") <= 0)
        )
        logic_error_count = logic_errors_lf.select(pl.len()).collect().item()
        
        # 5. Duplicates Check
        unique_rows = lf.select(["ticker", "date"]).unique().select(pl.len()).collect().item()
        duplicate_count = total_rows - unique_rows

        # 6. Yearly Distribution Check
        yearly_counts = lf.group_by("year").agg(pl.len().alias("count")).sort("year").collect()
        yearly_dict = {str(row["year"]): row["count"] for row in yearly_counts.to_dicts()}

        # 결과 생성
        suspicious_samples = None
        if logic_error_count > 0:
            suspicious_samples = logic_errors_lf.limit(5).collect()
            
        result = ValidationResult(
            total_rows=total_rows,
            missing_counts=null_counts,
            zero_counts=zero_counts,
            logic_error_count=logic_error_count,
            duplicate_count=duplicate_count,
            suspicious_samples=suspicious_samples
        )
        
        self._print_report(result, yearly_dict)
        return result

    def _print_report(self, res: ValidationResult, yearly_dict: dict = None):
        """검증 결과 출력"""
        status = "[PASS]" if res.is_valid else "[FAIL]"
        
        report = [
            "=" * 60,
            f" [Data Validation Report] Status: {status}",
            "=" * 60,
            f"Total Rows Checked: {res.total_rows:,}",
            f"-" * 60,
        ]
        
        if yearly_dict:
            report.append("Yearly Distribution:")
            for yr, cnt in sorted(yearly_dict.items()):
                report.append(f"  - {yr}: {cnt:,} rows")
            report.append("-" * 60)

        report.extend([
            f"Duplicates: {res.duplicate_count:,} {'(CRITICAL)' if res.duplicate_count > 0 else ''}",
            f"Logic Errors: {res.logic_error_count:,}",
            f"-" * 60,
            "Data Quality (Zero Value Content):"
        ])
        
        # 주요 컬럼별 0값 비율 출력 (데이터 유무 확인용)
        for col in ["pbr", "per", "roe", "foreign_net_buy", "institution_net_buy"]:
            z_cnt = res.zero_counts.get(col, 0)
            z_pct = (z_cnt / res.total_rows * 100) if res.total_rows > 0 else 0
            report.append(f"  - {col:<20}: {z_cnt:,} zeros ({z_pct:.2f}%)")
        
        if res.missing_counts:
            report.append(f"\nMissing Values (Nulls): {res.missing_counts}")
            
        if not res.is_valid:
            if res.zero_counts.get("pbr", 0) == res.total_rows:
                report.append("\n[CRITICAL] PBR 데이터가 모두 0입니다! 재무 데이터 수집/매핑을 확인하세요.")

        if res.suspicious_samples is not None:
            report.append("\n[Sample Logic Errors]")
            report.append(str(res.suspicious_samples))
            
        report_text = "\n".join(report)
        print(report_text, flush=True)
        
        with open("validation_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

if __name__ == "__main__":
    print("DEBUG: Script started", flush=True)
    try:
        validator = DataValidator()
        print("DEBUG: Validator initialized", flush=True)
        validator.run_check()
        print("DEBUG: Validation finished", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = f"DEBUG: Error occurred: {e}\n{traceback.format_exc()}"
        print(msg, flush=True)
        with open("validation_error.txt", "w", encoding="utf-8") as f:
            f.write(msg)

"""OpenDART API 기반 재무 데이터 수집기"""

import os
import logging
import requests
import pandas as pd
import polars as pl
from typing import Optional, Dict, List, Any
import zipfile
import io
import xml.etree.ElementTree as ET
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger("data.collectors.dart")

class OpenDartCollector:
    """
    OpenDART API를 사용하여 기업의 재무제표(계정 정보)를 수집합니다.
    
    기능:
    1. 고유번호(corp_code) 수집: DART는 ticker 대신 고유 corp_code 사용
    2. 단일회사 주요계정 조회: 매출액, 영업이익, 당기순이익, 자산총계, 부채총계, 자본총계, 자본금 등
    3. 분기/반기/사업보고서 자동 식별
    """
    
    BASE_URL = "https://opendart.fss.or.kr/api"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENDART_API_KEY")
        if not self.api_key:
            raise ValueError("OPENDART_API_KEY not found in environment variables.")
            
        self.corp_codes = {} # ticker -> corp_code 매핑
        self.blacklist_path = Path("data/dart_blacklist.json")
        self.blacklist_tickers = self._load_blacklist()
        
        # [Cache] 파일 I/O 반복 제거를 위한 캐시
        self._cached_financial_data = None

    def _load_blacklist(self) -> set:
        """블랙리스트(수집 불가 종목) 로드"""
        if self.blacklist_path.exists():
            try:
                with open(self.blacklist_path, "r") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _add_to_blacklist(self, ticker: str):
        """종목을 블랙리스트에 추가하고 저장"""
        self.blacklist_tickers.add(ticker)
        self.blacklist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.blacklist_path, "w") as f:
            json.dump(list(self.blacklist_tickers), f)
        
    def load_corp_codes(self):
        """DART 고유번호 XML 다운로드 및 파싱 (최초 1회 실행 필요)"""
        url = f"{self.BASE_URL}/corpCode.xml"
        params = {"crtfc_key": self.api_key}
        
        try:
            logger.info("Downloading DART Corp Codes...")
            resp = requests.get(url, params=params)
            
            if resp.status_code != 200:
                logger.error(f"Failed to download corp codes: {resp.status_code}")
                return
                
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                xml_data = zf.read("CORPCODE.xml")
                
            root = ET.fromstring(xml_data)
            
            count = 0
            for child in root.findall("list"):
                stock_code = child.findtext("stock_code")
                corp_code = child.findtext("corp_code")
                
                if stock_code and stock_code.strip():
                    self.corp_codes[stock_code.strip()] = corp_code
                    count += 1
                    
            logger.info(f"Loaded {count} corp codes.")
            
        except Exception as e:
            logger.error(f"Error loading corp codes: {e}")

    def collect_financial_stat_batch(self, tickers: List[str], year: str, reprt_code: str = "11011") -> pd.DataFrame:
        """
        다중 회사의 주요 계정 조회 (최대 100개)
        """
        if not self.corp_codes:
            self.load_corp_codes()
            
        # Ticker -> CorpCode 변환
        target_corp_codes = []
        valid_tickers = []
        
        for t in tickers:
            if t in self.blacklist_tickers:
                continue
            code = self.corp_codes.get(t)
            if code:
                target_corp_codes.append(code)
                valid_tickers.append(t)
            else:
                self._add_to_blacklist(t)

        if not target_corp_codes:
            return pd.DataFrame()
            
        # 쉼표로 구분된 문자열 생성
        corp_code_str = ",".join(target_corp_codes)
        
        url = f"{self.BASE_URL}/fnlttMultiAcnt.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code_str,
            "bsns_year": year,
            "reprt_code": reprt_code
        }
        
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if data['status'] != '000':
                # 에러 발생 시 개별 처리 필요할 수 있음
                logger.debug(f"Batch API Error {data['status']}: {data['message']}")
                return pd.DataFrame()
                
            records = data.get('list', [])
            if not records:
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            
            # corp_code -> ticker 역매핑 생성
            corp_to_ticker = {v: k for k, v in self.corp_codes.items() if k in valid_tickers}
            df['ticker'] = df['corp_code'].map(corp_to_ticker)
            
            # CFS (연결) 우선, OFS (별도) 후순위
            final_rows = []
            
            # 매핑 정의 (IFRS 표준 ID 우선, 실패 시 한글명 보완)
            id_map = {
                "ifrs-full_Assets": "total_assets",
                "ifrs_Assets": "total_assets",
                "ifrs-full_Liabilities": "total_liabilities",
                "ifrs_Liabilities": "total_liabilities",
                "ifrs-full_Equity": "total_equity",
                "ifrs_Equity": "total_equity",
                "ifrs-full_IssuedCapital": "capital", 
                "ifrs_IssuedCapital": "capital",
                "ifrs-full_Revenue": "revenue",
                "ifrs_Revenue": "revenue",
                "dart_OperatingIncomeLoss": "operating_income", 
                "ifrs-full_ProfitLoss": "net_income",
                "ifrs_ProfitLoss": "net_income"
            }
            
            # 한글명 정규화 매핑 (공백 제거된 키)
            name_min_map = {
                "자산총계": "total_assets",
                "부채총계": "total_liabilities",
                "자본총계": "total_equity",
                "자본금": "capital",
                "매출액": "revenue",
                "영업이익": "operating_income",
                "당기순이익": "net_income",
                "당기순이익(손실)": "net_income",
                "연결당기순이익": "net_income"
            }

            def resolve_account(row):
                # 1. ID Check
                acct_id = row.get('account_id', '')
                if acct_id in id_map:
                    return id_map[acct_id]
                
                # 2. Name Check (Normalize: Remove all spaces)
                raw_nm = row.get('account_nm', '')
                if not raw_nm: return None
                
                clean_nm = str(raw_nm).replace(" ", "").strip()
                
                if clean_nm in name_min_map:
                    return name_min_map[clean_nm]
                
                # 3. Keyword Check (보수적 접근)
                if "자산총계" in clean_nm: return "total_assets"
                if "부채총계" in clean_nm: return "total_liabilities"
                if "자본총계" in clean_nm: return "total_equity"
                if "자본금" == clean_nm: return "capital"
                if "매출액" in clean_nm: return "revenue"
                if "영업이익" in clean_nm and "영업이익(손실)" in clean_nm: return "operating_income"
                if "영업이익" == clean_nm: return "operating_income"
                if "당기순이익" in clean_nm: return "net_income"
                
                return None

            target_fields = set(id_map.values())

            for ticker, group in df.groupby('ticker'):
                # 연결(CFS) 우선 선택
                if 'CFS' in group['fs_div'].values:
                    sub_df = group[group['fs_div'] == 'CFS'].copy()
                else:
                    sub_df = group[group['fs_div'] == 'OFS'].copy()
                
                # 계정 식별 (새 컬럼 'target_field')
                sub_df['target_field'] = sub_df.apply(resolve_account, axis=1)
                
                # 유효한 계정이 하나도 없으면 Skip
                valid_rows = sub_df.dropna(subset=['target_field'])
                if valid_rows.empty:
                    continue
                
                # 데이터 추출
                row = {'ticker': ticker, 'year': year, 'reprt_code': reprt_code}
                
                for _, r in valid_rows.iterrows():
                    raw_val = r.get('thstrm_amount', '0')
                    # 콤마 제거 및 공백 처리
                    if isinstance(raw_val, str):
                        raw_val = raw_val.replace(',', '').strip()
                    
                    val = pd.to_numeric(raw_val, errors='coerce')
                    if pd.isna(val): val = 0.0
                    
                    # 이미 값이 있으면 덮어쓰지 않음 (중복 계정 방지)
                    field = r['target_field']
                    if field not in row:
                        row[field] = val
                    
                # 누락된 컬럼 0.0 채우기
                for col in target_fields:
                    if col not in row:
                        row[col] = 0.0
                        
                final_rows.append(row)
                
            return pd.DataFrame(final_rows)

        except Exception as e:
            logger.error(f"Error fetching batch financial data: {e}")
            return pd.DataFrame()

    def collect_financial_stat(self, ticker: str, year: str, reprt_code: str = "11011") -> pd.DataFrame:
        """
        단일 회사의 주요 계정 조회 (연결 우선)
        """

    def get_fallback_financials(self, ticker: str, year: str, reprt_code: str) -> pd.DataFrame:
        """
        데이터 수집 실패 시 필터에서 자동 탈락되도록 '최악의 값'으로 채운 DataFrame 반환
        PBR=999, PER=999, 자본잠식률=100(잠식), 순이익=-999 등으로 설정하여
        Quality Fitler 등에서 걸러지게 유도함.
        """
        fallback_data = {
            "ticker": [ticker],
            "year": [year],
            "reprt_code": [reprt_code],
            "total_assets": [0.0],
            "total_liabilities": [0.0],
            "total_equity": [1.0], # 0으로 나누기 방지
            "capital": [100.0],    # 자본금 > 자본총계 (자본잠식 상태 유도)
            "revenue": [0.0],
            "operating_income": [-999999.0],
            "net_income": [-999999.0]
        }
        return pd.DataFrame(fallback_data)

    def save_financial_data(self, df: pl.DataFrame, path: str = "data/financials.parquet"):
        """수집된 재무 데이터를 로컬에 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_parquet(path)
        logger.info(f"Saved financial data to {path}")

    def load_financial_data(self, path: str = "data/financials.parquet", as_of_date: Optional[str] = None) -> pl.DataFrame:
        """
        로컬에서 재무 데이터 로드
        
        Args:
            as_of_date (str, optional): 'YYYYMMDD' 형식. 이 날짜 기준으로 공시된 최신 데이터만 반환.
                                        None이면 절대적 최신 데이터 반환.
        """
        if not os.path.exists(path):
            logger.warning(f"Financial data file not found: {path}")
            return pl.DataFrame()
            
        # [Cache] 메모리에 있으면 재사용
        if self._cached_financial_data is not None:
            df = self._cached_financial_data
        else:
            df = pl.read_parquet(path)
            self._cached_financial_data = df # 캐싱
            
        if df.is_empty():
            return df

        # Point-in-Time 필터링 (생존 편향 및 룩어헤드 편향 방지)
        if as_of_date:
            target_dt = datetime.strptime(as_of_date, "%Y%m%d")
            
            # 각 보고서별 법정 공시 기한 (근사치) 적용
            # 11013(1Q): 5월 15일, 11012(반기): 8월 15일, 11014(3Q): 11월 15일, 11011(연간): 익년 3월 31일
            def get_disclosure_date(row):
                year = int(row['year'])
                code = row['reprt_code']
                if code == "11013": return datetime(year, 5, 15)
                if code == "11012": return datetime(year, 8, 15)
                if code == "11014": return datetime(year, 11, 15)
                if code == "11011": return datetime(year + 1, 3, 31)
                return datetime(year + 1, 12, 31) # Alway future

            # Polars에서 속도를 위해 연산 (UDF 대신 직접 연산 가능하면 좋으나 복잡하므로 간단히 처리)
            # 여기서는 편의상 필터링 로직 구현
            df = df.with_columns([
                pl.struct(["year", "reprt_code"]).map_elements(
                    lambda x: get_disclosure_date(x), return_dtype=pl.Datetime
                ).alias("disclosure_date")
            ])
            
            # 공시일이 타겟 날짜보다 이전인 데이터만 남김
            df = df.filter(pl.col("disclosure_date") <= target_dt)
            
            if df.is_empty():
                return df

        # 중복 제거: 각 티커별로 가장 최신(공시일 기준) 데이터 1개만 선택
        report_rank = {"11011": 4, "11014": 3, "11012": 2, "11013": 1}
        df = df.with_columns([
            pl.col("reprt_code").map_elements(lambda x: report_rank.get(x, 0), return_dtype=pl.Int32).alias("rank")
        ])
        
        # 년도 내림차순 -> 보고서 순위 내림차순 정렬 후 첫 번째 값 취함
        df = df.sort(["ticker", "year", "rank"], descending=True)
        df = df.unique(subset=["ticker"], keep="first")
        
        return df.drop(["rank", "disclosure_date"]) if "disclosure_date" in df.columns else df.drop("rank")



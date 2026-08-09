"""네이버 금융 크롤링 기반 섹터(업종) 정보 수집기 (AsyncIO)"""

import logging
import asyncio
import aiohttp
import polars as pl
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import time
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

logger = logging.getLogger("data.collectors.naver_finance")
# XML 파싱 경고 무시 (내장 html.parser 사용 시 발생)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class NaverFinanceCollector:
    """
    네이버 금융 업종별 시세 페이지를 크롤링하여 전 종목의 섹터 매핑 정보를 수집합니다.
    
    [전략]
    1. 업종별 리스트 페이지(sise_group.naver?type=upjong)에서 모든 업종의 고유 번호와 이름을 추출.
    2. 각 업종 상세 페이지(sise_group_detail.naver?type=upjong)에 접속하여 소속 종목 코드를 추출.
    3. Ticker -> Sector 매핑 테이블을 반환.
    4. 지수(KOSPI, KOSDAQ) 과거 이력 대량 수집 (Chart Data API 활용).
    """
    
    BASE_URL = "https://finance.naver.com"
    CHART_URL = "https://fchart.stock.naver.com/sise.nhn"
    # 'upjong' 파라미터가 있어야 정상적인 업종 리스트가 출력됨
    INDUSTRY_LIST_URL = f"{BASE_URL}/sise/sise_group.naver?type=upjong"
    
    def __init__(self, concurrency: int = 10):
        """
        Args:
            concurrency: 업종별 상세 페이지 수집 시 동시 요청 수
        """
        self.concurrency = concurrency
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        }
        self._cache_df: Optional[pl.DataFrame] = None

    async def collect_sector_mapping(self) -> pl.DataFrame:
        """
        전 종목의 섹터(업종) 매핑 데이터를 수집합니다.
        
        Returns:
            pl.DataFrame: [ticker, sector]
        """
        logger.info("Starting sector mapping collection via Naver Finance...")
        start_time = time.time()
        
        # SSL 검증 비활성화 (일부 환경 호환성) 및 DNS 캐시 활용
        connector = aiohttp.TCPConnector(ssl=False, limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
            # 1. 업종 리스트 및 URL 추출
            industries = await self._fetch_industry_list(session)
            if not industries:
                logger.error("Failed to fetch industry list from Naver.")
                return pl.DataFrame()
            
            logger.info(f"Found {len(industries)} industries. Starting detail crawl with concurrency={self.concurrency}...")

            # 2. 업종별 상세 페이지 병렬 수집
            semaphore = asyncio.Semaphore(self.concurrency)
            tasks = [
                self._fetch_industry_detail(session, semaphore, ind['name'], ind['url'])
                for ind in industries
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 리스트 평탄화
            flat_results = [item for sublist in results for item in sublist]
            
        if not flat_results:
            logger.warning("No sector mapping data collected from detail pages.")
            return pl.DataFrame()
            
        df = pl.DataFrame(flat_results)
        # 중복 제거 (여러 업종에 중복 노출되는 종목이 있을 수 있음)
        df = df.unique(subset=["ticker"])
        
        logger.info(f"Sector mapping completed. Total {len(df)} stocks in {time.time() - start_time:.2f}s")
        self._cache_df = df
        return df

    async def collect_index_data(self, symbol_name: str = "KOSPI", count: int = 3000) -> pl.DataFrame:
        """
        네이버 금융 차트 데이터를 통해 특정 지수의 과거 이력을 수집합니다.
        
        Args:
            symbol_name: "KOSPI" 또는 "KOSDAQ"
            count: 수집할 데이터 개수 (최근 기준)
            
        Returns:
            pl.DataFrame: [date, ticker, open, high, low, close, volume]
        """
        # 네이버 내부 심볼 매핑
        symbol_map = {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}
        target_symbol = symbol_map.get(symbol_name.upper(), "KOSPI")
        
        params = {
            "symbol": target_symbol,
            "timeframe": "day",
            "count": str(count),
            "requestType": "0"
        }
        
        logger.info(f"Fetching index history for {target_symbol} (count={count})...")
        
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
            try:
                async with session.get(self.CHART_URL, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Naver chart API returned status {response.status}")
                        return pl.DataFrame()
                    xml_text = await response.text()
                    
                return await asyncio.to_thread(self._parse_index_xml, xml_text, symbol_name)
            except Exception as e:
                logger.error(f"Error fetching index history: {e}")
                return pl.DataFrame()

    def _parse_index_xml(self, xml_text: str, symbol_name: str) -> pl.DataFrame:
        """네이버 지수 XML 파싱"""
        data = []
        try:
            # 'xml' 파서 대신 기본 'html.parser' 사용 (의존성 최소화)
            soup = BeautifulSoup(xml_text, 'html.parser')
            items = soup.find_all("item")
            
            for item in items:
                # data="날짜|시가|고가|저가|종가|거래량"
                vals = item.get("data").split("|")
                if len(vals) < 6: continue
                
                data.append({
                    "date": vals[0], # YYYYMMDD
                    "ticker": symbol_name.upper(),
                    "open": float(vals[1]),
                    "high": float(vals[2]),
                    "low": float(vals[3]),
                    "close": float(vals[4]),
                    "volume": float(vals[5])
                })
        except Exception as e:
            logger.error(f"Error parsing index XML: {e}")
            
        if not data:
            return pl.DataFrame()
            
        df = pl.DataFrame(data)
        # 타입 변환
        df = df.with_columns([
            pl.col("date").str.strptime(pl.Datetime, "%Y%m%d"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ])
        return df

    async def _fetch_industry_list(self, session: aiohttp.ClientSession) -> List[Dict]:
        """업종 리스트 페이지에서 업종명과 상세 URL 추출"""
        try:
            async with session.get(self.INDUSTRY_LIST_URL) as response:
                if response.status != 200:
                    logger.error(f"Industry list page returned status {response.status}")
                    return []
                content = await response.read()
                # 네이버 금융은 EUC-KR 인코딩 사용
                html = content.decode('euc-kr', errors='replace')
                
            soup = BeautifulSoup(html, 'html.parser')
            # 업종 테이블은 보통 class="type_1"
            table = soup.select_one("table.type_1")
            if not table:
                logger.error("Could not find table.type_1 in industry list page.")
                return []
                
            industries = []
            links = table.select("td > a")
            for a in links:
                href = a.get('href')
                name = a.text.strip()
                # 업종 상세 페이지 링크 패턴 확인 (상대 경로이므로 BASE_URL 결합)
                if href and '/sise/sise_group_detail.naver' in href:
                    full_url = href if href.startswith('http') else f"{self.BASE_URL}{href}"
                    industries.append({
                        "name": name,
                        "url": full_url
                    })
            return industries
        except Exception as e:
            logger.error(f"Error fetching industry list: {e}")
            return []

    async def _fetch_industry_detail(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, sector_name: str, url: str) -> List[Dict]:
        """특정 업종 상세 페이지에서 종목 코드 추출"""
        async with semaphore:
            try:
                # 인간다움을 모방하기 위한 미세 지연 (필요 시 주석 해제)
                # await asyncio.sleep(0.05)
                async with session.get(url) as response:
                    if response.status != 200:
                        return []
                    content = await response.read()
                    html = content.decode('euc-kr', errors='replace')
                    
                return await asyncio.to_thread(self._parse_industry_detail, html, sector_name)
            except Exception as e:
                logger.warning(f"Error fetching detail for sector '{sector_name}': {e}")
                return []

    def _parse_industry_detail(self, html: str, sector_name: str) -> List[Dict]:
        """업종 상세 페이지 HTML 파싱 (Ticker 추출)"""
        items = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # 업종 상세 페이지의 종목 리스트 테이블은 class="type_5"
            table = soup.select_one("table.type_5")
            if not table:
                return []
                
            # 종목 링크 추출 (td.name div.name_area a 구조)
            # a[href*="code="]를 사용하여 티커가 포함된 링크만 선별
            links = table.select("td.name div.name_area a")
            if not links:
                # 구조가 약간 다를 경우를 대비한 Fallback (tltle 등)
                links = table.select("a[href*='code=']")

            for a in links:
                href = a.get('href', '')
                if 'code=' in href:
                    ticker = href.split('code=')[-1].split('&')[0] # 파라미터가 더 있을 경우 대비
                    if ticker:
                        items.append({
                            "ticker": ticker,
                            "sector": sector_name
                        })
        except Exception:
            pass
        return items

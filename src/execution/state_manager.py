import sqlite3
from pathlib import Path
from config.base import BASE_DIR

class StateManager:
    """주문 및 포지션 상태 관리를 위한 SQLite 래퍼"""
    
    def __init__(self, db_name: str = "trading_state.db"):
        self.db_path = BASE_DIR / db_name
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    entry_date TEXT,
                    entry_price REAL,
                    quantity INTEGER,
                    status TEXT,
                    last_updated TEXT
                )
            ''')
            conn.commit()
            
    def update_position(self, ticker, price, quantity, status="OPEN"):
        """포지션 상태 업데이트"""
        # 구현 예정
        pass
        
    def get_open_positions(self):
        """현재 보유 포지션 조회"""
        # 구현 예정
        pass

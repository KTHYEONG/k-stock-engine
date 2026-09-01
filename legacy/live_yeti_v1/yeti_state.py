import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from legacy.config.base import BASE_DIR


@dataclass
class PositionState:
    ticker: str
    entry_date: str
    entry_price: float
    quantity: int
    max_price: float
    hold_days: int
    status: str = "OPEN"
    last_updated: str = ""


class StateManager:
    """SQLite-based local state for live trading positions and orders."""

    def __init__(self, db_name: str = "data/trading_state.db"):
        self.db_path = BASE_DIR / db_name
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    max_price REAL NOT NULL,
                    hold_days INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    last_updated TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS order_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    order_no TEXT,
                    status TEXT NOT NULL,
                    reason TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_open_positions(self) -> Dict[str, PositionState]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'OPEN'"
            ).fetchall()
        out: Dict[str, PositionState] = {}
        for row in rows:
            out[row["ticker"]] = PositionState(
                ticker=row["ticker"],
                entry_date=row["entry_date"],
                entry_price=float(row["entry_price"]),
                quantity=int(row["quantity"]),
                max_price=float(row["max_price"]),
                hold_days=int(row["hold_days"]),
                status=row["status"],
                last_updated=row["last_updated"],
            )
        return out

    def upsert_position(self, state: PositionState) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO positions (
                    ticker, entry_date, entry_price, quantity,
                    max_price, hold_days, status, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    entry_date = excluded.entry_date,
                    entry_price = excluded.entry_price,
                    quantity = excluded.quantity,
                    max_price = excluded.max_price,
                    hold_days = excluded.hold_days,
                    status = excluded.status,
                    last_updated = excluded.last_updated
                """,
                (
                    state.ticker,
                    state.entry_date,
                    float(state.entry_price),
                    int(state.quantity),
                    float(state.max_price),
                    int(state.hold_days),
                    state.status,
                    now if not state.last_updated else state.last_updated,
                ),
            )
            conn.commit()

    def remove_position(self, ticker: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
            conn.commit()

    def set_max_price(self, ticker: str, price: float) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET max_price = CASE WHEN max_price < ? THEN ? ELSE max_price END,
                    last_updated = ?
                WHERE ticker = ?
                """,
                (float(price), float(price), now, ticker),
            )
            conn.commit()

    def mark_cycle(self, cycle_date: str) -> bool:
        """Increment hold_days once per signal date. Returns True when updated."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT v FROM meta WHERE k = 'last_cycle_date'"
            ).fetchone()
            if row and row["v"] == cycle_date:
                return False
            conn.execute(
                "UPDATE positions SET hold_days = hold_days + 1 WHERE status = 'OPEN'"
            )
            conn.execute(
                """
                INSERT INTO meta (k, v) VALUES ('last_cycle_date', ?)
                ON CONFLICT(k) DO UPDATE SET v = excluded.v
                """,
                (cycle_date,),
            )
            conn.commit()
        return True

    def log_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: Optional[float],
        status: str,
        order_no: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO order_logs (
                    created_at, ticker, side, quantity, price,
                    order_no, status, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    ticker,
                    side,
                    int(quantity),
                    None if price is None else float(price),
                    order_no,
                    status,
                    reason,
                ),
            )
            conn.commit()

    def sync_with_broker_positions(
        self,
        broker_positions: Dict[str, int],
        price_map: Optional[Dict[str, float]] = None,
        as_of: Optional[str] = None,
    ) -> None:
        """
        Align local DB with broker holdings.
        - Adds missing positions.
        - Updates quantity for existing positions.
        - Removes stale local positions not held in broker.
        """
        price_map = price_map or {}
        as_of = as_of or datetime.now().strftime("%Y%m%d")
        now = datetime.now().isoformat(timespec="seconds")
        local = self.get_open_positions()

        with self._connect() as conn:
            for ticker, qty in broker_positions.items():
                if qty <= 0:
                    continue
                if ticker in local:
                    pos = local[ticker]
                    max_price = max(pos.max_price, float(price_map.get(ticker, pos.max_price)))
                    conn.execute(
                        """
                        UPDATE positions
                        SET quantity = ?, max_price = ?, status = 'OPEN', last_updated = ?
                        WHERE ticker = ?
                        """,
                        (int(qty), float(max_price), now, ticker),
                    )
                else:
                    px = float(price_map.get(ticker, 0.0))
                    conn.execute(
                        """
                        INSERT INTO positions (
                            ticker, entry_date, entry_price, quantity,
                            max_price, hold_days, status, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?)
                        """,
                        (ticker, as_of, px, int(qty), px, 1, now),
                    )

            to_delete = [t for t in local.keys() if broker_positions.get(t, 0) <= 0]
            for ticker in to_delete:
                conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
            conn.commit()

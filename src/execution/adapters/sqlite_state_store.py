"""SQLite-backed order state store with auditable transitions."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from src.core.instruments import AssetKind
from src.execution.domain.orders import OrderSide, OrderState, OrderStateRecord

_ALLOWED: dict[tuple[OrderState, OrderState], bool] = {
    (OrderState.NEW, OrderState.VALIDATED): True,
    (OrderState.VALIDATED, OrderState.SUBMITTED): True,
    (OrderState.SUBMITTED, OrderState.FILLED): True,
    (OrderState.SUBMITTED, OrderState.REJECTED): True,
    (OrderState.SUBMITTED, OrderState.CANCELED): True,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    asset_kind TEXT NOT NULL,
    side TEXT NOT NULL,
    state TEXT NOT NULL,
    submitted_quantity REAL NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0.0,
    filled_price REAL,
    reason TEXT NOT NULL DEFAULT '',
    seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS order_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    asset_kind TEXT NOT NULL,
    side TEXT NOT NULL,
    state TEXT NOT NULL,
    submitted_quantity REAL NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0.0,
    filled_price REAL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY
);
CREATE INDEX IF NOT EXISTS idx_order_history_order_id ON order_history (order_id, id);
"""


def _encode(record: OrderStateRecord) -> tuple[object, ...]:
    return (
        record.order_id,
        record.intent_id,
        record.instrument_id,
        record.asset_kind.value,
        record.side.value,
        record.state.value,
        float(record.submitted_quantity),
        float(record.filled_quantity),
        record.filled_price,
        record.reason,
    )


def _decode(row: sqlite3.Row) -> OrderStateRecord:
    price = row["filled_price"]
    return OrderStateRecord(
        order_id=str(row["order_id"]),
        intent_id=str(row["intent_id"]),
        instrument_id=str(row["instrument_id"]),
        asset_kind=AssetKind(str(row["asset_kind"])),
        side=OrderSide(str(row["side"])),
        state=OrderState(str(row["state"])),
        submitted_quantity=float(row["submitted_quantity"]),
        filled_quantity=float(row["filled_quantity"]),
        filled_price=None if price is None else float(price),
        reason=str(row["reason"] or ""),
    )


class SqliteStateStore:
    """Persistent ``StateStorePort`` with transition guards and history."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def save(self, record: OrderStateRecord) -> None:
        with self._conn:
            cur = self._conn.execute("SELECT * FROM orders WHERE order_id = ?", (record.order_id,))
            existing = cur.fetchone()
            if existing is None:
                seq_row = self._conn.execute("SELECT COALESCE(MAX(seq), -1) AS m FROM orders").fetchone()
                seq = int(seq_row["m"]) + 1
                self._conn.execute(
                    "INSERT INTO orders (order_id, intent_id, instrument_id, asset_kind, side,"
                    " state, submitted_quantity, filled_quantity, filled_price, reason, seq)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*_encode(record), seq),
                )
                return
            prev = _decode(existing)
            self._check_transition(prev, record)
            self._conn.execute(
                "INSERT INTO order_history (order_id, intent_id, instrument_id, asset_kind, side,"
                " state, submitted_quantity, filled_quantity, filled_price, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _encode(prev),
            )
            self._conn.execute(
                "UPDATE orders SET intent_id=?, instrument_id=?, asset_kind=?, side=?, state=?,"
                " submitted_quantity=?, filled_quantity=?, filled_price=?, reason=? WHERE order_id=?",
                (
                    record.intent_id,
                    record.instrument_id,
                    record.asset_kind.value,
                    record.side.value,
                    record.state.value,
                    float(record.submitted_quantity),
                    float(record.filled_quantity),
                    record.filled_price,
                    record.reason,
                    record.order_id,
                ),
            )

    @staticmethod
    def _check_transition(prev: OrderStateRecord, nxt: OrderStateRecord) -> None:
        if (prev.state, nxt.state) in _ALLOWED:
            return
        if prev.state is OrderState.SUBMITTED and nxt.state is OrderState.SUBMITTED:
            if float(nxt.filled_quantity) >= float(prev.filled_quantity):
                return
            raise ValueError("filled_quantity must not decrease on SUBMITTED progress")
        raise ValueError(f"illegal transition {prev.state.value} -> {nxt.state.value}")

    def get(self, order_id: str) -> OrderStateRecord | None:
        cur = self._conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        row = cur.fetchone()
        return None if row is None else _decode(row)

    def is_duplicate_intent(self, idempotency_key: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM idempotency_keys WHERE idempotency_key = ?", (idempotency_key,)
        )
        return cur.fetchone() is not None

    def mark_intent(self, idempotency_key: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys (idempotency_key) VALUES (?)",
                (idempotency_key,),
            )

    def records(self, *, session: date) -> tuple[OrderStateRecord, ...]:
        """Order states whose order id belongs to the session, in insertion order."""
        token = session.isoformat()
        cur = self._conn.execute("SELECT * FROM orders ORDER BY seq ASC")
        out: list[OrderStateRecord] = []
        for row in cur.fetchall():
            record = _decode(row)
            if token in record.order_id:
                out.append(record)
        return tuple(out)

    def history(self, order_id: str) -> tuple[OrderStateRecord, ...]:
        cur = self._conn.execute(
            "SELECT * FROM order_history WHERE order_id = ? ORDER BY id ASC", (order_id,)
        )
        return tuple(_decode(row) for row in cur.fetchall())

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Candle:
    begin: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    value: float | None

    def as_dict(self) -> dict:
        return {
            "begin": self.begin.strftime("%Y-%m-%d %H:%M") if isinstance(self.begin, datetime) else str(self.begin),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "value": self.value,
        }

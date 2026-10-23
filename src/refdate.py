"""
refdate.py
==========
Single source of truth for "today" inside the ranking pipeline.

Why this exists: recency signals ("days since last active") used to call
date.today(), so the same candidate file produced different scores depending
on the day the code was run. rank.py now sets the reference date ONCE from the
data itself (the latest last_active_date in the candidate file), which makes
every run on the same data deterministic.

Outside rank.py (e.g. unit tests that never call set_reference_date_from),
today() falls back to the real current date.
"""

from datetime import date, datetime
from typing import Iterable

_reference: date | None = None


def set_reference_date_from(date_strings: Iterable[str]) -> date | None:
    """Set the reference date to the latest valid YYYY-MM-DD in date_strings."""
    global _reference
    best: date | None = None
    for s in date_strings:
        try:
            d = datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if best is None or d > best:
            best = d
    _reference = best
    return best


def today() -> date:
    """Reference date if one was set, otherwise the real current date."""
    return _reference or date.today()

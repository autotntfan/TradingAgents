"""
finmind_utils.py
─────────────────────────────────────────────────────────
Utility functions for retrieving **daily OHLCV** data of
Taiwan-listed equities (TWSE / TPEx) via the FinMind API.

You can set the API key (i.e. `token`) as an environment variable.
```bash
export FINMIND_TOKEN="your_token_here"
```
Test if the token is set:
```python
from tradingagents.dataflows.finmind_utils import get_tw_stock_data
print(get_tw_stock_data("2330", "2025-07-01", "2025-07-05").head())
``` 

Only one public function is exposed:

    get_tw_stock_data(stock_id, start_date, end_date, token=None) -> pd.DataFrame

The returned DataFrame follows the same column order as yfinance:
['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

FinMind docs: https://finmind.github.io/
"""

from __future__ import annotations
import os
from functools import lru_cache
from datetime import date, datetime, timedelta
from .utils import parse_ticker
import pandas as pd
import requests

_BASE = "https://api.finmindtrade.com/api/v4/data"

def _daterange(start: date, end: date):
    """Yield each day from start to end (inclusive)."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def _finmind_query(dataset: str, **params) -> pd.DataFrame:
    """
    Query the FinMind REST endpoint and converts the JSON payload into a Pandas DataFrame.

    Raises
    ------
    RuntimeError
        If FinMind returns a non-200 status.
    """
    token = params.pop("token", "") or os.getenv("FINMIND_TOKEN", "")
    resp = requests.get(_BASE, params={"dataset": dataset, "token": token, **params}, timeout=10).json()
    print(resp)
    if resp["status"] != 200:
        raise RuntimeError(f"FinMind error: {resp.get('msg')}")
    return pd.DataFrame(resp["data"])


@lru_cache(maxsize=128)
def get_tw_stock_data(
    stock_id: str,
    start_date: str,
    end_date: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV for a Taiwan stock.

    Parameters
    ----------
    stock_id : str
        Four-digit ticker without the '.TW' suffix (e.g., ``"2330"``).
    start_date : str
        ISO date string ``"YYYY-MM-DD"``.
    end_date : str | None
        ISO date string; defaults to today.
    token : str | None
        FinMind API token.  If omitted, the function falls back
        to the environment variable ``FINMIND_TOKEN``.

    Returns
    -------
    pd.DataFrame
        Index: timezone-aware ``DatetimeIndex`` (Asia/Taipei)  
        Columns: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'

    Raises
    ------
    ValueError
        If FinMind returns an empty dataset.
    """
    is_tw, base_stock_id = parse_ticker(stock_id)
    assert is_tw, "get_tw_stock_news only accepts .TW tickers"
    
    if end_date is None:
        end_date = date.today().isoformat()

    df = _finmind_query(
        "TaiwanStockPrice",
        data_id=base_stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {base_stock_id} between " f"{start_date} and {end_date}"
        )

    # Harmonise column names with yfinance
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume",
        }
    )

    # FinMind has no adjusted close; use the raw close
    df["Adj Close"] = df["Close"]

    # Set index and localise to Asia/Taipei
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize("Asia/Taipei")
    df = df.set_index("Date")

    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

def get_tw_stock_news(
    stock_id: str,
    start_date: str,
    end_date: str,
    limit: int = 30,
    token: str | None = None,
) -> list[dict]:
    """
    Fetch TaiwanStockNews day-by-day and return unified schema
    (link/title/snippet/date/source) identical to googlenews_utils.getNewsData().
    """
    is_tw, base_stock_id = parse_ticker(stock_id)
    assert is_tw, "get_tw_stock_news only accepts .TW tickers"
    
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    all_rows: list[dict] = []
    for d in _daterange(start, end):
        # No end_date, due to `FinMind error: the dataset TaiwanStockNews size is too large, we only send one day data, so end_date parameter need be none.`
        df = _finmind_query(
            "TaiwanStockNews",
            data_id=base_stock_id,
            start_date=d.isoformat(),   
            token=token,
        ) 
        '''
        return dict example: 
        {'data': 
            [{
                'date': '2025-07-01 01:03:00',
                'stock_id': '2330',
                'link': 'https://finance.ettoday.net/news/2987963',
                'source': 'ETtoday財經雲',
                'title': '快訊/台積電漲20元至1080\u3000台股大漲逾240點 - ETtoday財經雲'
            }, ....
            ]
        } 
        '''
        if not df.empty:
            df = df.fillna("")
            all_rows.extend(
                {
                    "link": row["link"],
                    "title": row["title"],
                    "snippet": row.get("content", ""),
                    "date": row["date"],
                    "source": row.get("source", "FinMind"),
                }
                for _, row in df.iterrows()
            )
        if len(all_rows) >= limit:
            break

    all_rows = sorted(all_rows, key=lambda x: x["date"], reverse=True)[:limit]
    return all_rows
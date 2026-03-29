"""
Apex-Oracle — Multi-Stock Data Fetcher
Fetches OHLCV data for all stocks in the universe via yfinance.
Stores per-stock data in data/{ticker}/ folders.
"""
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from config import STOCK_UNIVERSE, DATA_DIR, HISTORICAL_START_DATE


def ensure_dirs():
    """Create data directories for each stock."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in STOCK_UNIVERSE:
        safe_name = ticker.replace(".", "_")
        (DATA_DIR / safe_name).mkdir(exist_ok=True)


def fetch_stock_data(ticker: str, start_date: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single stock.
    If data already exists, fetches only new data since last date.
    """
    safe_name = ticker.replace(".", "_")
    csv_path = DATA_DIR / safe_name / "raw_data.csv"

    if start_date is None:
        start_date = HISTORICAL_START_DATE

    # If file exists, only fetch new data
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
        if not existing.empty:
            last_date = existing["Date"].max()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"[{ticker}] Fetching new data from {start_date}")
    else:
        existing = pd.DataFrame()
        logger.info(f"[{ticker}] Fetching full history from {start_date}")

    try:
        data = yf.download(ticker, start=start_date, progress=False)

        if data.empty:
            logger.info(f"[{ticker}] No new data available")
            return existing if not existing.empty else pd.DataFrame()

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

        # Append to existing data
        if not existing.empty:
            data = pd.concat([existing, data], ignore_index=True)
            data = data.drop_duplicates(subset=["Date"], keep="last")
            data = data.sort_values("Date").reset_index(drop=True)

        # Save
        data.to_csv(csv_path, index=False)
        logger.info(f"[{ticker}] Saved {len(data)} rows to {csv_path}")
        return data

    except Exception as e:
        logger.error(f"[{ticker}] Error fetching data: {e}")
        if not existing.empty:
            return existing
        return pd.DataFrame()


def fetch_news_headlines(ticker: str) -> list:
    """
    Fetch recent news headlines for a stock via RSS/GNews.
    Returns list of dicts with 'title', 'published', 'source'.
    """
    import feedparser

    company_name = STOCK_UNIVERSE.get(ticker, ticker.split(".")[0])
    search_query = company_name.replace(" ", "+")

    # Use Google News RSS feed
    rss_url = f"https://news.google.com/rss/search?q={search_query}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(rss_url)
        headlines = []
        for entry in feed.entries[:20]:  # Last 20 headlines
            headlines.append({
                "title": entry.get("title", ""),
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
            })

        # Save headlines
        safe_name = ticker.replace(".", "_")
        news_path = DATA_DIR / safe_name / "news_headlines.json"
        with open(news_path, "w", encoding="utf-8") as f:
            json.dump(headlines, f, indent=2, ensure_ascii=False)

        logger.info(f"[{ticker}] Fetched {len(headlines)} news headlines")
        return headlines

    except Exception as e:
        logger.error(f"[{ticker}] Error fetching news: {e}")
        return []

def fetch_fundamental_data(ticker: str) -> dict:
    """Fetch fundamental data (P/E ratio, earnings growth) via yfinance."""
    safe_name = ticker.replace(".", "_")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            "pe_ratio": info.get("trailingPE", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
            "quarterly_revenue_growth": info.get("revenueGrowth", 0),
            "market_cap": info.get("marketCap", 0),
        }
        with open(DATA_DIR / safe_name / "fundamentals.json", "w") as f:
            json.dump(fundamentals, f, indent=2)
        logger.info(f"[{ticker}] Fetched fundamental data")
        return fundamentals
    except Exception as e:
        logger.error(f"[{ticker}] Error fetching fundamentals: {e}")
        return {}


def fetch_all_stocks():
    """Fetch data + news for all stocks in the universe."""
    ensure_dirs()
    results = {}

    for ticker in STOCK_UNIVERSE:
        logger.info(f"{'='*50}")
        logger.info(f"Fetching: {ticker} ({STOCK_UNIVERSE[ticker]})")

        # Fetch price data
        df = fetch_stock_data(ticker)
        results[ticker] = {
            "rows": len(df) if not df.empty else 0,
            "status": "ok" if not df.empty else "empty"
        }

        # Fetch news
        headlines = fetch_news_headlines(ticker)
        results[ticker]["headlines"] = len(headlines)

        # Fetch fundamentals
        fundamentals = fetch_fundamental_data(ticker)
        results[ticker]["fundamentals"] = "ok" if fundamentals else "failed"

    # Log fetch summary
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    log_path = DATA_DIR / "fetch_log.json"
    logs = []
    if log_path.exists():
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(log_entry)
    # Keep last 100 log entries
    logs = logs[-100:]

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    logger.info(f"Fetch cycle complete. Results: {results}")
    return results


if __name__ == "__main__":
    fetch_all_stocks()
"""
Apex-Oracle — Sentiment Analyzer
Scores news headlines using VADER (lightweight) or FinBERT (accurate).
Outputs daily sentiment scores per stock.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger

from config import DATA_DIR, STOCK_UNIVERSE


def analyze_with_vader(headlines: list) -> dict:
    """
    Score headlines using VADER sentiment analyzer (fast, no GPU needed).
    Returns aggregate sentiment scores.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = []

    for item in headlines:
        title = item.get("title", "")
        if not title:
            continue
        score = analyzer.polarity_scores(title)
        scores.append({
            "title": title,
            "compound": score["compound"],
            "positive": score["pos"],
            "negative": score["neg"],
            "neutral": score["neu"],
        })

    if not scores:
        return {
            "sentiment_score": 0.0,
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_neutral": 1.0,
            "num_headlines": 0,
        }

    avg_compound = np.mean([s["compound"] for s in scores])
    avg_pos = np.mean([s["positive"] for s in scores])
    avg_neg = np.mean([s["negative"] for s in scores])
    avg_neu = np.mean([s["neutral"] for s in scores])

    return {
        "sentiment_score": float(avg_compound),
        "sentiment_positive": float(avg_pos),
        "sentiment_negative": float(avg_neg),
        "sentiment_neutral": float(avg_neu),
        "num_headlines": len(scores),
    }


def analyze_with_finbert(headlines: list) -> dict:
    """
    Score headlines using FinBERT (finance-specific BERT model).
    More accurate but requires ~440MB model download.
    Falls back to VADER if unavailable.
    """
    try:
        from transformers import pipeline

        finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU
        )

        scores = []
        for item in headlines:
            title = item.get("title", "")
            if not title:
                continue
            try:
                result = finbert(title[:512])[0]  # Truncate to max length
                label = result["label"].lower()
                score_val = result["score"]

                scores.append({
                    "title": title,
                    "label": label,
                    "confidence": score_val,
                    "positive": score_val if label == "positive" else 0.0,
                    "negative": score_val if label == "negative" else 0.0,
                    "neutral": score_val if label == "neutral" else 0.0,
                })
            except Exception:
                continue

        if not scores:
            return analyze_with_vader(headlines)

        # Map labels to a compound score: positive=+1, neutral=0, negative=-1
        compound_scores = []
        for s in scores:
            if s["label"] == "positive":
                compound_scores.append(s["confidence"])
            elif s["label"] == "negative":
                compound_scores.append(-s["confidence"])
            else:
                compound_scores.append(0.0)

        return {
            "sentiment_score": float(np.mean(compound_scores)),
            "sentiment_positive": float(np.mean([s["positive"] for s in scores])),
            "sentiment_negative": float(np.mean([s["negative"] for s in scores])),
            "sentiment_neutral": float(np.mean([s["neutral"] for s in scores])),
            "num_headlines": len(scores),
        }

    except ImportError:
        logger.warning("FinBERT not available, falling back to VADER")
        return analyze_with_vader(headlines)
    except Exception as e:
        logger.warning(f"FinBERT error: {e}, falling back to VADER")
        return analyze_with_vader(headlines)


def analyze_sentiment(ticker: str, use_finbert: bool = False) -> dict:
    """
    Analyze sentiment for a single stock's news headlines.
    Saves daily sentiment scores to CSV for feature engineering.
    """
    safe_name = ticker.replace(".", "_")
    news_path = DATA_DIR / safe_name / "news_headlines.json"

    if not news_path.exists():
        logger.warning(f"[{ticker}] No news headlines found")
        return {"sentiment_score": 0.0}

    with open(news_path, "r", encoding="utf-8") as f:
        headlines = json.load(f)

    if not headlines:
        return {"sentiment_score": 0.0}

    # Analyze
    if use_finbert:
        scores = analyze_with_finbert(headlines)
    else:
        scores = analyze_with_vader(headlines)

    # Save daily sentiment to CSV (append)
    today = datetime.now().strftime("%Y-%m-%d")
    sentiment_row = {
        "Date": today,
        **scores
    }

    sentiment_path = DATA_DIR / safe_name / "sentiment_scores.csv"
    if sentiment_path.exists():
        df = pd.read_csv(sentiment_path, parse_dates=["Date"])
        # Update today's row or append
        df = df[df["Date"].astype(str) != today]
        df = pd.concat([df, pd.DataFrame([sentiment_row])], ignore_index=True)
    else:
        df = pd.DataFrame([sentiment_row])

    df.to_csv(sentiment_path, index=False)
    logger.info(f"[{ticker}] Sentiment: {scores['sentiment_score']:.3f} "
                f"(from {scores['num_headlines']} headlines)")

    return scores


def analyze_all_stocks(use_finbert: bool = False):
    """Analyze sentiment for all stocks in the universe."""
    results = {}
    for ticker in STOCK_UNIVERSE:
        scores = analyze_sentiment(ticker, use_finbert=use_finbert)
        results[ticker] = scores

    return results


if __name__ == "__main__":
    analyze_all_stocks(use_finbert=False)

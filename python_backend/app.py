"""
Apex-Oracle — FastAPI Backend
REST API + WebSocket for real-time prediction delivery.
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from config import DATA_DIR, STOCK_UNIVERSE, MODELS_DIR
from scheduler import start_scheduler, stop_scheduler, scheduler_instance, connected_clients


# ── App Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on app startup, stop on shutdown."""
    logger.info("Apex-Oracle starting up...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("Apex-Oracle shut down.")


app = FastAPI(
    title="Apex-Oracle API",
    description="Advanced Predictive Multi-Signal Engine with Explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Helper ────────────────────────────────────────────────
def _load_json(path: Path, default=None):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default or {}
    return default or {}


# ── Routes ────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve frontend."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/api/stocks")
async def get_stocks():
    """List all tracked stocks."""
    return {
        "stocks": [
            {"ticker": t, "name": n} for t, n in STOCK_UNIVERSE.items()
        ]
    }


@app.get("/api/prediction/{ticker}")
async def get_prediction(ticker: str):
    """Get latest prediction for a stock."""
    safe_name = ticker.replace(".", "_")
    pred = _load_json(DATA_DIR / safe_name / "latest_prediction.json")
    if not pred:
        return {"error": f"No prediction available for {ticker}. Run the pipeline first."}
    return pred


@app.get("/api/regime/{ticker}")
async def get_regime(ticker: str):
    """Get current market regime for a stock."""
    safe_name = ticker.replace(".", "_")
    pred = _load_json(DATA_DIR / safe_name / "latest_prediction.json")
    if pred and "regime" in pred:
        return pred["regime"]

    # Try regime history
    import pandas as pd
    regime_path = DATA_DIR / safe_name / "regime_history.csv"
    if regime_path.exists():
        df = pd.read_csv(regime_path)
        if not df.empty:
            return {
                "current": df["regime"].iloc[-1],
                "history": df.tail(30).to_dict(orient="records"),
            }
    return {"current": "UNKNOWN", "message": "No regime data available"}


@app.get("/api/historical/{ticker}")
async def get_historical(ticker: str, limit: int = 200):
    """Get historical price data with features."""
    import pandas as pd
    safe_name = ticker.replace(".", "_")
    features_path = DATA_DIR / safe_name / "features.csv"

    if features_path.exists():
        df = pd.read_csv(features_path)
        df = df.tail(limit)
        return {
            "ticker": ticker,
            "data": df[["Date", "Close", "High", "Low", "Open", "Volume"]].to_dict(orient="records"),
            "count": len(df),
        }

    raw_path = DATA_DIR / safe_name / "raw_data.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        df = df.tail(limit)
        return {
            "ticker": ticker,
            "data": df.to_dict(orient="records"),
            "count": len(df),
        }

    return {"error": f"No data for {ticker}"}

@app.get("/api/results")
async def get_results():
    """Get all models' backtesting results."""
    results_path = DATA_DIR / "results.json"
    results = _load_json(results_path)
    if not results:
        return {"error": "No backtesting results available. Run evaluation first."}
    return results


@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    """Get recent alerts."""
    from alert_system import get_recent_alerts
    return {"alerts": get_recent_alerts(limit)}


@app.get("/api/models/{ticker}")
async def get_models(ticker: str):
    """Get all 8 models' info for a stock."""
    safe_name = ticker.replace(".", "_")
    pred = _load_json(DATA_DIR / safe_name / "latest_prediction.json")
    if pred and "models" in pred:
        return {
            "regression": pred["models"]["regression"],
            "classification": pred["models"]["classification"],
            "weights": pred["models"].get("regression_weights", {}),
            "agreement": pred.get("agreement", {}),
        }
    return {"error": "No model data available"}


@app.get("/api/sentiment/{ticker}")
async def get_sentiment(ticker: str):
    """Get sentiment data for a stock."""
    import pandas as pd
    safe_name = ticker.replace(".", "_")
    sentiment_path = DATA_DIR / safe_name / "sentiment_scores.csv"

    if sentiment_path.exists():
        df = pd.read_csv(sentiment_path)
        return {
            "ticker": ticker,
            "latest": df.iloc[-1].to_dict() if not df.empty else {},
            "history": df.tail(30).to_dict(orient="records"),
        }

    # Try headlines
    news = _load_json(DATA_DIR / safe_name / "news_headlines.json", [])
    return {"ticker": ticker, "headlines": news[:10], "latest": {}}


@app.get("/api/explainability/{ticker}")
async def get_explainability(ticker: str):
    """Get SHAP + feature importance + narrative."""
    safe_name = ticker.replace(".", "_")
    expl = _load_json(DATA_DIR / safe_name / "latest_explanation.json")
    if expl:
        return expl
    return {"error": "No explanation available. Run the pipeline first."}


@app.get("/api/classification/{ticker}")
async def get_classification(ticker: str):
    """Get direction votes from all classifiers."""
    safe_name = ticker.replace(".", "_")
    pred = _load_json(DATA_DIR / safe_name / "latest_prediction.json")
    if pred and "models" in pred:
        return {
            "classification": pred["models"]["classification"],
            "agreement": pred.get("agreement", {}),
        }
    return {"error": "No classification data available"}


@app.get("/api/status")
async def get_status():
    """Get system health status."""
    return {
        "system": "Apex-Oracle",
        "version": "1.0.0",
        "scheduler": scheduler_instance.get_status(),
        "stocks_tracked": len(STOCK_UNIVERSE),
        "connected_clients": len(connected_clients),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/run-pipeline")
async def run_pipeline_now():
    """Manually trigger the prediction pipeline."""
    import asyncio
    asyncio.create_task(scheduler_instance.job_fetch_and_predict())
    return {"message": "Pipeline triggered. Results will be pushed via WebSocket."}


@app.post("/api/retrain")
async def retrain_now():
    """Manually trigger model retraining."""
    import asyncio
    asyncio.create_task(scheduler_instance.job_retrain_models())
    return {"message": "Retraining triggered. This may take several minutes."}


# ── WebSocket ─────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time prediction updates."""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"WebSocket client connected ({len(connected_clients)} total)")

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            # Client can request specific stock updates
            if data.startswith("subscribe:"):
                ticker = data.split(":")[1]
                safe_name = ticker.replace(".", "_")
                pred = _load_json(DATA_DIR / safe_name / "latest_prediction.json")
                if pred:
                    await websocket.send_json(pred)
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected ({len(connected_clients)} total)")


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

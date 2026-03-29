"""
Apex-Oracle — Automated Scheduler
Central automation engine using APScheduler.
Runs data fetching, prediction, and model retraining on schedule.
"""
import asyncio
import gc
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
import psutil

from config import (
    FETCH_INTERVAL_MINUTES, POST_MARKET_INTERVAL_MINUTES,
    RETRAIN_DAY, RETRAIN_HOUR,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    STOCK_UNIVERSE,
)

# WebSocket connections managed by app.py — imported at runtime
connected_clients = set()


class AMSFRXPScheduler:
    """Manages all automated tasks with memory-safe execution."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.last_run = None
        self.is_running = False

    def start(self):
        """Start the scheduler with all configured jobs."""
        logger.info("Starting Apex-Oracle Scheduler...")

        # 1. Market hours: fetch + predict every 5 min (Mon-Fri, 9:15-15:30 IST)
        self.scheduler.add_job(
            self.job_fetch_and_predict,
            CronTrigger(
                day_of_week="mon-fri",
                hour=f"{MARKET_OPEN_HOUR}-{MARKET_CLOSE_HOUR}",
                minute=f"*/{FETCH_INTERVAL_MINUTES}",
            ),
            id="market_hours_pipeline",
            replace_existing=True,
        )

        # 2. Post-market: every 30 min after close
        self.scheduler.add_job(
            self.job_fetch_and_predict,
            CronTrigger(
                day_of_week="mon-fri",
                hour=f"{MARKET_CLOSE_HOUR + 1}-23",
                minute=f"*/{POST_MARKET_INTERVAL_MINUTES}",
            ),
            id="post_market_pipeline",
            replace_existing=True,
        )

        # 3. Weekly model retraining (Sunday 2 AM)
        self.scheduler.add_job(
            self.job_retrain_models,
            CronTrigger(day_of_week=RETRAIN_DAY, hour=RETRAIN_HOUR, minute=0),
            id="weekly_retrain",
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info("Scheduler started with 3 jobs")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    async def _check_memory(self):
        """Monitor memory usage."""
        mem = psutil.virtual_memory()
        logger.info(f"Memory: {mem.percent}% used, {mem.available / (1024**3):.1f} GB available")
        if mem.percent > 85:
            logger.warning("High memory usage! Running garbage collection.")
            gc.collect()

    async def _notify_clients(self, data: dict):
        """Push update to all connected WebSocket clients."""
        import json
        message = json.dumps(data)
        disconnected = set()
        for ws in connected_clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        connected_clients -= disconnected

    async def job_fetch_and_predict(self):
        """Main pipeline: fetch data → features → sentiment → predict → push."""
        if self.is_running:
            logger.warning("Previous pipeline still running, skipping.")
            return

        self.is_running = True
        start_time = datetime.now()
        logger.info(f"{'='*60}")
        logger.info(f"Pipeline started at {start_time.strftime('%H:%M:%S')}")

        try:
            # Step 1: Fetch data
            from fetch_data import fetch_all_stocks
            fetch_results = fetch_all_stocks()
            gc.collect()

            # Step 2: Analyze sentiment
            from sentiment_analyzer import analyze_all_stocks
            analyze_all_stocks(use_finbert=False)  # Use VADER for speed
            gc.collect()

            # Step 3: Build features
            from feature_engineering import build_all_features
            build_all_features()
            gc.collect()

            # Step 4: Run fusion engine for all stocks
            from fusion_engine import FusionEngine
            engine = FusionEngine()
            predictions = engine.fuse_all_stocks()
            gc.collect()

            # Step 5: Generate alerts
            from alert_system import generate_alerts
            all_alerts = []
            for ticker, pred in predictions.items():
                if "error" not in pred:
                    alerts = generate_alerts(ticker, pred)
                    all_alerts.extend(alerts)

            # Step 6: Generate explanations
            from explainability import generate_full_explanation
            for ticker, pred in predictions.items():
                if "error" not in pred:
                    generate_full_explanation(ticker, pred)

            # Step 7: Push to WebSocket clients
            duration = (datetime.now() - start_time).total_seconds()
            update = {
                "type": "pipeline_complete",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "predictions": {
                    t: p.get("prediction", {}) for t, p in predictions.items()
                    if "error" not in p
                },
                "alerts": all_alerts[-5:],  # Last 5 alerts
            }
            await self._notify_clients(update)

            self.last_run = datetime.now()
            logger.info(f"Pipeline complete in {duration:.1f}s")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self.is_running = False
            await self._check_memory()

    async def job_retrain_models(self):
        """Weekly model retraining."""
        logger.info("Starting weekly model retraining...")
        await self._check_memory()

        try:
            from regime_detector import train_all_regimes
            from lstm_model import LSTMModel
            from xgboost_model import XGBoostModel
            from naive_bayes_model import NaiveBayesModel
            from random_forest_model import RandomForestModel
            from logistic_regression_model import LogisticRegressionModel
            from evaluate_models import evaluate

            # Retrain regime detector
            train_all_regimes()
            gc.collect()

            # Retrain each model type for all stocks
            for ticker in STOCK_UNIVERSE:
                logger.info(f"Retraining for {ticker}...")

                LSTMModel().train(ticker); gc.collect()
                XGBoostModel().train(ticker); gc.collect()
                NaiveBayesModel().train(ticker); gc.collect()
                SVMModel().train(ticker); gc.collect()
                RandomForestModel().train(ticker); gc.collect()
                LogisticRegressionModel().train(ticker); gc.collect()

                await self._check_memory()

            logger.info("Evaluating all models and generating results.json...")
            evaluate()

            logger.info("Weekly retraining complete!")

            await self._notify_clients({
                "type": "retrain_complete",
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.error(f"Retraining error: {e}")

    def get_status(self) -> dict:
        """Get scheduler status for API."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
            })

        return {
            "running": self.scheduler.running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "is_pipeline_running": self.is_running,
            "jobs": jobs,
            "memory_percent": psutil.virtual_memory().percent,
        }


# Singleton instance
scheduler_instance = AMSFRXPScheduler()


def start_scheduler():
    scheduler_instance.start()


def stop_scheduler():
    scheduler_instance.stop()

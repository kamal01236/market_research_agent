"""Job scheduler for periodic tasks."""
import asyncio
from datetime import datetime, time, timedelta
from typing import List, Optional

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from market_research_agent.config import Settings
from market_research_agent.providers import YahooFinanceAdapter
from market_research_agent.features.technical import TechnicalFeatures
from market_research_agent.store import FeatureStore
from market_research_agent.scoring import FactorScorer

class JobScheduler:
    """Scheduler for periodic data fetching and feature computation."""
    
    def __init__(self, settings: Settings, db_url: str = None):
        self.settings = settings
        self.scheduler = AsyncIOScheduler()
        self.provider = YahooFinanceAdapter()
        # Allow db_url override for testing; fallback to default if not provided
        if db_url is None:
            db_url = "postgresql+psycopg2://postgres:postgres@localhost:5432/market_research"
        self.feature_store = FeatureStore(db_url)
        self.feature_computer = TechnicalFeatures()
        self.scorer = FactorScorer()
        
        # Initialize timezone
        self.tz = pytz.timezone("Asia/Kolkata")
        
        # Trading hours in IST
        self.market_open = time(9, 15)  # 9:15 AM
        self.market_close = time(15, 30)  # 3:30 PM
        
    async def fetch_and_compute(
        self,
        symbols: List[str],
        compute_scores: bool = True
    ):
        """Fetch data, compute features, and optionally scores."""
        try:
            # Get last 30 days of data
            end_date = datetime.now(self.tz)
            start_date = end_date - timedelta(days=30)
            
            # Fetch market data
            data = await self.provider.fetch_data(
                symbols,
                start_date,
                end_date
            )
            
            # Compute and store features
            for sym, df in data.items():
                try:
                    # Validate data
                    valid, errors = await self.provider.validate_data(df)
                    if not valid:
                        logger.warning(f"Data validation failed for {sym}: {errors}")
                        continue
                        
                    # Compute features
                    features = await self.feature_computer.compute(df)
                    
                    # Store features
                    await self.feature_store.store_features(sym, features)
                    
                    if compute_scores:
                        # Compute scores
                        factors = await self.scorer.get_available_factors()
                        scores = await self.scorer.compute_scores(features, factors)
                        await self.feature_store.store_scores(sym, scores)
                        
                    logger.info(f"Successfully processed {sym}")
                    
                except Exception as e:
                    logger.error(f"Error processing {sym}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in fetch_and_compute: {str(e)}")
            
    async def schedule_jobs(self):
        """Schedule periodic jobs."""
        # Market open job - 9:15 AM IST
        self.scheduler.add_job(
            self.fetch_and_compute,
            CronTrigger(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                timezone=self.tz
            ),
            args=[self.settings.symbols],
            id="market_open",
            replace_existing=True
        )
        
        # Market close job - 3:30 PM IST
        self.scheduler.add_job(
            self.fetch_and_compute,
            CronTrigger(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                timezone=self.tz
            ),
            args=[self.settings.symbols, True],
            id="market_close",
            replace_existing=True
        )
        
        # Intraday updates - Every 15 minutes during market hours
        self.scheduler.add_job(
            self.fetch_and_compute,
            CronTrigger(
                hour=f"{self.market_open.hour}-{self.market_close.hour}",
                minute="*/15",
                timezone=self.tz
            ),
            args=[self.settings.symbols, False],
            id="intraday_update",
            replace_existing=True
        )
        
    def start(self):
        """Start the scheduler."""
        logger.info("Starting job scheduler...")
        self.scheduler.start()
        
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping job scheduler...")
        self.scheduler.shutdown()
        
    async def run_now(
        self,
        symbols: Optional[List[str]] = None,
        compute_scores: bool = True
    ):
        """Run jobs immediately."""
        symbols = symbols or self.settings.symbols
        await self.fetch_and_compute(symbols, compute_scores)
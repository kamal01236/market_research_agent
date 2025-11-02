"""Command line interface for the market research agent."""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import click
import yaml
from loguru import logger

from .api import app
from .config import Settings
from .providers import YahooFinanceAdapter
from .features.technical import TechnicalFeatures
from .store import FeatureStore
from .scoring import FactorScorer
from .scheduler import JobScheduler

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

def load_settings(config_path: str) -> Settings:
    """Load settings from config file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    return Settings.from_yaml(config_path)

async def run_data_collection(
    settings: Settings,
    symbols: Optional[List[str]] = None,
    days: int = 30
):
    """Run data collection for specified symbols."""
    provider = YahooFinanceAdapter(
        batch_size=settings.provider.batch_size,
        rate_limit=settings.provider.rate_limit
    )
    
    feature_computer = TechnicalFeatures()
    feature_store = FeatureStore()
    scorer = FactorScorer()
    
    symbols = symbols or settings.symbols
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    try:
        # Fetch market data
        data = await provider.fetch_data(symbols, start_date, end_date)
        
        for sym, df in data.items():
            try:
                # Validate data
                valid, errors = await provider.validate_data(df)
                if not valid:
                    logger.warning(f"Data validation failed for {sym}: {errors}")
                    continue
                
                # Compute features
                features = await feature_computer.compute(df)
                
                # Store features
                await feature_store.store_features(sym, features)
                
                # Compute and store scores
                factors = await scorer.get_available_factors()
                scores = await scorer.compute_scores(features, factors)
                await feature_store.store_scores(sym, scores)
                
                logger.info(f"Successfully processed {sym}")
                
            except Exception as e:
                logger.error(f"Error processing {sym}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        raise click.ClickException(str(e))

@click.group()
def cli():
    """Market Research Agent CLI."""
    pass

@cli.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="config.yml",
    help="Path to config file",
)
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    help="Specific symbols to process",
)
@click.option(
    "--days",
    "-d",
    type=int,
    default=30,
    help="Number of days to fetch",
)
def collect(config: str, symbols: tuple, days: int):
    """Collect and process market data."""
    settings = load_settings(config)
    symbol_list = list(symbols) if symbols else None
    asyncio.run(run_data_collection(settings, symbol_list, days))

@cli.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="config.yml",
    help="Path to config file",
)
def serve(config: str):
    """Start the API server."""
    import uvicorn
    
    settings = load_settings(config)
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s | %(levelname)s | %(message)s"
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=log_config
    )

@cli.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default="config.yml",
    help="Path to config file",
)
def scheduler(config: str):
    """Start the job scheduler."""
    settings = load_settings(config)
    scheduler = JobScheduler(settings)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Stopping scheduler...")
        scheduler.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    async def init_scheduler():
        await scheduler.schedule_jobs()
        scheduler.start()
        
        # Run initial collection
        await scheduler.run_now()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    try:
        asyncio.run(init_scheduler())
    except KeyboardInterrupt:
        scheduler.stop()
        sys.exit(0)

if __name__ == "__main__":
    cli()
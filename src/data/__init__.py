"""
Data module for local data storage and retrieval.

This module provides functionality to:
1. Screen and identify dividend-paying stocks (universe construction)
2. Download historical price data from Alpaca and yfinance
3. Download dividend history
4. Store and load data locally for fast backtesting
"""

# Only import loader by default (no external API dependencies)
from .loader import DataLoader

# Lazy imports for modules that require yfinance/alpaca
def get_universe_screener():
    """Lazy import to avoid requiring yfinance at module load."""
    from .universe import UniverseScreener
    return UniverseScreener

def get_data_downloader():
    """Lazy import to avoid requiring yfinance/alpaca at module load."""
    from .downloader import DataDownloader
    return DataDownloader

__all__ = ['DataLoader', 'get_universe_screener', 'get_data_downloader']

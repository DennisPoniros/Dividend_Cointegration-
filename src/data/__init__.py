"""
Data module for local data storage and retrieval.

This module provides functionality to:
1. Screen and identify dividend-paying stocks (universe construction)
2. Download historical price data from Alpaca and yfinance
3. Download dividend history
4. Store and load data locally for fast backtesting
"""

from .universe import UniverseScreener
from .downloader import DataDownloader
from .loader import DataLoader

__all__ = ['UniverseScreener', 'DataDownloader', 'DataLoader']

"""Processing module - dithering engine and workers."""

from .engine import DitheringEngine
from .pipeline import ProcessingPipeline
from .worker import ProcessingWorker

__all__ = ["DitheringEngine", "ProcessingPipeline", "ProcessingWorker"]

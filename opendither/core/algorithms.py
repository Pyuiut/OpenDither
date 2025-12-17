"""Dithering algorithm definitions and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Callable, Optional
import numpy as np


class AlgorithmCategory(Enum):
    """Categories for organizing dithering algorithms."""
    ERROR_DIFFUSION = "Error Diffusion"
    ORDERED = "Ordered/Bitmap"
    HALFTONE = "Halftone"
    PATTERN = "Pattern"
    MODULATION = "Modulation"
    SPECIAL = "Special Effects"


@dataclass
class Algorithm:
    """Definition of a dithering algorithm."""
    name: str
    category: AlgorithmCategory
    description: str = ""
    parameters: Dict[str, float] = field(default_factory=dict)


class AlgorithmLibrary:
    """Registry of all available dithering algorithms."""

    def __init__(self) -> None:
        self._algorithms: Dict[str, Algorithm] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register curated set of built-in algorithms."""
        # Error Diffusion (distinct looks)
        error_diffusion = [
            ("Floyd-Steinberg", "Classic error diffusion with diagonal distribution"),
            ("Jarvis-Judice-Ninke", "Extended kernel for ultra-smooth gradients"),
            ("Stucki", "Sharper detail with balanced diffusion"),
            ("Burkes", "Fast two-row diffusion with crisp edges"),
            ("Sierra Lite", "Single-row Sierra variant with softer grain"),
            ("Atkinson", "Vintage Macintosh look with bold pixels"),
        ]
        for name, desc in error_diffusion:
            self.register(Algorithm(name, AlgorithmCategory.ERROR_DIFFUSION, desc))

        # Ordered/Bitmap (most useful matrices)
        ordered = [
            ("Bayer 2x2", "Compact matrix for graphic styles"),
            ("Bayer 4x4", "Balanced ordered dither"),
            ("Bayer 8x8", "Large matrix for ultra-smooth gradients"),
            ("Clustered Dot 8x8", "Analog print-style clustered dots"),
        ]
        for name, desc in ordered:
            self.register(Algorithm(name, AlgorithmCategory.ORDERED, desc))

        # Halftone / tone-mapping
        halftone = [
            ("Circular Halftone", "Classic offset-print halftone dots"),
            ("Line Halftone", "Horizontal line halftone"),
            ("Diamond Halftone", "Diamond-shaped halftone pattern"),
        ]
        for name, desc in halftone:
            self.register(Algorithm(name, AlgorithmCategory.HALFTONE, desc))

        # Patterned styles (implemented variants)
        pattern = [
            ("Crosshatch", "Traditional crosshatch engraving style"),
            ("Stipple", "Random dot stippling effect"),
            ("Checkerboard", "Simple checkerboard pattern"),
        ]
        for name, desc in pattern:
            self.register(Algorithm(name, AlgorithmCategory.PATTERN, desc))

        # Special Effects (curated unique looks)
        special = [
            ("Glitch Blocks", "Random block displacement glitch"),
            ("Scanlines", "CRT scanline simulation"),
            ("Pixelate", "Pixel art style reduction"),
            ("VHS", "Analog tape degradation effect"),
            ("Bit Crush", "Bit depth reduction effect"),
        ]
        for name, desc in special:
            self.register(Algorithm(name, AlgorithmCategory.SPECIAL, desc))

    def register(self, algorithm: Algorithm) -> None:
        """Register an algorithm."""
        self._algorithms[algorithm.name] = algorithm

    def get(self, name: str) -> Optional[Algorithm]:
        """Get an algorithm by name."""
        return self._algorithms.get(name)

    def all(self) -> List[Algorithm]:
        """Get all registered algorithms."""
        return list(self._algorithms.values())

    def by_category(self) -> Dict[AlgorithmCategory, List[Algorithm]]:
        """Get algorithms grouped by category."""
        result: Dict[AlgorithmCategory, List[Algorithm]] = {}
        for algo in self._algorithms.values():
            if algo.category not in result:
                result[algo.category] = []
            result[algo.category].append(algo)
        return result

    def names(self) -> List[str]:
        """Get all algorithm names."""
        return list(self._algorithms.keys())

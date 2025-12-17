"""Preset management for saving and loading configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Preset:
    """A saved configuration preset."""
    name: str
    algorithm: str
    parameters: Dict[str, float] = field(default_factory=dict)
    palette: Optional[str] = None
    category: str = "Custom"
    description: str = ""


class PresetLibrary:
    """Registry and persistence for presets."""

    def __init__(self) -> None:
        self._presets: Dict[str, Preset] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in presets."""
        defaults = [
            # === FAVORITES ===
            Preset(
                "Classic B&W",
                "Floyd-Steinberg",
                {"scale": 1.0, "contrast": 1.0, "dither_strength": 1.0},
                "1-Bit Black & White",
                "Favorites",
                "Traditional black and white newspaper style"
            ),
            Preset(
                "High Contrast",
                "Atkinson",
                {"scale": 1.0, "contrast": 1.5, "dither_strength": 1.2},
                "1-Bit Black & White",
                "Favorites",
                "Bold high-contrast dithering"
            ),
            Preset(
                "Soft Dither",
                "Jarvis",
                {"scale": 1.0, "contrast": 0.9, "blur": 0.05, "dither_strength": 0.8},
                None,
                "Favorites",
                "Gentle dithering with soft edges"
            ),
            
            # === RETRO ===
            Preset(
                "Game Boy",
                "Bayer 4x4",
                {"scale": 1.0, "contrast": 1.2, "dither_strength": 1.0},
                "Game Boy",
                "Retro",
                "Classic handheld gaming look"
            ),
            Preset(
                "Commodore 64",
                "Bayer 8x8",
                {"scale": 1.0, "contrast": 1.1, "dither_strength": 1.0},
                "C64",
                "Retro",
                "8-bit home computer style"
            ),
            Preset(
                "NES Style",
                "Floyd-Steinberg",
                {"scale": 1.0, "contrast": 1.3, "saturation": 1.2},
                "NES",
                "Retro",
                "8-bit Nintendo Entertainment System"
            ),
            Preset(
                "CGA Mode",
                "Ordered 2x2",
                {"scale": 1.0, "contrast": 1.4, "dither_strength": 1.0},
                "CGA",
                "Retro",
                "IBM PC CGA graphics mode"
            ),
            
            # === ARTISTIC ===
            Preset(
                "Film Grain",
                "Noise Grain",
                {"scale": 1.0, "contrast": 0.9, "blur": 0.05, "grain": 0.3},
                None,
                "Artistic",
                "Analog film grain effect"
            ),
            Preset(
                "Newspaper Print",
                "Halftone",
                {"scale": 1.0, "contrast": 1.1, "pattern_scale": 1.5},
                "1-Bit Black & White",
                "Artistic",
                "Classic newspaper halftone dots"
            ),
            Preset(
                "Crosshatch Sketch",
                "Crosshatch",
                {"scale": 1.0, "contrast": 1.2, "sharpness": 0.2},
                "1-Bit Black & White",
                "Artistic",
                "Hand-drawn crosshatch effect"
            ),
            Preset(
                "Stipple Art",
                "Stipple",
                {"scale": 1.0, "contrast": 1.0, "dither_strength": 1.0},
                None,
                "Artistic",
                "Pointillism stipple effect"
            ),
            Preset(
                "Blueprint",
                "Bayer 4x4",
                {"scale": 1.0, "contrast": 1.3, "temperature": 0.3, "tint": 0.4},
                "Blueprint",
                "Artistic",
                "Technical blueprint style"
            ),
            Preset(
                "Sepia Vintage",
                "Floyd-Steinberg",
                {"scale": 1.0, "contrast": 0.9, "saturation": 0.3, "temperature": 1.3},
                "Sepia",
                "Artistic",
                "Warm vintage sepia tones"
            ),
            
            # === EFFECTS ===
            Preset(
                "CRT Monitor",
                "Scanlines",
                {"scale": 1.0, "contrast": 1.1, "glow": 0.1},
                None,
                "Effects",
                "Retro CRT monitor simulation"
            ),
            Preset(
                "VHS Tape",
                "VHS",
                {"scale": 1.0, "contrast": 0.9, "blur": 0.1, "grain": 0.2},
                None,
                "Effects",
                "Nostalgic VHS tape distortion"
            ),
            Preset(
                "Glitch Art",
                "Glitch",
                {"scale": 1.0, "contrast": 1.0, "dither_strength": 1.5},
                None,
                "Effects",
                "Digital glitch aesthetic"
            ),
            Preset(
                "Pixelate Retro",
                "Pixelate",
                {"scale": 0.25, "contrast": 1.2},
                None,
                "Effects",
                "Chunky pixel art style"
            ),
            
            # === MODERN ===
            Preset(
                "Cyberpunk Neon",
                "Floyd-Steinberg",
                {"scale": 1.0, "contrast": 1.4, "saturation": 1.5, "glow": 0.3},
                "Cyberpunk",
                "Modern",
                "Neon cyberpunk aesthetic"
            ),
            Preset(
                "Vapor Wave",
                "Bayer 4x4",
                {"scale": 1.0, "contrast": 1.1, "saturation": 1.3, "hue_shift": 0.5},
                "Vapor Wave",
                "Modern",
                "Retro-futuristic vaporwave style"
            ),
            Preset(
                "Minimal Dark",
                "Atkinson",
                {"scale": 1.0, "contrast": 1.3, "brightness": 0.8},
                "Minimal Dark",
                "Modern",
                "Clean minimal dark aesthetic"
            ),
            Preset(
                "Lo-Fi",
                "Sierra",
                {"scale": 1.0, "contrast": 0.85, "blur": 0.1, "grain": 0.15, "vignette": 0.3},
                None,
                "Modern",
                "Chill lo-fi aesthetic"
            ),
        ]
        for preset in defaults:
            self.register(preset)

    def register(self, preset: Preset) -> None:
        """Register or update a preset."""
        self._presets[preset.name] = preset

    def get(self, name: str) -> Optional[Preset]:
        """Get a preset by name."""
        return self._presets.get(name)

    def delete(self, name: str) -> bool:
        """Delete a preset. Returns True if deleted."""
        if name in self._presets:
            del self._presets[name]
            return True
        return False

    def all(self) -> List[Preset]:
        """Get all presets."""
        return list(self._presets.values())

    def categories(self) -> Dict[str, List[Preset]]:
        """Get presets grouped by category."""
        result: Dict[str, List[Preset]] = {}
        for preset in self._presets.values():
            if preset.category not in result:
                result[preset.category] = []
            result[preset.category].append(preset)
        return result

    def names(self) -> List[str]:
        """Get all preset names."""
        return list(self._presets.keys())

    def save_to_file(self, path: Path) -> None:
        """Export all presets to a JSON file."""
        data = [asdict(p) for p in self._presets.values()]
        path.write_text(json.dumps(data, indent=2))

    def load_from_file(self, path: Path) -> int:
        """Import presets from a JSON file. Returns count imported."""
        data = json.loads(path.read_text())
        count = 0
        for item in data:
            preset = Preset(**item)
            self.register(preset)
            count += 1
        return count

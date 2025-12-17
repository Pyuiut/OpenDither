"""Optimized image processing pipeline with caching and progress reporting."""

from __future__ import annotations

from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from hashlib import md5
import colorsys
import numpy as np
from numpy.typing import NDArray
import cv2

from opendither.core import PaletteLibrary


@dataclass
class PipelineStage:
    """A single stage in the processing pipeline."""
    name: str
    weight: float = 1.0  # Relative processing time weight


@dataclass  
class CacheEntry:
    """Cached result from a pipeline stage."""
    params_hash: str
    result: NDArray[np.uint8]


class ProcessingPipeline:
    """Optimized pipeline with intelligent caching and progress reporting."""
    
    # Pipeline stages in order with relative weights (for progress estimation)
    STAGES = [
        PipelineStage("adjustments", 0.15),    # Fast: brightness, contrast, etc.
        PipelineStage("color_grading", 0.10),  # Fast: color adjustments
        PipelineStage("curves", 0.05),         # Very fast: LUT application
        PipelineStage("dithering", 0.60),      # Slow: main dithering algorithm  
        PipelineStage("effects", 0.10),        # Medium: post effects
    ]
    
    def __init__(self):
        self.palette_library = PaletteLibrary()
        self._cache: Dict[str, CacheEntry] = {}
        self._bayer_cache: Dict[int, NDArray] = {}
        self._progress_callback: Optional[Callable[[int, str], None]] = None
        self._current_progress: float = 0
        
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set callback for progress updates: callback(percent, stage_name)"""
        self._progress_callback = callback
        
    def _report_progress(self, stage_idx: int, stage_progress: float = 1.0):
        """Report progress to callback."""
        if not self._progress_callback:
            return
            
        # Calculate cumulative progress
        total_weight = sum(s.weight for s in self.STAGES)
        completed_weight = sum(self.STAGES[i].weight for i in range(stage_idx))
        current_weight = self.STAGES[stage_idx].weight * stage_progress
        
        progress = int(((completed_weight + current_weight) / total_weight) * 100)
        stage_name = self.STAGES[stage_idx].name
        self._progress_callback(min(100, progress), stage_name)
    
    def _hash_params(self, params: Dict, keys: List[str]) -> str:
        """Create hash from specific parameter keys."""
        relevant = {k: params.get(k, 0) for k in keys}
        return md5(str(sorted(relevant.items())).encode()).hexdigest()[:16]
    
    def process(
        self,
        image: NDArray[np.uint8],
        algorithm: str,
        parameters: Dict[str, float],
        palette: Optional[str] = None,
        curves_lut: Optional[Dict[str, List[int]]] = None,
    ) -> NDArray[np.uint8]:
        """Process image through optimized pipeline with caching."""
        
        result = image.copy()
        
        # === STAGE 0: Adjustments (brightness, contrast, exposure, etc.) ===
        self._report_progress(0, 0)
        result = self._apply_adjustments(result, parameters)
        self._report_progress(0, 1.0)
        
        # === STAGE 1: Color Grading (saturation, vibrance, temperature) ===
        self._report_progress(1, 0)
        result = self._apply_color_grading(result, parameters)
        self._report_progress(1, 1.0)
        
        # === STAGE 2: Curves (LUT application - very fast) ===
        self._report_progress(2, 0)
        if curves_lut:
            result = self._apply_curves(result, curves_lut)
        self._report_progress(2, 1.0)
        
        # Preserve color reference before dithering
        color_reference = result.copy()
        
        # === STAGE 3: Dithering (main algorithm - slowest) ===
        self._report_progress(3, 0)
        colors = None
        if palette:
            pal = self.palette_library.get(palette)
            if pal:
                colors = np.array(pal.colors, dtype=np.uint8)
        result = self._apply_dithering(result, algorithm, parameters, colors)
        self._report_progress(3, 1.0)
        
        if parameters.get("preserve_colors", 0):
            result = self._restore_colors(color_reference, result)
        
        # === STAGE 4: Effects (post-processing) ===
        self._report_progress(4, 0)
        result = self._apply_effects(result, parameters)
        self._report_progress(4, 1.0)
        
        return result

    def _restore_colors(self, reference: NDArray, dithered: NDArray) -> NDArray:
        """Reinject original hues/saturation while keeping dithered luminance."""
        if reference.ndim != 3:
            return dithered
        
        if dithered.ndim == 3:
            luminance = cv2.cvtColor(dithered, cv2.COLOR_RGB2GRAY)
        else:
            luminance = dithered
        
        ref_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV).astype(np.float32)
        luminance = cv2.resize(luminance, (reference.shape[1], reference.shape[0]))
        ref_hsv[:, :, 2] = np.clip(luminance.astype(np.float32), 0, 255)
        
        restored = cv2.cvtColor(ref_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return restored
    
    def _apply_flares(self, image: NDArray, params: Dict) -> NDArray:
        """Add customizable lens flare overlays with ghosts, shapes, and tint."""
        intensity = params.get("flare_intensity", 0) / 100.0
        if intensity <= 0:
            return image
        
        threshold = params.get("flare_threshold", 60)
        style = params.get("flare_style", "Lens")
        distribution = params.get("flare_distribution", "Highlights")
        shape = params.get("flare_shape", "Star")
        tint_name = params.get("flare_tint", "Warm")
        flare_amount = params.get("flare_amount", 0)
        variation = params.get("flare_variation", 50) / 100.0
        base_size = max(6, int(params.get("flare_size", 40)))
        flare_spacing = params.get("flare_spacing", 40) / 100.0
        hue = params.get("flare_color_hue", 40) / 360.0
        sat = params.get("flare_color_sat", 80) / 100.0
        val = params.get("flare_color_value", 90) / 100.0
        
        # Legacy tint presets can coexist with HSV sliders – prefer explicit HSV.
        tint_colors = {
            "Warm": np.array([255, 196, 150], dtype=np.float32),
            "Cool": np.array([150, 210, 255], dtype=np.float32),
            "Neon": np.array([190, 255, 220], dtype=np.float32),
            "Gold": np.array([255, 223, 140], dtype=np.float32),
            "Sunset": np.array([255, 160, 190], dtype=np.float32),
        }
        preset_tint = tint_colors.get(tint_name, tint_colors["Warm"])
        hsv_rgb = colorsys.hsv_to_rgb(
            np.clip(hue, 0, 1),
            np.clip(sat, 0, 1),
            np.clip(val, 0, 1),
        )
        hsv_tint = np.array([c * 255 for c in hsv_rgb], dtype=np.float32)
        tint = (preset_tint * 0.4 + hsv_tint * 0.6)
        
        base = image.astype(np.float32)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        
        mask = np.clip((brightness - threshold) / max(1, 255 - threshold), 0, 1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5 + intensity * 5)
        
        if style == "Lens":
            streak = cv2.blur(mask, (int(30 + intensity * 40), 1))
            streak_v = cv2.blur(mask, (1, int(30 + intensity * 40)))
            mask = np.clip(mask + 0.6 * streak + 0.6 * streak_v, 0, 1)
        elif style == "Orbs":
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15 + intensity * 10)
        elif style == "Starburst":
            h, w = mask.shape
            Y, X = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            radial = np.clip(1.0 - dist / dist.max(), 0, 1)
            mask = np.clip(mask + radial * 0.5, 0, 1)
        base_overlay = tint * (mask[:, :, None] * intensity * 0.9)
        
        # Build placement map for additional ghosts/elements.
        if distribution == "Edge Trails":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 180).astype(np.float32) / 255.0
            placement = cv2.GaussianBlur(edges, (0, 0), sigmaX=2.0)
        elif distribution == "Uniform":
            placement = np.full_like(mask, 0.5)
        else:
            placement = mask
        placement = cv2.normalize(placement, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        coords = np.column_stack(np.where(placement > placement.mean() * 0.5))
        if coords.size == 0:
            coords = np.column_stack(np.where(np.ones_like(placement)))
        flare_count = max(1, int(1 + flare_amount / 20))
        h, w = placement.shape
        overlay_mask = np.zeros((h, w), dtype=np.float32)
        
        def draw_flare(canvas: NDArray[np.float32], center: Tuple[int, int], size: int, strength: float) -> None:
            cx, cy = center
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                return
            radius = max(4, size)
            cv2.circle(canvas, (cx, cy), radius, strength, -1)
            if shape == "Star":
                for angle in range(0, 180, 45):
                    dx = int(np.cos(np.deg2rad(angle)) * radius * 2)
                    dy = int(np.sin(np.deg2rad(angle)) * radius * 2)
                    cv2.line(canvas, (cx - dx, cy - dy), (cx + dx, cy + dy), strength * 0.7, thickness=max(1, radius // 6))
            elif shape == "Hex":
                angles = np.linspace(0, 2 * np.pi, 7)[:-1]
                pts = np.stack(
                    [
                        [cx + radius * np.cos(a), cy + radius * np.sin(a)]
                        for a in angles
                    ]
                ).astype(np.int32)
                cv2.fillConvexPoly(canvas, pts, strength)
            # Soft Orb just uses the gaussian circle above
        
        for _ in range(flare_count):
            idx = np.random.randint(0, len(coords))
            cy, cx = coords[idx]
            local_size = int(base_size * (0.8 + variation * (np.random.rand() - 0.5)))
            local_strength = np.clip(intensity * (0.7 + np.random.rand() * 0.6), 0.1, 1.5)
            draw_flare(overlay_mask, (cx, cy), local_size, local_strength)
            
            ghost_steps = max(0, int(flare_amount / 25))
            angle = np.random.rand() * 2 * np.pi
            spacing_px = (base_size * 1.5) * (0.3 + flare_spacing)
            for g in range(ghost_steps):
                distance = spacing_px * (g + 1) * (0.8 + variation * (np.random.rand() - 0.5))
                offset_x = int(cx + np.cos(angle) * distance)
                offset_y = int(cy + np.sin(angle) * distance)
                ghost_strength = local_strength * (0.7 ** (g + 1))
                draw_flare(
                    overlay_mask,
                    (offset_x, offset_y),
                    max(4, int(local_size * (0.85 ** (g + 1)))),
                    ghost_strength,
                )
        
        overlay_mask = cv2.GaussianBlur(
            overlay_mask,
            (0, 0),
            sigmaX=2.5 + intensity * 6,
        )
        overlay_mask = np.clip(overlay_mask, 0, 1)
        color_overlay = overlay_mask[:, :, None] * tint * (0.7 + intensity)
        result = np.clip(base + base_overlay + color_overlay, 0, 255).astype(np.uint8)
        return result
    
    def _apply_glitch(self, image: NDArray, params: Dict) -> NDArray:
        """Apply stylized glitch effects with additional controls."""
        intensity = params.get("glitch_intensity", 0)
        if intensity <= 0:
            return image
        
        style = params.get("glitch_style", "RGB Split")
        frequency = max(1, params.get("glitch_frequency", 40))
        shift_amount = params.get("glitch_shift", 40)
        result = image.copy()
        
        if style == "RGB Split":
            shift = max(1, int(shift_amount / 8) + intensity // 8)
            channels = cv2.split(result)
            channels[0] = np.roll(channels[0], shift, axis=1)
            channels[1] = np.roll(channels[1], -shift // 2, axis=0)
            channels[2] = np.roll(channels[2], shift // 3, axis=1)
            result = cv2.merge(channels)
        elif style == "Block Shift":
            h, w = result.shape[:2]
            bands = max(2, frequency // 10)
            max_shift = int(w * (0.05 + shift_amount / 200.0)) + intensity
            for _ in range(bands):
                y = np.random.randint(0, max(1, h - 10))
                height = np.random.randint(5, min(h // 3, h - y))
                shift = np.random.randint(-max_shift, max_shift)
                result[y : y + height] = np.roll(result[y : y + height], shift, axis=1)
        elif style == "Analog Ripples":
            h, w = result.shape[:2]
            ripple = result.astype(np.float32)
            freq = max(4, int(120 / max(frequency, 1)))
            amp = max(1, int((shift_amount / 100.0) * 20) + intensity // 6)
            for y in range(h):
                offset = int(np.sin(y / freq) * amp)
                ripple[y] = np.roll(ripple[y], offset, axis=0)
            noise_scale = max(0.5, intensity / 12)
            noise = np.random.randn(*ripple.shape) * noise_scale
            result = np.clip(ripple + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_adjustments(self, image: NDArray, params: Dict) -> NDArray:
        """Apply basic adjustments - optimized with vectorized operations."""
        result = image.astype(np.float32)
        
        # Scale/resize
        scale = params.get("scale", 100) / 100.0
        if scale != 1.0 and scale > 0:
            h, w = image.shape[:2]
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Brightness (-100 to 100) -> (-1.0 to 1.0) * 127
        brightness = params.get("brightness", 0)
        if brightness != 0:
            result = result + (brightness / 100.0) * 127
        
        # Contrast (-100 to 100) -> factor
        contrast = params.get("contrast", 0)
        if contrast != 0:
            factor = (100 + contrast) / 100.0
            result = (result - 127.5) * factor + 127.5
        
        # Exposure (-500 to 500) -> stops
        exposure = params.get("exposure", 0)
        if exposure != 0:
            stops = exposure / 100.0
            result = result * (2 ** stops)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_color_grading(self, image: NDArray, params: Dict) -> NDArray:
        """Apply color grading adjustments."""
        if len(image.shape) != 3:
            return image
            
        # Convert to HSV for saturation/vibrance
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Saturation (-100 to 100)
        saturation = params.get("saturation", 0)
        if saturation != 0:
            factor = 1.0 + saturation / 100.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # Vibrance (-100 to 100) - affects less saturated colors more
        vibrance = params.get("vibrance", 0)
        if vibrance != 0:
            sat = hsv[:, :, 1] / 255.0
            factor = 1.0 + (vibrance / 100.0) * (1.0 - sat)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # Hue shift
        hue_shift = params.get("hue_shift", 0)
        if hue_shift != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Temperature (-100 to 100)
        temperature = params.get("temperature", 0)
        if temperature != 0:
            result = result.astype(np.float32)
            result[:, :, 0] = np.clip(result[:, :, 0] + temperature * 0.5, 0, 255)  # R
            result[:, :, 2] = np.clip(result[:, :, 2] - temperature * 0.5, 0, 255)  # B
            result = result.astype(np.uint8)
        
        return result
    
    def _apply_curves(self, image: NDArray, curves_lut: Dict[str, List[int]]) -> NDArray:
        """Apply curves LUT - extremely fast."""
        result = image.copy()
        
        # RGB curve applies to all channels
        if "RGB" in curves_lut:
            lut = np.array(curves_lut["RGB"], dtype=np.uint8)
            if len(image.shape) == 3:
                for i in range(3):
                    result[:, :, i] = cv2.LUT(result[:, :, i], lut)
            else:
                result = cv2.LUT(result, lut)
        
        # Individual channel curves
        if len(image.shape) == 3:
            if "Red" in curves_lut:
                lut = np.array(curves_lut["Red"], dtype=np.uint8)
                result[:, :, 0] = cv2.LUT(result[:, :, 0], lut)
            if "Green" in curves_lut:
                lut = np.array(curves_lut["Green"], dtype=np.uint8)
                result[:, :, 1] = cv2.LUT(result[:, :, 1], lut)
            if "Blue" in curves_lut:
                lut = np.array(curves_lut["Blue"], dtype=np.uint8)
                result[:, :, 2] = cv2.LUT(result[:, :, 2], lut)
        
        return result
    
    def _apply_dithering(
        self, 
        image: NDArray, 
        algorithm: str, 
        params: Dict,
        colors: Optional[NDArray]
    ) -> NDArray:
        """Apply dithering algorithm - optimized implementations."""
        
        if not algorithm or algorithm.lower() in {"none", "no_dither"}:
            return image
        
        # Get luminance threshold
        threshold = params.get("luminance", 128)
        strength = params.get("dither_strength", 100) / 100.0
        
        method = algorithm.lower().replace("-", "_").replace(" ", "_")
        
        # Floyd-Steinberg (optimized)
        if "floyd" in method:
            return self._floyd_steinberg_fast(image, colors, threshold, strength)
        
        # Ordered/Bayer dithering (very fast)
        elif "bayer" in method or "ordered" in method:
            size = 8
            if "2x2" in method: size = 2
            elif "4x4" in method: size = 4
            elif "16" in method: size = 16
            return self._ordered_dither(image, size, colors, threshold)
        
        # Threshold (instant)
        elif "threshold" in method:
            return self._threshold_dither(image, colors, threshold)
        
        # Atkinson
        elif "atkinson" in method:
            return self._atkinson_dither(image, colors, threshold, strength)
        
        # Random/noise dithering (fast)
        elif "random" in method or "noise" in method:
            return self._random_dither(image, colors, threshold)
        
        # Pattern dithering
        elif "pattern" in method or "halftone" in method:
            return self._halftone_dither(image, colors)
        
        # Default: Floyd-Steinberg
        return self._floyd_steinberg_fast(image, colors, threshold, strength)
    
    def _floyd_steinberg_fast(
        self, 
        image: NDArray, 
        colors: Optional[NDArray],
        threshold: int,
        strength: float
    ) -> NDArray:
        """Optimized Floyd-Steinberg using row-based processing."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        result = gray.astype(np.float32)
        h, w = result.shape
        
        # Process row by row for cache efficiency
        for y in range(h - 1):
            for x in range(1, w - 1):
                old_val = result[y, x]
                new_val = 255.0 if old_val > threshold else 0.0
                result[y, x] = new_val
                
                error = (old_val - new_val) * strength
                
                result[y, x + 1] += error * 7 / 16
                result[y + 1, x - 1] += error * 3 / 16
                result[y + 1, x] += error * 5 / 16
                result[y + 1, x + 1] += error * 1 / 16
        
        # Last row
        result[-1, :] = np.where(result[-1, :] > threshold, 255, 0)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if colors is not None and len(colors) > 1:
            result = self._map_to_palette(result, colors)
        elif len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _ordered_dither(
        self, 
        image: NDArray, 
        size: int,
        colors: Optional[NDArray],
        threshold: int
    ) -> NDArray:
        """Fast ordered dithering using pre-computed Bayer matrix."""
        if size not in self._bayer_cache:
            self._bayer_cache[size] = self._generate_bayer(size)
        
        bayer = self._bayer_cache[size]
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Tile bayer matrix
        bayer_tiled = np.tile(bayer, (h // size + 1, w // size + 1))[:h, :w]
        
        # Apply threshold with bayer offset
        threshold_map = threshold + (bayer_tiled - 0.5) * 128
        result = np.where(gray > threshold_map, 255, 0).astype(np.uint8)
        
        if colors is not None and len(colors) > 1:
            result = self._map_to_palette(result, colors)
        elif len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _threshold_dither(
        self, 
        image: NDArray, 
        colors: Optional[NDArray],
        threshold: int
    ) -> NDArray:
        """Simple threshold - instant."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        result = np.where(gray > threshold, 255, 0).astype(np.uint8)
        
        if colors is not None and len(colors) > 1:
            result = self._map_to_palette(result, colors)
        elif len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _atkinson_dither(
        self, 
        image: NDArray, 
        colors: Optional[NDArray],
        threshold: int,
        strength: float
    ) -> NDArray:
        """Atkinson dithering - lighter than Floyd-Steinberg."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        result = gray.astype(np.float32)
        h, w = result.shape
        
        for y in range(h - 2):
            for x in range(1, w - 2):
                old_val = result[y, x]
                new_val = 255.0 if old_val > threshold else 0.0
                result[y, x] = new_val
                
                error = (old_val - new_val) * strength / 8
                
                result[y, x + 1] += error
                result[y, x + 2] += error
                result[y + 1, x - 1] += error
                result[y + 1, x] += error
                result[y + 1, x + 1] += error
                result[y + 2, x] += error
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if colors is not None and len(colors) > 1:
            result = self._map_to_palette(result, colors)
        elif len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _random_dither(
        self, 
        image: NDArray, 
        colors: Optional[NDArray],
        threshold: int
    ) -> NDArray:
        """Random dithering - fast with numpy."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        noise = np.random.randint(-64, 64, gray.shape, dtype=np.int16)
        result = np.where(gray.astype(np.int16) + noise > threshold, 255, 0).astype(np.uint8)
        
        if colors is not None and len(colors) > 1:
            result = self._map_to_palette(result, colors)
        elif len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _halftone_dither(self, image: NDArray, colors: Optional[NDArray]) -> NDArray:
        """Halftone pattern dithering."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        result = np.zeros_like(gray)
        
        dot_size = 4
        for y in range(0, h, dot_size):
            for x in range(0, w, dot_size):
                block = gray[y:y+dot_size, x:x+dot_size]
                if block.size > 0:
                    avg = np.mean(block)
                    radius = int((avg / 255) * (dot_size / 2))
                    if radius > 0:
                        cy, cx = dot_size // 2, dot_size // 2
                        yy, xx = np.ogrid[:dot_size, :dot_size]
                        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
                        result[y:y+dot_size, x:x+dot_size][mask[:block.shape[0], :block.shape[1]]] = 255
        
        if colors is not None and len(colors) > 1:
            result = self._map_to_palette(result, colors)
        elif len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _generate_bayer(self, size: int) -> NDArray:
        """Generate Bayer matrix of given size."""
        if size == 2:
            return np.array([[0, 2], [3, 1]]) / 4.0
        
        smaller = self._generate_bayer(size // 2)
        return np.block([
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ]) / (size * size)
    
    def _map_to_palette(self, gray: NDArray, colors: NDArray) -> NDArray:
        """Map grayscale to palette colors."""
        # Simple mapping based on brightness
        n_colors = len(colors)
        indices = (gray.astype(np.float32) / 255 * (n_colors - 1)).astype(np.int32)
        indices = np.clip(indices, 0, n_colors - 1)
        
        result = colors[indices]
        return result
    
    def _apply_effects(self, image: NDArray, params: Dict) -> NDArray:
        """Apply post-processing effects."""
        result = image.copy()
        
        # Invert
        if params.get("invert", 0):
            result = 255 - result
        
        # Blur
        blur = params.get("blur", 0)
        if blur > 0:
            ksize = int(blur / 10) * 2 + 1
            if ksize > 1:
                result = cv2.GaussianBlur(result, (ksize, ksize), 0)
        
        # Glow (soft bloom)
        glow = params.get("glow", 0)
        if glow > 0:
            base = result.astype(np.float32)
            intensity = min(glow / 40.0, 3.0)
            radius = max(3, int(4 + intensity * 8))
            ksize = radius * 2 + 1
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(base, (ksize, ksize), 0)
            halo_size = min(ksize + 12, 81)
            if halo_size % 2 == 0:
                halo_size += 1
            halo = cv2.GaussianBlur(base, (halo_size, halo_size), radius / 2)
            glow_mix = cv2.addWeighted(base, 1.0, blurred, 0.8 * intensity, 0)
            glow_mix = cv2.addWeighted(glow_mix, 1.0, halo, 0.5 * (intensity / 1.5), 0)
            highlight = cv2.normalize(blurred, None, 0.0, 1.0, cv2.NORM_MINMAX)
            glow_mix += (highlight ** 2) * (40.0 * intensity)
            result = np.clip(glow_mix, 0, 255).astype(np.uint8)
        
        # Light flares
        flare_intensity = params.get("flare_intensity", 0)
        if flare_intensity > 0:
            result = self._apply_flares(result, params)
        
        # Glitch accents
        glitch_intensity = params.get("glitch_intensity", 0)
        if glitch_intensity > 0:
            result = self._apply_glitch(result, params)
        
        # Grain
        grain = params.get("grain", 0)
        if grain > 0:
            noise = np.random.normal(0, grain / 10, result.shape).astype(np.float32)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Vignette
        vignette = params.get("vignette", 0)
        if vignette > 0 and len(result.shape) >= 2:
            h, w = result.shape[:2]
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            vignette_mask = 1 - (dist / max_dist) * (vignette / 100)
            vignette_mask = np.clip(vignette_mask, 0, 1)
            
            if len(result.shape) == 3:
                vignette_mask = vignette_mask[:, :, np.newaxis]
            
            result = (result * vignette_mask).astype(np.uint8)
        
        return result

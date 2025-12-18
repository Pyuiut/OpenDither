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
        """Add realistic lens flares with star patterns at bright light sources."""
        intensity = params.get("flare_intensity", 0) / 100.0
        if intensity <= 0:
            return image
        
        threshold = params.get("flare_threshold", 60)
        style = params.get("flare_style", "Lens")
        shape = params.get("flare_shape", "Star")
        flare_amount = max(1, int(params.get("flare_amount", 0) / 10) + 1)
        base_size = max(10, int(params.get("flare_size", 40)))
        hue = params.get("flare_color_hue", 40) / 360.0
        sat = params.get("flare_color_sat", 80) / 100.0
        val = params.get("flare_color_value", 90) / 100.0
        
        h, w = image.shape[:2]
        result = image.astype(np.float32)
        
        # Create flare color from HSV
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        flare_color = np.array([rgb[0] * 255, rgb[1] * 255, rgb[2] * 255], dtype=np.float32)
        
        # Detect bright spots (light sources)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to find bright areas
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright areas
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get centers of bright spots
        light_sources = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                brightness = gray[cy, cx] if 0 <= cy < h and 0 <= cx < w else 128
                light_sources.append((cx, cy, area, brightness))
        
        # Sort by brightness and take top sources
        light_sources.sort(key=lambda x: x[3], reverse=True)
        light_sources = light_sources[:flare_amount]
        
        # Create flare overlay
        flare_layer = np.zeros((h, w), dtype=np.float32)
        
        for cx, cy, area, brightness in light_sources:
            source_intensity = (brightness / 255.0) * intensity
            size = int(base_size * (1 + area / 1000))
            
            # Draw star rays
            if shape == "Star" or style == "Starburst":
                num_rays = 6 if shape == "Star" else 8
                for i in range(num_rays):
                    angle = (i / num_rays) * np.pi + np.pi / 12  # Offset for aesthetics
                    ray_length = size * 3
                    
                    # Draw ray with gradient
                    for dist in range(1, int(ray_length)):
                        falloff = 1.0 - (dist / ray_length)
                        px = int(cx + np.cos(angle) * dist)
                        py = int(cy + np.sin(angle) * dist)
                        if 0 <= px < w and 0 <= py < h:
                            flare_layer[py, px] += source_intensity * falloff * 0.8
                        # Opposite direction
                        px2 = int(cx - np.cos(angle) * dist)
                        py2 = int(cy - np.sin(angle) * dist)
                        if 0 <= px2 < w and 0 <= py2 < h:
                            flare_layer[py2, px2] += source_intensity * falloff * 0.8
            
            # Draw central glow
            glow_size = size * 2
            Y, X = np.ogrid[max(0, cy-glow_size):min(h, cy+glow_size), 
                           max(0, cx-glow_size):min(w, cx+glow_size)]
            dist_sq = (X - cx)**2 + (Y - cy)**2
            glow = np.exp(-dist_sq / (size**2 * 2)) * source_intensity
            
            y_start, y_end = max(0, cy-glow_size), min(h, cy+glow_size)
            x_start, x_end = max(0, cx-glow_size), min(w, cx+glow_size)
            flare_layer[y_start:y_end, x_start:x_end] += glow
            
            # Lens ghost reflections (circles on opposite side of image center)
            if style == "Lens":
                img_cx, img_cy = w // 2, h // 2
                # Vector from center to light source
                dx = cx - img_cx
                dy = cy - img_cy
                
                # Create ghost reflections on opposite side
                for ghost_i in range(3):
                    ghost_dist = 0.3 + ghost_i * 0.4
                    ghost_x = int(img_cx - dx * ghost_dist)
                    ghost_y = int(img_cy - dy * ghost_dist)
                    ghost_size = int(size * (0.5 - ghost_i * 0.1))
                    ghost_intensity = source_intensity * (0.3 - ghost_i * 0.08)
                    
                    if 0 <= ghost_x < w and 0 <= ghost_y < h and ghost_size > 2:
                        # Draw ghost circle
                        Y2, X2 = np.ogrid[max(0, ghost_y-ghost_size):min(h, ghost_y+ghost_size),
                                         max(0, ghost_x-ghost_size):min(w, ghost_x+ghost_size)]
                        dist_sq2 = (X2 - ghost_x)**2 + (Y2 - ghost_y)**2
                        # Ring shape for ghost
                        ring = np.exp(-((np.sqrt(dist_sq2) - ghost_size*0.7)**2) / (ghost_size*0.3)**2)
                        ring *= ghost_intensity
                        
                        y2_start, y2_end = max(0, ghost_y-ghost_size), min(h, ghost_y+ghost_size)
                        x2_start, x2_end = max(0, ghost_x-ghost_size), min(w, ghost_x+ghost_size)
                        flare_layer[y2_start:y2_end, x2_start:x2_end] += ring
            
            # Hex aperture ghosts
            elif shape == "Hex":
                for hex_i in range(6):
                    angle = hex_i * np.pi / 3
                    hx = int(cx + np.cos(angle) * size * 1.5)
                    hy = int(cy + np.sin(angle) * size * 1.5)
                    if 0 <= hx < w and 0 <= hy < h:
                        cv2.circle(flare_layer, (hx, hy), size // 3, source_intensity * 0.3, -1)
        
        # Blur the flare layer for softness
        ksize = max(3, (base_size // 2) * 2 + 1)
        flare_layer = cv2.GaussianBlur(flare_layer, (ksize, ksize), 0)
        
        # Apply flare with color
        if result.ndim == 3:
            for c in range(3):
                result[:, :, c] += flare_layer * flare_color[c]
        else:
            result += flare_layer * 255
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
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
        
        # Epsilon Glow (Dither Boy style)
        if params.get("epsilon_glow_enabled", 0):
            result = self._apply_epsilon_glow(result, params)
        
        # JPEG Glitch Effects
        result = self._apply_jpeg_glitch(result, params)
        
        # Chromatic Aberration
        if params.get("chromatic_enabled", 0):
            result = self._apply_chromatic_aberration(result, params)
        
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

    def _apply_epsilon_glow(self, image: NDArray, params: Dict) -> NDArray:
        """Apply epsilon glow effect - bloom from bright areas."""
        threshold = params.get("epsilon_threshold", 25) / 100.0 * 255
        smoothing = max(1, int(params.get("epsilon_smoothing", 25)))
        radius = max(1, int(params.get("epsilon_radius", 25)))
        intensity = params.get("epsilon_intensity", 500) / 100.0
        
        result = image.astype(np.float32)
        
        # Extract luminance
        if image.ndim == 3:
            luminance = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            luminance = image.astype(np.float32)
        
        # Create bright mask - soft threshold
        bright_mask = np.clip((luminance - threshold) / (255 - threshold + 1), 0, 1)
        
        # Extract bright pixels from original image
        if image.ndim == 3:
            bright_image = image.astype(np.float32) * bright_mask[:,:,np.newaxis]
        else:
            bright_image = image.astype(np.float32) * bright_mask
        
        # Multi-pass blur for soft glow
        glow = bright_image.copy()
        for i in range(3):
            ksize = (radius * 2 + 1) * (i + 1)
            if ksize % 2 == 0:
                ksize += 1
            ksize = min(ksize, 251)  # OpenCV limit
            glow = cv2.GaussianBlur(glow, (ksize, ksize), 0)
        
        # Smooth the glow further
        if smoothing > 1:
            smooth_k = smoothing * 2 + 1
            if smooth_k % 2 == 0:
                smooth_k += 1
            smooth_k = min(smooth_k, 251)
            glow = cv2.GaussianBlur(glow, (smooth_k, smooth_k), 0)
        
        # Add glow to original with intensity
        result = result + glow * intensity
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_jpeg_glitch(self, image: NDArray, params: Dict) -> NDArray:
        """Apply JPEG-style glitch effects."""
        result = image.copy()
        h, w = image.shape[:2]
        
        # Block shift - shift horizontal bands randomly
        block_shift = int(params.get("block_shift", 0))
        if block_shift > 0:
            block_size = 8
            rng = np.random.RandomState(42)
            for y in range(0, h, block_size):
                if rng.rand() < 0.4:  # 40% chance
                    shift = rng.randint(-block_shift * 3, block_shift * 3 + 1)
                    end_y = min(y + block_size, h)
                    result[y:end_y] = np.roll(result[y:end_y], shift, axis=1)
        
        # Channel swap - swap RGB channels in random regions
        channel_swap = int(params.get("channel_swap", 0))
        if channel_swap > 0 and result.ndim == 3:
            rng = np.random.RandomState(43)
            num_regions = channel_swap // 5 + 1
            for _ in range(num_regions):
                y1 = rng.randint(0, max(1, h - 20))
                x1 = rng.randint(0, max(1, w - 40))
                rh = rng.randint(10, min(50, h - y1))
                rw = rng.randint(20, min(100, w - x1))
                # Rotate channels
                temp = result[y1:y1+rh, x1:x1+rw].copy()
                result[y1:y1+rh, x1:x1+rw, 0] = temp[:, :, 1]
                result[y1:y1+rh, x1:x1+rw, 1] = temp[:, :, 2]
                result[y1:y1+rh, x1:x1+rw, 2] = temp[:, :, 0]
        
        # Scanline offset - create CRT-like scanline displacement
        scanline_offset = int(params.get("scanline_offset", 0))
        if scanline_offset > 0:
            for y in range(0, h, 2):
                shift = int(np.sin(y * 0.1) * scanline_offset * 0.5)
                result[y] = np.roll(result[y], shift, axis=0)
        
        # Block scramble - swap random 8x8 blocks (JPEG-like)
        block_scramble = int(params.get("block_scramble", 0))
        if block_scramble > 0 and h > 16 and w > 16:
            block_size = 8
            rng = np.random.RandomState(44)
            num_swaps = block_scramble
            for _ in range(num_swaps):
                y1 = rng.randint(0, (h - block_size) // block_size) * block_size
                x1 = rng.randint(0, (w - block_size) // block_size) * block_size
                y2 = rng.randint(0, (h - block_size) // block_size) * block_size
                x2 = rng.randint(0, (w - block_size) // block_size) * block_size
                
                block1 = result[y1:y1+block_size, x1:x1+block_size].copy()
                block2 = result[y2:y2+block_size, x2:x2+block_size].copy()
                result[y1:y1+block_size, x1:x1+block_size] = block2
                result[y2:y2+block_size, x2:x2+block_size] = block1
        
        # Interlace corruption - darken alternating lines + random shifts
        interlace = int(params.get("interlace_corruption", 0))
        if interlace > 0:
            # Darken odd lines
            darken = 1.0 - (interlace / 200.0)
            result[1::2] = (result[1::2].astype(np.float32) * darken).astype(np.uint8)
            
            # Random line corruption
            rng = np.random.RandomState(45)
            num_corrupt = interlace // 10 + 1
            for _ in range(num_corrupt):
                y = rng.randint(0, h)
                shift = rng.randint(-interlace, interlace + 1)
                result[y] = np.roll(result[y], shift, axis=0)
        
        return result

    def _apply_chromatic_aberration(self, image: NDArray, params: Dict) -> NDArray:
        """Apply chromatic aberration (RGB channel displacement)."""
        if image.ndim != 3:
            return image
        
        max_displace = int(params.get("chromatic_max_displace", 20))
        red_offset = int((params.get("chromatic_red", 50) - 50) / 50.0 * max_displace)
        green_offset = int((params.get("chromatic_green", 50) - 50) / 50.0 * max_displace)
        blue_offset = int((params.get("chromatic_blue", 50) - 50) / 50.0 * max_displace)
        
        result = image.copy()
        
        if red_offset != 0:
            result[:, :, 0] = np.roll(image[:, :, 0], red_offset, axis=1)
        if green_offset != 0:
            result[:, :, 1] = np.roll(image[:, :, 1], green_offset, axis=1)
        if blue_offset != 0:
            result[:, :, 2] = np.roll(image[:, :, 2], blue_offset, axis=1)
        
        return result

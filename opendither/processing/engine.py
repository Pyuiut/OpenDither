"""Core dithering engine with optimized algorithm implementations."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2

from opendither.core import AlgorithmLibrary, PaletteLibrary


class DitheringEngine:
    """Main processing engine for dithering operations - OPTIMIZED."""

    def __init__(self) -> None:
        self.algorithm_library = AlgorithmLibrary()
        self.palette_library = PaletteLibrary()
        self._bayer_cache: Dict[int, NDArray] = {}

    def process(
        self,
        image: NDArray[np.uint8],
        algorithm: str,
        parameters: Dict[str, float],
        palette: Optional[str] = None,
        preview_mode: bool = False,
    ) -> NDArray[np.uint8]:
        """Apply dithering to an image.
        
        Args:
            preview_mode: If True, process at lower resolution for speed
        """
        # Downscale for preview mode (much faster)
        original_size = None
        if preview_mode and image.shape[0] > 800:
            original_size = (image.shape[1], image.shape[0])
            scale = 800 / image.shape[0]
            new_size = (int(image.shape[1] * scale), 800)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # Pre-processing
        result = self._preprocess(image.copy(), parameters)
        
        # Get palette colors if specified
        colors = None
        if palette:
            pal = self.palette_library.get(palette)
            if pal:
                colors = np.array(pal.colors, dtype=np.uint8)
        
        # Apply dithering algorithm
        result = self._apply_algorithm(result, algorithm, parameters, colors)
        
        # Post-processing effects
        result = self._postprocess(result, parameters)
        
        # Upscale back if was preview
        if original_size:
            result = cv2.resize(result, original_size, interpolation=cv2.INTER_NEAREST)
        
        return result

    def _preprocess(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """Apply preprocessing adjustments."""
        result = image.astype(np.float32)
        
        # Scale (resize)
        scale = params.get("scale", 1.0)
        if scale != 1.0 and scale > 0:
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Contrast
        contrast = params.get("contrast", 1.0)
        if contrast != 1.0:
            result = (result - 128) * contrast + 128
        
        # Midtones
        midtones = params.get("midtones", 0.5)
        if midtones != 0.5:
            gamma = 1.0 / (midtones * 2 + 0.1)
            result = 255 * np.power(result / 255, gamma)
        
        # Highlights
        highlights = params.get("highlights", 0.5)
        if highlights != 0.5:
            mask = result > 128
            result[mask] = result[mask] + (highlights - 0.5) * 100
        
        # Shadows
        shadows = params.get("shadows", 0.5)
        if shadows != 0.5:
            mask = result < 128
            result[mask] = result[mask] + (shadows - 0.5) * 100
        
        # Blur
        blur = params.get("blur", 0.0)
        if blur > 0:
            ksize = int(blur * 10) * 2 + 1
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_algorithm(
        self,
        image: NDArray[np.uint8],
        algorithm: str,
        params: Dict[str, float],
        colors: Optional[NDArray[np.uint8]] = None,
    ) -> NDArray[np.uint8]:
        """Apply the selected dithering algorithm."""
        # Convert to grayscale for processing if no palette
        if colors is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = None
        
        # Route to appropriate algorithm implementation
        algo_lower = algorithm.lower()
        
        # Error Diffusion
        if "floyd" in algo_lower:
            return self._floyd_steinberg(image, colors)
        elif "jarvis" in algo_lower:
            return self._jarvis(image, colors)
        elif "stucki" in algo_lower:
            return self._stucki(image, colors)
        elif "burkes" in algo_lower:
            return self._burkes(image, colors)
        elif "sierra" in algo_lower:
            return self._sierra(image, colors)
        elif "atkinson" in algo_lower:
            return self._atkinson(image, colors)
        
        # Ordered Dithering
        elif "bayer" in algo_lower:
            size = 4
            if "2x2" in algorithm:
                size = 2
            elif "8x8" in algorithm:
                size = 8
            elif "16x16" in algorithm:
                size = 16
            return self._bayer(image, size, colors)
        
        # Pattern
        elif "crosshatch" in algo_lower:
            return self._crosshatch(image, colors)
        elif "stipple" in algo_lower:
            return self._stipple(image, colors)
        elif "checkerboard" in algo_lower:
            return self._checkerboard(image, colors)
        
        # Halftone
        elif "halftone" in algo_lower or "clustered" in algo_lower:
            return self._halftone(image, colors)
        
        # Special Effects
        elif "scanline" in algo_lower:
            return self._scanlines(image, params)
        elif "noise" in algo_lower or "grain" in algo_lower:
            return self._noise_grain(image, params)
        elif "glitch" in algo_lower:
            return self._glitch(image, params)
        elif "pixelate" in algo_lower:
            return self._pixelate(image, params)
        elif "posterize" in algo_lower:
            return self._posterize(image, params)
        elif "vhs" in algo_lower:
            return self._vhs(image, params)
        elif "crt" in algo_lower:
            return self._crt(image, params)
        elif "chromatic" in algo_lower:
            return self._chromatic_aberration(image, params)
        elif "bit" in algo_lower and "crush" in algo_lower:
            return self._bit_crush(image, params)
        
        # Default: Floyd-Steinberg
        return self._floyd_steinberg(image, colors)

    def _postprocess(
        self,
        image: NDArray[np.uint8],
        params: Dict[str, float],
    ) -> NDArray[np.uint8]:
        """Apply post-processing effects."""
        result = image.copy()
        
        # Invert
        if params.get("invert", 0.0) > 0.5:
            result = 255 - result
        
        # Glow effect
        glow = params.get("glow", 0.0)
        if glow > 0:
            blurred = cv2.GaussianBlur(result, (21, 21), 0)
            result = cv2.addWeighted(result, 1.0, blurred, glow, 0)
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    # ========== OPTIMIZED Error Diffusion Algorithms ==========

    def _floyd_steinberg(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Floyd-Steinberg - OPTIMIZED with row-based processing."""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        # Process row by row for better cache performance
        for y in range(h):
            row = img[y].copy()
            for x in range(w):
                old_val = row[x].copy()
                new_val = self._quantize_fast(old_val, colors)
                row[x] = new_val
                error = old_val - new_val
                
                if x + 1 < w:
                    row[x + 1] = np.clip(row[x + 1] + error * 0.4375, 0, 255)
            
            img[y] = row
            
            # Diffuse to next row
            if y + 1 < h:
                err_row = image[y].astype(np.float32) - img[y]
                img[y + 1, :-1] = np.clip(img[y + 1, :-1] + err_row[1:] * 0.1875, 0, 255)
                img[y + 1] = np.clip(img[y + 1] + err_row * 0.3125, 0, 255)
                img[y + 1, 1:] = np.clip(img[y + 1, 1:] + err_row[:-1] * 0.0625, 0, 255)
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _quantize_fast(
        self,
        pixel: NDArray[np.float32],
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.float32]:
        """Fast quantization."""
        if colors is None:
            avg = np.mean(pixel)
            return np.array([255.0, 255.0, 255.0]) if avg > 127 else np.array([0.0, 0.0, 0.0])
        
        # Vectorized distance calculation
        diff = colors.astype(np.float32) - pixel
        distances = np.sum(diff * diff, axis=1)
        return colors[np.argmin(distances)].astype(np.float32)

    def _jarvis(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Jarvis-Judice-Ninke error diffusion."""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        kernel = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ]) / 48.0
        
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x].copy()
                new_pixel = self._find_closest_color(old_pixel, colors)
                img[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                for ky in range(3):
                    for kx in range(5):
                        ny, nx = y + ky, x + kx - 2
                        if 0 <= ny < h and 0 <= nx < w and kernel[ky, kx] > 0:
                            img[ny, nx] += error * kernel[ky, kx]
        
        return np.clip(img, 0, 255).astype(np.uint8)

    def _stucki(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Stucki error diffusion."""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        kernel = np.array([
            [0, 0, 0, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1]
        ]) / 42.0
        
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x].copy()
                new_pixel = self._find_closest_color(old_pixel, colors)
                img[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                for ky in range(3):
                    for kx in range(5):
                        ny, nx = y + ky, x + kx - 2
                        if 0 <= ny < h and 0 <= nx < w and kernel[ky, kx] > 0:
                            img[ny, nx] += error * kernel[ky, kx]
        
        return np.clip(img, 0, 255).astype(np.uint8)

    def _burkes(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Burkes error diffusion."""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x].copy()
                new_pixel = self._find_closest_color(old_pixel, colors)
                img[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    img[y, x + 1] += error * 8 / 32
                if x + 2 < w:
                    img[y, x + 2] += error * 4 / 32
                if y + 1 < h:
                    for dx, weight in [(-2, 2), (-1, 4), (0, 8), (1, 4), (2, 2)]:
                        nx = x + dx
                        if 0 <= nx < w:
                            img[y + 1, nx] += error * weight / 32
        
        return np.clip(img, 0, 255).astype(np.uint8)

    def _sierra(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Sierra error diffusion."""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x].copy()
                new_pixel = self._find_closest_color(old_pixel, colors)
                img[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                weights = [
                    (0, 1, 5), (0, 2, 3),
                    (1, -2, 2), (1, -1, 4), (1, 0, 5), (1, 1, 4), (1, 2, 2),
                    (2, -1, 2), (2, 0, 3), (2, 1, 2)
                ]
                for dy, dx, weight in weights:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        img[ny, nx] += error * weight / 32
        
        return np.clip(img, 0, 255).astype(np.uint8)

    def _atkinson(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Atkinson error diffusion (Macintosh style)."""
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        for y in range(h):
            for x in range(w):
                old_pixel = img[y, x].copy()
                new_pixel = self._find_closest_color(old_pixel, colors)
                img[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                # Atkinson only diffuses 6/8 of error (loses 1/4)
                positions = [
                    (0, 1), (0, 2),
                    (1, -1), (1, 0), (1, 1),
                    (2, 0)
                ]
                for dy, dx in positions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        img[ny, nx] += error / 8
        
        return np.clip(img, 0, 255).astype(np.uint8)

    # ========== Ordered Dithering ==========

    def _bayer(
        self, 
        image: NDArray[np.uint8], 
        size: int = 4,
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Bayer ordered dithering."""
        # Generate Bayer matrix
        if size == 2:
            bayer = np.array([[0, 2], [3, 1]]) / 4.0
        elif size == 4:
            bayer = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ]) / 16.0
        elif size == 8:
            b4 = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ])
            bayer = np.zeros((8, 8))
            bayer[:4, :4] = b4 * 4
            bayer[:4, 4:] = b4 * 4 + 2
            bayer[4:, :4] = b4 * 4 + 3
            bayer[4:, 4:] = b4 * 4 + 1
            bayer /= 64.0
        else:  # 16x16
            bayer = self._generate_bayer(4) / 256.0
        
        h, w = image.shape[:2]
        result = image.astype(np.float32)
        
        # Tile the bayer matrix
        bayer_tiled = np.tile(bayer, (h // size + 1, w // size + 1))[:h, :w]
        
        # Apply threshold
        threshold = (bayer_tiled - 0.5) * 128
        
        if len(result.shape) == 3:
            threshold = threshold[:, :, np.newaxis]
        
        result = result + threshold
        
        # Quantize
        if colors is not None:
            result = self._quantize_to_palette(result, colors)
        else:
            result = np.where(result > 127, 255, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _generate_bayer(self, n: int) -> NDArray[np.float32]:
        """Generate a 2^n x 2^n Bayer matrix."""
        if n == 0:
            return np.array([[0]])
        smaller = self._generate_bayer(n - 1)
        size = 2 ** n
        result = np.zeros((size, size))
        half = size // 2
        result[:half, :half] = 4 * smaller
        result[:half, half:] = 4 * smaller + 2
        result[half:, :half] = 4 * smaller + 3
        result[half:, half:] = 4 * smaller + 1
        return result

    # ========== Pattern Dithering ==========

    def _crosshatch(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Crosshatch pattern dithering."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Create hatching based on brightness levels
        for y in range(h):
            for x in range(w):
                val = gray[y, x]
                if val < 64:
                    result[y, x] = [0, 0, 0]
                elif val < 128:
                    if (x + y) % 4 == 0 or (x - y) % 4 == 0:
                        result[y, x] = [0, 0, 0]
                elif val < 192:
                    if (x + y) % 8 == 0:
                        result[y, x] = [0, 0, 0]
        
        return result

    def _stipple(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Random stipple dithering."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        noise = np.random.random(gray.shape) * 255
        result = np.where(gray > noise, 255, 0).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    def _checkerboard(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Checkerboard pattern dithering."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        checker = np.indices((h, w)).sum(axis=0) % 2
        threshold = 128 + (checker * 2 - 1) * 64
        result = np.where(gray > threshold, 255, 0).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    # ========== Halftone ==========

    def _halftone(
        self, 
        image: NDArray[np.uint8], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.uint8]:
        """Circular halftone pattern."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        result = np.ones((h, w), dtype=np.uint8) * 255
        
        cell_size = 4
        for y in range(0, h, cell_size):
            for x in range(0, w, cell_size):
                region = gray[y:y+cell_size, x:x+cell_size]
                if region.size == 0:
                    continue
                avg = np.mean(region)
                radius = int((1 - avg / 255) * cell_size / 2)
                if radius > 0:
                    cy, cx = y + cell_size // 2, x + cell_size // 2
                    cv2.circle(result, (cx, cy), radius, 0, -1)
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    # ========== Special Effects ==========

    def _scanlines(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """CRT scanline effect."""
        result = image.copy()
        h = result.shape[0]
        for y in range(0, h, 2):
            result[y] = (result[y].astype(np.float32) * 0.7).astype(np.uint8)
        return result

    def _noise_grain(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """Film grain noise effect."""
        intensity = params.get("glitch", 0.3) * 100
        noise = np.random.randn(*image.shape) * intensity
        result = image.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _glitch(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """Random glitch block effect."""
        result = image.copy()
        h, w = result.shape[:2]
        intensity = params.get("glitch", 0.5)
        num_glitches = int(intensity * 20)
        
        for _ in range(num_glitches):
            y = np.random.randint(0, h - 10)
            height = np.random.randint(2, 20)
            shift = np.random.randint(-50, 50)
            result[y:y+height] = np.roll(result[y:y+height], shift, axis=1)
        
        return result

    def _pixelate(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """Pixelation effect."""
        scale = max(1, int(params.get("scale", 1.0) * 8))
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _posterize(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """Color posterization effect."""
        levels = max(2, int(params.get("contrast", 1.0) * 4))
        divisor = 256 // levels
        return (image // divisor * divisor).astype(np.uint8)

    def _vhs(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """VHS tape degradation effect."""
        result = image.copy()
        # Color bleeding
        result[:, :, 0] = np.roll(result[:, :, 0], 2, axis=1)
        result[:, :, 2] = np.roll(result[:, :, 2], -2, axis=1)
        # Add noise
        noise = np.random.randn(*result.shape) * 15
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Scanlines
        result[::2] = (result[::2].astype(np.float32) * 0.9).astype(np.uint8)
        return result

    def _crt(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """CRT monitor simulation."""
        result = image.copy()
        h, w = result.shape[:2]
        # RGB subpixel pattern
        for x in range(w):
            channel = x % 3
            mask = np.ones(3) * 0.7
            mask[channel] = 1.0
            result[:, x] = (result[:, x].astype(np.float32) * mask).astype(np.uint8)
        # Scanlines
        result[::2] = (result[::2].astype(np.float32) * 0.8).astype(np.uint8)
        return result

    def _chromatic_aberration(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """RGB channel separation effect."""
        offset = max(1, int(params.get("glitch", 0.5) * 10))
        result = image.copy()
        result[:, :, 0] = np.roll(image[:, :, 0], offset, axis=1)
        result[:, :, 2] = np.roll(image[:, :, 2], -offset, axis=1)
        return result

    def _bit_crush(
        self, 
        image: NDArray[np.uint8], 
        params: Dict[str, float]
    ) -> NDArray[np.uint8]:
        """Bit depth reduction effect."""
        bits = max(1, int(params.get("contrast", 1.0) * 4))
        divisor = 2 ** (8 - bits)
        return ((image // divisor) * divisor).astype(np.uint8)

    # ========== Utility Methods ==========

    def _find_closest_color(
        self, 
        pixel: NDArray[np.float32], 
        colors: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.float32]:
        """Find the closest color in palette or B&W."""
        if colors is None:
            # Simple black/white threshold
            if len(pixel.shape) == 0 or pixel.ndim == 0:
                return np.array([255.0, 255.0, 255.0]) if pixel > 127 else np.array([0.0, 0.0, 0.0])
            avg = np.mean(pixel)
            return np.array([255.0, 255.0, 255.0]) if avg > 127 else np.array([0.0, 0.0, 0.0])
        
        # Find closest color in palette
        distances = np.sum((colors.astype(np.float32) - pixel) ** 2, axis=1)
        return colors[np.argmin(distances)].astype(np.float32)

    def _quantize_to_palette(
        self, 
        image: NDArray[np.float32], 
        colors: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """Quantize image to palette colors."""
        h, w = image.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                pixel = image[y, x]
                distances = np.sum((colors.astype(np.float32) - pixel) ** 2, axis=1)
                result[y, x] = colors[np.argmin(distances)]
        
        return result

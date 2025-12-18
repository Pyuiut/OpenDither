"""Processing pipeline with Dither Boy-style effects."""

from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np
from numpy.typing import NDArray
import cv2

from opendither.core import PaletteLibrary


class ProcessingPipeline:
    """Pipeline with epsilon glow, JPEG glitch, chromatic aberration, and advanced dithering."""
    
    def __init__(self):
        self.palette_library = PaletteLibrary()
        self._progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def process(
        self,
        image: NDArray[np.uint8],
        algorithm: str,
        parameters: Dict[str, float],
        palette: Optional[str] = None,
        curves_lut: Optional[Dict[str, List[int]]] = None,
    ) -> NDArray[np.uint8]:
        """Process image through the pipeline."""
        
        result = image.copy()
        
        # Get palette colors if specified
        colors = None
        if palette:
            pal = self.palette_library.get(palette)
            if pal:
                colors = np.array(pal.colors, dtype=np.uint8)
        
        # Apply pre-dither effects (contrast, midtones, etc.)
        if parameters.get("effects_enabled", 0):
            result = self._apply_tone_effects(result, parameters)
        
        # Apply blur if specified
        blur = parameters.get("blur", 0)
        if blur > 0:
            ksize = int(blur) * 2 + 1
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)
        
        # Apply dithering
        result = self._apply_dithering(result, algorithm, parameters, colors)
        
        # Apply epsilon glow
        if parameters.get("epsilon_glow_enabled", 0):
            result = self._apply_epsilon_glow(result, parameters)
        
        # Apply JPEG glitch effects
        result = self._apply_jpeg_glitch(result, parameters)
        
        # Apply chromatic aberration
        if parameters.get("chromatic_enabled", 0):
            result = self._apply_chromatic_aberration(result, parameters)
        
        # Invert if requested
        if parameters.get("invert", 0):
            result = 255 - result
        
        return result

    def _apply_tone_effects(self, image: NDArray, params: Dict) -> NDArray:
        """Apply contrast, midtones, highlights, luminance threshold."""
        result = image.astype(np.float32)
        
        # Contrast (centered at 50)
        contrast = (params.get("contrast", 50) - 50) / 50.0  # -1 to 1
        if abs(contrast) > 0.01:
            factor = 1.0 + contrast
            result = 128 + (result - 128) * factor
        
        # Midtones (gamma adjustment, centered at 50)
        midtones = params.get("midtones", 50)
        if midtones != 50:
            gamma = 1.0 + (50 - midtones) / 50.0  # Inverted: lower value = brighter
            gamma = max(0.2, min(3.0, gamma))
            result = 255.0 * np.power(result / 255.0, gamma)
        
        # Highlights (brighten bright areas)
        highlights = (params.get("highlights", 50) - 50) / 50.0
        if abs(highlights) > 0.01:
            mask = result / 255.0
            result = result + highlights * 50 * mask
        
        # Luminance threshold
        lum_thresh = params.get("luminance_threshold", 50)
        if lum_thresh != 50:
            # Soft threshold effect
            threshold = lum_thresh * 2.55  # 0-255
            result = np.where(result < threshold, result * 0.8, result)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_dithering(
        self,
        image: NDArray,
        algorithm: str,
        params: Dict,
        colors: Optional[NDArray] = None
    ) -> NDArray:
        """Apply dithering algorithm."""
        
        if algorithm == "None":
            if colors is not None:
                return self._quantize_to_palette(image, colors)
            return image
        
        scale = max(1, int(params.get("scale", 2)))
        line_scale = max(1, int(params.get("line_scale", 1)))
        
        # Downscale for pixelation effect
        h, w = image.shape[:2]
        small_w, small_h = max(1, w // scale), max(1, h // scale)
        
        # Downscale
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Apply dithering algorithm
        if algorithm == "Modulated Diffuse Y":
            dithered = self._modulated_diffuse_y(small, colors, line_scale)
        elif algorithm == "Waveform":
            dithered = self._waveform_dither(small, colors, line_scale)
        elif algorithm == "Floyd-Steinberg":
            dithered = self._floyd_steinberg(small, colors)
        elif algorithm == "Atkinson":
            dithered = self._atkinson_dither(small, colors)
        elif algorithm.startswith("Bayer"):
            size = int(algorithm.split()[1].split("x")[0])
            dithered = self._bayer_dither(small, size, colors)
        elif algorithm == "Halftone Dot":
            dithered = self._halftone_dot(small, colors)
        elif algorithm == "Halftone Line":
            dithered = self._halftone_line(small, colors, line_scale)
        elif algorithm == "Blue Noise":
            dithered = self._blue_noise_dither(small, colors)
        elif algorithm == "Error Diffusion":
            dithered = self._error_diffusion(small, colors)
        else:
            dithered = small
        
        # Upscale back with nearest neighbor for crisp pixels
        result = cv2.resize(dithered, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return result

    def _modulated_diffuse_y(
        self,
        image: NDArray,
        colors: Optional[NDArray],
        line_scale: int = 1
    ) -> NDArray:
        """Modulated diffusion dithering with Y-axis wave pattern."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create modulated threshold pattern
        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)
        
        # Wave modulation based on position
        wave_freq = 0.1 * line_scale
        wave = np.sin(y_coords * wave_freq + x_coords * 0.05) * 30
        wave += np.sin(y_coords * wave_freq * 2.3) * 20
        
        threshold = 128 + wave
        
        # Apply threshold with error diffusion
        error = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                old_val = gray[y, x] + error[y, x]
                new_val = 255 if old_val > threshold[y, x] else 0
                
                quant_error = old_val - new_val
                
                # Distribute error
                if x + 1 < w:
                    error[y, x + 1] += quant_error * 0.4375
                if y + 1 < h:
                    if x > 0:
                        error[y + 1, x - 1] += quant_error * 0.1875
                    error[y + 1, x] += quant_error * 0.3125
                    if x + 1 < w:
                        error[y + 1, x + 1] += quant_error * 0.0625
                
                gray[y, x] = new_val
        
        # Map to colors
        if colors is not None and len(colors) >= 2:
            dark_color = colors[0]
            light_color = colors[-1]
            result[gray == 0] = dark_color
            result[gray == 255] = light_color
        else:
            result[gray == 0] = [0, 0, 0]
            result[gray == 255] = [255, 255, 255]
        
        return result

    def _waveform_dither(
        self,
        image: NDArray,
        colors: Optional[NDArray],
        line_scale: int = 1
    ) -> NDArray:
        """Waveform-based dithering with horizontal line patterns."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create waveform pattern
        wave_density = max(1, 15 // line_scale)
        
        for y in range(h):
            row_intensity = gray[y, :].astype(np.float32) / 255.0
            
            # Create wave pattern based on intensity
            wave_height = (1.0 - row_intensity) * wave_density
            
            for x in range(w):
                # Distance from wave center
                wave_y = y % wave_density
                center = wave_density / 2
                dist = abs(wave_y - center)
                
                if dist < wave_height[x]:
                    # Inside wave - use dark color
                    if colors is not None and len(colors) >= 2:
                        result[y, x] = colors[0]
                    else:
                        result[y, x] = [0, 0, 0]
                else:
                    # Outside wave - use light color
                    if colors is not None and len(colors) >= 2:
                        result[y, x] = colors[-1]
                    else:
                        result[y, x] = [255, 255, 255]
        
        return result

    def _floyd_steinberg(self, image: NDArray, colors: Optional[NDArray]) -> NDArray:
        """Classic Floyd-Steinberg error diffusion."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h, w = gray.shape
        
        for y in range(h):
            for x in range(w):
                old = gray[y, x]
                new = 255 if old > 127 else 0
                gray[y, x] = new
                error = old - new
                
                if x + 1 < w:
                    gray[y, x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        gray[y + 1, x - 1] += error * 3 / 16
                    gray[y + 1, x] += error * 5 / 16
                    if x + 1 < w:
                        gray[y + 1, x + 1] += error * 1 / 16
        
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        if colors is not None and len(colors) >= 2:
            result = np.zeros((h, w, 3), dtype=np.uint8)
            result[gray == 0] = colors[0]
            result[gray == 255] = colors[-1]
            return result
        
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _atkinson_dither(self, image: NDArray, colors: Optional[NDArray]) -> NDArray:
        """Atkinson dithering (lighter than Floyd-Steinberg)."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h, w = gray.shape
        
        for y in range(h):
            for x in range(w):
                old = gray[y, x]
                new = 255 if old > 127 else 0
                gray[y, x] = new
                error = (old - new) / 8
                
                if x + 1 < w:
                    gray[y, x + 1] += error
                if x + 2 < w:
                    gray[y, x + 2] += error
                if y + 1 < h:
                    if x > 0:
                        gray[y + 1, x - 1] += error
                    gray[y + 1, x] += error
                    if x + 1 < w:
                        gray[y + 1, x + 1] += error
                if y + 2 < h:
                    gray[y + 2, x] += error
        
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        if colors is not None and len(colors) >= 2:
            result = np.zeros((h, w, 3), dtype=np.uint8)
            result[gray == 0] = colors[0]
            result[gray == 255] = colors[-1]
            return result
        
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _bayer_dither(self, image: NDArray, size: int, colors: Optional[NDArray]) -> NDArray:
        """Ordered Bayer dithering."""
        bayer_matrices = {
            2: np.array([[0, 2], [3, 1]]) / 4.0,
            4: np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ]) / 16.0,
            8: self._generate_bayer_matrix(8) / 64.0,
        }
        
        matrix = bayer_matrices.get(size, bayer_matrices[4])
        
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Tile the Bayer matrix
        tiled = np.tile(matrix, (h // size + 1, w // size + 1))[:h, :w]
        
        # Apply threshold
        threshold = (tiled * 255).astype(np.float32)
        dithered = (gray > threshold).astype(np.uint8) * 255
        
        if colors is not None and len(colors) >= 2:
            result = np.zeros((h, w, 3), dtype=np.uint8)
            result[dithered == 0] = colors[0]
            result[dithered == 255] = colors[-1]
            return result
        
        return cv2.cvtColor(dithered, cv2.COLOR_GRAY2RGB)

    def _generate_bayer_matrix(self, size: int) -> NDArray:
        """Generate Bayer matrix of given size."""
        if size == 1:
            return np.array([[0]])
        
        smaller = self._generate_bayer_matrix(size // 2)
        return np.block([
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ])

    def _halftone_dot(self, image: NDArray, colors: Optional[NDArray]) -> NDArray:
        """Halftone dot pattern dithering."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        cell_size = 4
        
        for cy in range(0, h, cell_size):
            for cx in range(0, w, cell_size):
                # Get average intensity for cell
                cell = gray[cy:cy+cell_size, cx:cx+cell_size]
                avg = np.mean(cell) / 255.0
                
                # Draw circle based on intensity
                radius = int((1 - avg) * cell_size / 2)
                
                for dy in range(cell_size):
                    for dx in range(cell_size):
                        if cy + dy >= h or cx + dx >= w:
                            continue
                        
                        dist = np.sqrt((dx - cell_size/2)**2 + (dy - cell_size/2)**2)
                        
                        if dist < radius:
                            if colors is not None and len(colors) >= 2:
                                result[cy + dy, cx + dx] = colors[0]
                            else:
                                result[cy + dy, cx + dx] = [0, 0, 0]
                        else:
                            if colors is not None and len(colors) >= 2:
                                result[cy + dy, cx + dx] = colors[-1]
                            else:
                                result[cy + dy, cx + dx] = [255, 255, 255]
        
        return result

    def _halftone_line(self, image: NDArray, colors: Optional[NDArray], line_scale: int) -> NDArray:
        """Halftone line pattern dithering."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        line_spacing = max(2, 4 // line_scale)
        
        for y in range(h):
            line_phase = y % line_spacing
            threshold = (line_phase / line_spacing) * 255
            
            for x in range(w):
                if gray[y, x] < threshold:
                    if colors is not None and len(colors) >= 2:
                        result[y, x] = colors[0]
                    else:
                        result[y, x] = [0, 0, 0]
                else:
                    if colors is not None and len(colors) >= 2:
                        result[y, x] = colors[-1]
                    else:
                        result[y, x] = [255, 255, 255]
        
        return result

    def _blue_noise_dither(self, image: NDArray, colors: Optional[NDArray]) -> NDArray:
        """Blue noise dithering using random threshold."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Generate blue noise-like pattern
        np.random.seed(42)  # Consistent pattern
        noise = np.random.rand(h, w) * 255
        
        dithered = (gray > noise).astype(np.uint8) * 255
        
        if colors is not None and len(colors) >= 2:
            result = np.zeros((h, w, 3), dtype=np.uint8)
            result[dithered == 0] = colors[0]
            result[dithered == 255] = colors[-1]
            return result
        
        return cv2.cvtColor(dithered, cv2.COLOR_GRAY2RGB)

    def _error_diffusion(self, image: NDArray, colors: Optional[NDArray]) -> NDArray:
        """Generic error diffusion dithering."""
        return self._floyd_steinberg(image, colors)

    def _quantize_to_palette(self, image: NDArray, colors: NDArray) -> NDArray:
        """Quantize image to palette colors without dithering."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w = image.shape[:2]
        flat = image.reshape(-1, 3).astype(np.float32)
        
        # Find nearest color for each pixel
        result = np.zeros_like(flat)
        for i, pixel in enumerate(flat):
            distances = np.sum((colors.astype(np.float32) - pixel) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            result[i] = colors[nearest_idx]
        
        return result.reshape(h, w, 3).astype(np.uint8)

    def _apply_epsilon_glow(self, image: NDArray, params: Dict) -> NDArray:
        """Apply epsilon glow effect (Dither Boy signature effect)."""
        threshold = params.get("epsilon_threshold", 25)
        smoothing = params.get("epsilon_smoothing", 25)
        radius = max(1, int(params.get("epsilon_radius", 25)))
        intensity = params.get("epsilon_intensity", 500) / 100.0
        aspect = params.get("epsilon_aspect", 100) / 100.0
        direction = params.get("epsilon_direction", 0)
        falloff = params.get("epsilon_falloff", 10)
        epsilon = params.get("epsilon_value", 50)
        distance_scale = params.get("epsilon_distance_scale", 150)
        
        result = image.astype(np.float32)
        
        # Extract bright areas
        if image.ndim == 3:
            luminance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            luminance = image.copy()
        
        # Threshold for glow source
        thresh_val = threshold * 2.55
        _, bright_mask = cv2.threshold(luminance, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Smooth the mask
        if smoothing > 0:
            smooth_size = smoothing * 2 + 1
            bright_mask = cv2.GaussianBlur(bright_mask, (smooth_size, smooth_size), 0)
        
        # Create directional kernel for glow
        ksize = radius * 2 + 1
        
        # Apply aspect ratio and direction
        kernel_x = int(ksize * aspect)
        kernel_y = ksize
        
        if kernel_x % 2 == 0:
            kernel_x += 1
        
        # Create glow
        glow = cv2.GaussianBlur(bright_mask.astype(np.float32), (kernel_x, kernel_y), 0)
        
        # Apply distance falloff
        if falloff > 1:
            glow = np.power(glow / 255.0, falloff / 10.0) * 255
        
        # Normalize and apply intensity
        glow = glow / 255.0 * intensity
        
        # Add glow to image
        if result.ndim == 3:
            for c in range(3):
                result[:, :, c] = result[:, :, c] + glow * (epsilon / 50.0)
        else:
            result = result + glow * (epsilon / 50.0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_jpeg_glitch(self, image: NDArray, params: Dict) -> NDArray:
        """Apply JPEG-style glitch effects."""
        result = image.copy()
        
        # Block shift
        block_shift = int(params.get("block_shift", 0))
        if block_shift > 0:
            result = self._apply_block_shift(result, block_shift)
        
        # Channel swap
        channel_swap = params.get("channel_swap", 0) / 100.0
        if channel_swap > 0:
            result = self._apply_channel_swap(result, channel_swap)
        
        # Scanline offset
        scanline_offset = int(params.get("scanline_offset", 0))
        if scanline_offset > 0:
            result = self._apply_scanlines(result, scanline_offset)
        
        # Block scramble
        block_scramble = int(params.get("block_scramble", 0))
        if block_scramble > 0:
            result = self._apply_block_scramble(result, block_scramble)
        
        # Interlace corruption
        interlace = params.get("interlace_corruption", 0) / 100.0
        if interlace > 0:
            result = self._apply_interlace(result, interlace)
        
        return result

    def _apply_block_shift(self, image: NDArray, amount: int) -> NDArray:
        """Shift random blocks horizontally."""
        result = image.copy()
        h, w = image.shape[:2]
        
        block_size = 8
        np.random.seed(42)
        
        for y in range(0, h, block_size):
            if np.random.rand() < 0.3:  # 30% chance to shift
                shift = np.random.randint(-amount, amount + 1)
                result[y:y+block_size] = np.roll(result[y:y+block_size], shift, axis=1)
        
        return result

    def _apply_channel_swap(self, image: NDArray, amount: float) -> NDArray:
        """Randomly swap color channels."""
        if image.ndim != 3:
            return image
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Random regions to swap
        np.random.seed(43)
        num_regions = int(amount * 20)
        
        for _ in range(num_regions):
            y = np.random.randint(0, h - 16)
            x = np.random.randint(0, w - 16)
            region_h = np.random.randint(8, 32)
            region_w = np.random.randint(8, 64)
            
            y2 = min(y + region_h, h)
            x2 = min(x + region_w, w)
            
            # Swap channels
            swap_type = np.random.randint(0, 3)
            if swap_type == 0:
                result[y:y2, x:x2, 0], result[y:y2, x:x2, 1] = \
                    result[y:y2, x:x2, 1].copy(), result[y:y2, x:x2, 0].copy()
            elif swap_type == 1:
                result[y:y2, x:x2, 1], result[y:y2, x:x2, 2] = \
                    result[y:y2, x:x2, 2].copy(), result[y:y2, x:x2, 1].copy()
            else:
                result[y:y2, x:x2, 0], result[y:y2, x:x2, 2] = \
                    result[y:y2, x:x2, 2].copy(), result[y:y2, x:x2, 0].copy()
        
        return result

    def _apply_scanlines(self, image: NDArray, offset: int) -> NDArray:
        """Apply scanline effect with offset."""
        result = image.copy()
        h = image.shape[0]
        
        for y in range(0, h, 2):
            if y + 1 < h:
                shift = (offset * (y % 4 - 2)) // 2
                result[y] = np.roll(result[y], shift, axis=0)
        
        return result

    def _apply_block_scramble(self, image: NDArray, amount: int) -> NDArray:
        """Scramble random blocks."""
        result = image.copy()
        h, w = image.shape[:2]
        
        block_size = 16
        np.random.seed(44)
        
        num_scrambles = amount * 2
        
        for _ in range(num_scrambles):
            y1 = np.random.randint(0, h - block_size)
            x1 = np.random.randint(0, w - block_size)
            y2 = np.random.randint(0, h - block_size)
            x2 = np.random.randint(0, w - block_size)
            
            # Swap blocks
            block1 = result[y1:y1+block_size, x1:x1+block_size].copy()
            block2 = result[y2:y2+block_size, x2:x2+block_size].copy()
            
            result[y1:y1+block_size, x1:x1+block_size] = block2
            result[y2:y2+block_size, x2:x2+block_size] = block1
        
        return result

    def _apply_interlace(self, image: NDArray, amount: float) -> NDArray:
        """Apply interlace corruption effect."""
        result = image.copy()
        h = image.shape[0]
        
        # Darken every other line
        for y in range(0, h, 2):
            result[y] = (result[y].astype(np.float32) * (1.0 - amount * 0.5)).astype(np.uint8)
        
        # Random line corruption
        np.random.seed(45)
        num_corrupt = int(amount * h * 0.1)
        
        for _ in range(num_corrupt):
            y = np.random.randint(0, h)
            if np.random.rand() < 0.5:
                result[y] = np.roll(result[y], np.random.randint(-20, 20), axis=0)
        
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
        
        # Shift each channel
        if red_offset != 0:
            result[:, :, 0] = np.roll(image[:, :, 0], red_offset, axis=1)
        if green_offset != 0:
            result[:, :, 1] = np.roll(image[:, :, 1], green_offset, axis=1)
        if blue_offset != 0:
            result[:, :, 2] = np.roll(image[:, :, 2], blue_offset, axis=1)
        
        return result

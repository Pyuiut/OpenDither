"""Real-time histogram widget for image analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QLinearGradient
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QFrame, QLabel


class HistogramCanvas(QWidget):
    """Canvas for drawing histogram."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 100)
        self.setMaximumHeight(120)
        
        self.histogram_r: Optional[NDArray] = None
        self.histogram_g: Optional[NDArray] = None
        self.histogram_b: Optional[NDArray] = None
        self.histogram_l: Optional[NDArray] = None
        
        self.show_rgb = True
        self.show_luminance = True
    
    def set_image(self, image: Optional[NDArray[np.uint8]]):
        """Calculate histogram from image."""
        if image is None:
            self.histogram_r = None
            self.histogram_g = None
            self.histogram_b = None
            self.histogram_l = None
            self.update()
            return
        
        if len(image.shape) == 3:
            # RGB image
            self.histogram_r = np.histogram(image[:, :, 0], bins=256, range=(0, 256))[0]
            self.histogram_g = np.histogram(image[:, :, 1], bins=256, range=(0, 256))[0]
            self.histogram_b = np.histogram(image[:, :, 2], bins=256, range=(0, 256))[0]
            
            # Luminance
            luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            self.histogram_l = np.histogram(luminance, bins=256, range=(0, 256))[0]
        else:
            # Grayscale
            self.histogram_l = np.histogram(image, bins=256, range=(0, 256))[0]
            self.histogram_r = None
            self.histogram_g = None
            self.histogram_b = None
        
        self.update()
    
    def paintEvent(self, event):
        """Draw the histogram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 5
        
        # Background
        painter.fillRect(0, 0, w, h, QColor("#18181b"))
        
        # Border
        painter.setPen(QPen(QColor("#27272a"), 1))
        painter.drawRect(margin, margin, w - 2 * margin, h - 2 * margin)
        
        if self.histogram_l is None and self.histogram_r is None:
            # No data
            painter.setPen(QColor("#52525b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            return
        
        # Find max for scaling
        max_val = 1
        if self.show_luminance and self.histogram_l is not None:
            max_val = max(max_val, np.max(self.histogram_l))
        if self.show_rgb:
            if self.histogram_r is not None:
                max_val = max(max_val, np.max(self.histogram_r))
            if self.histogram_g is not None:
                max_val = max(max_val, np.max(self.histogram_g))
            if self.histogram_b is not None:
                max_val = max(max_val, np.max(self.histogram_b))
        
        draw_w = w - 2 * margin
        draw_h = h - 2 * margin
        
        def draw_histogram(hist, color, alpha=100):
            if hist is None:
                return
            
            path = QPainterPath()
            path.moveTo(margin, h - margin)
            
            for i, val in enumerate(hist):
                x = margin + (i / 255) * draw_w
                y = h - margin - (val / max_val) * draw_h * 0.95
                path.lineTo(x, y)
            
            path.lineTo(margin + draw_w, h - margin)
            path.closeSubpath()
            
            # Fill
            fill_color = QColor(color)
            fill_color.setAlpha(alpha)
            painter.fillPath(path, fill_color)
            
            # Stroke
            stroke_color = QColor(color)
            stroke_color.setAlpha(min(255, alpha + 80))
            painter.setPen(QPen(stroke_color, 1))
            painter.drawPath(path)
        
        # Draw RGB channels
        if self.show_rgb:
            draw_histogram(self.histogram_r, "#ef4444", 60)
            draw_histogram(self.histogram_g, "#22c55e", 60)
            draw_histogram(self.histogram_b, "#3b82f6", 60)
        
        # Draw luminance
        if self.show_luminance:
            draw_histogram(self.histogram_l, "#ffffff", 40)


class HistogramWidget(QFrame):
    """Complete histogram widget with controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("histogramWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        # Title
        title = QLabel("Histogram")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        
        # Canvas
        self.canvas = HistogramCanvas()
        layout.addWidget(self.canvas)
        
        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(12)
        
        self.rgb_cb = QCheckBox("RGB")
        self.rgb_cb.setChecked(True)
        self.rgb_cb.stateChanged.connect(self._on_rgb_changed)
        controls.addWidget(self.rgb_cb)
        
        self.lum_cb = QCheckBox("Lum")
        self.lum_cb.setChecked(True)
        self.lum_cb.stateChanged.connect(self._on_lum_changed)
        controls.addWidget(self.lum_cb)
        
        controls.addStretch()
        layout.addLayout(controls)
    
    def set_image(self, image: Optional[NDArray[np.uint8]]):
        """Update histogram with new image."""
        self.canvas.set_image(image)
    
    def _on_rgb_changed(self, state):
        self.canvas.show_rgb = bool(state)
        self.canvas.update()
    
    def _on_lum_changed(self, state):
        self.canvas.show_luminance = bool(state)
        self.canvas.update()

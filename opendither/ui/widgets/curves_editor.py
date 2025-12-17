"""Interactive curves editor widget for color grading."""

from __future__ import annotations

from typing import List, Tuple, Optional
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QLinearGradient, QBrush
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QFrame


class CurvePoint:
    """A control point on the curve."""
    def __init__(self, x: float, y: float):
        self.x = x  # 0.0 to 1.0
        self.y = y  # 0.0 to 1.0


class CurvesCanvas(QWidget):
    """Canvas for drawing and editing curves."""
    
    curveChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMaximumHeight(200)
        
        # Control points for the curve (default: linear)
        self.points: List[CurvePoint] = [
            CurvePoint(0.0, 0.0),
            CurvePoint(1.0, 1.0)
        ]
        
        self.selected_point: Optional[int] = None
        self.hover_point: Optional[int] = None
        self.channel = "RGB"  # RGB, Red, Green, Blue
        
        self.setMouseTracking(True)
        
    def set_channel(self, channel: str):
        """Set the active channel."""
        self.channel = channel
        self.update()
    
    def reset_curve(self):
        """Reset to linear curve."""
        self.points = [CurvePoint(0.0, 0.0), CurvePoint(1.0, 1.0)]
        self.curveChanged.emit()
        self.update()
    
    def get_curve_values(self) -> List[int]:
        """Get 256 values representing the curve lookup table."""
        lut = []
        for i in range(256):
            x = i / 255.0
            y = self._interpolate(x)
            lut.append(int(max(0, min(255, y * 255))))
        return lut
    
    def _interpolate(self, x: float) -> float:
        """Interpolate y value for given x using cubic spline."""
        if len(self.points) < 2:
            return x
        
        # Find surrounding points
        sorted_points = sorted(self.points, key=lambda p: p.x)
        
        # Find the segment
        for i in range(len(sorted_points) - 1):
            if sorted_points[i].x <= x <= sorted_points[i + 1].x:
                p0 = sorted_points[i]
                p1 = sorted_points[i + 1]
                
                if p1.x == p0.x:
                    return p0.y
                
                t = (x - p0.x) / (p1.x - p0.x)
                # Smooth interpolation
                t = t * t * (3 - 2 * t)  # Smoothstep
                return p0.y + t * (p1.y - p0.y)
        
        # Outside range
        if x <= sorted_points[0].x:
            return sorted_points[0].y
        return sorted_points[-1].y
    
    def paintEvent(self, event):
        """Draw the curves editor."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 10
        
        # Background
        painter.fillRect(0, 0, w, h, QColor("#18181b"))
        
        # Grid
        painter.setPen(QPen(QColor("#27272a"), 1))
        for i in range(5):
            x = margin + (w - 2 * margin) * i / 4
            y = margin + (h - 2 * margin) * i / 4
            painter.drawLine(int(x), margin, int(x), h - margin)
            painter.drawLine(margin, int(y), w - margin, int(y))
        
        # Diagonal reference line
        painter.setPen(QPen(QColor("#3f3f46"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin, h - margin, w - margin, margin)
        
        # Draw histogram background hint
        gradient = QLinearGradient(0, h - margin, 0, margin)
        gradient.setColorAt(0, QColor(0, 0, 0, 30))
        gradient.setColorAt(1, QColor(255, 255, 255, 30))
        painter.fillRect(margin, margin, w - 2 * margin, h - 2 * margin, gradient)
        
        # Draw the curve
        channel_colors = {
            "RGB": "#fafafa",
            "Red": "#ef4444",
            "Green": "#22c55e",
            "Blue": "#3b82f6"
        }
        curve_color = QColor(channel_colors.get(self.channel, "#fafafa"))
        painter.setPen(QPen(curve_color, 2))
        
        path = QPainterPath()
        first = True
        for i in range(w - 2 * margin):
            x_norm = i / (w - 2 * margin - 1)
            y_norm = self._interpolate(x_norm)
            
            px = margin + i
            py = h - margin - y_norm * (h - 2 * margin)
            
            if first:
                path.moveTo(px, py)
                first = False
            else:
                path.lineTo(px, py)
        
        painter.drawPath(path)
        
        # Draw control points
        sorted_points = sorted(self.points, key=lambda p: p.x)
        for i, point in enumerate(sorted_points):
            px = margin + point.x * (w - 2 * margin)
            py = h - margin - point.y * (h - 2 * margin)
            
            # Point style based on state
            if i == self.selected_point:
                painter.setBrush(QBrush(curve_color))
                painter.setPen(QPen(QColor("#ffffff"), 2))
                radius = 7
            elif i == self.hover_point:
                painter.setBrush(QBrush(QColor("#3f3f46")))
                painter.setPen(QPen(curve_color, 2))
                radius = 6
            else:
                painter.setBrush(QBrush(QColor("#27272a")))
                painter.setPen(QPen(curve_color, 2))
                radius = 5
            
            painter.drawEllipse(QPointF(px, py), radius, radius)
        
        # Border
        painter.setPen(QPen(QColor("#27272a"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(margin, margin, w - 2 * margin, h - 2 * margin)
    
    def _point_at_pos(self, pos) -> Optional[int]:
        """Find point near position."""
        w, h = self.width(), self.height()
        margin = 10
        
        sorted_points = sorted(enumerate(self.points), key=lambda x: x[1].x)
        
        for orig_idx, point in sorted_points:
            px = margin + point.x * (w - 2 * margin)
            py = h - margin - point.y * (h - 2 * margin)
            
            dist = ((pos.x() - px) ** 2 + (pos.y() - py) ** 2) ** 0.5
            if dist < 12:
                return orig_idx
        return None
    
    def _pos_to_curve(self, pos) -> Tuple[float, float]:
        """Convert widget position to curve coordinates."""
        w, h = self.width(), self.height()
        margin = 10
        
        x = (pos.x() - margin) / (w - 2 * margin)
        y = 1.0 - (pos.y() - margin) / (h - 2 * margin)
        
        return max(0, min(1, x)), max(0, min(1, y))
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            point_idx = self._point_at_pos(event.pos())
            
            if point_idx is not None:
                self.selected_point = point_idx
            else:
                # Add new point
                x, y = self._pos_to_curve(event.pos())
                self.points.append(CurvePoint(x, y))
                self.selected_point = len(self.points) - 1
                self.curveChanged.emit()
            
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            # Remove point (except first and last)
            point_idx = self._point_at_pos(event.pos())
            if point_idx is not None and len(self.points) > 2:
                # Don't remove endpoints
                point = self.points[point_idx]
                if point.x not in (0.0, 1.0):
                    del self.points[point_idx]
                    self.selected_point = None
                    self.curveChanged.emit()
                    self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.selected_point is not None:
            x, y = self._pos_to_curve(event.pos())
            
            point = self.points[self.selected_point]
            # Lock x for endpoints
            if point.x == 0.0:
                x = 0.0
            elif point.x == 1.0:
                x = 1.0
            
            point.x = x
            point.y = y
            self.curveChanged.emit()
            self.update()
        else:
            # Hover effect
            self.hover_point = self._point_at_pos(event.pos())
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_point = None
            self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave."""
        self.hover_point = None
        self.update()


class CurvesEditor(QFrame):
    """Complete curves editor widget with channel selection."""
    
    curveChanged = pyqtSignal(str, list)  # channel, lut
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("curvesEditor")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Channel selector
        controls = QHBoxLayout()
        controls.setSpacing(8)
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["RGB", "Red", "Green", "Blue"])
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        controls.addWidget(self.channel_combo)
        
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("secondaryButton")
        reset_btn.setFixedWidth(60)
        reset_btn.clicked.connect(self._on_reset)
        controls.addWidget(reset_btn)
        
        layout.addLayout(controls)
        
        # Curves canvas
        self.canvas = CurvesCanvas()
        self.canvas.curveChanged.connect(self._on_curve_changed)
        layout.addWidget(self.canvas)
        
        # Store curves for each channel
        self.channel_curves = {
            "RGB": [CurvePoint(0, 0), CurvePoint(1, 1)],
            "Red": [CurvePoint(0, 0), CurvePoint(1, 1)],
            "Green": [CurvePoint(0, 0), CurvePoint(1, 1)],
            "Blue": [CurvePoint(0, 0), CurvePoint(1, 1)],
        }
    
    def _on_channel_changed(self, channel: str):
        """Handle channel selection change."""
        # Save current channel's curve
        current = self.channel_combo.currentText()
        self.channel_curves[current] = self.canvas.points.copy()
        
        # Load new channel's curve
        self.canvas.points = self.channel_curves.get(channel, [CurvePoint(0, 0), CurvePoint(1, 1)])
        self.canvas.set_channel(channel)
    
    def _on_reset(self):
        """Reset current channel curve."""
        self.canvas.reset_curve()
        channel = self.channel_combo.currentText()
        self.channel_curves[channel] = [CurvePoint(0, 0), CurvePoint(1, 1)]
        self._on_curve_changed()
    
    def _on_curve_changed(self):
        """Emit curve change signal."""
        channel = self.channel_combo.currentText()
        lut = self.canvas.get_curve_values()
        self.curveChanged.emit(channel, lut)
    
    def get_all_luts(self) -> dict:
        """Get LUTs for all channels."""
        return {
            channel: self._get_channel_lut(channel)
            for channel in ["RGB", "Red", "Green", "Blue"]
        }
    
    def _get_channel_lut(self, channel: str) -> List[int]:
        """Get LUT for specific channel."""
        if channel == self.channel_combo.currentText():
            return self.canvas.get_curve_values()
        
        # Calculate from stored points
        points = self.channel_curves.get(channel, [CurvePoint(0, 0), CurvePoint(1, 1)])
        # Simplified linear interpolation
        lut = []
        sorted_points = sorted(points, key=lambda p: p.x)
        for i in range(256):
            x = i / 255.0
            # Find segment
            y = x  # Default linear
            for j in range(len(sorted_points) - 1):
                if sorted_points[j].x <= x <= sorted_points[j + 1].x:
                    p0, p1 = sorted_points[j], sorted_points[j + 1]
                    if p1.x != p0.x:
                        t = (x - p0.x) / (p1.x - p0.x)
                        t = t * t * (3 - 2 * t)
                        y = p0.y + t * (p1.y - p0.y)
                    break
            lut.append(int(max(0, min(255, y * 255))))
        return lut

"""Split view widget for before/after comparison."""

from __future__ import annotations

from typing import Optional
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QImage, QPen, QColor, QCursor, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QButtonGroup, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsLineItem
)


class SplitViewCanvas(QWidget):
    """Canvas for split view comparison."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        self.before_image: Optional[QImage] = None
        self.after_image: Optional[QImage] = None
        
        self.split_position = 0.5  # 0.0 to 1.0
        self.split_mode = "horizontal"  # horizontal, vertical, side_by_side
        self.dragging = False
        
        self.setMouseTracking(True)
    
    def set_images(self, before: Optional[QImage], after: Optional[QImage]):
        """Set before and after images."""
        self.before_image = before
        self.after_image = after
        self.update()
    
    def set_mode(self, mode: str):
        """Set split mode."""
        self.split_mode = mode
        self.update()
    
    def paintEvent(self, event):
        """Draw the split view."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        w, h = self.width(), self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, QColor("#09090b"))
        
        if self.before_image is None and self.after_image is None:
            painter.setPen(QColor("#52525b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No images to compare")
            return
        
        if self.split_mode == "side_by_side":
            self._draw_side_by_side(painter, w, h)
        elif self.split_mode == "vertical":
            self._draw_vertical_split(painter, w, h)
        else:
            self._draw_horizontal_split(painter, w, h)
    
    def _scale_image_to_fit(self, image: QImage, max_w: int, max_h: int) -> QImage:
        """Scale image to fit within bounds while maintaining aspect ratio."""
        if image is None:
            return None
        return image.scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    def _draw_horizontal_split(self, painter: QPainter, w: int, h: int):
        """Draw horizontal split (left/right)."""
        split_x = int(w * self.split_position)
        
        # Scale images to fit
        if self.before_image:
            before_scaled = self._scale_image_to_fit(self.before_image, w, h)
            if before_scaled:
                # Center the image
                x_off = (w - before_scaled.width()) // 2
                y_off = (h - before_scaled.height()) // 2
                
                # Clip to left side
                painter.setClipRect(0, 0, split_x, h)
                painter.drawImage(x_off, y_off, before_scaled)
        
        if self.after_image:
            after_scaled = self._scale_image_to_fit(self.after_image, w, h)
            if after_scaled:
                x_off = (w - after_scaled.width()) // 2
                y_off = (h - after_scaled.height()) // 2
                
                # Clip to right side
                painter.setClipRect(split_x, 0, w - split_x, h)
                painter.drawImage(x_off, y_off, after_scaled)
        
        painter.setClipping(False)
        
        # Draw split line
        painter.setPen(QPen(QColor("#fafafa"), 2))
        painter.drawLine(split_x, 0, split_x, h)
        
        # Draw handle
        handle_y = h // 2
        painter.setBrush(QColor("#fafafa"))
        painter.drawEllipse(QPointF(split_x, handle_y), 8, 8)
        
        # Labels
        painter.setPen(QColor("#fafafa"))
        painter.drawText(10, 25, "Before")
        painter.drawText(w - 50, 25, "After")
    
    def _draw_vertical_split(self, painter: QPainter, w: int, h: int):
        """Draw vertical split (top/bottom)."""
        split_y = int(h * self.split_position)
        
        if self.before_image:
            before_scaled = self._scale_image_to_fit(self.before_image, w, h)
            if before_scaled:
                x_off = (w - before_scaled.width()) // 2
                y_off = (h - before_scaled.height()) // 2
                
                painter.setClipRect(0, 0, w, split_y)
                painter.drawImage(x_off, y_off, before_scaled)
        
        if self.after_image:
            after_scaled = self._scale_image_to_fit(self.after_image, w, h)
            if after_scaled:
                x_off = (w - after_scaled.width()) // 2
                y_off = (h - after_scaled.height()) // 2
                
                painter.setClipRect(0, split_y, w, h - split_y)
                painter.drawImage(x_off, y_off, after_scaled)
        
        painter.setClipping(False)
        
        # Draw split line
        painter.setPen(QPen(QColor("#fafafa"), 2))
        painter.drawLine(0, split_y, w, split_y)
        
        # Draw handle
        handle_x = w // 2
        painter.setBrush(QColor("#fafafa"))
        painter.drawEllipse(QPointF(handle_x, split_y), 8, 8)
    
    def _draw_side_by_side(self, painter: QPainter, w: int, h: int):
        """Draw side by side comparison."""
        half_w = w // 2 - 5
        
        if self.before_image:
            before_scaled = self._scale_image_to_fit(self.before_image, half_w, h)
            if before_scaled:
                x_off = (half_w - before_scaled.width()) // 2
                y_off = (h - before_scaled.height()) // 2
                painter.drawImage(x_off, y_off, before_scaled)
        
        if self.after_image:
            after_scaled = self._scale_image_to_fit(self.after_image, half_w, h)
            if after_scaled:
                x_off = half_w + 10 + (half_w - after_scaled.width()) // 2
                y_off = (h - after_scaled.height()) // 2
                painter.drawImage(x_off, y_off, after_scaled)
        
        # Separator
        painter.setPen(QPen(QColor("#27272a"), 2))
        painter.drawLine(w // 2, 0, w // 2, h)
        
        # Labels
        painter.setPen(QColor("#71717a"))
        painter.drawText(10, 25, "Before")
        painter.drawText(w // 2 + 10, 25, "After")
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.split_mode != "side_by_side":
                self.dragging = True
                self._update_split_position(event.pos())
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.dragging:
            self._update_split_position(event.pos())
        else:
            # Update cursor
            if self.split_mode != "side_by_side":
                if self.split_mode == "horizontal":
                    self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
                else:
                    self.setCursor(QCursor(Qt.CursorShape.SplitVCursor))
            else:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
    
    def _update_split_position(self, pos):
        """Update split position from mouse position."""
        if self.split_mode == "horizontal":
            self.split_position = max(0.05, min(0.95, pos.x() / self.width()))
        else:
            self.split_position = max(0.05, min(0.95, pos.y() / self.height()))
        self.update()


class SplitViewWidget(QFrame):
    """Complete split view widget with mode controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("splitViewWidget")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Mode buttons
        controls = QHBoxLayout()
        controls.setSpacing(4)
        
        self.btn_group = QButtonGroup(self)
        
        modes = [
            ("⇆", "horizontal", "Horizontal split"),
            ("⇅", "vertical", "Vertical split"),
            ("▣", "side_by_side", "Side by side"),
        ]
        
        for text, mode, tooltip in modes:
            btn = QPushButton(text)
            btn.setObjectName("secondaryButton")
            btn.setFixedSize(36, 28)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setProperty("mode", mode)
            btn.clicked.connect(lambda checked, m=mode: self._on_mode_changed(m))
            self.btn_group.addButton(btn)
            controls.addWidget(btn)
            
            if mode == "horizontal":
                btn.setChecked(True)
        
        controls.addStretch()
        
        # Reset button
        reset_btn = QPushButton("50%")
        reset_btn.setObjectName("secondaryButton")
        reset_btn.setFixedWidth(40)
        reset_btn.clicked.connect(self._reset_position)
        controls.addWidget(reset_btn)
        
        layout.addLayout(controls)
        
        # Canvas
        self.canvas = SplitViewCanvas()
        layout.addWidget(self.canvas, 1)
    
    def set_images(self, before: Optional[QImage], after: Optional[QImage]):
        """Set before and after images."""
        self.canvas.set_images(before, after)
    
    def _on_mode_changed(self, mode: str):
        """Handle mode change."""
        self.canvas.set_mode(mode)
    
    def _reset_position(self):
        """Reset split position to center."""
        self.canvas.split_position = 0.5
        self.canvas.update()

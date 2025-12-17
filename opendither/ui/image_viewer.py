"""Interactive image viewer with pan and zoom support."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ImageViewer(QGraphicsView):
    """Zoomable and pannable image viewer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)
        
        # Enable drag/pan
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Dark background
        self.setBackgroundBrush(Qt.GlobalColor.black)
        
        self._zoom_level = 0
        self._max_zoom = 10
        self._min_zoom = -10

    def set_image(self, image: QImage, preserve_view: bool = False) -> None:
        """Set the displayed image."""
        pixmap = QPixmap.fromImage(image)
        
        prev_center = None
        prev_transform = None
        prev_zoom = None
        
        if (
            preserve_view
            and not self._pixmap_item.pixmap().isNull()
            and not pixmap.isNull()
        ):
            prev_center = self.mapToScene(self.viewport().rect().center())
            prev_transform = self.transform()
            prev_zoom = self._zoom_level
        
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        
        if prev_center is not None and prev_transform is not None:
            if not self.sceneRect().contains(prev_center):
                prev_center = self.sceneRect().center()
            self.setTransform(prev_transform)
            self.centerOn(prev_center)
            self._zoom_level = prev_zoom if prev_zoom is not None else 0
        else:
            self.resetTransform()
            self.fit_in_view()
            self._zoom_level = 0

    def fit_in_view(self) -> None:
        """Fit the image to the view."""
        if self._pixmap_item.pixmap().isNull():
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        """Zoom in by a fixed factor."""
        if self._zoom_level < self._max_zoom:
            self._zoom_level += 1
            self.scale(1.25, 1.25)

    def zoom_out(self) -> None:
        """Zoom out by a fixed factor."""
        if self._zoom_level > self._min_zoom:
            self._zoom_level -= 1
            self.scale(0.8, 0.8)

    def reset_zoom(self) -> None:
        """Reset zoom to fit view."""
        self.resetTransform()
        self.fit_in_view()
        self._zoom_level = 0

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        if self._pixmap_item.pixmap().isNull():
            return
        
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def has_image(self) -> bool:
        """Check if an image is loaded."""
        return not self._pixmap_item.pixmap().isNull()

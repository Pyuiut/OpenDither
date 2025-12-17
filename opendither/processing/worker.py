"""Background worker for async image processing with progress reporting."""

from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .pipeline import ProcessingPipeline


class ProcessingWorker(QObject):
    """Worker object for background dithering operations with real progress."""
    
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)  # percent, stage_name

    def __init__(
        self,
        image: NDArray[np.uint8],
        algorithm: str,
        parameters: Dict[str, float],
        palette: Optional[str] = None,
        curves_lut: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        super().__init__()
        self.image = image
        self.algorithm = algorithm
        self.parameters = parameters
        self.palette = palette
        self.curves_lut = curves_lut
        self.pipeline = ProcessingPipeline()
        
        # Connect progress callback
        self.pipeline.set_progress_callback(self._on_progress)

    def _on_progress(self, percent: int, stage: str):
        """Forward progress to Qt signal."""
        self.progress.emit(percent, stage)

    def run(self) -> None:
        """Execute the processing task."""
        try:
            self.progress.emit(0, "Starting...")
            
            result = self.pipeline.process(
                self.image,
                self.algorithm,
                self.parameters,
                self.palette,
                self.curves_lut,
            )
            
            self.progress.emit(100, "Complete")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def create_worker_thread(
    image: NDArray[np.uint8],
    algorithm: str,
    parameters: Dict[str, float],
    palette: Optional[str] = None,
    curves_lut: Optional[Dict[str, List[int]]] = None,
) -> tuple[QThread, ProcessingWorker]:
    """Create a worker thread for processing."""
    thread = QThread()
    worker = ProcessingWorker(image, algorithm, parameters, palette, curves_lut)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    return thread, worker

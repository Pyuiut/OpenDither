"""Main application window - Premium UI with optimized performance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import Qt, QThread, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap, QPainter, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsBlurEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QGraphicsOpacityEffect,
    QButtonGroup,
    QWidget,
)

from opendither.core import AlgorithmLibrary, PaletteLibrary, PresetLibrary, Preset
from opendither.processing import DitheringEngine
from opendither.processing.worker import create_worker_thread
from .image_viewer import ImageViewer
from .styles import DARK_THEME
from .widgets import CurvesEditor, HistogramWidget, SplitViewWidget


class MainWindow(QMainWindow):
    """Main application window with premium UI."""

    def __init__(self) -> None:
        super().__init__()
        
        # Initialize libraries
        self.algorithm_library = AlgorithmLibrary()
        self.palette_library = PaletteLibrary()
        self.preset_library = PresetLibrary()
        self.engine = DitheringEngine()
        
        # Image state
        self.original_image: Optional[NDArray[np.uint8]] = None
        self.processed_image: Optional[NDArray[np.uint8]] = None
        self.current_algorithm: str = "None"
        self.current_palette: Optional[str] = None
        
        # Parameters - ALL START AT NEUTRAL (0)
        self.parameters: Dict[str, float] = {
            # Basic adjustments (neutral = 0)
            "brightness": 0, "contrast": 0, "exposure": 0,
            # Vibrance
            "vibrance": 0, "saturation": 0,
            # Tones
            "highlights": 0, "shadows": 0, "whites": 0, "blacks": 0,
            # B&W mix
            "bw_reds": 40, "bw_yellows": 60, "bw_greens": 40,
            "bw_cyans": 60, "bw_blues": 20, "bw_magentas": 80,
            # Detail
            "sharpness": 0, "clarity": 0, "dehaze": 0,
            # Transform
            "scale": 100, "rotation": 0,
            # Color
            "hue_shift": 0, "temperature": 0, "tint": 0,
            # Levels
            "levels_black": 0, "levels_white": 255, "levels_gamma": 100,
            "out_black": 0, "out_white": 255,
            # Dither specific
            "luminance": 128, "dither_strength": 100,
            "pattern_scale": 100, "color_depth": 100,
            # Effects
            "blur": 0, "glow": 0, "grain": 0, "vignette": 0,
            "flare_intensity": 0, "flare_threshold": 60,
            "flare_amount": 0, "flare_variation": 50, "flare_size": 40,
            "flare_color_hue": 40, "flare_color_sat": 80, "flare_color_value": 90,
            "flare_spacing": 40,
            "glitch_intensity": 0, "glitch_frequency": 40, "glitch_shift": 40,
            # Export
            "export_quality": 95, "export_dpi": 300,
            "export_width": 1920, "export_height": 1080,
            # Flags
            "invert": 0,
            "preserve_colors": 0,
            "flare_style": "Lens",
            "flare_tint": "Warm",
            "flare_distribution": "Highlights",
            "flare_shape": "Star",
            "glitch_style": "RGB Split",
        }
        self._default_parameters = self.parameters.copy()
        
        # Curves LUT for RGB channels
        self._curve_luts: Dict[str, list] = {}
        
        # Processing state
        self.worker_thread: Optional[QThread] = None
        self.worker = None
        self._pending_update = False
        self._live_preview = True
        self._preview_mode = True  # Fast preview mode
        self.current_section_index = 0
        
        # Debounce timer for slider changes
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)  # 150ms debounce
        self._debounce_timer.timeout.connect(self._do_update)
        
        # Setup UI
        self._setup_window()
        self._create_menu_bar()
        self._create_main_layout()
        self._apply_theme()
        
        # Enable drag and drop
        self.setAcceptDrops(True)

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.setWindowTitle("OpenDither")
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)

    def _apply_theme(self) -> None:
        """Apply dark theme stylesheet."""
        self.setStyleSheet(DARK_THEME)

    def _create_menu_bar(self) -> None:
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        import_action = QAction("Import Image", self)
        import_action.setShortcut(QKeySequence("Ctrl+O"))
        import_action.triggered.connect(self._import_image)
        file_menu.addAction(import_action)
        
        export_action = QAction("Export Image", self)
        export_action.setShortcut(QKeySequence("Ctrl+S"))
        export_action.triggered.connect(self._export_image)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Undo", lambda: None).setShortcut(QKeySequence("Ctrl+Z"))
        edit_menu.addAction("Redo", lambda: None).setShortcut(QKeySequence("Ctrl+Y"))
        
        # Batch menu
        batch_menu = menubar.addMenu("Batch")
        batch_menu.addAction("Process Folder...", self._batch_process)
        batch_menu.addAction("Process Video...", self._process_video)
        
        # Adjustments menu
        adj_menu = menubar.addMenu("Adjustments")
        adj_menu.addAction("Reset All", self._reset_all)
        adj_menu.addAction("Apply Current", self._apply_processing)
        
        # Themes menu
        themes_menu = menubar.addMenu("Themes")
        themes_menu.addAction("Dark (Default)", lambda: None)
        themes_menu.addAction("Light", lambda: None)
        
        # Extras menu
        extras_menu = menubar.addMenu("Extras")
        extras_menu.addAction("Export SVG Vector...", self._export_svg)
        extras_menu.addAction("Import Presets...", self._import_presets)
        extras_menu.addAction("Export Presets...", self._export_presets)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_action = QAction("User Guide", self)
        help_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)
        help_menu.addAction("About", self._show_about)

    def _create_main_layout(self) -> None:
        """Create the main UI layout with fixed control panel."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left: Viewer container with loading overlay
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        
        self.viewer = ImageViewer()
        viewer_layout.addWidget(self.viewer)
        
        # Loading overlay (initially hidden)
        self.loading_overlay = self._create_loading_overlay()
        self.loading_overlay.setParent(viewer_container)
        self.loading_overlay.hide()
        
        main_layout.addWidget(viewer_container, 1)
        
        # Right: Control panel (fixed width)
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel, 0)
        
        # Status bar
        self.statusBar().showMessage("✨ Ready — Drop an image to start")
        
        # Progress bar in status bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def _create_loading_overlay(self) -> QWidget:
        """Create a loading overlay with blur effect and progress."""
        overlay = QFrame()
        overlay.setObjectName("loadingOverlay")
        overlay.setStyleSheet("""
            QFrame#loadingOverlay {
                background-color: rgba(9, 9, 11, 0.85);
            }
        """)
        
        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Container for loading content
        container = QFrame()
        container.setObjectName("loadingContainer")
        container.setFixedSize(280, 120)
        container.setStyleSheet("""
            QFrame#loadingContainer {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 16px;
            }
        """)
        
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.setSpacing(16)
        
        # Loading text
        self.loading_label = QLabel("Processing...")
        self.loading_label.setStyleSheet("color: #fafafa; font-size: 14px; font-weight: 500;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.loading_label)
        
        # Progress bar
        self.overlay_progress = QProgressBar()
        self.overlay_progress.setFixedWidth(200)
        self.overlay_progress.setFixedHeight(6)
        self.overlay_progress.setRange(0, 0)  # Indeterminate
        self.overlay_progress.setTextVisible(False)
        container_layout.addWidget(self.overlay_progress, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(container)
        
        return overlay
    
    def resizeEvent(self, event):
        """Handle resize to keep overlay covering viewer."""
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay') and hasattr(self, 'viewer'):
            # Match overlay size to viewer
            self.loading_overlay.setGeometry(self.viewer.geometry())
        self._update_nav_indicator_geometry(animate=False)

    def _create_control_panel(self) -> QWidget:
        """Create modern control panel with navigation rail and sections."""
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setFixedWidth(420)
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        
        # === TOP ACTIONS BAR ===
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 16, 16, 16)
        top_layout.setSpacing(10)
        
        self.import_btn = QPushButton("📥 Import")
        self.import_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self._import_image)
        top_layout.addWidget(self.import_btn)
        
        self.export_btn = QPushButton("📤 Export")
        self.export_btn.clicked.connect(self._export_image)
        self.export_btn.setEnabled(False)
        top_layout.addWidget(self.export_btn)
        
        self.apply_btn = QPushButton("⚡ Apply")
        self.apply_btn.setObjectName("accentButton")
        self.apply_btn.clicked.connect(self._force_apply)
        top_layout.addWidget(self.apply_btn)
        
        panel_layout.addWidget(top_bar)
        
        # === NAVIGATION + STACK ===
        sections = [
            ("style", "🎨 Style", self._create_style_tab()),
            ("adjust", "🛠 Adjust", self._create_adjust_tab()),
            ("color", "🌈 Color", self._create_color_tab()),
            ("curves", "📈 Curves", self._create_curves_tab()),
            ("fx", "✨ FX", self._create_effects_tab()),
            ("compare", "🌓 Compare", self._create_compare_tab()),
            ("export", "📦 Export", self._create_export_tab()),
            ("settings", "⚙ Settings", self._create_settings_tab()),
        ]
        
        body_frame = QFrame()
        body_layout = QHBoxLayout(body_frame)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        
        # Navigation rail
        nav_wrapper = QFrame()
        nav_wrapper.setObjectName("navWrapper")
        nav_wrapper.setFixedWidth(150)
        nav_layout = QVBoxLayout(nav_wrapper)
        nav_layout.setContentsMargins(18, 18, 12, 18)
        nav_layout.setSpacing(12)
        
        nav_label = QLabel("Navigate")
        nav_label.setObjectName("navTitle")
        nav_layout.addWidget(nav_label)
        
        self.nav_buttons_container = QWidget()
        self.nav_buttons_container.setObjectName("navButtons")
        nav_buttons_layout = QVBoxLayout(self.nav_buttons_container)
        nav_buttons_layout.setContentsMargins(0, 0, 0, 0)
        nav_buttons_layout.setSpacing(6)
        
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.section_buttons: list[QPushButton] = []
        
        for idx, (_, label, _) in enumerate(sections):
            btn = QPushButton(label)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            self.nav_group.addButton(btn, idx)
            nav_buttons_layout.addWidget(btn)
            self.section_buttons.append(btn)
        
        nav_buttons_layout.addStretch()
        nav_layout.addWidget(self.nav_buttons_container)
        nav_layout.addStretch()
        
        self.nav_indicator = QFrame(self.nav_buttons_container)
        self.nav_indicator.setObjectName("navIndicator")
        self.nav_indicator.hide()
        self.nav_indicator.lower()
        self.nav_indicator_anim = QPropertyAnimation(self.nav_indicator, b"geometry", self)
        self.nav_indicator_anim.setDuration(220)
        self.nav_indicator_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        
        self.nav_group.idClicked.connect(self._on_section_selected)
        
        # Content stack with fade effect
        self.section_stack = QStackedWidget()
        self.section_stack.setObjectName("sectionStack")
        self.stack_effect = QGraphicsOpacityEffect(self.section_stack)
        self.section_stack.setGraphicsEffect(self.stack_effect)
        self.stack_effect.setOpacity(1.0)
        self.stack_fade_anim = QPropertyAnimation(self.stack_effect, b"opacity", self)
        self.stack_fade_anim.setDuration(220)
        self.stack_fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        for _, _, widget in sections:
            self.section_stack.addWidget(widget)
        
        body_layout.addWidget(nav_wrapper)
        body_layout.addWidget(self.section_stack, 1)
        panel_layout.addWidget(body_frame, 1)
        
        # Initialize selection after layout settles
        QTimer.singleShot(0, lambda: self._set_section(0, animate=False))
        
        # === BOTTOM ACTIONS ===
        bottom_bar = QFrame()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 8, 12, 12)
        bottom_layout.setSpacing(8)
        
        save_btn = QPushButton("💾 Save Preset")
        save_btn.clicked.connect(self._save_preset)
        bottom_layout.addWidget(save_btn)
        
        reset_btn = QPushButton("↺ Reset")
        reset_btn.setObjectName("secondaryButton")
        reset_btn.clicked.connect(self._reset_all)
        bottom_layout.addWidget(reset_btn)
        
        panel_layout.addWidget(bottom_bar)
        
        return panel
    
    def _on_section_selected(self, index: int) -> None:
        """Handle navigation button selection."""
        self._set_section(index)
    
    def _set_section(self, index: int, animate: bool = True) -> None:
        """Switch to the requested section with subtle animation."""
        if not hasattr(self, "section_stack"):
            return
        if index < 0 or index >= self.section_stack.count():
            return
        
        if hasattr(self, "nav_group"):
            button = self.nav_group.button(index)
            if button and not button.isChecked():
                button.setChecked(True)
        
        if self.current_section_index == index and animate:
            return
        
        self.current_section_index = index
        self.section_stack.setCurrentIndex(index)
        self._animate_stack_transition(animate)
        self._update_nav_indicator_geometry(animate=animate)
    
    def _animate_stack_transition(self, animate: bool) -> None:
        """Fade the stacked widget when switching sections."""
        if not hasattr(self, "stack_effect"):
            return
        
        self.stack_fade_anim.stop()
        if not animate:
            self.stack_effect.setOpacity(1.0)
            return
        
        self.stack_effect.setOpacity(0.0)
        self.stack_fade_anim.setStartValue(0.0)
        self.stack_fade_anim.setEndValue(1.0)
        self.stack_fade_anim.start()
    
    def _update_nav_indicator_geometry(self, animate: bool = True) -> None:
        """Move the glowing indicator to align with the active nav button."""
        if not hasattr(self, "nav_indicator") or not hasattr(self, "nav_buttons_container"):
            return
        if not self.section_buttons:
            return
        if self.current_section_index >= len(self.section_buttons):
            return
        
        button = self.section_buttons[self.current_section_index]
        if not button.isVisible():
            return
        
        btn_geo = button.geometry()
        indicator_geo = QRect(
            6,
            btn_geo.y() - 2,
            self.nav_buttons_container.width() - 12,
            btn_geo.height() + 4,
        )
        
        self.nav_indicator.show()
        self.nav_indicator.raise_()
        self.nav_indicator_anim.stop()
        
        if not animate or not self.nav_indicator.isVisible():
            self.nav_indicator.setGeometry(indicator_geo)
            return
        
        self.nav_indicator_anim.setStartValue(self.nav_indicator.geometry())
        self.nav_indicator_anim.setEndValue(indicator_geo)
        self.nav_indicator_anim.start()
    
    def _create_style_tab(self) -> QWidget:
        """Create the Style tab with algorithm and preset selection."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Algorithm section
        algo_card = self._create_card()
        algo_layout = QVBoxLayout(algo_card)
        algo_layout.setContentsMargins(0, 0, 0, 0)
        algo_layout.setSpacing(12)
        
        algo_title = QLabel("Dithering Algorithm")
        algo_title.setObjectName("cardTitle")
        algo_layout.addWidget(algo_title)
        
        self.style_combo = QComboBox()
        self.style_combo.addItem("None")
        self.style_combo.addItem("Floyd-Steinberg")
        self.style_combo.addItem("Jarvis-Judice-Ninke")
        self.style_combo.addItem("Stucki")
        self.style_combo.addItem("Burkes")
        self.style_combo.addItem("Sierra Lite")
        self.style_combo.addItem("Atkinson")
        self.style_combo.addItem("Bayer 2x2")
        self.style_combo.addItem("Bayer 4x4")
        self.style_combo.addItem("Bayer 8x8")
        self.style_combo.addItem("Clustered Dot 8x8")
        self.style_combo.addItem("Circular Halftone")
        self.style_combo.addItem("Line Halftone")
        self.style_combo.addItem("Diamond Halftone")
        self.style_combo.addItem("Crosshatch")
        self.style_combo.addItem("Stipple")
        self.style_combo.addItem("Checkerboard")
        self.style_combo.addItem("Glitch Blocks")
        self.style_combo.addItem("Scanlines")
        self.style_combo.addItem("Pixelate")
        self.style_combo.addItem("VHS")
        self.style_combo.addItem("Bit Crush")
        self.style_combo.setCurrentText("None")
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        algo_layout.addWidget(self.style_combo)
        
        # Dither parameters
        algo_layout.addLayout(self._create_slider("Strength", "dither_strength", 0, 500, 100))
        algo_layout.addLayout(self._create_slider("Pattern Scale", "pattern_scale", 10, 500, 100))
        algo_layout.addLayout(self._create_slider("Color Depth", "color_depth", 1, 256, 100))
        
        layout.addWidget(algo_card)
        
        # Preset section
        preset_card = self._create_card()
        preset_layout = QVBoxLayout(preset_card)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(12)
        
        preset_title = QLabel("Presets")
        preset_title.setObjectName("cardTitle")
        preset_layout.addWidget(preset_title)
        
        self.preset_combo = QComboBox()
        self._refresh_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        
        layout.addWidget(preset_card)
        layout.addStretch()
        
        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab
    
    def _create_adjust_tab(self) -> QWidget:
        """Create the Adjust tab with Photoshop-like adjustments."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # === LUMINOSITY/CONTRAST ===
        lum_card = self._create_card()
        lum_layout = QVBoxLayout(lum_card)
        lum_layout.setContentsMargins(0, 0, 0, 0)
        lum_layout.setSpacing(6)
        
        lum_title = QLabel("Luminosity / Contrast")
        lum_title.setObjectName("cardTitle")
        lum_layout.addWidget(lum_title)
        
        lum_layout.addLayout(self._create_slider("Brightness", "brightness", -100, 100, 0))
        lum_layout.addLayout(self._create_slider("Contrast", "contrast", -100, 100, 0))
        lum_layout.addLayout(self._create_slider("Exposure", "exposure", -500, 500, 0))
        
        layout.addWidget(lum_card)
        
        # === VIBRANCE ===
        vib_card = self._create_card()
        vib_layout = QVBoxLayout(vib_card)
        vib_layout.setContentsMargins(0, 0, 0, 0)
        vib_layout.setSpacing(6)
        
        vib_title = QLabel("Vibrance")
        vib_title.setObjectName("cardTitle")
        vib_layout.addWidget(vib_title)
        
        vib_layout.addLayout(self._create_slider("Vibrance", "vibrance", -100, 100, 0))
        vib_layout.addLayout(self._create_slider("Saturation", "saturation", -100, 100, 0))
        
        layout.addWidget(vib_card)
        
        # === TONES (HIGHLIGHTS/SHADOWS) ===
        tone_card = self._create_card()
        tone_layout = QVBoxLayout(tone_card)
        tone_layout.setContentsMargins(0, 0, 0, 0)
        tone_layout.setSpacing(6)
        
        tone_title = QLabel("Tones")
        tone_title.setObjectName("cardTitle")
        tone_layout.addWidget(tone_title)
        
        tone_layout.addLayout(self._create_slider("Highlights", "highlights", -100, 100, 0))
        tone_layout.addLayout(self._create_slider("Shadows", "shadows", -100, 100, 0))
        tone_layout.addLayout(self._create_slider("Whites", "whites", -100, 100, 0))
        tone_layout.addLayout(self._create_slider("Blacks", "blacks", -100, 100, 0))
        
        layout.addWidget(tone_card)
        
        # === BLACK & WHITE ===
        bw_card = self._create_card()
        bw_layout = QVBoxLayout(bw_card)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.setSpacing(6)
        
        bw_title = QLabel("Black & White Mix")
        bw_title.setObjectName("cardTitle")
        bw_layout.addWidget(bw_title)
        
        self.bw_enabled_cb = QCheckBox("Enable B&W conversion")
        bw_layout.addWidget(self.bw_enabled_cb)
        
        bw_layout.addLayout(self._create_slider("Reds", "bw_reds", -200, 300, 40))
        bw_layout.addLayout(self._create_slider("Yellows", "bw_yellows", -200, 300, 60))
        bw_layout.addLayout(self._create_slider("Greens", "bw_greens", -200, 300, 40))
        bw_layout.addLayout(self._create_slider("Cyans", "bw_cyans", -200, 300, 60))
        bw_layout.addLayout(self._create_slider("Blues", "bw_blues", -200, 300, 20))
        bw_layout.addLayout(self._create_slider("Magentas", "bw_magentas", -200, 300, 80))
        
        layout.addWidget(bw_card)
        
        # === DETAIL ===
        detail_card = self._create_card()
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(6)
        
        detail_title = QLabel("Detail")
        detail_title.setObjectName("cardTitle")
        detail_layout.addWidget(detail_title)
        
        detail_layout.addLayout(self._create_slider("Sharpness", "sharpness", 0, 150, 0))
        detail_layout.addLayout(self._create_slider("Clarity", "clarity", -100, 100, 0))
        detail_layout.addLayout(self._create_slider("Dehaze", "dehaze", -100, 100, 0))
        
        layout.addWidget(detail_card)
        
        # === TRANSFORM ===
        scale_card = self._create_card()
        scale_layout = QVBoxLayout(scale_card)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(6)
        
        scale_title = QLabel("Transform")
        scale_title.setObjectName("cardTitle")
        scale_layout.addWidget(scale_title)
        
        scale_layout.addLayout(self._create_slider("Scale %", "scale", 10, 1000, 100))
        scale_layout.addLayout(self._create_slider("Rotation", "rotation", -180, 180, 0))
        
        layout.addWidget(scale_card)
        layout.addStretch()
        
        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab
    
    def _create_color_tab(self) -> QWidget:
        """Create the Color tab with palette and color grading."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Palette section
        palette_card = self._create_card()
        palette_layout = QVBoxLayout(palette_card)
        palette_layout.setContentsMargins(0, 0, 0, 0)
        palette_layout.setSpacing(8)
        
        palette_title = QLabel("Color Palette")
        palette_title.setObjectName("cardTitle")
        palette_layout.addWidget(palette_title)
        
        # Category selector
        cat_row = QHBoxLayout()
        cat_label = QLabel("Category:")
        cat_label.setFixedWidth(60)
        cat_row.addWidget(cat_label)
        
        self.palette_cat_combo = QComboBox()
        self.palette_cat_combo.addItem("All")
        for category in sorted(self.palette_library.categories().keys()):
            self.palette_cat_combo.addItem(category)
        self.palette_cat_combo.currentTextChanged.connect(self._on_palette_category_changed)
        cat_row.addWidget(self.palette_cat_combo, 1)
        palette_layout.addLayout(cat_row)
        
        # Palette selector
        pal_row = QHBoxLayout()
        pal_label = QLabel("Palette:")
        pal_label.setFixedWidth(60)
        pal_row.addWidget(pal_label)
        
        self.palette_combo = QComboBox()
        self.palette_combo.addItem("None")
        # Add all palettes initially
        for name in self.palette_library.names():
            self.palette_combo.addItem(name)
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        pal_row.addWidget(self.palette_combo, 1)
        palette_layout.addLayout(pal_row)
        
        # Palette preview (color swatches)
        self.palette_preview = QFrame()
        self.palette_preview.setFixedHeight(30)
        self.palette_preview.setStyleSheet("background: #27272a; border-radius: 4px;")
        palette_layout.addWidget(self.palette_preview)
        
        self.invert_cb = QCheckBox("Invert colors")
        self.invert_cb.stateChanged.connect(self._on_invert_changed)
        palette_layout.addWidget(self.invert_cb)
        
        self.preserve_colors_cb = QCheckBox("Keep original colors")
        self.preserve_colors_cb.setChecked(bool(self.parameters.get("preserve_colors", 0)))
        self.preserve_colors_cb.stateChanged.connect(self._on_preserve_colors_changed)
        palette_layout.addWidget(self.preserve_colors_cb)
        
        layout.addWidget(palette_card)
        
        # Color grading
        grading_card = self._create_card()
        grading_layout = QVBoxLayout(grading_card)
        grading_layout.setContentsMargins(0, 0, 0, 0)
        grading_layout.setSpacing(6)
        
        grading_title = QLabel("Color Grading")
        grading_title.setObjectName("cardTitle")
        grading_layout.addWidget(grading_title)
        
        grading_layout.addLayout(self._create_slider("Hue Shift", "hue_shift", -180, 180, 0))
        grading_layout.addLayout(self._create_slider("Temperature", "temperature", -100, 100, 0))
        grading_layout.addLayout(self._create_slider("Tint", "tint", -100, 100, 0))
        
        layout.addWidget(grading_card)
        
        # Gradient Map
        gradient_card = self._create_card()
        gradient_layout = QVBoxLayout(gradient_card)
        gradient_layout.setContentsMargins(0, 0, 0, 0)
        gradient_layout.setSpacing(8)
        
        gradient_title = QLabel("Gradient Map")
        gradient_title.setObjectName("cardTitle")
        gradient_layout.addWidget(gradient_title)
        
        self.gradient_combo = QComboBox()
        self.gradient_combo.addItems([
            "None",
            "Black → White",
            "Sepia",
            "Duotone Blue",
            "Duotone Purple", 
            "Sunset",
            "Cool Tones",
            "Warm Tones",
            "Cyanotype",
            "Infrared",
        ])
        self.gradient_combo.currentTextChanged.connect(self._on_gradient_changed)
        gradient_layout.addWidget(self.gradient_combo)
        
        # Gradient preview
        self.gradient_preview = QFrame()
        self.gradient_preview.setFixedHeight(20)
        self.gradient_preview.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #000000, stop:1 #ffffff); border-radius: 4px;"
        )
        gradient_layout.addWidget(self.gradient_preview)
        
        layout.addWidget(gradient_card)
        layout.addStretch()
        
        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab
    
    def _create_effects_tab(self) -> QWidget:
        """Create the Effects tab with special effects."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Blur/Glow effects
        blur_card = self._create_card()
        blur_layout = QVBoxLayout(blur_card)
        blur_layout.setContentsMargins(0, 0, 0, 0)
        blur_layout.setSpacing(8)
        
        blur_title = QLabel("Blur & Glow")
        blur_title.setObjectName("cardTitle")
        blur_layout.addWidget(blur_title)
        
        blur_layout.addLayout(self._create_slider("Blur", "blur", 0, 100, 0))
        blur_layout.addLayout(self._create_slider("Glow", "glow", 0, 200, 0))
        
        layout.addWidget(blur_card)
        
        # Flare controls
        flare_card = self._create_card()
        flare_layout = QVBoxLayout(flare_card)
        flare_layout.setContentsMargins(0, 0, 0, 0)
        flare_layout.setSpacing(8)
        
        flare_title = QLabel("Lens Flares")
        flare_title.setObjectName("cardTitle")
        flare_layout.addWidget(flare_title)
        
        flare_layout.addLayout(self._create_slider("Intensity", "flare_intensity", 0, 100, int(self.parameters.get("flare_intensity", 0))))
        flare_layout.addLayout(self._create_slider("Highlight Threshold", "flare_threshold", 0, 255, int(self.parameters.get("flare_threshold", 60))))
        
        flare_style_row = QHBoxLayout()
        flare_style_label = QLabel("Style")
        flare_style_row.addWidget(flare_style_label)
        flare_style_row.addStretch()
        self.flare_style_combo = QComboBox()
        self.flare_style_combo.addItems(["Lens", "Orbs", "Starburst"])
        self.flare_style_combo.setCurrentText(self.parameters.get("flare_style", "Lens"))
        self.flare_style_combo.currentTextChanged.connect(self._on_flare_style_changed)
        flare_style_row.addWidget(self.flare_style_combo)
        flare_layout.addLayout(flare_style_row)
        
        flare_tint_row = QHBoxLayout()
        flare_tint_label = QLabel("Tint")
        flare_tint_row.addWidget(flare_tint_label)
        flare_tint_row.addStretch()
        self.flare_tint_combo = QComboBox()
        self.flare_tint_combo.addItems(["Warm", "Cool", "Neon", "Gold", "Sunset"])
        self.flare_tint_combo.setCurrentText(self.parameters.get("flare_tint", "Warm"))
        self.flare_tint_combo.currentTextChanged.connect(self._on_flare_tint_changed)
        flare_tint_row.addWidget(self.flare_tint_combo)
        flare_layout.addLayout(flare_tint_row)

        flare_distribution_row = QHBoxLayout()
        flare_distribution_label = QLabel("Distribution")
        flare_distribution_row.addWidget(flare_distribution_label)
        flare_distribution_row.addStretch()
        self.flare_distribution_combo = QComboBox()
        self.flare_distribution_combo.addItems(["Highlights", "Uniform", "Edge Trails"])
        self.flare_distribution_combo.setCurrentText(self.parameters.get("flare_distribution", "Highlights"))
        self.flare_distribution_combo.currentTextChanged.connect(self._on_flare_distribution_changed)
        flare_distribution_row.addWidget(self.flare_distribution_combo)
        flare_layout.addLayout(flare_distribution_row)

        flare_shape_row = QHBoxLayout()
        flare_shape_label = QLabel("Shape")
        flare_shape_row.addWidget(flare_shape_label)
        flare_shape_row.addStretch()
        self.flare_shape_combo = QComboBox()
        self.flare_shape_combo.addItems(["Star", "Hex", "Soft Orb"])
        self.flare_shape_combo.setCurrentText(self.parameters.get("flare_shape", "Star"))
        self.flare_shape_combo.currentTextChanged.connect(self._on_flare_shape_changed)
        flare_shape_row.addWidget(self.flare_shape_combo)
        flare_layout.addLayout(flare_shape_row)

        flare_layout.addLayout(self._create_slider("Ghost Count", "flare_amount", 0, 100, int(self.parameters.get("flare_amount", 0))))
        flare_layout.addLayout(self._create_slider("Variation", "flare_variation", 0, 100, int(self.parameters.get("flare_variation", 50))))
        flare_layout.addLayout(self._create_slider("Element Size", "flare_size", 5, 300, int(self.parameters.get("flare_size", 40))))
        flare_layout.addLayout(self._create_slider("Hue", "flare_color_hue", 0, 360, int(self.parameters.get("flare_color_hue", 40))))
        flare_layout.addLayout(self._create_slider("Saturation", "flare_color_sat", 0, 100, int(self.parameters.get("flare_color_sat", 80))))
        flare_layout.addLayout(self._create_slider("Value", "flare_color_value", 0, 100, int(self.parameters.get("flare_color_value", 90))))
        flare_layout.addLayout(self._create_slider("Spacing", "flare_spacing", 0, 100, int(self.parameters.get("flare_spacing", 40))))
        
        layout.addWidget(flare_card)
        
        # Glitch lab
        glitch_card = self._create_card()
        glitch_layout = QVBoxLayout(glitch_card)
        glitch_layout.setContentsMargins(0, 0, 0, 0)
        glitch_layout.setSpacing(8)
        
        glitch_title = QLabel("Glitch Lab")
        glitch_title.setObjectName("cardTitle")
        glitch_layout.addWidget(glitch_title)
        
        glitch_layout.addLayout(self._create_slider("Glitch Amount", "glitch_intensity", 0, 100, int(self.parameters.get("glitch_intensity", 0))))
        glitch_layout.addLayout(self._create_slider("Scan Frequency", "glitch_frequency", 0, 100, int(self.parameters.get("glitch_frequency", 40))))
        glitch_layout.addLayout(self._create_slider("Color Shift", "glitch_shift", 0, 100, int(self.parameters.get("glitch_shift", 40))))
        
        glitch_row = QHBoxLayout()
        glitch_label = QLabel("Style")
        glitch_row.addWidget(glitch_label)
        glitch_row.addStretch()
        self.glitch_style_combo = QComboBox()
        self.glitch_style_combo.addItems(["RGB Split", "Block Shift", "Analog Ripples"])
        self.glitch_style_combo.setCurrentText(self.parameters.get("glitch_style", "RGB Split"))
        self.glitch_style_combo.currentTextChanged.connect(self._on_glitch_style_changed)
        glitch_row.addWidget(self.glitch_style_combo)
        glitch_layout.addLayout(glitch_row)
        
        layout.addWidget(glitch_card)
        
        # Film effects
        film_card = self._create_card()
        film_layout = QVBoxLayout(film_card)
        film_layout.setContentsMargins(0, 0, 0, 0)
        film_layout.setSpacing(8)
        
        film_title = QLabel("Film & Texture")
        film_title.setObjectName("cardTitle")
        film_layout.addWidget(film_title)
        
        film_layout.addLayout(self._create_slider("Grain", "grain", 0, 500, 0))
        film_layout.addLayout(self._create_slider("Vignette", "vignette", 0, 200, 0))
        
        layout.addWidget(film_card)
        layout.addStretch()
        
        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab
    
    def _create_curves_tab(self) -> QWidget:
        """Create the Curves & Levels tab with interactive editors."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Histogram
        hist_card = self._create_card()
        hist_layout = QVBoxLayout(hist_card)
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.setSpacing(8)
        
        self.histogram = HistogramWidget()
        hist_layout.addWidget(self.histogram)
        layout.addWidget(hist_card)
        
        # Curves editor
        curves_card = self._create_card()
        curves_layout = QVBoxLayout(curves_card)
        curves_layout.setContentsMargins(0, 0, 0, 0)
        curves_layout.setSpacing(8)
        
        curves_title = QLabel("Curves")
        curves_title.setObjectName("cardTitle")
        curves_layout.addWidget(curves_title)
        
        self.curves_editor = CurvesEditor()
        self.curves_editor.curveChanged.connect(self._on_curve_changed)
        curves_layout.addWidget(self.curves_editor)
        
        layout.addWidget(curves_card)
        
        # Levels
        levels_card = self._create_card()
        levels_layout = QVBoxLayout(levels_card)
        levels_layout.setContentsMargins(0, 0, 0, 0)
        levels_layout.setSpacing(8)
        
        levels_title = QLabel("Levels")
        levels_title.setObjectName("cardTitle")
        levels_layout.addWidget(levels_title)
        
        levels_layout.addLayout(self._create_slider("Input Black", "levels_black", 0, 255, 0))
        levels_layout.addLayout(self._create_slider("Input White", "levels_white", 0, 255, 255))
        levels_layout.addLayout(self._create_slider("Gamma", "levels_gamma", 10, 300, 100))
        levels_layout.addLayout(self._create_slider("Output Black", "out_black", 0, 255, 0))
        levels_layout.addLayout(self._create_slider("Output White", "out_white", 0, 255, 255))
        
        layout.addWidget(levels_card)
        layout.addStretch()
        
        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab
    
    def _create_compare_tab(self) -> QWidget:
        """Create the Compare tab with split view."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Split view widget
        self.split_view = SplitViewWidget()
        layout.addWidget(self.split_view, 1)
        
        # Update button
        update_btn = QPushButton("Update Comparison")
        update_btn.setObjectName("primaryButton")
        update_btn.clicked.connect(self._update_comparison)
        layout.addWidget(update_btn)
        
        return tab
    
    def _create_export_tab(self) -> QWidget:
        """Create the Export tab with quality and format options."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Format selection
        format_card = self._create_card()
        format_layout = QVBoxLayout(format_card)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(10)
        
        format_title = QLabel("Export Format")
        format_title.setObjectName("cardTitle")
        format_layout.addWidget(format_title)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([
            "PNG (Lossless)",
            "JPEG (Quality)",
            "TIFF (Uncompressed)",
            "BMP (Raw)",
            "WebP (Modern)",
            "PDF (Vector)",
            "SVG (Vector)"
        ])
        format_layout.addWidget(self.export_format_combo)
        
        layout.addWidget(format_card)
        
        # Quality settings
        quality_card = self._create_card()
        quality_layout = QVBoxLayout(quality_card)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(8)
        
        quality_title = QLabel("Quality Settings")
        quality_title.setObjectName("cardTitle")
        quality_layout.addWidget(quality_title)
        
        quality_layout.addLayout(self._create_slider("Quality %", "export_quality", 1, 100, 95))
        quality_layout.addLayout(self._create_slider("DPI", "export_dpi", 72, 600, 300))
        
        self.export_metadata_cb = QCheckBox("Include metadata")
        self.export_metadata_cb.setChecked(True)
        quality_layout.addWidget(self.export_metadata_cb)
        
        self.export_color_profile_cb = QCheckBox("Embed color profile (sRGB)")
        self.export_color_profile_cb.setChecked(True)
        quality_layout.addWidget(self.export_color_profile_cb)
        
        layout.addWidget(quality_card)
        
        # Resize options
        resize_card = self._create_card()
        resize_layout = QVBoxLayout(resize_card)
        resize_layout.setContentsMargins(0, 0, 0, 0)
        resize_layout.setSpacing(8)
        
        resize_title = QLabel("Resize on Export")
        resize_title.setObjectName("cardTitle")
        resize_layout.addWidget(resize_title)
        
        self.export_resize_cb = QCheckBox("Resize output")
        resize_layout.addWidget(self.export_resize_cb)
        
        resize_layout.addLayout(self._create_slider("Width", "export_width", 100, 8000, 1920))
        resize_layout.addLayout(self._create_slider("Height", "export_height", 100, 8000, 1080))
        
        layout.addWidget(resize_card)
        
        # Export buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        export_btn = QPushButton("💾 Export Image")
        export_btn.setObjectName("primaryButton")
        export_btn.clicked.connect(self._export_image)
        btn_row.addWidget(export_btn)
        
        batch_btn = QPushButton("📁 Batch Export")
        batch_btn.clicked.connect(self._batch_process)
        btn_row.addWidget(batch_btn)
        
        layout.addLayout(btn_row)
        layout.addStretch()
        
        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab
    
    def _create_settings_tab(self) -> QWidget:
        """Create the Settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # Preview settings
        preview_card = self._create_card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)
        
        preview_title = QLabel("Preview")
        preview_title.setObjectName("cardTitle")
        preview_layout.addWidget(preview_title)
        
        self.live_preview_cb = QCheckBox("Live preview (auto-update)")
        self.live_preview_cb.setChecked(True)
        self.live_preview_cb.stateChanged.connect(self._on_live_preview_changed)
        preview_layout.addWidget(self.live_preview_cb)
        
        self.fast_preview_cb = QCheckBox("Fast mode (lower quality)")
        self.fast_preview_cb.setChecked(True)
        self.fast_preview_cb.stateChanged.connect(self._on_fast_preview_changed)
        preview_layout.addWidget(self.fast_preview_cb)
        
        layout.addWidget(preview_card)
        
        # Zoom
        zoom_card = self._create_card()
        zoom_layout = QVBoxLayout(zoom_card)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(10)
        
        zoom_title = QLabel("Zoom")
        zoom_title.setObjectName("cardTitle")
        zoom_layout.addWidget(zoom_title)
        
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(8)
        for text, slot in [("−", self.viewer.zoom_out), ("Fit", self.viewer.reset_zoom), ("+", self.viewer.zoom_in)]:
            btn = QPushButton(text)
            btn.setObjectName("secondaryButton")
            btn.clicked.connect(slot)
            zoom_row.addWidget(btn)
        zoom_layout.addLayout(zoom_row)
        
        layout.addWidget(zoom_card)
        
        # Info
        info_label = QLabel("OpenDither v2.0\nPress F1 for help")
        info_label.setObjectName("hintLabel")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        layout.addStretch()
        return tab

    def _create_card(self) -> QFrame:
        """Create a styled card container."""
        card = QFrame()
        card.setObjectName("card")
        card.setContentsMargins(0, 0, 0, 0)
        return card

    def _create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        return line

    def _create_slider(
        self, label: str, key: str, min_val: int, max_val: int, default: int
    ) -> QVBoxLayout:
        """Create a compact labeled slider."""
        container = QVBoxLayout()
        container.setSpacing(4)
        
        # Label row with value
        label_row = QHBoxLayout()
        name_label = QLabel(label)
        label_row.addWidget(name_label)
        label_row.addStretch()
        
        value_label = QLabel(str(default))
        value_label.setObjectName("valueLabel")
        label_row.addWidget(value_label)
        container.addLayout(label_row)
        
        # Slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(lambda v: self._on_slider_changed(key, v))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        container.addWidget(slider)
        
        # Store reference
        setattr(self, f"slider_{key}", slider)
        
        return container

    # ========== Event Handlers ==========

    def _on_style_changed(self, text: str) -> None:
        """Handle style/algorithm selection."""
        if text == "None":
            self.current_algorithm = "None"
        else:
            self.current_algorithm = text
        self._schedule_update()

    def _on_preset_changed(self, text: str) -> None:
        """Handle preset selection."""
        if text == "None":
            return
        
        preset = self.preset_library.get(text)
        if not preset:
            return
        
        # Apply preset settings
        self.current_algorithm = preset.algorithm
        self.style_combo.setCurrentText(preset.algorithm)
        
        if preset.palette:
            self.current_palette = preset.palette
            self.palette_combo.setCurrentText(preset.palette)
        
        # Apply parameters
        for key, value in preset.parameters.items():
            if hasattr(self, f"slider_{key}"):
                slider = getattr(self, f"slider_{key}")
                slider.setValue(int(value * 100))
        
        self._schedule_update()

    def _on_palette_category_changed(self, text: str) -> None:
        """Handle palette category selection."""
        self.palette_combo.clear()
        self.palette_combo.addItem("None")
        
        if text == "All":
            # Show all palettes
            for name in self.palette_library.names():
                self.palette_combo.addItem(name)
        elif text != "None":
            categories = self.palette_library.categories()
            if text in categories:
                for palette in categories[text]:
                    self.palette_combo.addItem(palette.name)

    def _on_palette_changed(self, text: str) -> None:
        """Handle palette selection."""
        self.current_palette = text if text != "None" else None
        
        # Update palette preview
        self._update_palette_preview(text)
        self._schedule_update()
    
    def _update_palette_preview(self, palette_name: str) -> None:
        """Update the palette color swatches preview."""
        if palette_name == "None" or not palette_name:
            self.palette_preview.setStyleSheet("background: #27272a; border-radius: 4px;")
            return
        
        palette = self.palette_library.get(palette_name)
        if not palette:
            return
        
        # Create gradient from palette colors
        colors = palette.colors
        n = len(colors)
        stops = []
        for i, color in enumerate(colors):
            pos = i / (n - 1) if n > 1 else 0
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            stops.append(f"stop:{pos:.2f} {hex_color}")
        
        gradient = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)}); border-radius: 4px;"
        self.palette_preview.setStyleSheet(gradient)
    
    def _on_gradient_changed(self, text: str) -> None:
        """Handle gradient map selection."""
        gradients = {
            "None": [(0, 0, 0), (255, 255, 255)],
            "Black → White": [(0, 0, 0), (255, 255, 255)],
            "Sepia": [(44, 33, 21), (255, 243, 224)],
            "Duotone Blue": [(15, 23, 42), (59, 130, 246)],
            "Duotone Purple": [(30, 15, 45), (168, 85, 247)],
            "Sunset": [(45, 10, 30), (255, 100, 50), (255, 200, 100)],
            "Cool Tones": [(10, 30, 50), (50, 120, 180), (200, 230, 255)],
            "Warm Tones": [(50, 20, 10), (180, 80, 40), (255, 220, 180)],
            "Cyanotype": [(10, 20, 40), (50, 130, 180)],
            "Infrared": [(20, 0, 30), (255, 100, 150), (255, 255, 200)],
        }
        
        colors = gradients.get(text, gradients["None"])
        n = len(colors)
        stops = []
        for i, color in enumerate(colors):
            pos = i / (n - 1) if n > 1 else 0
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            stops.append(f"stop:{pos:.2f} {hex_color}")
        
        gradient = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)}); border-radius: 4px;"
        self.gradient_preview.setStyleSheet(gradient)
        
        # Store gradient for processing
        if text != "None":
            self.parameters["gradient_map"] = text
        else:
            self.parameters.pop("gradient_map", None)
        
        self._schedule_update()

    def _on_slider_changed(self, key: str, value: int) -> None:
        """Handle slider value changes - store raw value, not normalized."""
        self.parameters[key] = value
        self._schedule_update()

    def _on_invert_changed(self, state: int) -> None:
        """Handle invert toggle."""
        self.parameters["invert"] = int(bool(state))
        self._schedule_update()
    
    def _on_preserve_colors_changed(self, state: int) -> None:
        """Handle keep original colors toggle."""
        self.parameters["preserve_colors"] = int(bool(state))
        self._schedule_update()
    
    def _on_flare_style_changed(self, text: str) -> None:
        """Handle flare style selection."""
        self.parameters["flare_style"] = text
        self._schedule_update()
    
    def _on_flare_tint_changed(self, text: str) -> None:
        """Handle flare tint selection."""
        self.parameters["flare_tint"] = text
        self._schedule_update()
    
    def _on_flare_distribution_changed(self, text: str) -> None:
        """Handle flare distribution selection."""
        self.parameters["flare_distribution"] = text
        self._schedule_update()
    
    def _on_flare_shape_changed(self, text: str) -> None:
        """Handle flare shape selection."""
        self.parameters["flare_shape"] = text
        self._schedule_update()
    
    def _on_glitch_style_changed(self, text: str) -> None:
        """Handle glitch style selection."""
        self.parameters["glitch_style"] = text
        self._schedule_update()

    def _on_live_preview_changed(self, state: int) -> None:
        """Handle live preview toggle."""
        self._live_preview = bool(state)

    def _on_fast_preview_changed(self, state: int) -> None:
        """Handle fast preview toggle."""
        self._preview_mode = bool(state)

    def _on_curve_changed(self, channel: str, lut: list) -> None:
        """Handle curve change from curves editor."""
        # Store the LUT for this channel
        if not hasattr(self, '_curve_luts'):
            self._curve_luts = {}
        self._curve_luts[channel] = lut
        self._schedule_update()
    
    def _update_comparison(self) -> None:
        """Update the split view comparison."""
        if self.original_image is None:
            return
        
        # Convert numpy arrays to QImage
        before_qimg = self._numpy_to_qimage(self.original_image)
        
        if self.processed_image is not None:
            after_qimg = self._numpy_to_qimage(self.processed_image)
        else:
            after_qimg = before_qimg
        
        self.split_view.set_images(before_qimg, after_qimg)
    
    def _numpy_to_qimage(self, image: NDArray[np.uint8]) -> QImage:
        """Convert numpy array to QImage."""
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = w * 3
            return QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            return QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    
    def _update_histogram(self) -> None:
        """Update histogram with current image."""
        if hasattr(self, 'histogram'):
            img = self.processed_image if self.processed_image is not None else self.original_image
            self.histogram.set_image(img)

    def _refresh_preset_combo(self) -> None:
        """Refresh the preset combo box."""
        self.preset_combo.clear()
        self.preset_combo.addItem("None")
        for name in self.preset_library.names():
            self.preset_combo.addItem(name)

    def _force_apply(self) -> None:
        """Force apply processing (ignores live preview setting)."""
        if self.original_image is None:
            self.statusBar().showMessage("No image loaded")
            return
        self._do_update()

    # ========== Processing ==========

    def _schedule_update(self) -> None:
        """Schedule a processing update with debouncing."""
        if self.original_image is None:
            return
        
        if not self._live_preview:
            return  # Don't auto-update if live preview is off
        
        # Use debounce timer to avoid too many updates
        self._debounce_timer.start()

    def _do_update(self) -> None:
        """Actually perform the update (called after debounce)."""
        if self.original_image is None:
            return
        
        if self.worker_thread and self.worker_thread.isRunning():
            self._pending_update = True
            return
        
        self._apply_processing()

    def _show_loading(self, message: str = "Processing...") -> None:
        """Show the loading overlay."""
        self.loading_label.setText(message)
        self.loading_overlay.setGeometry(self.viewer.geometry())
        self.loading_overlay.show()
        self.loading_overlay.raise_()
    
    def _hide_loading(self) -> None:
        """Hide the loading overlay."""
        self.loading_overlay.hide()

    def _apply_processing(self) -> None:
        """Apply current settings to the image."""
        if self.original_image is None:
            return
        
        # Show loading overlay
        self._show_loading(f"Applying {self.current_algorithm}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)  # Real progress 0-100%
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Processing... 0%")
        
        # Create worker thread with curves LUT
        curves_lut = self._curve_luts if self._curve_luts else None
        
        self.worker_thread, self.worker = create_worker_thread(
            self.original_image,
            self.current_algorithm,
            self.parameters.copy(),
            self.current_palette,
            curves_lut,
        )
        
        self.worker.finished.connect(self._on_processing_finished)
        self.worker.error.connect(self._on_processing_error)
        self.worker.progress.connect(self._on_processing_progress)
        self.worker_thread.start()
    
    def _on_processing_progress(self, percent: int, stage: str) -> None:
        """Handle progress updates from worker."""
        self.progress_bar.setValue(percent)
        self.loading_label.setText(f"{stage.title()}... {percent}%")
        self.statusBar().showMessage(f"Processing {stage}... {percent}%")

    def _on_processing_finished(self, result: NDArray[np.uint8]) -> None:
        """Handle processing completion."""
        self.processed_image = result
        self._update_viewer(result, preserve_view=True)
        
        # Update histogram
        self._update_histogram()
        
        # Hide loading overlay
        self._hide_loading()
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("✓ Ready")
        
        # Clean up
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
        
        # Process pending update if any
        if self._pending_update:
            self._pending_update = False
            self._apply_processing()

    def _on_processing_error(self, error: str) -> None:
        """Handle processing error."""
        self._pending_update = False
        self._hide_loading()
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"✗ Error: {error}")
        
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    def _update_viewer(self, image: NDArray[np.uint8], preserve_view: bool = False) -> None:
        """Update the image viewer."""
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = w * 3
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        self.viewer.set_image(qimage, preserve_view=preserve_view)

    # ========== File Operations ==========

    def _import_image(self) -> None:
        """Import an image file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"
        )
        if not path:
            return
        
        self._load_image(Path(path))

    def _load_image(self, path: Path) -> None:
        """Load an image from path."""
        image = cv2.imread(str(path))
        if image is None:
            QMessageBox.warning(self, "Error", f"Could not load image: {path}")
            return
        
        # Convert BGR to RGB
        self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.processed_image = self.original_image.copy()
        
        self._update_viewer(self.original_image)
        self.export_btn.setEnabled(True)
        
        h, w = self.original_image.shape[:2]
        self.setWindowTitle(f"OpenDither - {path.name} ({w}x{h})")
        self.statusBar().showMessage(f"Loaded: {path.name}")
        
        # Apply initial processing
        self._schedule_update()

    def _export_image(self) -> None:
        """Export the processed image."""
        if self.processed_image is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "dithered.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if not path:
            return
        
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        self.statusBar().showMessage(f"Exported: {Path(path).name}")

    def _export_svg(self) -> None:
        """Export as SVG vector."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No image to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SVG", "dithered.svg", "SVG (*.svg)"
        )
        if not path:
            return
        
        # Simple SVG export (placeholder - would need proper implementation)
        QMessageBox.information(self, "SVG Export", "SVG export feature coming soon!")

    # ========== Preset Operations ==========

    def _save_preset(self) -> None:
        """Save current settings as a preset."""
        name, ok = QInputDialog.getText(
            self, 
            "Save Preset", 
            "Enter a name for your preset:",
            text=f"{self.current_algorithm} Custom"
        )
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # Check if preset exists
        if self.preset_library.get(name):
            reply = QMessageBox.question(
                self,
                "Overwrite Preset",
                f"A preset named '{name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        preset = Preset(
            name=name,
            algorithm=self.current_algorithm,
            parameters=self.parameters.copy(),
            palette=self.current_palette,
            category="Custom",
            description=f"Created from {self.current_algorithm}"
        )
        self.preset_library.register(preset)
        
        # Refresh combo
        self._refresh_preset_combo()
        self.preset_combo.setCurrentText(name)
        
        QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully!")
        self.statusBar().showMessage(f"Preset saved: {name}")

    def _import_presets(self) -> None:
        """Import presets from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Presets", "", "JSON (*.json)"
        )
        if not path:
            return
        
        try:
            count = self.preset_library.load_from_file(Path(path))
            self._refresh_preset_combo()
            QMessageBox.information(self, "Success", f"Imported {count} presets successfully!")
            self.statusBar().showMessage(f"Imported {count} presets")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import presets: {e}")

    def _export_presets(self) -> None:
        """Export presets to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Presets", "presets.json", "JSON (*.json)"
        )
        if not path:
            return
        
        try:
            self.preset_library.save_to_file(Path(path))
            self.statusBar().showMessage(f"Exported presets to {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export presets: {e}")

    def _reset_all(self) -> None:
        """Reset all adjustments to defaults (neutral = 0)."""
        defaults = self._default_parameters.copy()
        
        # Reset all sliders
        for key, default in defaults.items():
            if hasattr(self, f"slider_{key}"):
                slider = getattr(self, f"slider_{key}")
                slider.blockSignals(True)
                if isinstance(default, (int, float)):
                    slider.setValue(int(default))
                slider.blockSignals(False)
        
        # Reset parameters dict
        self.parameters = defaults.copy()
        self.parameters.pop("gradient_map", None)
        
        # Reset curves
        self._curve_luts = {}
        if hasattr(self, 'curves_editor'):
            self.curves_editor.canvas.reset_curve()
        
        # Reset UI elements
        self.current_algorithm = "None"
        self.invert_cb.setChecked(False)
        self.style_combo.setCurrentText("None")
        self.preset_combo.setCurrentIndex(0)
        self.palette_combo.setCurrentIndex(0)
        self.palette_cat_combo.setCurrentIndex(0)
        
        if hasattr(self, 'gradient_combo'):
            self.gradient_combo.setCurrentIndex(0)
        if hasattr(self, 'palette_preview'):
            self.palette_preview.setStyleSheet("background: #27272a; border-radius: 4px;")
        
        if hasattr(self, 'gradient_combo'):
            self.gradient_combo.setCurrentIndex(0)
        
        # Update viewer with original
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self._update_viewer(self.original_image)
            self._update_histogram()
        
        if hasattr(self, 'preserve_colors_cb'):
            self.preserve_colors_cb.setChecked(False)
        if hasattr(self, 'flare_style_combo'):
            self.flare_style_combo.setCurrentText("Lens")
        if hasattr(self, 'flare_tint_combo'):
            self.flare_tint_combo.setCurrentText("Warm")
        if hasattr(self, 'flare_distribution_combo'):
            self.flare_distribution_combo.setCurrentText("Highlights")
        if hasattr(self, 'flare_shape_combo'):
            self.flare_shape_combo.setCurrentText("Star")
        if hasattr(self, 'glitch_style_combo'):
            self.glitch_style_combo.setCurrentText("RGB Split")
        
        self.statusBar().showMessage("✓ All settings reset to neutral")

    def _batch_process(self) -> None:
        """Process all images in a selected folder."""
        input_dir = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not input_dir:
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return
        
        input_path = Path(input_dir)
        images = [
            *input_path.glob("*.png"),
            *input_path.glob("*.jpg"),
            *input_path.glob("*.jpeg"),
            *input_path.glob("*.bmp"),
            *input_path.glob("*.tif"),
            *input_path.glob("*.tiff"),
            *input_path.glob("*.webp"),
        ]
        
        if not images:
            QMessageBox.warning(self, "Warning", "No images found in the selected folder.")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(images))
        
        processed = 0
        for i, img_path in enumerate(images, start=1):
            self.progress_bar.setValue(i)
            self.statusBar().showMessage(f"Processing {img_path.name} ({i}/{len(images)})")
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.engine.process(
                image_rgb,
                self.current_algorithm,
                self.parameters.copy(),
                self.current_palette,
            )
            
            output_path = Path(output_dir) / img_path.name
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_bgr)
            processed += 1
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Batch complete: {processed} images processed")
        QMessageBox.information(self, "Batch Complete", f"Processed {processed} images.")

    def _process_video(self) -> None:
        """Process a video file."""
        QMessageBox.information(self, "Video Processing", "Video processing feature coming soon!")

    # ========== Help ==========

    def _show_help(self) -> None:
        """Show help dialog."""
        help_text = """
<h2>OpenDither User Guide</h2>

<h3>Quick Start</h3>
<ol>
<li>Import an image using <b>Import</b> button or <b>Ctrl+O</b></li>
<li>Select a <b>Style</b> (dithering algorithm)</li>
<li>Adjust sliders to fine-tune the effect</li>
<li>Export your result using <b>Export</b> or <b>Ctrl+S</b></li>
</ol>

<h3>Keyboard Shortcuts</h3>
<ul>
<li><b>Ctrl+O</b> - Import image</li>
<li><b>Ctrl+S</b> - Export image</li>
<li><b>Ctrl+Q</b> - Quit</li>
<li><b>Ctrl+Shift+P</b> - Show this help</li>
</ul>

<h3>Features</h3>
<ul>
<li><b>50+ Dithering Algorithms</b> - Error diffusion, ordered, halftone, and more</li>
<li><b>Color Palettes</b> - Retro computing and artistic palettes</li>
<li><b>Presets</b> - Save and load your favorite settings</li>
<li><b>Batch Processing</b> - Process entire folders</li>
<li><b>Live Preview</b> - See changes in real-time</li>
</ul>
"""
        QMessageBox.about(self, "OpenDither Help", help_text)

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """
<h2>OpenDither v1.0.0</h2>
<p>Professional retro image dithering application.</p>
<p>Featuring 50+ unique dithering algorithms for authentic retro effects.</p>
<p><b>Categories:</b></p>
<ul>
<li>15 Error Diffusion Algorithms</li>
<li>10 Ordered/Bitmap Dithers</li>
<li>5 Halftone Effects</li>
<li>10 Pattern Dithers</li>
<li>10 Modulation Effects</li>
<li>10 Special Effects</li>
</ul>
<p>© 2024 OpenDither Team</p>
"""
        QMessageBox.about(self, "About OpenDither", about_text)

    # ========== Drag and Drop ==========

    def dragEnterEvent(self, event) -> None:
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        """Handle file drop."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self._load_image(Path(path))
                event.acceptProposedAction()
                return
        event.ignore()

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Clean up worker thread
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

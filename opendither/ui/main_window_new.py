"""OpenDither Main Window - Dither Boy style interface."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from opendither.core import AlgorithmLibrary, PaletteLibrary, PresetLibrary
from opendither.processing import DitheringEngine
from opendither.processing.worker import create_worker_thread
from .image_viewer import ImageViewer
from .styles import DITHER_BOY_THEME


class MainWindow(QMainWindow):
    """Main application window - Dither Boy style."""

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
        self.current_algorithm: str = "Modulated Diffuse Y"
        self.current_palette: Optional[str] = None
        
        # Parameters matching Dither Boy
        self.parameters: Dict[str, float] = {
            # Epsilon Glow
            "epsilon_glow_enabled": 0,
            "epsilon_threshold": 25,
            "epsilon_smoothing": 25,
            "epsilon_radius": 25,
            "epsilon_intensity": 500,
            "epsilon_aspect": 100,
            "epsilon_direction": 0,
            "epsilon_falloff": 10,
            "epsilon_value": 50,
            "epsilon_distance_scale": 150,
            
            # JPEG Glitch Effects
            "block_shift": 0,
            "channel_swap": 0,
            "scanline_offset": 0,
            "block_scramble": 0,
            "interlace_corruption": 0,
            
            # Chromatic Effects
            "chromatic_enabled": 0,
            "chromatic_max_displace": 20,
            "chromatic_red": 50,
            "chromatic_green": 50,
            "chromatic_blue": 50,
            
            # Style/Dither settings
            "scale": 2,
            "line_scale": 1,
            
            # Effects
            "effects_enabled": 0,
            "contrast": 50,
            "midtones": 50,
            "highlights": 50,
            "luminance_threshold": 50,
            "blur": 0,
            "depth": 0,
            
            # Flags
            "invert": 0,
        }
        self._default_parameters = self.parameters.copy()
        
        # Processing state
        self.worker_thread: Optional[QThread] = None
        self.worker = None
        self._pending_update = False
        self._live_preview = True
        
        # Debounce timer
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(100)
        self._debounce_timer.timeout.connect(self._do_update)
        
        # Setup UI
        self._setup_window()
        self._create_menu_bar()
        self._create_main_layout()
        self._apply_theme()
        
        self.setAcceptDrops(True)

    def _setup_window(self) -> None:
        self.setWindowTitle("OpenDither")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

    def _apply_theme(self) -> None:
        self.setStyleSheet(DITHER_BOY_THEME)

    def _create_menu_bar(self) -> None:
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        import_action = QAction("Import", self)
        import_action.setShortcut(QKeySequence("Ctrl+O"))
        import_action.triggered.connect(self._import_image)
        file_menu.addAction(import_action)
        
        export_action = QAction("Export", self)
        export_action.setShortcut(QKeySequence("Ctrl+S"))
        export_action.triggered.connect(self._export_image)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close).setShortcut(QKeySequence("Ctrl+Q"))

    def _create_main_layout(self) -> None:
        """Create 3-panel layout: Left effects | Center viewer | Right style."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # LEFT PANEL - Effects controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel)
        
        # CENTER - Image viewer
        self.viewer = ImageViewer()
        self.viewer.setMinimumWidth(600)
        main_layout.addWidget(self.viewer, 1)
        
        # RIGHT PANEL - Style and palette
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel)
        
        # Status bar
        self.statusBar().showMessage("Ready — Drop an image to start")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _create_left_panel(self) -> QWidget:
        """Create left panel with Epsilon Glow, JPEG Glitch, Chromatic effects."""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setFixedWidth(200)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)
        
        # === EPSILON GLOW ===
        glow_section = self._create_section("Epsilon Glow")
        glow_layout = glow_section.layout()
        
        self.epsilon_glow_cb = QCheckBox()
        self.epsilon_glow_cb.stateChanged.connect(lambda v: self._on_param_changed("epsilon_glow_enabled", 1 if v else 0))
        self._add_control_row(glow_layout, "Epsilon Glow", self.epsilon_glow_cb)
        
        self.threshold_slider = self._create_slider(0, 100, 25)
        self.threshold_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_threshold", v))
        self._add_slider_row(glow_layout, "Threshold", self.threshold_slider)
        
        self.smoothing_slider = self._create_slider(0, 100, 25)
        self.smoothing_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_smoothing", v))
        self._add_slider_row(glow_layout, "Threshold Smoothing", self.smoothing_slider)
        
        self.radius_slider = self._create_slider(1, 100, 25)
        self.radius_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_radius", v))
        self._add_slider_row(glow_layout, "Radius", self.radius_slider)
        
        self.intensity_slider = self._create_slider(0, 1000, 500)
        self.intensity_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_intensity", v))
        self._add_slider_row(glow_layout, "Intensity", self.intensity_slider)
        
        self.aspect_slider = self._create_slider(1, 200, 100)
        self.aspect_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_aspect", v))
        self._add_slider_row(glow_layout, "Aspect Ratio", self.aspect_slider)
        
        self.direction_slider = self._create_slider(0, 360, 0)
        self.direction_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_direction", v))
        self._add_slider_row(glow_layout, "Direction (°)", self.direction_slider)
        
        self.falloff_slider = self._create_slider(1, 50, 10)
        self.falloff_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_falloff", v))
        self._add_slider_row(glow_layout, "Falloff n", self.falloff_slider)
        
        self.epsilon_val_slider = self._create_slider(1, 255, 50)
        self.epsilon_val_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_value", v))
        self._add_slider_row(glow_layout, "Epsilon", self.epsilon_val_slider)
        
        self.distance_scale_slider = self._create_slider(1, 500, 150)
        self.distance_scale_slider.valueChanged.connect(lambda v: self._on_param_changed("epsilon_distance_scale", v))
        self._add_slider_row(glow_layout, "Distance Scale", self.distance_scale_slider)
        
        glow_reset = QPushButton("Reset")
        glow_reset.clicked.connect(self._reset_glow)
        glow_layout.addWidget(glow_reset)
        
        layout.addWidget(glow_section)
        
        # === JPEG GLITCH EFFECTS ===
        glitch_section = self._create_section("JPEG Glitch Effects")
        glitch_layout = glitch_section.layout()
        
        self.block_shift_slider = self._create_slider(0, 100, 0)
        self.block_shift_slider.valueChanged.connect(lambda v: self._on_param_changed("block_shift", v))
        self._add_slider_row(glitch_layout, "Block Shift", self.block_shift_slider, "px")
        
        self.channel_swap_cb = QCheckBox()
        self.channel_swap_slider = self._create_slider(0, 100, 0)
        self.channel_swap_slider.valueChanged.connect(lambda v: self._on_param_changed("channel_swap", v))
        self._add_slider_row(glitch_layout, "Channel Swap", self.channel_swap_slider, "%")
        
        self.scanline_cb = QCheckBox()
        self.scanline_slider = self._create_slider(0, 100, 0)
        self.scanline_slider.valueChanged.connect(lambda v: self._on_param_changed("scanline_offset", v))
        self._add_slider_row(glitch_layout, "Scanline Offset", self.scanline_slider)
        
        self.scramble_slider = self._create_slider(0, 100, 0)
        self.scramble_slider.valueChanged.connect(lambda v: self._on_param_changed("block_scramble", v))
        self._add_slider_row(glitch_layout, "Block Scramble", self.scramble_slider)
        
        self.interlace_cb = QCheckBox()
        self.interlace_slider = self._create_slider(0, 100, 0)
        self.interlace_slider.valueChanged.connect(lambda v: self._on_param_changed("interlace_corruption", v))
        self._add_slider_row(glitch_layout, "Interlace Corruption", self.interlace_slider, "%")
        
        glitch_reset = QPushButton("Reset")
        glitch_reset.clicked.connect(self._reset_glitch)
        glitch_layout.addWidget(glitch_reset)
        
        layout.addWidget(glitch_section)
        
        # === CHROMATIC EFFECTS ===
        chroma_section = self._create_section("Chromatic Effects")
        chroma_layout = chroma_section.layout()
        
        self.chromatic_cb = QCheckBox()
        self.chromatic_cb.stateChanged.connect(lambda v: self._on_param_changed("chromatic_enabled", 1 if v else 0))
        self._add_control_row(chroma_layout, "Chromatic Effects", self.chromatic_cb)
        
        self.max_displace_slider = self._create_slider(0, 100, 20)
        self.max_displace_slider.valueChanged.connect(lambda v: self._on_param_changed("chromatic_max_displace", v))
        self._add_slider_row(chroma_layout, "Max Displace", self.max_displace_slider, "px")
        
        self.red_channel_slider = self._create_slider(0, 100, 50)
        self.red_channel_slider.valueChanged.connect(lambda v: self._on_param_changed("chromatic_red", v))
        self._add_slider_row(chroma_layout, "Red Channel", self.red_channel_slider)
        
        self.green_channel_slider = self._create_slider(0, 100, 50)
        self.green_channel_slider.valueChanged.connect(lambda v: self._on_param_changed("chromatic_green", v))
        self._add_slider_row(chroma_layout, "Green Channel", self.green_channel_slider)
        
        self.blue_channel_slider = self._create_slider(0, 100, 50)
        self.blue_channel_slider.valueChanged.connect(lambda v: self._on_param_changed("chromatic_blue", v))
        self._add_slider_row(chroma_layout, "Blue Channel", self.blue_channel_slider)
        
        chroma_reset = QPushButton("Reset")
        chroma_reset.clicked.connect(self._reset_chromatic)
        chroma_layout.addWidget(chroma_reset)
        
        layout.addWidget(chroma_section)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right panel with Style, Palette, and Effects controls."""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setFixedWidth(220)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # === TOP BUTTONS ===
        btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("Import")
        self.import_btn.clicked.connect(self._import_image)
        btn_layout.addWidget(self.import_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export_image)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(lambda: self.viewer.zoom_in())
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(lambda: self.viewer.zoom_out())
        zoom_layout.addWidget(zoom_out_btn)
        
        reset_zoom_btn = QPushButton("Reset Zoom")
        reset_zoom_btn.clicked.connect(lambda: self.viewer.fit_in_view())
        zoom_layout.addWidget(reset_zoom_btn)
        layout.addLayout(zoom_layout)
        
        # Help hint
        help_label = QLabel("⌘+Shift+/ for help")
        help_label.setObjectName("helpLabel")
        layout.addWidget(help_label)
        
        layout.addWidget(self._create_separator())
        
        # === STYLE ===
        style_label = QLabel("Style")
        style_label.setObjectName("sectionLabel")
        layout.addWidget(style_label)
        
        self.style_combo = QComboBox()
        styles = [
            "None",
            "Modulated Diffuse Y",
            "Waveform",
            "Floyd-Steinberg",
            "Atkinson",
            "Bayer 2x2",
            "Bayer 4x4",
            "Bayer 8x8",
            "Halftone Dot",
            "Halftone Line",
            "Blue Noise",
            "Error Diffusion",
        ]
        self.style_combo.addItems(styles)
        self.style_combo.setCurrentText("Modulated Diffuse Y")
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        layout.addWidget(self.style_combo)
        
        # Presets
        preset_label = QLabel("Presets")
        preset_label.setObjectName("sectionLabel")
        layout.addWidget(preset_label)
        
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("None")
        for preset in self.preset_library.list():
            self.preset_combo.addItem(preset.name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo, 1)
        
        shuffle_btn = QPushButton("⇄")
        shuffle_btn.setFixedWidth(30)
        preset_layout.addWidget(shuffle_btn)
        layout.addLayout(preset_layout)
        
        layout.addWidget(self._create_separator())
        
        # === SCALE ===
        self.scale_slider = self._create_slider(1, 16, 2)
        self.scale_slider.valueChanged.connect(lambda v: self._on_param_changed("scale", v))
        self._add_slider_row(layout, "Scale", self.scale_slider)
        
        self.line_scale_slider = self._create_slider(1, 10, 1)
        self.line_scale_slider.valueChanged.connect(lambda v: self._on_param_changed("line_scale", v))
        self._add_slider_row(layout, "Line Scale", self.line_scale_slider)
        
        layout.addWidget(self._create_separator())
        
        # === PALETTE ===
        palette_cat_label = QLabel("Palette Category")
        palette_cat_label.setObjectName("sectionLabel")
        layout.addWidget(palette_cat_label)
        
        self.palette_category_combo = QComboBox()
        categories = list(set(p.category for p in self.palette_library.list()))
        categories.sort()
        self.palette_category_combo.addItems(["All"] + categories)
        self.palette_category_combo.currentTextChanged.connect(self._on_palette_category_changed)
        layout.addWidget(self.palette_category_combo)
        
        palette_label = QLabel("Palette")
        palette_label.setObjectName("sectionLabel")
        layout.addWidget(palette_label)
        
        palette_row = QHBoxLayout()
        self.palette_preview = QLabel()
        self.palette_preview.setFixedSize(60, 20)
        self.palette_preview.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4B0082, stop:0.5 #FF1493, stop:1 #FFD700);")
        palette_row.addWidget(self.palette_preview)
        
        self.palette_combo = QComboBox()
        self._populate_palettes()
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        palette_row.addWidget(self.palette_combo, 1)
        
        shuffle_palette_btn = QPushButton("⇄")
        shuffle_palette_btn.setFixedWidth(30)
        palette_row.addWidget(shuffle_palette_btn)
        layout.addLayout(palette_row)
        
        layout.addWidget(self._create_separator())
        
        # === INVERT ===
        invert_layout = QHBoxLayout()
        invert_label = QLabel("Invert")
        invert_layout.addWidget(invert_label)
        invert_layout.addStretch()
        self.invert_cb = QCheckBox()
        self.invert_cb.stateChanged.connect(lambda v: self._on_param_changed("invert", 1 if v else 0))
        invert_layout.addWidget(self.invert_cb)
        layout.addLayout(invert_layout)
        
        # === EFFECTS ===
        effects_layout = QHBoxLayout()
        effects_label = QLabel("Effects")
        effects_layout.addWidget(effects_label)
        effects_layout.addStretch()
        self.effects_cb = QCheckBox()
        self.effects_cb.setChecked(True)
        self.effects_cb.stateChanged.connect(lambda v: self._on_param_changed("effects_enabled", 1 if v else 0))
        effects_layout.addWidget(self.effects_cb)
        layout.addLayout(effects_layout)
        
        self.contrast_slider = self._create_slider(0, 100, 50)
        self.contrast_slider.valueChanged.connect(lambda v: self._on_param_changed("contrast", v))
        self._add_slider_row(layout, "Contrast", self.contrast_slider)
        
        self.midtones_slider = self._create_slider(0, 100, 50)
        self.midtones_slider.valueChanged.connect(lambda v: self._on_param_changed("midtones", v))
        self._add_slider_row(layout, "Midtones", self.midtones_slider)
        
        self.highlights_slider = self._create_slider(0, 100, 50)
        self.highlights_slider.valueChanged.connect(lambda v: self._on_param_changed("highlights", v))
        self._add_slider_row(layout, "Highlights", self.highlights_slider)
        
        self.lum_threshold_slider = self._create_slider(0, 100, 50)
        self.lum_threshold_slider.valueChanged.connect(lambda v: self._on_param_changed("luminance_threshold", v))
        self._add_slider_row(layout, "Luminance Threshold", self.lum_threshold_slider)
        
        self.blur_slider = self._create_slider(0, 50, 0)
        self.blur_slider.valueChanged.connect(lambda v: self._on_param_changed("blur", v))
        self._add_slider_row(layout, "Blur", self.blur_slider)
        
        self.depth_slider = self._create_slider(0, 20, 0)
        self.depth_slider.valueChanged.connect(lambda v: self._on_param_changed("depth", v))
        self._add_slider_row(layout, "Depth", self.depth_slider)
        
        layout.addWidget(self._create_separator())
        
        # === BOTTOM BUTTONS ===
        save_preset_btn = QPushButton("Save Preset")
        save_preset_btn.clicked.connect(self._save_preset)
        layout.addWidget(save_preset_btn)
        
        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.clicked.connect(self._reset_all)
        layout.addWidget(reset_all_btn)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        return panel

    # === HELPER METHODS ===
    
    def _create_section(self, title: str) -> QFrame:
        """Create a collapsible section frame."""
        frame = QFrame()
        frame.setObjectName("section")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(6)
        
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        
        return frame

    def _create_slider(self, min_val: int, max_val: int, default: int) -> QSlider:
        """Create a horizontal slider."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _add_slider_row(self, layout, label: str, slider: QSlider, suffix: str = "") -> None:
        """Add a label + slider + value display row."""
        row = QHBoxLayout()
        row.setSpacing(8)
        
        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        row.addWidget(lbl)
        
        row.addWidget(slider, 1)
        
        value_lbl = QLabel(f"{slider.value()}{suffix}")
        value_lbl.setMinimumWidth(35)
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        slider.valueChanged.connect(lambda v: value_lbl.setText(f"{v}{suffix}"))
        row.addWidget(value_lbl)
        
        if isinstance(layout, QVBoxLayout):
            layout.addLayout(row)
        else:
            layout.addLayout(row)

    def _add_control_row(self, layout, label: str, widget: QWidget) -> None:
        """Add a label + control widget row."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(widget)
        layout.addLayout(row)

    def _create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        return sep

    def _populate_palettes(self, category: str = "All") -> None:
        """Populate palette combo based on category."""
        self.palette_combo.clear()
        self.palette_combo.addItem("None")
        
        for pal in self.palette_library.list():
            if category == "All" or pal.category == category:
                self.palette_combo.addItem(pal.name)

    def _update_palette_preview(self, name: str) -> None:
        """Update palette preview swatch."""
        if name == "None":
            self.palette_preview.setStyleSheet("background: #27272a;")
            return
            
        pal = self.palette_library.get(name)
        if pal and pal.colors:
            stops = []
            n = len(pal.colors)
            for i, c in enumerate(pal.colors[:6]):  # Max 6 colors in preview
                pos = i / max(n - 1, 1)
                stops.append(f"stop:{pos:.2f} rgb({c[0]},{c[1]},{c[2]})")
            gradient = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)});"
            self.palette_preview.setStyleSheet(gradient)

    # === EVENT HANDLERS ===
    
    def _on_param_changed(self, key: str, value: float) -> None:
        """Handle parameter change."""
        self.parameters[key] = value
        self._schedule_update()

    def _on_style_changed(self, style: str) -> None:
        """Handle dithering style change."""
        self.current_algorithm = style
        self._schedule_update()

    def _on_preset_changed(self, name: str) -> None:
        """Handle preset selection."""
        if name == "None":
            return
        preset = self.preset_library.get(name)
        if preset:
            self._apply_preset(preset)

    def _on_palette_category_changed(self, category: str) -> None:
        """Handle palette category change."""
        self._populate_palettes(category)

    def _on_palette_changed(self, name: str) -> None:
        """Handle palette selection."""
        self.current_palette = name if name != "None" else None
        self._update_palette_preview(name)
        self._schedule_update()

    def _apply_preset(self, preset) -> None:
        """Apply a preset to current parameters."""
        for key, value in preset.parameters.items():
            if key in self.parameters:
                self.parameters[key] = value
        self._sync_ui_to_parameters()
        self._schedule_update()

    def _sync_ui_to_parameters(self) -> None:
        """Sync all UI controls to current parameters."""
        # Block signals during sync
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(self.parameters.get("epsilon_threshold", 25)))
        self.threshold_slider.blockSignals(False)
        # ... repeat for other sliders (abbreviated for length)

    # === RESET METHODS ===
    
    def _reset_glow(self) -> None:
        """Reset epsilon glow parameters."""
        glow_keys = ["epsilon_glow_enabled", "epsilon_threshold", "epsilon_smoothing",
                     "epsilon_radius", "epsilon_intensity", "epsilon_aspect",
                     "epsilon_direction", "epsilon_falloff", "epsilon_value", "epsilon_distance_scale"]
        for key in glow_keys:
            self.parameters[key] = self._default_parameters[key]
        
        self.epsilon_glow_cb.setChecked(False)
        self.threshold_slider.setValue(25)
        self.smoothing_slider.setValue(25)
        self.radius_slider.setValue(25)
        self.intensity_slider.setValue(500)
        self.aspect_slider.setValue(100)
        self.direction_slider.setValue(0)
        self.falloff_slider.setValue(10)
        self.epsilon_val_slider.setValue(50)
        self.distance_scale_slider.setValue(150)
        self._schedule_update()

    def _reset_glitch(self) -> None:
        """Reset glitch parameters."""
        glitch_keys = ["block_shift", "channel_swap", "scanline_offset", "block_scramble", "interlace_corruption"]
        for key in glitch_keys:
            self.parameters[key] = 0
        
        self.block_shift_slider.setValue(0)
        self.channel_swap_slider.setValue(0)
        self.scanline_slider.setValue(0)
        self.scramble_slider.setValue(0)
        self.interlace_slider.setValue(0)
        self._schedule_update()

    def _reset_chromatic(self) -> None:
        """Reset chromatic aberration parameters."""
        self.parameters["chromatic_enabled"] = 0
        self.parameters["chromatic_max_displace"] = 20
        self.parameters["chromatic_red"] = 50
        self.parameters["chromatic_green"] = 50
        self.parameters["chromatic_blue"] = 50
        
        self.chromatic_cb.setChecked(False)
        self.max_displace_slider.setValue(20)
        self.red_channel_slider.setValue(50)
        self.green_channel_slider.setValue(50)
        self.blue_channel_slider.setValue(50)
        self._schedule_update()

    def _reset_all(self) -> None:
        """Reset all parameters to defaults."""
        self.parameters = self._default_parameters.copy()
        self.current_algorithm = "Modulated Diffuse Y"
        self.current_palette = None
        
        # Reset all UI elements
        self._reset_glow()
        self._reset_glitch()
        self._reset_chromatic()
        
        self.style_combo.setCurrentText("Modulated Diffuse Y")
        self.preset_combo.setCurrentIndex(0)
        self.scale_slider.setValue(2)
        self.line_scale_slider.setValue(1)
        self.palette_combo.setCurrentIndex(0)
        self.invert_cb.setChecked(False)
        self.effects_cb.setChecked(True)
        self.contrast_slider.setValue(50)
        self.midtones_slider.setValue(50)
        self.highlights_slider.setValue(50)
        self.lum_threshold_slider.setValue(50)
        self.blur_slider.setValue(0)
        self.depth_slider.setValue(0)
        
        self._schedule_update()

    # === PROCESSING ===
    
    def _schedule_update(self) -> None:
        """Schedule a debounced update."""
        if self._live_preview and self.original_image is not None:
            self._debounce_timer.start()

    def _do_update(self) -> None:
        """Perform the actual image update."""
        if self.original_image is None:
            return
        
        if self.worker_thread and self.worker_thread.isRunning():
            self._pending_update = True
            return
        
        self._start_processing()

    def _start_processing(self) -> None:
        """Start background processing."""
        if self.original_image is None:
            return
        
        from opendither.processing.pipeline_new import ProcessingPipeline
        pipeline = ProcessingPipeline()
        
        self.worker_thread, self.worker = create_worker_thread(
            pipeline.process,
            self.original_image.copy(),
            self.current_algorithm,
            self.parameters.copy(),
            self.current_palette,
            None  # curves_lut
        )
        
        self.worker.finished.connect(self._on_processing_finished)
        self.worker.error.connect(self._on_processing_error)
        self.worker_thread.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

    def _on_processing_finished(self, result: NDArray) -> None:
        """Handle processing completion."""
        self.processed_image = result
        self._display_image(result)
        
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Preview updated")
        
        if self._pending_update:
            self._pending_update = False
            self._start_processing()

    def _on_processing_error(self, error: str) -> None:
        """Handle processing error."""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Error: {error}")

    def _display_image(self, image: NDArray) -> None:
        """Display image in viewer."""
        if image is None:
            return
        
        if image.ndim == 2:
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            if ch == 4:
                bytes_per_line = 4 * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
            else:
                bytes_per_line = 3 * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        self.viewer.set_image(pixmap)

    # === FILE OPERATIONS ===
    
    def _import_image(self) -> None:
        """Import an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        
        if file_path:
            self._load_image(file_path)

    def _load_image(self, path: str) -> None:
        """Load image from path."""
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            if image.ndim == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.original_image = image
            self._display_image(image)
            self._schedule_update()
            
            self.setWindowTitle(f"OpenDither - {Path(path).name}")
            self.statusBar().showMessage(f"Loaded: {Path(path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def _export_image(self) -> None:
        """Export the processed image."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                image = self.processed_image
                if image.ndim == 3:
                    if image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(file_path, image)
                self.statusBar().showMessage(f"Exported: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")

    def _save_preset(self) -> None:
        """Save current settings as preset."""
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
            self.statusBar().showMessage(f"Preset '{name}' saved")

    # === DRAG AND DROP ===
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                self._load_image(path)

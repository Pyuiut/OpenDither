"""Dither Boy-inspired dark theme for OpenDither."""

DITHER_BOY_THEME = """
/* ============ GLOBAL - Dark Theme ============ */
* {
    outline: none;
}

QMainWindow {
    background-color: #1a1a1e;
}

QWidget {
    background-color: transparent;
    color: #e0e0e0;
    font-family: "SF Pro Text", "Segoe UI", -apple-system, sans-serif;
    font-size: 11px;
}

/* ============ MENU BAR ============ */
QMenuBar {
    background-color: #1a1a1e;
    color: #a0a0a0;
    border-bottom: 1px solid #2a2a30;
    padding: 4px 8px;
    font-size: 12px;
}

QMenuBar::item {
    padding: 4px 10px;
    background: transparent;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #2a2a30;
    color: #ffffff;
}

QMenu {
    background-color: #1e1e22;
    border: 1px solid #2a2a30;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 16px;
    border-radius: 4px;
    margin: 2px;
}

QMenu::item:selected {
    background-color: #3a3a44;
}

/* ============ PANELS ============ */
QFrame#leftPanel, QFrame#rightPanel {
    background-color: #1a1a1e;
    border: none;
}

QFrame#leftPanel {
    border-right: 1px solid #2a2a30;
}

QFrame#rightPanel {
    border-left: 1px solid #2a2a30;
}

/* ============ SCROLL AREA ============ */
QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollBar:vertical {
    background-color: #1a1a1e;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #3a3a44;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4a4a54;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1a1a1e;
    height: 8px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background-color: #3a3a44;
    border-radius: 4px;
    min-width: 30px;
}

/* ============ SECTIONS ============ */
QFrame#section {
    background-color: transparent;
    margin-bottom: 8px;
}

QLabel#sectionTitle {
    color: #808088;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-bottom: 6px;
}

QLabel#sectionLabel {
    color: #a0a0a8;
    font-size: 11px;
    margin-top: 4px;
}

QLabel#helpLabel {
    color: #606068;
    font-size: 10px;
    font-style: italic;
}

/* ============ BUTTONS ============ */
QPushButton {
    background-color: #2a2a30;
    border: 1px solid #3a3a44;
    border-radius: 4px;
    padding: 6px 12px;
    color: #d0d0d8;
    font-weight: 500;
    font-size: 11px;
}

QPushButton:hover {
    background-color: #3a3a44;
    border-color: #4a4a54;
}

QPushButton:pressed {
    background-color: #1e1e22;
}

QPushButton:disabled {
    background-color: #1e1e22;
    color: #505058;
    border-color: #2a2a30;
}

/* ============ SLIDERS ============ */
QSlider::groove:horizontal {
    background-color: #2a2a30;
    height: 4px;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background-color: #5080c0;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    background-color: #60a0e0;
}

QSlider::sub-page:horizontal {
    background-color: #4070a0;
    border-radius: 2px;
}

/* ============ COMBO BOX ============ */
QComboBox {
    background-color: #2a2a30;
    border: 1px solid #3a3a44;
    border-radius: 4px;
    padding: 5px 10px;
    color: #d0d0d8;
    font-size: 11px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #4a4a54;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #808088;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e22;
    border: 1px solid #3a3a44;
    border-radius: 4px;
    selection-background-color: #3a3a44;
    padding: 4px;
}

/* ============ CHECK BOX ============ */
QCheckBox {
    spacing: 6px;
    color: #d0d0d8;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #3a3a44;
    background-color: #2a2a30;
}

QCheckBox::indicator:checked {
    background-color: #5080c0;
    border-color: #5080c0;
}

QCheckBox::indicator:hover {
    border-color: #5080c0;
}

/* ============ LABELS ============ */
QLabel {
    color: #b0b0b8;
    font-size: 11px;
}

/* ============ SEPARATORS ============ */
QFrame#separator {
    background-color: #2a2a30;
    max-height: 1px;
    margin: 8px 0;
}

QFrame[frameShape="4"] {
    background-color: #2a2a30;
    max-height: 1px;
}

/* ============ PROGRESS BAR ============ */
QProgressBar {
    background-color: #2a2a30;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #5080c0;
    border-radius: 3px;
}

/* ============ STATUS BAR ============ */
QStatusBar {
    background-color: #1a1a1e;
    border-top: 1px solid #2a2a30;
    color: #808088;
    font-size: 11px;
}

/* ============ TOOLTIPS ============ */
QToolTip {
    background-color: #2a2a30;
    border: 1px solid #3a3a44;
    border-radius: 4px;
    color: #d0d0d8;
    padding: 4px 8px;
    font-size: 11px;
}

/* ============ IMAGE VIEWER ============ */
QGraphicsView {
    background-color: #0e0e10;
    border: none;
}
"""

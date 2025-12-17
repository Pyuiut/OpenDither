"""Ultra-modern minimal dark theme for OpenDither."""

DARK_THEME = """
/* ============ GLOBAL - Zinc Dark Theme ============ */
* {
    outline: none;
}

QMainWindow {
    background-color: #09090b;
}

QWidget {
    background-color: transparent;
    color: #fafafa;
    font-family: "Inter", "Segoe UI", -apple-system, sans-serif;
    font-size: 12px;
}

/* ============ MENU BAR ============ */
QMenuBar {
    background-color: #09090b;
    color: #a1a1aa;
    border-bottom: 1px solid #27272a;
    padding: 8px 12px;
    font-size: 13px;
}

QMenuBar::item {
    padding: 6px 12px;
    background: transparent;
    border-radius: 6px;
}

QMenuBar::item:selected {
    background-color: #27272a;
    color: #fafafa;
}

QMenu {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 6px;
}

QMenu::item {
    padding: 10px 20px;
    border-radius: 8px;
    margin: 2px;
}

QMenu::item:selected {
    background-color: #27272a;
}

QMenu::separator {
    height: 1px;
    background-color: #27272a;
    margin: 6px 12px;
}

/* ============ BUTTONS ============ */
QPushButton {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    padding: 10px 16px;
    color: #fafafa;
    font-weight: 500;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #3f3f46;
    border-color: #52525b;
}

QPushButton:pressed {
    background-color: #18181b;
}

QPushButton:disabled {
    background-color: #18181b;
    color: #52525b;
    border-color: #27272a;
}

QPushButton#primaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
    border: none;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #818cf8, stop:1 #a78bfa);
}

QPushButton#secondaryButton {
    background-color: transparent;
    border: 1px solid #3f3f46;
    color: #a1a1aa;
}

QPushButton#secondaryButton:hover {
    background-color: #27272a;
    color: #fafafa;
}

/* ============ COMBO BOX ============ */
QComboBox {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 10px 14px;
    color: #fafafa;
    font-size: 12px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #6366f1;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #71717a;
}

QComboBox QAbstractItemView {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 8px;
    selection-background-color: #27272a;
    padding: 4px;
}

/* ============ SLIDERS ============ */
QSlider::groove:horizontal {
    height: 4px;
    background-color: #27272a;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
    border-radius: 8px;
    border: none;
}

QSlider::handle:horizontal:hover {
    width: 18px;
    height: 18px;
    margin: -7px 0;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
    border-radius: 2px;
}

/* ============ CHECKBOX ============ */
QCheckBox {
    spacing: 10px;
    color: #a1a1aa;
    font-size: 12px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #3f3f46;
    border-radius: 6px;
    background-color: #18181b;
}

QCheckBox::indicator:hover {
    border-color: #6366f1;
}

QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
    border: none;
}

/* ============ LABELS ============ */
QLabel {
    color: #71717a;
    background: transparent;
}

QLabel#cardTitle {
    color: #fafafa;
    font-size: 13px;
    font-weight: 600;
}

QLabel#valueLabel {
    color: #fafafa;
    font-weight: 600;
    font-size: 12px;
    min-width: 32px;
}

QLabel#hintLabel {
    color: #52525b;
    font-size: 11px;
}

/* ============ CONTROL PANEL ============ */
QFrame#controlPanel {
    background-color: #050509;
    border-left: 1px solid #1f1f28;
}

QFrame#topBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111324, stop:1 #111116);
    border-bottom: 1px solid #1f1f28;
}

QFrame#card {
    background: rgba(20, 20, 28, 0.9);
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 18px;
    padding: 18px;
}

QFrame#card:hover {
    border-color: rgba(99, 102, 241, 0.25);
    background: rgba(25, 25, 34, 0.95);
}

QFrame#navWrapper {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #07070d, stop:1 #0e0e18);
    border-right: 1px solid #1f1f28;
}

QLabel#navTitle {
    color: #9ca3af;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

QWidget#navButtons {
    background: transparent;
}

QPushButton#navButton {
    text-align: left;
    padding: 12px 14px;
    border-radius: 12px;
    border: 1px solid transparent;
    background: rgba(255, 255, 255, 0.02);
    color: #a5a6ff;
    font-weight: 500;
}

QPushButton#navButton:hover {
    background: rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.35);
    color: #f8fafc;
}

QPushButton#navButton:checked {
    background: rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.8);
    color: #ffffff;
}

QFrame#navIndicator {
    background: qradialgradient(
        cx:0.5, cy:0.5, radius:1,
        stop:0 rgba(99, 102, 241, 0.45),
        stop:1 rgba(99, 102, 241, 0.05)
    );
    border-radius: 14px;
}

QStackedWidget#sectionStack {
    background: #050509;
}

/* ============ ACCENT BUTTON ============ */
QPushButton#accentButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
    border: none;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 16px;
}

QPushButton#accentButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #818cf8, stop:1 #a78bfa);
}

QPushButton#accentButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
}

/* ============ SCROLL AREA ============ */
QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background-color: transparent;
    width: 8px;
    margin: 4px 2px;
}

QScrollBar::handle:vertical {
    background-color: #3f3f46;
    border-radius: 4px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background-color: #52525b;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
}

/* ============ STATUS BAR ============ */
QStatusBar {
    background-color: #09090b;
    color: #52525b;
    border-top: 1px solid #27272a;
    padding: 6px 16px;
    font-size: 11px;
}

QStatusBar::item {
    border: none;
}

/* ============ PROGRESS BAR ============ */
QProgressBar {
    background-color: #27272a;
    border: none;
    border-radius: 4px;
    height: 4px;
    text-align: center;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
    border-radius: 4px;
}

/* ============ GRAPHICS VIEW ============ */
QGraphicsView {
    background-color: #09090b;
    border: none;
}

/* ============ GROUP BOX ============ */
QGroupBox {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 12px;
    margin-top: 8px;
    padding: 16px;
    padding-top: 32px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    top: 8px;
    color: #71717a;
    font-size: 11px;
    letter-spacing: 1px;
}

/* ============ TOOLTIPS ============ */
QToolTip {
    background-color: #27272a;
    color: #fafafa;
    border: 1px solid #3f3f46;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 11px;
}

/* ============ DIALOGS ============ */
QMessageBox, QInputDialog {
    background-color: #18181b;
}

QMessageBox QLabel {
    color: #fafafa;
    font-size: 13px;
}

QLineEdit {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 10px 14px;
    color: #fafafa;
    font-size: 12px;
}

QLineEdit:focus {
    border-color: #6366f1;
}
"""

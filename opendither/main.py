"""Application entry point."""

import sys
from PyQt6.QtWidgets import QApplication

from opendither.ui import MainWindow


def main() -> int:
    """Initialize and run the application."""
    # High DPI support
    app = QApplication(sys.argv)
    app.setApplicationName("OpenDither")
    app.setOrganizationName("OpenDither")
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

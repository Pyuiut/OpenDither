#!/usr/bin/env python3
"""
OpenDither - Advanced image and video dithering software.
"""

import sys
import os
from opendither.main import main

if __name__ == "__main__":
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Set the application name for QSettings
    from PyQt6.QtCore import QCoreApplication, QSettings
    QCoreApplication.setApplicationName("OpenDither")
    QCoreApplication.setOrganizationName("OpenDither")
    
    # Run the application
    main()

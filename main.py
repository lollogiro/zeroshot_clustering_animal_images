"""
Zero-Shot Clustering Pipeline for Animal Images

Entry point for the GUI application.

Usage:
    python main.py          # Launch GUI
    python main.py --cli    # Launch CLI version
"""

import sys


def main():
    """Launch the GUI or CLI based on arguments."""
    if "--cli" in sys.argv:
        from ui_pipeline import run_ui_pipeline
        run_ui_pipeline()
    else:
        from gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple Web Interface Launcher (Clean Output)
===========================================

Launches the OMR web interface with minimal console output.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the web interface quietly."""
    # Set environment variables for clean output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Interface path
    interface_path = Path(__file__).parent / "web_interface.py"
    
    # Detect Python executable
    python_paths = [
        Path(__file__).parent / ".venv" / "Scripts" / "python.exe",  # Windows venv
        Path(__file__).parent / ".venv" / "bin" / "python",         # Unix venv
        Path(sys.executable),                                        # Current Python
    ]
    
    python_path = None
    for path in python_paths:
        if path.exists():
            python_path = str(path)
            break
    
    if not python_path:
        python_path = "python"
    
    print("🎼 Launching OMR Web Interface...")
    print("🌐 Opening at: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop")
    print()
    
    # Launch Streamlit
    cmd = [python_path, "-m", "streamlit", "run", str(interface_path), "--server.headless", "true"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
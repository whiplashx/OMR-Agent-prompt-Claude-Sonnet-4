#!/usr/bin/env python3
"""
OMR Web Interface Launcher
=========================

Launch the interactive web interface for the OMR Pipeline.
This script starts the Streamlit web application for manual correction and review.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def check_omr_components():
    """Check if OMR components are available."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from omr_pipeline import OMRPipeline
        print("✅ OMR Pipeline components available")
        return True
    except Exception as e:
        print(f"⚠️ OMR components issue: {e}")
        return False

def launch_web_interface():
    """Launch the Streamlit web interface."""
    print("🚀 Launching OMR Web Interface...")
    
    # Path to the enhanced interface file
    interface_path = Path(__file__).parent / "web_interface.py"
    
    if not interface_path.exists():
        print(f"❌ Interface file not found: {interface_path}")
        return False
    
    try:
        # Detect the best Python executable to use
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
            python_path = "python"  # Fallback to system Python
        
        print(f"🐍 Using Python: {python_path}")
        
        # Build Streamlit command
        cmd = [python_path, "-m", "streamlit", "run", str(interface_path)]
        
        print(f"🔄 Running command: {' '.join(cmd)}")
        print("📱 Opening web browser...")
        print("🌐 Interface will be available at: http://localhost:8501")
        print("⏹️ Press Ctrl+C to stop the server")
        
        # Launch Streamlit
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch interface: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False

def main():
    """Main launcher function."""
    print("🎼 OMR Pipeline Web Interface Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_streamlit():
        print("💡 Install Streamlit: pip install streamlit")
        return 1
    
    if not check_omr_components():
        print("💡 Make sure you're in the OMR directory")
        return 1
    
    print("\n🎯 Web Interface Features:")
    print("   📸 Upload sheet music images")
    print("   🔍 Review detected symbols")
    print("   ✏️ Manual correction tools")
    print("   📊 Confidence analysis")
    print("   💾 Export corrected MusicXML")
    print("   🎵 Interactive visualization")
    
    # Launch the interface
    print("\n" + "=" * 50)
    if launch_web_interface():
        print("✅ Web interface launched successfully!")
        return 0
    else:
        print("❌ Failed to launch web interface")
        return 1

if __name__ == "__main__":
    sys.exit(main())
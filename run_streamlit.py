#!/usr/bin/env python3
"""
Streamlit deployment script for EdPrep AI
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    print("🚀 Starting EdPrep AI Streamlit App...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Run the app
    print("🌐 Launching Streamlit app...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()

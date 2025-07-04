#!/usr/bin/env python3
"""
FilRelex ML Setup and Test Script
This script will help you get the ML component working.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "dgl", "-f", "https://data.dgl.ai/wheels/repo.html"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "pandas", "numpy", "scikit-learn", "matplotlib", 
                             "sqlalchemy", "psycopg2-binary", "tqdm"])
        print("✅ Dependencies installed")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_setup():
    """Test the ML setup."""
    print("🔍 Testing setup...")
    try:
        import torch
        import dgl
        import pandas as pd
        import numpy as np
        from sqlalchemy import create_engine, text
        import json
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ DGL: {dgl.__version__}")
        print(f"✅ All imports successful")
        
        # Test database connection
        with open("my_db_config.json", "r") as f:
            config = json.load(f)["database"]
        
        engine = create_engine(
            f"postgresql://{config[\"user\"]}:{config[\"password\"]}@"
            f"{config[\"host\"]}:{config[\"port\"]}/{config[\"database\"]}"
            f"?sslmode={config[\"ssl_mode\"]}"
        )
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM words")).fetchone()
            print(f"✅ Database: {result[0]:,} words")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FilRelex ML Setup")
    print("=" * 30)
    
    if "--install" in sys.argv:
        install_dependencies()
    
    test_setup()


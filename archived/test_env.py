#!/usr/bin/env python3
"""Test environment variable loading."""

import os
from pathlib import Path

# Try to load .env file explicitly
try:
    from dotenv import load_dotenv
    
    # Load .env file from current directory
    env_path = Path('.env')
    if env_path.exists():
        print(f"Found .env file at: {env_path.absolute()}")
        load_dotenv(env_path)
        print("✅ .env file loaded")
    else:
        print(f"❌ .env file not found at: {env_path.absolute()}")
        
except ImportError:
    print("❌ python-dotenv not installed")

# Check environment variables
print(f"\nEnvironment variables:")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'NOT_SET')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT_SET')[:10]}..." if os.getenv('OPENAI_API_KEY') else "OPENAI_API_KEY: NOT_SET")
print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'NOT_SET')}")

# Check working directory
print(f"\nWorking directory: {os.getcwd()}")
print(f"Files in directory: {list(Path('.').glob('*'))}")

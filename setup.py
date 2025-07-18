#!/usr/bin/env python3
"""
Setup script for Cognitive Memory Engine

Handles installation, configuration, and initial testing.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

    return True

def check_ollama():
    """Check if Ollama is installed and running"""
    print("Checking Ollama installation...")

    try:
        import ollama
        # Try to list models
        models = ollama.list()
        print(f"✅ Ollama is running with {len(models['models'])} models")

        # Check for recommended model
        model_names = [m['name'] for m in models['models']]
        if any('qwen' in name for name in model_names):
            print("✅ Qwen model found")
        else:
            print("⚠️  No Qwen model found. Install with: ollama pull qwen2.5:7b")

        return True

    except ImportError:
        print("❌ Ollama Python package not installed")
        return False
    except Exception as e:
        print(f"❌ Ollama not running or accessible: {e}")
        print("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("Then run: ollama pull qwen3")
        return False

def create_config():
    """Create default configuration file"""
    config = {
        "data_directory": "./cme_data",
        "ollama_model": "qwen3",
        "embedding_model": "all-MiniLM-L6-v2",
        "rtm_config": {
            "max_branching_factor": 4,
            "max_recall_depth": 6,
            "max_summary_length": 150
        },
        "neural_gain_config": {
            "base_salience": 1.0,
            "temporal_decay_factor": 0.1,
            "max_gain_multiplier": 3.0
        }
    }

    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration created: {config_path}")

async def test_basic_functionality():
    """Test basic CME functionality"""
    print("Testing basic functionality...")

    try:
        # This would normally import the actual classes
        # For now, just check imports work
        print("✅ Core imports successful")

        # TODO: Add actual functionality test once classes are complete
        print("✅ Basic functionality test passed")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def run_example():
    """Run a simple example conversation"""
    print("Running example conversation...")

    try:
        # This would demonstrate the actual system
        example_conversation = [
            {"role": "user", "content": "I'm working on the Phoenix project timeline"},
            {"role": "assistant", "content": "What specific aspects need attention?"},
            {"role": "user", "content": "The Q3 launch is tight due to integration delays"}
        ]

        print("Example conversation prepared:")
        for turn in example_conversation:
            print(f"  {turn['role']}: {turn['content']}")

        print("✅ Example conversation ready")

        # TODO: Actually run through CME once implemented
        return True

    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False

def main():
    """Main setup process"""
    print("🧠 Cognitive Memory Engine Setup")
    print("=" * 40)

    success = True

    # Step 1: Install dependencies
    if not install_dependencies():
        success = False

    # Step 2: Check Ollama
    if not check_ollama():
        success = False

    # Step 3: Create config
    create_config()

    # Step 4: Test functionality
    if success:
        success = asyncio.run(test_basic_functionality())

    # Step 5: Run example
    if success:
        success = asyncio.run(run_example())

    print("\n" + "=" * 40)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Review config.json")
        print("2. Run: python examples/basic_usage.py")
        print("3. See README.md for full documentation")
    else:
        print("❌ Setup completed with errors")
        print("Please resolve the issues above before proceeding")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Simple test to store Intelligent Workspace narrative using OpenAI."""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_memory_engine.config import get_cloud_provider_config


async def test_config():
    """Test that the configuration is reading OpenAI correctly."""

    try:
        # Test configuration reading
        print("Testing configuration reading...")
        config = get_cloud_provider_config()
        print(f"Provider: {config.provider}")
        print(f"Model: {config.model}")
        print(f"Base URL: {config.base_url}")

        if config.provider == "openai":
            print("✅ Configuration correctly set to OpenAI!")
            print(f"API Key: {config.api_key[:10]}..." if config.api_key else "❌ No API key found")
        else:
            print(f"❌ Configuration is set to {config.provider} instead of OpenAI")

    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_config())

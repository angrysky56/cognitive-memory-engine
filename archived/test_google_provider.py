#!/usr/bin/env python3
"""
Test script for the updated GoogleProvider implementation.
"""
import asyncio
import os

from src.cognitive_memory_engine.llm_providers import GoogleProvider


class TestConfig:
    """Simple config class for testing."""
    def __init__(self):
        self.provider = "google"
        self.api_key = os.getenv("GOOGLE_API_KEY", "test-key")
        self.model = "gemini-2.0-flash-001"
        self.max_tokens = 1000


async def test_google_provider():
    """Test the GoogleProvider implementation."""
    config = TestConfig()

    try:
        # Create provider
        provider = GoogleProvider(config)

        # Test validation
        print("Testing validation...")
        provider.validate()
        print("✓ Validation passed")

        # Test list_models
        print("Testing list_models...")
        models = await provider.list_models()
        print(f"✓ Available models: {models}")

        # Test generate (if API key is available)
        if config.api_key != "test-key":
            print("Testing generate...")
            response = await provider.generate("Hello, how are you?", temperature=0.5)
            print(f"✓ Generated response: {response[:100]}...")
        else:
            print("⚠ Skipping generate test - no API key provided")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("To install the required package, run: pip install google-genai")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Testing GoogleProvider with new unified SDK...")
    asyncio.run(test_google_provider())

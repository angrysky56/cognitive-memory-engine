#!/usr/bin/env python3
"""
List available models from all configured LLM providers.

Usage:
    python list_models.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognitive_memory_engine.config import get_cloud_provider_config
from cognitive_memory_engine.llm_providers import LLMProviderFactory, list_all_available_models


async def main():
    """List all available models from configured providers."""
    print("🤖 Cognitive Memory Engine - Available Models")
    print("=" * 50)

    # Get current configuration
    config = get_cloud_provider_config()
    print(f"\nCurrent provider: {config.provider}")
    print(f"Current model: {config.model}")

    # Try to list available providers
    print(f"\nAvailable providers: {', '.join(LLMProviderFactory.list_providers())}")

    # List models from all providers
    print("\n📋 Checking available models from all providers...")
    print("(This may take a moment)\n")

    try:
        all_models = await list_all_available_models(config)

        if not all_models:
            print("❌ No models found. Make sure you have API keys configured.")
            print("\nTo configure providers, set these environment variables:")
            print("  - OPENAI_API_KEY for OpenAI")
            print("  - GOOGLE_API_KEY for Google Gemini")
            print("  - ANTHROPIC_API_KEY for Anthropic Claude")
            print("  - Or run Ollama locally for free models")
            return

        for provider, models in all_models.items():
            print(f"\n✅ {provider.upper()} Models ({len(models)} available):")
            for model in sorted(models)[:10]:  # Show first 10 models
                print(f"   - {model}")
            if len(models) > 10:
                print(f"   ... and {len(models) - 10} more")

    except Exception as e:
        print(f"\n❌ Error listing models: {e}")
        print("\nMake sure you have the required packages installed:")
        print("  pip install ollama openai google-generativeai anthropic")


if __name__ == "__main__":
    asyncio.run(main())

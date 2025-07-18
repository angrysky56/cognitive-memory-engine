"""
Configuration management for Cognitive Memory Engine

Loads configuration from environment variables with support for:
- Local Ollama models
- Cloud providers (OpenAI, Anthropic, Google)
- Provider-specific settings
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try parent directories
        for parent in Path('.').parents:
            env_path = parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                break
except ImportError:
    # dotenv not available, environment variables must be set manually
    pass
from pathlib import Path

from .types import NeuralGainConfig, RTMConfig, SystemConfig, TemporalScale


@dataclass
class CloudProviderConfig:
    """Configuration for cloud-based LLM providers"""
    provider: str = "ollama"  # ollama, openai, anthropic, google
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    max_tokens: int = 8164
    temperature: float = 0.7
    timeout: int = 60


def load_env_config() -> SystemConfig:
    """
    Load system configuration from environment variables.

    Supports both local Ollama and cloud providers.
    """
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    # Data storage
    data_directory = os.getenv("CME_DATA_DIR", "./cme_data")

    # Model configuration - check for cloud providers first
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "phi4-mini:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    elif provider == "google":
        model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-lite")
        base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1")
    else:
        # Fallback to Ollama
        model = os.getenv("OLLAMA_MODEL", "phi4-mini:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Embedding model
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # RTM Configuration
    rtm_config = RTMConfig(
        max_branching_factor=int(os.getenv("RTM_MAX_BRANCHING_FACTOR", "4")),
        max_recall_depth=int(os.getenv("RTM_MAX_RECALL_DEPTH", "6")),
        max_summary_length=int(os.getenv("RTM_MAX_SUMMARY_LENGTH", "150"))
    )

    # Neural Gain Configuration
    neural_gain_config = NeuralGainConfig(
        base_salience=float(os.getenv("NEURAL_GAIN_BASE_SALIENCE", "1.0")),
        temporal_decay_factor=float(os.getenv("NEURAL_GAIN_TEMPORAL_DECAY", "0.1")),
        max_gain_multiplier=float(os.getenv("NEURAL_GAIN_MAX_MULTIPLIER", "3.0"))
    )

    # Performance settings
    max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
    vector_similarity_threshold = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.7"))
    auto_archive_days = int(os.getenv("AUTO_ARCHIVE_DAYS", "30"))

    # Vector Manager Configuration
    vector_manager_type = os.getenv("VECTOR_MANAGER", "chroma").lower()

    # SVG Configuration
    svg_config = {
        "kernel_width": float(os.getenv("SVG_KERNEL_WIDTH", "1.0")),
        "max_out_degree": int(os.getenv("SVG_MAX_OUT_DEGREE", "32")),
        "epsilon": float(os.getenv("SVG_EPSILON", "0.1"))
    }

    return SystemConfig(
        data_directory=data_directory,
        llm_model=model,
        embedding_model=embedding_model,
        rtm_config=rtm_config,
        neural_gain_config=neural_gain_config,
        max_context_length=max_context_length,
        vector_similarity_threshold=vector_similarity_threshold,
        auto_archive_days=auto_archive_days,
        vector_manager=vector_manager_type,
        svg_config=svg_config
    )


def get_cloud_provider_config() -> CloudProviderConfig:
    """Get cloud provider specific configuration."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        return CloudProviderConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60"))
        )
    elif provider == "anthropic":
        return CloudProviderConfig(
            provider="anthropic",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("ANTHROPIC_TIMEOUT", "60"))
        )
    elif provider == "google":
        return CloudProviderConfig(
            provider="google",
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-001"),
            base_url=os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1"),
            max_tokens=int(os.getenv("GOOGLE_MAX_TOKENS", "100000")),
            temperature=float(os.getenv("GOOGLE_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("GOOGLE_TIMEOUT", "60"))
        )
    else:
        # Ollama configuration
        return CloudProviderConfig(
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "phi4-mini:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "64000")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "60"))
        )


def get_provider_specific_model_name() -> str:
    """Get the correct model name for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    elif provider == "google":
        return os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-001")
    else:
        return os.getenv("OLLAMA_MODEL", "phi4-mini:latest")

"""
Cognitive Memory Engine Exceptions

Custom exception classes for the CME system.
"""

class CMEError(Exception):
    """Base exception class for Cognitive Memory Engine errors."""

    def __init__(self, message: str, error_code: str | None = None):
        """
        Initialize CME error.

        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class CMEInitializationError(CMEError):
    """Raised when the engine fails to initialize properly."""
    pass


class CMEStorageError(CMEError):
    """Raised when storage operations fail."""
    pass


class CMEVectorError(CMEError):
    """Raised when vector operations fail."""
    pass


class CMETemporalError(CMEError):
    """Raised when temporal operations fail."""
    pass


class CMENarrativeError(CMEError):
    """Raised when narrative tree operations fail."""
    pass


class CMEResponseError(CMEError):
    """Raised when response generation fails."""
    pass


class CMEConfigurationError(CMEError):
    """Raised when configuration is invalid."""
    pass

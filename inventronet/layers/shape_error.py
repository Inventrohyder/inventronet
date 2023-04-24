class ShapeError(Exception):
    """An exception class for shape mismatch."""

    def __init__(self, message: str):
        """Initialize the exception with a message."""
        super().__init__(message)

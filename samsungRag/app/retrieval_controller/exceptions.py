"""Controller exceptions."""


class ControllerError(Exception):
    """Base error for retrieval-controller failures."""


class InvalidInputError(ControllerError):
    """Raised when input fails semantic validation beyond Pydantic."""

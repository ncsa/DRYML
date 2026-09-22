"""Errors shared by DRYML filesystem publication backends."""


class FilesystemError(OSError):
    """Base error for filesystem guarantees unavailable through native errors.

    Native failures such as :class:`FileExistsError`,
    :class:`FileNotFoundError`, :class:`PermissionError`, and cross-device
    :class:`OSError` remain unchanged. This type is reserved for adapter-level
    failures without a more useful native subclass.
    """


class FilesystemCapabilityError(FilesystemError):
    """Raised when the host cannot provide a required publication guarantee.

    The operation fails rather than falling back to copying, an existence check
    followed by a replacing rename, or a best-effort durability mode. The
    error does not imply rollback: a preceding publication step may already be
    visible.
    """

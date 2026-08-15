"""Application-level project-root path containment helpers."""

from pathlib import Path


def resolve_root(root: str | Path) -> Path:
    """Resolve and validate an existing project directory."""
    resolved = Path(root).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(str(resolved))
    return resolved


def eval_symlinks_or_ancestor(path: str | Path) -> Path:
    """Resolve symlinks through the closest existing path or symlink ancestor."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("path must be absolute")
    if candidate.exists():
        return candidate.resolve(strict=True)

    suffix: list[str] = []
    cursor = candidate
    while not cursor.exists():
        if cursor.is_symlink():
            return cursor.resolve(strict=False).joinpath(*reversed(suffix))
        parent = cursor.parent
        if parent == cursor:
            raise FileNotFoundError(str(candidate))
        suffix.append(cursor.name)
        cursor = parent
    return cursor.resolve(strict=True).joinpath(*reversed(suffix))


def sandbox_ok(root: Path, path: str) -> bool:
    """Return whether a possibly new path resolves beneath the project root."""
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    try:
        resolved_root = resolve_root(root)
        requested = Path(path).expanduser() if path else resolved_root
        if not requested.is_absolute():
            requested = resolved_root / requested
        resolved = eval_symlinks_or_ancestor(requested)
        return resolved.is_relative_to(resolved_root)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        return False


__all__ = ["eval_symlinks_or_ancestor", "resolve_root", "sandbox_ok"]

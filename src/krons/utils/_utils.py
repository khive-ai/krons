# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path as StdPath
from types import UnionType
from typing import Any, ParamSpec, TypeVar, Union, get_args, get_origin
from uuid import UUID, uuid4

from anyio import Path as AsyncPath

from krons.protocols import Observable

__all__ = (
    "create_path",
    "get_bins",
    "import_module",
    "is_import_installed",
    "now_utc",
    "to_uuid",
    "coerce_created_at",
    "load_type_from_string",
    "async_synchronized",
    "synchronized",
)

P = ParamSpec("P")
R = TypeVar("R")


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


async def create_path(
    directory: StdPath | AsyncPath | str,
    filename: str,
    extension: str | None = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
    timeout: float | None = None,
) -> AsyncPath:
    """Generate file path asynchronously with optional timeout.

    Args:
        directory: Base directory path
        filename: Target filename (may contain subdirectory with /)
        extension: File extension (if filename doesn't have one)
        timestamp: Add timestamp to filename
        dir_exist_ok: Allow existing directories
        file_exist_ok: Allow existing files
        time_prefix: Put timestamp before filename instead of after
        timestamp_format: Custom strftime format for timestamp
        random_hash_digits: Add random hash suffix (0 = disabled)
        timeout: Maximum time in seconds for async I/O operations (None = no timeout)

    Returns:
        AsyncPath to the created/validated file path

    Raises:
        ValueError: If filename contains backslash
        FileExistsError: If file exists and file_exist_ok is False
        TimeoutError: If timeout is exceeded
    """
    from .concurrency import move_on_after

    async def _impl() -> AsyncPath:
        nonlocal directory, filename

        if "/" in filename:
            sub_dir, filename = filename.split("/")[:-1], filename.split("/")[-1]
            directory = AsyncPath(directory) / "/".join(sub_dir)

        if "\\" in filename:
            raise ValueError("Filename cannot contain directory separators.")

        directory = AsyncPath(directory)
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
        else:
            name = filename
            ext = extension or ""
        ext = f".{ext.lstrip('.')}" if ext else ""

        if timestamp:
            ts_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
            name = f"{ts_str}_{name}" if time_prefix else f"{name}_{ts_str}"

        if random_hash_digits > 0:
            random_suffix = uuid4().hex[:random_hash_digits]
            name = f"{name}-{random_suffix}"

        full_path = directory / f"{name}{ext}"

        await full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)

        if await full_path.exists() and not file_exist_ok:
            raise FileExistsError(f"File {full_path} already exists.")

        return full_path

    if timeout is None:
        return await _impl()

    with move_on_after(timeout) as cancel_scope:
        result = await _impl()
    if cancel_scope.cancelled_caught:
        raise TimeoutError(f"create_path timed out after {timeout}s")
    return result


def get_bins(input_: list[str], upper: int) -> list[list[int]]:
    """Partition string list indices into bins by cumulative length.

    Args:
        input_: List of strings to partition.
        upper: Maximum cumulative length per bin.

    Returns:
        List of index lists, each bin's total string length < upper.
    """
    current = 0
    bins = []
    current_bin = []
    for idx, item in enumerate(input_):
        if current + len(item) < upper:
            current_bin.append(idx)
            current += len(item)
        else:
            bins.append(current_bin)
            current_bin = [idx]
            current = len(item)
    if current_bin:
        bins.append(current_bin)
    return bins


def import_module(
    package_name: str,
    module_name: str | None = None,
    import_name: str | list | None = None,
) -> Any:
    """Import module or attribute(s) by dotted path.

    Args:
        package_name: Base package name (e.g., 'kron').
        module_name: Optional submodule (e.g., 'utils').
        import_name: Attribute(s) to import from module.

    Returns:
        Module object, single attribute, or list of attributes.

    Raises:
        ImportError: If module or attribute not found.
    """
    try:
        full_import_path = f"{package_name}.{module_name}" if module_name else package_name

        if import_name:
            import_name = [import_name] if not isinstance(import_name, list) else import_name
            a = __import__(
                full_import_path,
                fromlist=import_name,
            )
            if len(import_name) == 1:
                return getattr(a, import_name[0])
            return [getattr(a, name) for name in import_name]
        else:
            return __import__(full_import_path)

    except ImportError as e:
        raise ImportError(f"Failed to import module {full_import_path}: {e}") from e


def is_import_installed(package_name: str) -> bool:
    """Check if package is installed."""
    return importlib.util.find_spec(package_name) is not None


_TYPE_CACHE: dict[str, type] = {}

_DEFAULT_ALLOWED_PREFIXES: frozenset[str] = frozenset({"krons."})
_ALLOWED_MODULE_PREFIXES: set[str] = set(_DEFAULT_ALLOWED_PREFIXES)


def register_type_prefix(prefix: str) -> None:
    """Register module prefix for dynamic type loading allowlist.

    Security: Only register prefixes for modules you control.

    Args:
        prefix: Module prefix to allow (e.g., "myapp.models.").
                Must end with "." to prevent prefix attacks.

    Raises:
        ValueError: If prefix doesn't end with ".".
    """
    if not prefix.endswith("."):
        raise ValueError(f"Prefix must end with '.': {prefix}")
    _ALLOWED_MODULE_PREFIXES.add(prefix)


def load_type_from_string(type_str: str) -> type:
    """Load type from fully qualified path (e.g., 'krons.core.Node').

    Security: Only allowlisted module prefixes can be loaded.

    Args:
        type_str: Fully qualified type path.

    Returns:
        The loaded type class.

    Raises:
        ValueError: If path invalid, not allowlisted, or type not found.
    """
    if type_str in _TYPE_CACHE:
        return _TYPE_CACHE[type_str]

    if not isinstance(type_str, str):
        raise ValueError(f"Expected string, got {type(type_str)}")

    if "." not in type_str:
        raise ValueError(f"Invalid type path (no module): {type_str}")

    if not any(type_str.startswith(prefix) for prefix in _ALLOWED_MODULE_PREFIXES):
        raise ValueError(
            f"Module '{type_str}' not in allowed prefixes: {sorted(_ALLOWED_MODULE_PREFIXES)}"
        )

    try:
        module_path, class_name = type_str.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        if module is None:
            raise ImportError(f"Module '{module_path}' not found")

        type_class = getattr(module, class_name)
        if not isinstance(type_class, type):
            raise ValueError(f"'{type_str}' is not a type")

        _TYPE_CACHE[type_str] = type_class
        return type_class

    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load type '{type_str}': {e}") from e


def extract_types(item_type: Any) -> set[type]:
    """Extract concrete types from type annotations.

    Handles Union, list, set, and single types recursively.

    Args:
        item_type: Type annotation (Union[X, Y], list[type], set[type], or type).

    Returns:
        Set of concrete types extracted from the annotation.
    """

    def is_union(t):
        origin = get_origin(t)
        return origin is Union or isinstance(t, UnionType)

    extracted: set[type] = set()

    if isinstance(item_type, set):
        for t in item_type:
            if is_union(t):
                extracted.update(get_args(t))
            else:
                extracted.add(t)
        return extracted

    if isinstance(item_type, list):
        for t in item_type:
            if is_union(t):
                extracted.update(get_args(t))
            else:
                extracted.add(t)
        return extracted

    if is_union(item_type):
        return set(get_args(item_type))

    return {item_type}


def to_uuid(value: Any) -> UUID:
    """Convert value to UUID instance.

    Args:
        value: UUID, UUID string, or Observable with .id attribute.

    Returns:
        UUID instance.

    Raises:
        ValueError: If value cannot be converted to UUID.
    """
    if isinstance(value, Observable):
        return value.id
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ValueError("Cannot get ID from item.")


def coerce_created_at(v) -> datetime:
    """Coerce value to UTC-aware datetime.

    Args:
        v: datetime, Unix timestamp (int/float), or ISO string.

    Returns:
        UTC-aware datetime instance.

    Raises:
        ValueError: If value cannot be parsed as datetime.
    """
    if isinstance(v, datetime):
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v

    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=UTC)

    if isinstance(v, str):
        with contextlib.suppress(ValueError):
            return datetime.fromtimestamp(float(v), tz=UTC)
        with contextlib.suppress(ValueError):
            return datetime.fromisoformat(v)
        raise ValueError(f"String '{v}' is neither timestamp nor ISO format")

    raise ValueError(f"Expected datetime/timestamp/string, got {type(v).__name__}")


def synchronized(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for thread-safe method execution.

    Requires decorated method's instance to have self._lock (threading.Lock).
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        with self._lock:
            return func(*args, **kwargs)

    return wrapper


def async_synchronized(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorator for async-safe method execution.

    Requires decorated method's instance to have self._async_lock (anyio.Lock).
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        async with self._async_lock:  # type: ignore[attr-defined]
            return await func(*args, **kwargs)

    return wrapper

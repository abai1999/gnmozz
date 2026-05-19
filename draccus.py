from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

T = TypeVar("T")


class ChoiceRegistry:
    _choice_registry: dict[str, type] = {}

    @classmethod
    def register_subclass(cls, name: str, subcls: type) -> None:
        cls._choice_registry[str(name)] = subcls

    @classmethod
    def get_choice_class(cls, name: str) -> type:
        try:
            return cls._choice_registry[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown choice class: {name!r}") from exc


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "name") and hasattr(obj, "value") and obj.__class__.__module__ != "builtins":
        return obj.value
    return obj


def wrap(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def dump(obj: Any, fp: Any) -> None:
    yaml.safe_dump(_to_jsonable(obj), fp, sort_keys=False)


def encode(obj: Any) -> Any:
    return _to_jsonable(obj)

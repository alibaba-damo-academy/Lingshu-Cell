from __future__ import annotations

from dataclasses import MISSING as DATACLASS_MISSING, field, fields, is_dataclass
from functools import wraps
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from omegaconf import DictConfig, ListConfig, MISSING, OmegaConf


def choice_field(*choices: str, default: object = DATACLASS_MISSING):
    """Declare one string field with an explicit allowed-value list."""
    if not choices:
        raise ValueError("choices cannot be empty")
    if len(set(choices)) != len(choices):
        raise ValueError(f"choices contain duplicates: {choices!r}")

    kwargs: dict[str, Any] = {"metadata": {"choices": tuple(choices)}}
    if default is not DATACLASS_MISSING:
        if default is not MISSING and default not in choices:
            raise ValueError(f"default={default!r} must be one of {choices!r}")
        kwargs["default"] = default
    return field(**kwargs)


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_dataclass_type(tp: Any) -> bool:
    tp = _unwrap_optional(tp)
    return isinstance(tp, type) and is_dataclass(tp)


def _field_required(schema_field: Any) -> bool:
    return schema_field.default is DATACLASS_MISSING and schema_field.default_factory is DATACLASS_MISSING


def validate_structured_choices(cfg: Any, schema_cls: type, path: str = "") -> None:
    """Validate explicit `choice_field(...)` metadata on one structured config tree."""
    schema_cls = _unwrap_optional(schema_cls)
    if not (isinstance(schema_cls, type) and is_dataclass(schema_cls)):
        raise TypeError(f"schema_cls must be a dataclass type, got {schema_cls!r}")

    type_hints = get_type_hints(schema_cls)
    for schema_field in fields(schema_cls):
        key = schema_field.name
        field_path = f"{path}.{key}" if path else key
        field_type = _unwrap_optional(type_hints.get(key, schema_field.type))

        if isinstance(cfg, DictConfig):
            missing = key not in cfg
            value = None if missing else cfg[key]
        else:
            missing = not hasattr(cfg, key)
            value = None if missing else getattr(cfg, key)

        if missing:
            if _field_required(schema_field):
                raise ValueError(f"Missing required field: {field_path}")
            continue

        if isinstance(cfg, DictConfig) and OmegaConf.is_missing(cfg, key):
            raise ValueError(f"Missing mandatory value at: {field_path}")

        choices = schema_field.metadata.get("choices")
        if choices is not None:
            if not isinstance(value, str):
                raise TypeError(
                    f"Field {field_path} should be str for choices validation, "
                    f"got {type(value).__name__}: {value!r}"
                )
            if value not in choices:
                raise ValueError(
                    f"Invalid value at {field_path}: {value!r}; expected one of {choices!r}"
                )

        if _is_dataclass_type(field_type):
            validate_structured_choices(value, field_type, field_path)
            continue

        origin = get_origin(field_type)
        if origin in (list, tuple):
            element_types = get_args(field_type)
            element_type = _unwrap_optional(element_types[0] if element_types else Any)
            if _is_dataclass_type(element_type):
                if not isinstance(value, (list, tuple, ListConfig)):
                    raise TypeError(
                        f"Field {field_path} should be a sequence, got {type(value).__name__}"
                    )
                for index, item in enumerate(value):
                    validate_structured_choices(item, element_type, f"{field_path}[{index}]")


def validated_main(schema_cls: type):
    """Wrap one Hydra main function with post-compose structured choice validation."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(cfg: DictConfig):
            validate_structured_choices(cfg, schema_cls)
            return fn(cfg)

        return wrapper

    return decorator

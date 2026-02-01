import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def map_positional_args(
    arguments: dict[str, Any], param_names: list[str]
) -> dict[str, Any]:
    """Map positional arguments (_pos_0, _pos_1, ...) to actual parameter names."""
    mapped = {}
    pos_count = 0

    for key, value in arguments.items():
        if key.startswith("_pos_"):
            if pos_count >= len(param_names):
                raise ValueError(
                    f"Too many positional arguments (expected {len(param_names)})"
                )
            mapped[param_names[pos_count]] = value
            pos_count += 1
        else:
            # Keep keyword arguments as-is
            mapped[key] = value

    return mapped


def _get_nested_fields_from_annotation(annotation) -> set[str]:
    """Extract all field names from an annotation that may be a Pydantic model or Union."""
    origin = get_origin(annotation)

    # Handle Union types (typing.Union or types.UnionType)
    if origin is Union or isinstance(annotation, types.UnionType):
        union_members = get_args(annotation)
        all_fields = set()
        for member in union_members:
            if member is type(None):
                continue
            # Recursively check nested unions or models
            nested = _get_nested_fields_from_annotation(member)
            all_fields.update(nested)
        return all_fields

    # Handle direct Pydantic model
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return set(annotation.model_fields.keys())

    # Handle Pydantic model class (not instance)
    if hasattr(annotation, "model_fields"):
        return set(annotation.model_fields.keys())

    return set()


def nest_arguments_by_schema(arguments: dict[str, Any], schema_cls) -> dict[str, Any]:
    """Restructure flat arguments into nested format based on schema structure."""
    if not schema_cls or not hasattr(schema_cls, "model_fields"):
        return arguments

    # Get top-level field names
    top_level_fields = set(schema_cls.model_fields.keys())

    # Find fields that are nested objects (Pydantic models or unions)
    nested_field_mappings = {}
    for field_name, field_info in schema_cls.model_fields.items():
        annotation = field_info.annotation
        nested_fields = _get_nested_fields_from_annotation(annotation)
        if nested_fields:
            nested_field_mappings[field_name] = nested_fields

    # If no nested fields detected, return as-is
    if not nested_field_mappings:
        return arguments

    # Separate top-level args from nested args
    result: dict[str, Any] = {}
    nested_args: dict[str, dict[str, Any]] = {}

    for key, value in arguments.items():
        if key in top_level_fields:
            # This is a top-level field
            result[key] = value
        else:
            # Check if this belongs to a nested field
            for nested_field, nested_keys in nested_field_mappings.items():
                if key in nested_keys:
                    if nested_field not in nested_args:
                        nested_args[nested_field] = {}
                    nested_args[nested_field][key] = value
                    break
            else:
                # Unknown field - keep at top level (will fail validation later)
                result[key] = value

    # Add nested structures to result
    result.update(nested_args)

    return result

import textwrap

from pydantic import BaseModel

from ._typescript import typescript_schema

__all__ = (
    "format_model_schema",
    "format_schema_pretty",
    "format_clean_multiline_strings",
)


def format_model_schema(request_model: type[BaseModel]) -> str:
    model_schema = request_model.model_json_schema()
    schema_text = ""
    if defs := model_schema.get("$defs"):
        for def_name, def_schema in defs.items():
            if def_ts := typescript_schema(def_schema):
                schema_text += f"\n{def_name}:\n" + textwrap.indent(def_ts, "  ")
    return schema_text


def format_schema_pretty(schema: dict, indent: int = 0) -> str:
    """Format schema dict with unquoted Python type values."""
    lines = ["{"]
    items = list(schema.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(value, dict):
            nested = format_schema_pretty(value, indent + 4)
            lines.append(f'{" " * (indent + 4)}"{key}": {nested}{comma}')
        elif isinstance(value, list) and value:
            if isinstance(value[0], dict):
                nested = format_schema_pretty(value[0], indent + 4)
                lines.append(f'{" " * (indent + 4)}"{key}": [{nested}]{comma}')
            else:
                lines.append(f'{" " * (indent + 4)}"{key}": [{value[0]}]{comma}')
        else:
            lines.append(f'{" " * (indent + 4)}"{key}": {value}{comma}')
    lines.append(f"{' ' * indent}}}")
    return "\n".join(lines)


def format_clean_multiline_strings(data: dict) -> dict:
    """Clean multiline strings for YAML block scalars (| not |-)."""
    cleaned: dict[str, object] = {}
    for k, v in data.items():
        if isinstance(v, str) and "\n" in v:
            # Strip trailing whitespace from each line, ensure ends with newline for "|"
            lines = "\n".join(line.rstrip() for line in v.split("\n"))
            cleaned[k] = lines if lines.endswith("\n") else lines + "\n"
        elif isinstance(v, list):
            cleaned[k] = [
                (
                    _clean_multiline(item)
                    if isinstance(item, str) and "\n" in item
                    else item
                )
                for item in v
            ]
        elif isinstance(v, dict):
            cleaned[k] = format_clean_multiline_strings(v)
        else:
            cleaned[k] = v
    return cleaned


def _clean_multiline(s: str) -> str:
    """Clean a multiline string: strip line trailing whitespace, ensure final newline."""
    lines = "\n".join(line.rstrip() for line in s.split("\n"))
    return lines if lines.endswith("\n") else lines + "\n"

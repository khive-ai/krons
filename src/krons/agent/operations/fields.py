from typing import TYPE_CHECKING, Any
from krons.core.types import HashableModel
from pydantic import Field, field_validator, BaseModel
from krons.utils import to_dict, to_list, extract_json
import re


class ActionRequestModel(HashableModel):
    """Represents a tool/action request"""

    function: str = Field(
        description=(
            "Name of the function to call from the provided `tool_schemas`. "
            "If no `tool_schemas` exist, set to None or leave blank. "
            "Never invent new function names outside what's given."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dictionary of arguments for the chosen function. "
            "Use only argument names/types defined in `tool_schemas`. "
            "Never introduce extra argument names."
        ),
    )

    @field_validator("arguments", mode="before")
    def validate_arguments(cls, value: Any) -> dict[str, Any]:
        """
        Coerce arguments into a dictionary if possible, recursively.

        Raises:
            ValueError if the data can't be coerced.
        """
        if isinstance(value, dict):
            return value
        return to_dict(
            value,
            fuzzy_parse=True,
            recursive=True,
            recursive_python_only=False,
        )

    @classmethod
    def create(cls, content: str):
        """
        Attempt to parse a string (usually from a conversation or JSON) into
        one or more ActionRequestModel instances.

        If no valid structure is found, returns an empty list.
        """

        def parse_action_request(content: str | dict) -> list[dict]:
            json_blocks = []

            if isinstance(content, BaseModel):
                json_blocks = [content.model_dump()]

            elif isinstance(content, str):
                json_blocks = extract_json(content, fuzzy_parse=True)
                if not json_blocks:
                    pattern2 = r"```python\s*(.*?)\s*```"
                    _d = re.findall(pattern2, content, re.DOTALL)
                    json_blocks = [
                        extract_json(match, fuzzy_parse=True) for match in _d
                    ]
                    json_blocks = to_list(json_blocks, dropna=True)

                print(json_blocks)

            elif content and isinstance(content, dict):
                json_blocks = [content]

            if json_blocks and not isinstance(json_blocks, list):
                json_blocks = [json_blocks]

            out = []

            for i in json_blocks:
                j = {}
                if isinstance(i, dict):
                    if "function" in i and isinstance(i["function"], dict):
                        if "name" in i["function"]:
                            i["function"] = i["function"]["name"]
                    for k, v in i.items():
                        k = (
                            k.replace("action_", "")
                            .replace("recipient_", "")
                            .replace("s", "")
                        )
                        if k in ["name", "function", "recipient"]:
                            j["function"] = v
                        elif k in ["parameter", "argument", "arg", "param"]:
                            j["arguments"] = to_dict(
                                v,
                                str_type="json",
                                fuzzy_parse=True,
                                suppress=True,
                            )
                    if (
                        j
                        and all(key in j for key in ["function", "arguments"])
                        and j["arguments"]
                    ):
                        out.append(j)

            return out

        try:
            ctx = parse_action_request(content)
            if ctx:
                return [cls.model_validate(i) for i in ctx]
            return []
        except Exception:
            return []

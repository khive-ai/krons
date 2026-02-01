from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, JsonValue

from krons.core.types import MaybeUnset, Unset
from krons.protocols import Deserializable, implements
from krons.resource.backend import is_unset
from krons.utils.schemas import (
    breakdown_pydantic_annotation,
    format_clean_multiline_strings,
    format_model_schema,
    format_schema_pretty,
    is_pydantic_model,
    minimal_yaml,
)

from .common import CustomRenderer
from .role import Role, RoledContent


@implements(Deserializable)
@dataclass(slots=True)
class Instruction(RoledContent):
    role: ClassVar[Role] = Role.USER

    primary: MaybeUnset[str] = Unset
    context: MaybeUnset[list] = Unset
    request_model: MaybeUnset[type[BaseModel]] = Unset
    tool_schemas: MaybeUnset[list[str | dict]] = Unset
    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset
    structure_format: MaybeUnset[Literal["json", "custom"]] = Unset
    custom_renderer: MaybeUnset[Callable[[type[BaseModel]], str]] = Unset

    @classmethod
    def create(
        cls,
        primary: MaybeUnset[str] = Unset,
        context: MaybeUnset[JsonValue] = Unset,
        tool_schemas: MaybeUnset[list[str | dict]] = Unset,
        request_model: MaybeUnset[type[BaseModel]] = Unset,
        images: MaybeUnset[list[str]] = Unset,
        image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset,
        structure_format: MaybeUnset[Literal["json", "custom"]] = Unset,
        custom_renderer: MaybeUnset[Callable[[type[BaseModel]], str]] = Unset,
    ):
        if is_unset(primary) and is_unset(request_model):
            raise ValueError("Either 'primary' or 'request_model' must be provided.")

        if not is_unset(request_model) and not is_pydantic_model(request_model):
            raise ValueError(
                "'request_model' must be a subclass of pydantic BaseModel."
            )

        if images is not None:
            from krons.utils.validators import validate_image_url

            for url in images:
                validate_image_url(url)

        if not cls._is_sentinel(context):
            context = [context] if not isinstance(context, list) else context

        return cls(
            primary=primary,
            context=context,
            tool_schemas=tool_schemas,
            request_model=request_model,
            images=images,
            image_detail=image_detail,
            structure_format=structure_format,
            custom_renderer=custom_renderer,
        )

    def _format_text_content(
        self,
        structure_format: Literal["json", "custom"],
        custom_renderer: MaybeUnset[CustomRenderer],
    ) -> str:
        if structure_format == "custom" and not callable(custom_renderer):
            raise ValueError(
                "Custom renderer must be provided when structure_format is 'custom'."
            )

        task_data = {
            "Primary Instruction": self.primary,
            "Context": self.context,
            "Tools": self.tool_schemas,
        }
        text = _format_task(
            {k: v for k, v in task_data.items() if not self._is_sentinel(v)}
        )

        if not self._is_sentinel(self.request_model):
            model = self.request_model
            text += format_model_schema(model)

            if structure_format == "custom":
                text += custom_renderer(model)
            elif structure_format == "json" or is_unset(structure_format):
                text += _format_json_response_structure(model)

        return text.strip()

    def render(
        self, structure_format=Unset, custom_renderer=Unset
    ) -> str | list[dict[str, Any]]:
        structure_format = (
            self.structure_format if is_unset(structure_format) else structure_format
        )
        custom_renderer = (
            self.custom_renderer if is_unset(custom_renderer) else custom_renderer
        )
        text = self._format_text_content(structure_format, custom_renderer)
        return text if is_unset(self.images) else self._format_image_content(text)


def _format_json_response_structure(request_model: type[BaseModel]) -> str:
    """Format response structure with Python types (unquoted)."""
    schema = breakdown_pydantic_annotation(request_model)
    json_schema = "\n\n## ResponseFormat\n"
    json_schema += "```json\n"
    json_schema += format_schema_pretty(schema, indent=0)
    json_schema += "\n```\nMUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT. Return ONLY valid JSON without markdown code blocks.\n"
    return json_schema


def _format_task(task_data: dict) -> str:
    text = "## Task\n"
    text += minimal_yaml(format_clean_multiline_strings(task_data))
    return text

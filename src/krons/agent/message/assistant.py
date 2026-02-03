from dataclasses import dataclass
from typing import Any, ClassVar

from typing_extensions import Self

from krons.core.types import MaybeUnset, Unset
from krons.resource.backend import NormalizedResponse

from .role import Role, RoledContent


@dataclass(slots=True)
class Assistant(RoledContent):
    """Assistant text response."""

    role: ClassVar[Role] = Role.ASSISTANT
    response: MaybeUnset[Any] = Unset

    _buffered_response: Any = Unset

    @classmethod
    def create(cls, response_object: NormalizedResponse, /) -> Self:
        self = cls(response=response_object.data)
        self._buffered_response = response_object
        return self

    @property
    def raw_response(self) -> dict[str, Any] | None:
        if isinstance(self._buffered_response, NormalizedResponse):
            return self._buffered_response.raw_response
        return None

    def render(self, *_args, **_kwargs) -> str:
        return str(self.response) if not self.is_sentinel_field("response") else ""

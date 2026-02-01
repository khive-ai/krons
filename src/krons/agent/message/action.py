from dataclasses import dataclass
from typing import Any, ClassVar

from krons.core.types import MaybeUnset, Unset
from krons.protocols import Deserializable, implements
from krons.utils.schemas import minimal_yaml

from .role import Role, RoledContent


@implements(Deserializable)
@dataclass(slots=True)
class ActionRequest(RoledContent):
    """Action/function call request."""

    role: ClassVar[Role] = Role.ACTION

    function: MaybeUnset[str] = Unset
    arguments: MaybeUnset[dict[str, Any]] = Unset

    def render(self, *_args, **_kwargs) -> str:
        doc: dict[str, Any] = {}
        if not self._is_sentinel(self.function):
            doc["function"] = self.function
        doc["arguments"] = {} if self._is_sentinel(self.arguments) else self.arguments
        return minimal_yaml(doc)

    @classmethod
    def create(
        cls, function: str | None = None, arguments: dict[str, Any] | None = None
    ) -> "ActionRequest":
        return cls(
            function=Unset if function is None else function,
            arguments=Unset if arguments is None else arguments,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionRequest":
        return cls.create(
            function=data.get("function"), arguments=data.get("arguments")
        )


@implements(Deserializable)
@dataclass(slots=True)
class ActionResponse(RoledContent):
    """Function call response."""

    role: ClassVar[Role] = Role.ACTION

    request_id: MaybeUnset[str] = Unset
    result: MaybeUnset[Any] = Unset
    error: MaybeUnset[str] = Unset

    def render(self, *_args, **_kwargs) -> str:
        doc: dict[str, Any] = {"success": self.success}
        if not self._is_sentinel(self.request_id):
            doc["request_id"] = str(self.request_id)[:8]
        if self.success:
            if not self._is_sentinel(self.result):
                doc["result"] = self.result
        else:
            doc["error"] = self.error
        return minimal_yaml(doc)

    @property
    def success(self) -> bool:
        return self._is_sentinel(self.error)

    @classmethod
    def create(
        cls,
        request_id: str | None = None,
        result: Any = Unset,
        error: str | None = None,
    ) -> "ActionResponse":
        return cls(
            request_id=Unset if request_id is None else request_id,
            result=result,
            error=Unset if error is None else error,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResponse":
        return cls.create(
            request_id=data.get("request_id"),
            result=data.get("result"),
            error=data.get("error"),
        )

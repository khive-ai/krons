from krons.errors import ValidationError
from krons.core.types import Enum, MaybeUnset, Unset, is_sentinel
from krons.resource import Calling
from typing import Callable


class ReturnAs(Enum):
    TEXT = "text"
    RAW = "raw"
    RESPONSE = "response"
    MESSAGE = "message"
    CALLING = "calling"
    CUSTOM = "custom"


def handle_return(
    calling: Calling,
    return_as: ReturnAs,
    /,
    *,
    return_parser: MaybeUnset[Callable] = Unset,
):
    """
    if return calling, or custom, no further processing is done.
    No validation is performed on the calling object in these cases.

    In other cases, the calling object will be validated to be complete
    with a normalized response.
    """

    if return_as == ReturnAs.CALLING:
        return calling

    if return_as == ReturnAs.CUSTOM:
        if is_sentinel(return_parser, {"none", "empty"}) or not callable(return_parser):
            raise ValidationError(
                "return_parser must be provided as a callable when return_as is 'custom'"
            )
        return return_parser(calling)

    calling.assert_is_normalized()
    response = calling.response

    match return_as:
        case ReturnAs.TEXT:
            return response.data
        case ReturnAs.RAW:
            return response.raw_response
        case ReturnAs.MESSAGE:
            from krons.agent.message.assistant import parse_to_assistant_response

            return parse_to_assistant_response(response)
        case _:
            raise ValidationError(f"Unsupported return_as: {return_as.value}")

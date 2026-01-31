from __future__ import annotations

import types
from functools import reduce
from typing import Any, Union, get_args, get_origin


def resolve_annotation_to_base_types(annotation: Any) -> dict[str, Any]:
    def resolve_nullable_inner_type(_anno: Any) -> tuple[bool, Any]:
        origin = get_origin(_anno)

        if origin is type(None):
            return True, type(None)

        if origin in (type(int | str), types.UnionType) or origin is Union:
            args = get_args(_anno)
            non_none_args = [a for a in args if a is not type(None)]
            if len(args) != len(non_none_args):
                if len(non_none_args) == 1:
                    return True, non_none_args[0]
                if non_none_args:
                    return True, reduce(lambda a, b: a | b, non_none_args)
            return False, _anno

        return False, _anno

    def resolve_listable_element_type(_anno: Any) -> Any:
        origin = get_origin(_anno)

        if origin is list:
            args = get_args(_anno)
            if args:
                return True, args[0]
            return True, Any

        return False, _anno

    _null, _inner = resolve_nullable_inner_type(annotation)
    _list, _elem = resolve_listable_element_type(_inner)

    return {
        "base_type": _elem,
        "nullable": _null,
        "listable": _list,
    }

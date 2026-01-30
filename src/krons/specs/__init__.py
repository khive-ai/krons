# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from .adapters.factory import AdapterType, get_adapter
from .catalog import AuditSpecs, CommonSpecs, ContentSpecs
from .operable import Operable
from .phrase import CrudOperation, CrudPattern, Phrase, phrase
from .protocol import SpecAdapter
from .spec import CommonMeta, Spec

__all__ = (
    "AdapterType",
    "AuditSpecs",
    "CommonMeta",
    "CommonSpecs",
    "ContentSpecs",
    "CrudOperation",
    "CrudPattern",
    "Operable",
    "Phrase",
    "Spec",
    "SpecAdapter",
    "get_adapter",
    "phrase",
)

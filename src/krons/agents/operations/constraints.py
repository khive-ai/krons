# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validation constraints for agent operations.

These functions validate preconditions and resolve parameters
for generate, communicate, act, and react operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from krons.errors import ConfigurationError, ValidationError

if TYPE_CHECKING:
    from krons.core.specs import Operable
    from krons.resources.backend import Calling, NormalizedResponse
    from krons.session import Branch, Session

__all__ = (
    "capabilities_must_be_subset_of_branch",
    "capabilities_must_be_subset_of_operable",
    "genai_model_must_be_configured",
    "resolve_branch_exists_in_session",
    "resolve_genai_model_exists_in_session",
    "resolve_generate_params",
    "resolve_parse_params",
    "resolve_response_is_normalized",
    "resource_must_be_accessible_by_branch",
    "resource_must_exist_in_session",
    "response_must_be_completed",
)


def resolve_branch_exists_in_session(
    session: Session,
    branch: Branch | str,
) -> Branch:
    """Resolve branch reference to actual Branch object.

    Args:
        session: Session containing branches
        branch: Branch object or branch ID string

    Returns:
        Resolved Branch object

    Raises:
        ValidationError: If branch not found in session
    """
    if isinstance(branch, str):
        # Look up branch by ID in session
        resolved = session.branches.get(branch)
        if resolved is None:
            raise ValidationError(
                f"Branch '{branch}' not found in session",
                details={"session_id": str(session.id), "branch_id": branch},
            )
        return resolved
    return branch


def resolve_genai_model_exists_in_session(
    session: Session,
    model_name: str | None = None,
) -> Any:
    """Resolve and validate that a GenAI model exists in session.

    Args:
        session: Session containing services
        model_name: Optional model name, uses default if not provided

    Returns:
        iModel instance from session

    Raises:
        ConfigurationError: If model not found or not configured
    """
    name = model_name or getattr(session, "default_generate_model", None)
    if name is None:
        raise ConfigurationError(
            "No model specified and no default_generate_model configured",
            details={"session_id": str(session.id)},
        )

    if not hasattr(session, "registry") or session.registry is None:
        raise ConfigurationError(
            "Session has no service registry",
            details={"session_id": str(session.id)},
        )

    model = session.registry.get(name)
    if model is None:
        raise ConfigurationError(
            f"Model '{name}' not found in session registry",
            details={
                "session_id": str(session.id),
                "model_name": name,
                "available": list(session.registry.list_names()),
            },
        )
    return model


def resolve_response_is_normalized(calling: Calling) -> NormalizedResponse:
    """Extract normalized response from a Calling event.

    Args:
        calling: Completed Calling event

    Returns:
        NormalizedResponse from the calling

    Raises:
        ValidationError: If response is not available or not normalized
    """
    from krons.core.types import is_sentinel
    from krons.resources.backend import NormalizedResponse

    response = calling.response
    if is_sentinel(response):
        raise ValidationError(
            "Calling has no response",
            details={"calling_id": str(calling.id)},
        )

    if not isinstance(response, NormalizedResponse):
        raise ValidationError(
            f"Response is not normalized: {type(response).__name__}",
            details={"calling_id": str(calling.id)},
        )
    return response


def response_must_be_completed(calling: Calling) -> None:
    """Validate that a Calling event completed successfully.

    Args:
        calling: Calling event to check

    Raises:
        ValidationError: If calling failed or has errors
    """
    from krons.core import EventStatus

    if calling.execution.status == EventStatus.FAILED:
        raise ValidationError(
            f"Calling failed: {calling.execution.error}",
            details={
                "calling_id": str(calling.id),
                "status": calling.execution.status.value,
            },
        )

    response = calling.response
    from krons.core.types import is_sentinel

    if not is_sentinel(response) and response.status == "error":
        raise ValidationError(
            f"Response error: {response.error}",
            details={"calling_id": str(calling.id)},
        )


def resource_must_exist_in_session(session: Session, name: str) -> Any:
    """Validate that a resource exists in session registry.

    Args:
        session: Session to check
        name: Resource name

    Returns:
        The resource from registry

    Raises:
        ValidationError: If resource not found
    """
    if not hasattr(session, "registry") or session.registry is None:
        raise ValidationError(
            "Session has no service registry",
            details={"session_id": str(session.id)},
        )

    resource = session.registry.get(name)
    if resource is None:
        raise ValidationError(
            f"Resource '{name}' not found in session",
            details={
                "session_id": str(session.id),
                "resource_name": name,
                "available": list(session.registry.list_names()),
            },
        )
    return resource


def resource_must_be_accessible_by_branch(branch: Branch, name: str) -> None:
    """Validate that a branch has access to a resource.

    Args:
        branch: Branch to check
        name: Resource name

    Raises:
        ValidationError: If branch doesn't have access
    """
    if hasattr(branch, "resources") and name not in branch.resources:
        raise ValidationError(
            f"Branch '{branch.id}' has no access to resource '{name}'",
            details={
                "branch_id": str(branch.id),
                "resource": name,
                "available": list(getattr(branch, "resources", [])),
            },
        )


def capabilities_must_be_subset_of_branch(
    branch: Branch,
    required: set[str],
) -> None:
    """Validate branch has all required capabilities.

    Args:
        branch: Branch to check
        required: Set of required capability names

    Raises:
        ValidationError: If branch missing capabilities
    """
    branch_caps = getattr(branch, "capabilities", set())
    missing = required - branch_caps
    if missing:
        raise ValidationError(
            f"Branch missing capabilities: {missing}",
            details={
                "branch_id": str(branch.id),
                "required": sorted(required),
                "available": sorted(branch_caps),
                "missing": sorted(missing),
            },
        )


def capabilities_must_be_subset_of_operable(
    operable: Operable,
    required: set[str],
) -> None:
    """Validate operable defines all required fields.

    Args:
        operable: Operable to check
        required: Set of required field names

    Raises:
        ValidationError: If operable missing fields
    """
    field_names = {s.name for s in operable.get_specs()}
    missing = required - field_names
    if missing:
        raise ValidationError(
            f"Operable missing fields: {missing}",
            details={
                "required": sorted(required),
                "available": sorted(field_names),
                "missing": sorted(missing),
            },
        )


def genai_model_must_be_configured(session: Session) -> None:
    """Validate session has a default GenAI model configured.

    Args:
        session: Session to check

    Raises:
        ConfigurationError: If no default model configured
    """
    if not getattr(session, "default_generate_model", None):
        raise ConfigurationError(
            "Session has no default_generate_model configured",
            details={"session_id": str(session.id)},
        )


def resolve_parse_params(
    params: Any,
    operable: Operable | None = None,
) -> dict[str, Any]:
    """Resolve and validate parse operation parameters.

    Args:
        params: ParseParams or dict
        operable: Optional operable for field validation

    Returns:
        Resolved parameters dict
    """
    if hasattr(params, "model_dump"):
        result = params.model_dump(exclude_none=True)
    elif isinstance(params, dict):
        result = dict(params)
    else:
        result = {"text": str(params)}

    if operable and "output_fields" in result:
        # Validate output fields exist in operable
        field_names = {s.name for s in operable.get_specs()}
        invalid = set(result["output_fields"]) - field_names
        if invalid:
            raise ValidationError(
                f"Output fields not in operable: {invalid}",
                details={"invalid": sorted(invalid), "available": sorted(field_names)},
            )

    return result


def resolve_generate_params(
    params: Any,
    session: Session,
    branch: Branch,
) -> dict[str, Any]:
    """Resolve and validate generate operation parameters.

    Args:
        params: GenerateParams or dict
        session: Session for model resolution
        branch: Branch for capability checks

    Returns:
        Resolved parameters dict
    """
    if hasattr(params, "model_dump"):
        result = params.model_dump(exclude_none=True)
    elif isinstance(params, dict):
        result = dict(params)
    else:
        raise ValidationError(
            f"Invalid params type: {type(params).__name__}",
            details={"expected": "GenerateParams or dict"},
        )

    # Resolve model if not specified
    if "model" not in result or result["model"] is None:
        result["model"] = getattr(session, "default_generate_model", None)

    return result

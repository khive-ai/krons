from krons.errors import AccessError, ConfigurationError, ExistsError, NotFoundError

__all__ = (
    "resource_must_exist",
    "resource_must_be_accessible",
    "capabilities_must_be_granted",
    "branch_name_must_be_unique",
    "genai_model_must_be_configured",
)


def resource_must_exist(session, name: str):
    """Validate resource exists in session. Raise NotFoundError if not."""
    if not session.services.has(name):
        raise NotFoundError(
            f"Service '{name}' not found in session services",
            details={"available": session.services.list_names()},
        )


def resource_must_be_accessible(branch, name: str) -> None:
    """Validate branch has resource access. Raise AccessError if not."""
    if name not in branch.resources:
        raise AccessError(
            f"Branch '{branch.name}' has no access to resource '{name}'",
            details={
                "branch": branch.name,
                "resource": name,
                "available": list(branch.resources),
            },
        )


def capabilities_must_be_granted(branch, capabilities: set[str]) -> None:
    """Validate branch has all capabilities. Raise AccessError listing missing."""
    if not capabilities.issubset(branch.capabilities):
        missing = capabilities - branch.capabilities
        raise AccessError(
            f"Branch '{branch.name}' missing capabilities: {missing}",
            details={
                "requested": sorted(capabilities),
                "available": sorted(branch.capabilities),
            },
        )


def branch_name_must_be_unique(session, name: str) -> None:
    try:
        session.communications.get_progression(name)
    except KeyError:
        raise ExistsError(f"Branch with name '{name}' already exists")


def genai_model_must_be_configured(session) -> None:
    """Validate session has a default GenAI model configured.

    Args:
        session: Session to check

    Raises:
        ConfigurationError: If no default model configured
    """
    if session.default_gen_model is None:
        raise ConfigurationError(
            "Session has no default_gen_model configured",
            details={"session_id": str(session.id)},
        )

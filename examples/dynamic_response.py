# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Dynamic LLM response model generation using Spec and Operable.

This example demonstrates how to create Pydantic models at runtime with
schema-driven structured outputs and validation using krons' Spec system.

Key concepts:
    - Spec: Framework-agnostic field specification
    - Operable: Ordered collection of Specs with adapter interface
    - Validators: Pydantic field_validator functions attached to Specs
    - compose_structure: Generate BaseModel from Specs at runtime
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from krons.core.specs import Operable, Spec


def create_response_model(
    fields: dict[str, type],
    validators: dict[str, Any] | None = None,
    model_name: str = "LLMResponse",
) -> type:
    """Build a Pydantic model dynamically from field definitions.

    Args:
        fields: Mapping of field names to Python types
        validators: Optional mapping of field names to validator functions.
            Validators should follow Pydantic's field_validator signature:
            ``def validator(cls, v) -> v``
        model_name: Name for the generated model class

    Returns:
        A dynamically created Pydantic BaseModel subclass

    Example:
        >>> Model = create_response_model(
        ...     {"answer": str, "confidence": float},
        ...     validators={"confidence": lambda cls, v: max(0, min(1, v))}
        ... )
        >>> instance = Model(answer="Yes", confidence=1.5)
        >>> instance.confidence  # Clamped to 1.0
        1.0
    """
    specs = []
    for name, field_type in fields.items():
        spec = Spec(field_type, name=name)
        if validators and name in validators:
            spec = spec.with_validator(validators[name])
        specs.append(spec)

    op = Operable(specs)
    return op.compose_structure(model_name)


def clamp_confidence(cls, v: float) -> float:
    """Clamp confidence score to [0, 1] range."""
    return max(0.0, min(1.0, v))


def non_empty_string(cls, v: str) -> str:
    """Ensure string is not empty after stripping whitespace."""
    v = v.strip()
    if not v:
        raise ValueError("Field cannot be empty or whitespace-only")
    return v


def main():
    """Demonstrate dynamic response model generation."""
    print("=" * 60)
    print("Dynamic LLM Response Model Generation with krons")
    print("=" * 60)

    # Define response schema with validators
    ResponseModel = create_response_model(
        fields={
            "reasoning": str,
            "answer": str,
            "confidence": float,
        },
        validators={
            "reasoning": non_empty_string,
            "answer": non_empty_string,
            "confidence": clamp_confidence,
        },
        model_name="StructuredResponse",
    )

    print(f"\nGenerated model: {ResponseModel.__name__}")
    print(f"Fields: {list(ResponseModel.model_fields.keys())}")

    # Example 1: Valid data
    print("\n--- Example 1: Valid data ---")
    valid_data = {
        "reasoning": "The sky appears blue due to Rayleigh scattering.",
        "answer": "Blue",
        "confidence": 0.95,
    }
    response = ResponseModel(**valid_data)
    print(f"Input:  {valid_data}")
    print(f"Output: reasoning={response.reasoning!r}")
    print(f"        answer={response.answer!r}")
    print(f"        confidence={response.confidence}")

    # Example 2: Confidence clamped to bounds
    print("\n--- Example 2: Confidence clamped ---")
    clamped_data = {
        "reasoning": "I am very certain about this.",
        "answer": "Yes",
        "confidence": 1.5,  # Will be clamped to 1.0
    }
    response = ResponseModel(**clamped_data)
    print(f"Input confidence:  {clamped_data['confidence']}")
    print(f"Output confidence: {response.confidence}")

    clamped_data_low = {
        "reasoning": "This is uncertain.",
        "answer": "Maybe",
        "confidence": -0.3,  # Will be clamped to 0.0
    }
    response = ResponseModel(**clamped_data_low)
    print(f"Input confidence:  {clamped_data_low['confidence']}")
    print(f"Output confidence: {response.confidence}")

    # Example 3: Validation error (empty string)
    print("\n--- Example 3: Validation error ---")
    invalid_data = {
        "reasoning": "   ",  # Empty after strip
        "answer": "Test",
        "confidence": 0.5,
    }
    try:
        ResponseModel(**invalid_data)
    except ValidationError as e:
        print(f"Input:  {invalid_data}")
        print(f"Error:  {e.errors()[0]['msg']}")

    # Example 4: Extending with optional fields
    print("\n--- Example 4: Model with optional fields ---")
    specs_with_optional = [
        Spec(str, name="query"),
        Spec(str, name="response"),
        Spec(list, name="sources").as_optional(),  # nullable + default=None
        Spec(dict, name="metadata").as_optional(),
    ]
    ExtendedModel = Operable(specs_with_optional).compose_structure("ExtendedResponse")

    extended = ExtendedModel(query="What is krons?", response="A composable framework")
    print(f"Fields: {list(ExtendedModel.model_fields.keys())}")
    print(f"sources (default): {extended.sources}")
    print(f"metadata (default): {extended.metadata}")

    # Example 5: Extract specs from existing model (round-trip)
    print("\n--- Example 5: Extract specs from model ---")
    extracted_op = Operable.from_structure(ResponseModel)
    print(f"Extracted field names: {extracted_op.allowed()}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

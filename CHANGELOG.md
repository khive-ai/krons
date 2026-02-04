# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-02-04

### Added

- `parse_forward_ref` canonical parser for ForwardRef annotations
- Support for `FK["Model"]` and `FK['Model']` string forward references
- Robust nullability detection: `Optional[X]`, `Union[X, None]`, `X | None`
- `meta_key` parameter in `ContentSpecs.get_specs()` for DB alias customization
- Duck typing (`_is_spec_like`) to avoid circular imports

### Changed

- DDL generation now uses config-driven audit columns (dict pattern)
- `_utils.py` uses shared `parse_forward_ref` instead of duplicate implementation
- Removed `include_audit_columns` parameter from `generate_ddl()` (now config-driven)

### Fixed

- ForwardRef parsing with `from __future__ import annotations` (PEP 563)
- Nested Union types in nullability detection (e.g., `Union[FK[User], None]`)

## [0.2.3] - 2026-02-04

### Added

- Multi-agent example patterns (code_review_panel, tech_debate)
- 10 comprehensive examples demonstrating framework capabilities

### Fixed

- Pile UUID access bug for Element retrieval

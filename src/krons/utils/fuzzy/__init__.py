from ._extract_json import extract_json
from ._fuzzy_json import fuzzy_json
from ._fuzzy_match import fuzzy_match_keys
from ._string_similarity import SimilarityAlgo, string_similarity
from ._to_dict import to_dict

# Alias for backward compatibility
fuzzy_validate_mapping = fuzzy_match_keys

__all__ = (
    "extract_json",
    "fuzzy_json",
    "fuzzy_match_keys",
    "fuzzy_validate_mapping",
    "string_similarity",
    "SimilarityAlgo",
    "to_dict",
)

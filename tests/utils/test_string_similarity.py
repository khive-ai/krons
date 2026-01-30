# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from krons.utils.fuzzy._string_similarity import SimilarityAlgo, string_similarity

# ============================================================================
# Test SimilarityAlgo static methods
# ============================================================================


def test_cosine_similarity_basic():
    """Test basic cosine similarity"""
    assert SimilarityAlgo.cosine_similarity("hello", "hello") == 1.0
    assert 0 < SimilarityAlgo.cosine_similarity("hello", "help") < 1.0


def test_cosine_similarity_empty():
    """Test cosine similarity with empty strings"""
    assert SimilarityAlgo.cosine_similarity("", "hello") == 0.0
    assert SimilarityAlgo.cosine_similarity("hello", "") == 0.0
    assert SimilarityAlgo.cosine_similarity("", "") == 0.0


def test_cosine_similarity_no_overlap():
    """Test cosine similarity with no character overlap"""
    assert SimilarityAlgo.cosine_similarity("abc", "def") == 0.0


def test_hamming_similarity():
    """Test hamming similarity"""
    assert SimilarityAlgo.hamming_similarity("hello", "hello") == 1.0
    assert SimilarityAlgo.hamming_similarity("hello", "hallo") == 0.8
    assert SimilarityAlgo.hamming_similarity("hello", "help") == 0.0  # Different lengths


def test_jaro_winkler_similarity():
    """Test jaro winkler similarity"""
    assert SimilarityAlgo.jaro_winkler_similarity("hello", "hello") == 1.0
    assert 0 < SimilarityAlgo.jaro_winkler_similarity("hello", "hallo") < 1.0


def test_jaro_winkler_invalid_scaling():
    """Test jaro winkler with invalid scaling"""
    with pytest.raises(ValueError, match="Scaling factor must be between"):
        SimilarityAlgo.jaro_winkler_similarity("hello", "world", scaling=0.3)


def test_sequence_matcher_similarity():
    """Test sequence matcher similarity"""
    assert SimilarityAlgo.sequence_matcher_similarity("hello", "hello") == 1.0
    assert 0 < SimilarityAlgo.sequence_matcher_similarity("hello", "hallo") < 1.0


# ============================================================================
# Test string_similarity function
# ============================================================================


def test_string_similarity_jaro_winkler():
    """Test string_similarity with jaro_winkler algorithm"""
    result = string_similarity("hello", ["hello", "help", "world"])
    assert "hello" in result


def test_string_similarity_threshold():
    """Test string_similarity with threshold"""
    result = string_similarity("hello", ["hello", "help", "world"], threshold=0.9)
    assert isinstance(result, list)


def test_string_similarity_most_similar():
    """Test return_most_similar option"""
    result = string_similarity("hello", ["hello", "help", "world"], return_most_similar=True)
    assert isinstance(result, str)


def test_string_similarity_invalid_threshold():
    """Test invalid threshold raises ValueError"""
    with pytest.raises(ValueError, match="threshold must be between"):
        string_similarity("hello", ["world"], threshold=1.5)


def test_string_similarity_empty_correct_words():
    """Test empty correct_words raises ValueError"""
    with pytest.raises(ValueError, match="correct_words must not be empty"):
        string_similarity("hello", [])


def test_string_similarity_case_sensitive():
    """Test case sensitive matching"""
    result = string_similarity("Hello", ["hello", "Hello"], case_sensitive=True)
    assert "Hello" in result


def test_string_similarity_hamming():
    """Test hamming algorithm"""
    result = string_similarity("hello", ["hello", "hallo"], algorithm=SimilarityAlgo.HAMMING)
    assert "hello" in result


def test_string_similarity_no_matches():
    """Test when no matches found"""
    result = string_similarity(
        "hello", ["xyz", "abc"], threshold=0.9, algorithm=SimilarityAlgo.LEVENSHTEIN
    )
    assert result is None


# ============================================================================
# Test SimilarityAlgo enum
# ============================================================================


def test_similarity_algo_enum_values():
    """Test SimilarityAlgo enum has correct values"""
    assert SimilarityAlgo.JARO_WINKLER.value == "jaro_winkler"
    assert SimilarityAlgo.LEVENSHTEIN.value == "levenshtein"
    assert SimilarityAlgo.SEQUENCE_MATCHER.value == "sequence_matcher"
    assert SimilarityAlgo.HAMMING.value == "hamming"
    assert SimilarityAlgo.COSINE.value == "cosine"


def test_string_similarity_with_enum():
    """Test string_similarity accepts SimilarityAlgo enum"""
    # Test with JARO_WINKLER enum
    result = string_similarity(
        "hello", ["hello", "help", "world"], algorithm=SimilarityAlgo.JARO_WINKLER
    )
    assert "hello" in result

    # Test with LEVENSHTEIN enum
    result = string_similarity(
        "hello", ["hello", "hallo"], algorithm=SimilarityAlgo.LEVENSHTEIN, threshold=0.8
    )
    assert isinstance(result, list)

    # Test with HAMMING enum
    result = string_similarity("hello", ["hello", "hallo"], algorithm=SimilarityAlgo.HAMMING)
    assert "hello" in result


def test_string_similarity_enum_backward_compatible():
    """Test that enum produce consistent results"""
    test_word = "color"
    correct_words = ["colour", "caller", "car"]

    result_enum = string_similarity(
        test_word,
        correct_words,
        algorithm=SimilarityAlgo.JARO_WINKLER,
        return_most_similar=True,
    )

    # Should return "colour" as most similar
    assert result_enum == "colour"


def test_string_similarity_hamming_different_lengths_skip():
    """Test hamming skips different length strings"""
    # When using hamming, strings of different lengths should be skipped
    # "hi" (len=2) vs "hello" (len=5) - should skip, but "hello" vs "hello" should match
    result = string_similarity(
        "hello", ["hi", "hello", "hey"], algorithm=SimilarityAlgo.HAMMING, threshold=0.5
    )
    # "hi" and "hey" have different lengths than "hello", so they should be skipped
    # Only "hello" should match
    assert result == ["hello"]


# ============================================================================
# Test from_potential_valid classmethod
# ============================================================================


def test_similarity_algo_from_potential_valid():
    """Test SimilarityAlgo.from_potential_valid classmethod"""
    # Test with string input
    algo = SimilarityAlgo.from_potential_valid("jaro_winkler")
    assert algo == SimilarityAlgo.JARO_WINKLER

    # Test with enum input (pass-through)
    algo = SimilarityAlgo.from_potential_valid(SimilarityAlgo.LEVENSHTEIN)
    assert algo == SimilarityAlgo.LEVENSHTEIN

    # Test with invalid string
    with pytest.raises(ValueError, match="Invalid similarity algorithm"):
        SimilarityAlgo.from_potential_valid("invalid_algorithm")

import pytest

from splitter import (
    MultilingualSplitter,
    markdown_to_text
)


@pytest.fixture(scope="module")
def splitter_instance():
    return MultilingualSplitter()


# ============================================================
# Basic Segmentation Tests
# ============================================================

def test_process_english_document(splitter_instance):

    result = splitter_instance.process_document(
        "Hello world.",
        translate=False
    )

    assert result is not None
    assert len(result["segments"]) > 0
    assert result["segments"][0]["lang"] == "en"


def test_process_german_document(splitter_instance):

    result = splitter_instance.process_document(
        "Dies ist ein deutscher Satz.",
        translate=False
    )

    assert result is not None
    assert len(result["segments"]) > 0
    assert result["segments"][0]["lang"] == "de"


def test_mixed_language_document(splitter_instance):

    text = """
    Hello world.

    Dies ist ein deutscher Satz.
    """

    result = splitter_instance.process_document(
        text,
        translate=False
    )

    languages = {
        segment["lang"]
        for segment in result["segments"]
    }

    assert "en" in languages
    assert "de" in languages


# ============================================================
# Markdown Normalization
# ============================================================

def test_markdown_normalization():

    text = """
    # Header

    **Hello World**
    """

    normalized = markdown_to_text(text)

    assert normalized is not None
    assert isinstance(normalized, str)
    assert len(normalized) > 0


# ============================================================
# Segment Metadata Validation
# ============================================================

def test_segment_positions(splitter_instance):

    result = splitter_instance.process_document(
        "Hello. Dies ist Deutsch.",
        translate=False
    )

    for segment in result["segments"]:

        assert segment["start_char"] >= 0
        assert segment["end_char"] > segment["start_char"]


def test_execution_time_present(splitter_instance):

    result = splitter_instance.process_document(
        "Hello world.",
        translate=False
    )

    assert "debug_execution_time_ms" in result
    assert result["debug_execution_time_ms"] >= 0


# ============================================================
# Payload Validation
# ============================================================

def test_vectorization_payload_without_translation(
    splitter_instance
):

    payload = splitter_instance.build_vectorization_payload(
        "Hello world.",
        translate=False
    )

    assert payload is not None
    assert "segments" in payload
    assert len(payload["segments"]) > 0


def test_language_map_matches_segments(
    splitter_instance
):

    payload = splitter_instance.build_vectorization_payload(
        "Hello. Dies ist Deutsch.",
        translate=False
    )

    language_map = payload["language_map"]
    segments = payload["segments"]

    assert len(language_map) == len(segments)

    for idx, segment in enumerate(segments):

        assert (
            language_map[idx]["start_char"]
            == segment["start_char"]
        )

        assert (
            language_map[idx]["end_char"]
            == segment["end_char"]
        )


# ============================================================
# Translation Validation
# ============================================================

def test_translation_payload(
    splitter_instance
):

    payload = splitter_instance.build_vectorization_payload(
        "Dies ist ein deutscher Satz.",
        translate=True
    )

    assert payload is not None

    assert "translated_text_en" in payload

    assert payload["translated_text_en"].strip() != ""


def test_non_german_passthrough_translation(
    splitter_instance
):

    payload = splitter_instance.build_vectorization_payload(
        "Hello world.",
        translate=True
    )

    assert payload["translated_text_en"].strip() != ""


# ============================================================
# Empty Input
# ============================================================

def test_empty_document(splitter_instance):

    result = splitter_instance.process_document(
        "",
        translate=False
    )

    assert result is not None
    assert "segments" in result


# ============================================================
# Long Document
# ============================================================

def test_long_document(splitter_instance):

    text = "Hello world. " * 200

    result = splitter_instance.process_document(
        text,
        translate=False
    )

    assert result is not None
    assert len(result["segments"]) > 0


# ============================================================
# Stress Test
# ============================================================

def test_splitter_stress_performance(
    splitter_instance
):
    """
    Splitter-only stress test.
    No translation.
    Safe for CI/CD.
    """

    stress_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Dies ist ein deutscher Satz im Dokument. "
    )

    large_text = stress_text * 300

    result = splitter_instance.process_document(
        large_text,
        translate=False
    )

    assert result is not None
    assert len(result["segments"]) > 0

    assert (
        result["debug_execution_time_ms"]
        < 1000
    ), (
        f"Splitter exceeded 1 second: "
        f"{result['debug_execution_time_ms']} ms"
    )
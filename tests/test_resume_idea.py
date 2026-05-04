"""Tests for idea-boundary aware resume slicing.

Semantic: REWIND to the start of the current sentence so unfinished ideas
are completed. Never skip content forward.
"""
import pytest

from services.presentation import resume_at_clean_boundary, resume_at_next_idea


SAMPLE_PARAGRAPHS = (
    "Three key elements of K-Nearest Neighbors. The first is the labeled training database. "
    "We collect examples whose class is already known.\n\n"
    "Second, we need a distance metric to measure similarity between samples. "
    "Euclidean distance is the most common choice for continuous features.\n\n"
    "Third, we choose the value of K — the number of nearest neighbors to consider. "
    "K is typically odd to avoid ties in classification."
)


class TestResumeAtCleanBoundary:
    def test_zero_offset_returns_full(self):
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, 0)
        assert out == SAMPLE_PARAGRAPHS

    def test_offset_at_end_returns_empty(self):
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, len(SAMPLE_PARAGRAPHS))
        assert out == ""

    def test_offset_past_end_returns_empty(self):
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, 99999)
        assert out == ""

    def test_empty_input(self):
        assert resume_at_clean_boundary("", 0) == ""
        assert resume_at_clean_boundary("", 100) == ""

    def test_rewinds_to_start_of_current_sentence(self):
        # Pause inside "labeled training database" (mid-sentence #2)
        offset = SAMPLE_PARAGRAPHS.find("training database") + 5
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, offset)
        # Should rewind to the start of sentence 2 ("The first is...")
        assert out.startswith("The first is the labeled training database.")
        # And contain the rest (sentence 3, paragraph 2, paragraph 3)
        assert "Second, we need" in out
        assert "Third, we choose" in out

    def test_rewinds_to_start_of_sentence_in_paragraph_2(self):
        # Pause inside paragraph 2 sentence 2 ("Euclidean distance...")
        offset = SAMPLE_PARAGRAPHS.find("most common")
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, offset)
        # Rewind to start of "Euclidean distance is..."
        assert out.startswith("Euclidean distance is")
        assert "Third, we choose" in out

    def test_unfinished_idea_is_completed_not_skipped(self):
        """The user's actual concern: if the idea isn't complete, finish it."""
        # Pause early in paragraph 2 — the rest of paragraph 2 should be heard
        offset = SAMPLE_PARAGRAPHS.find("Second,") + 10  # mid-sentence-1 of para 2
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, offset)
        # The full paragraph 2 must be present (we don't skip to paragraph 3)
        assert "Second, we need a distance metric" in out
        assert "Euclidean distance is the most common" in out
        # Paragraph 3 also there
        assert "Third, we choose" in out

    def test_rewind_distance_capped_at_600_chars(self):
        # Construct a single very long sentence (> 600 chars) without periods
        long_blob = "a" * 800 + ". next sentence."
        offset = 700  # deep inside the long blob
        out = resume_at_clean_boundary(long_blob, offset)
        # Should NOT rewind to position 0 (would be 700 chars of repetition).
        # Instead falls back to raw slice from offset 700.
        assert out.startswith("a" * 50)  # raw slice prefix
        assert len(out) <= len(long_blob) - offset + 1

    def test_strips_leading_whitespace(self):
        text = "Para one.   Para two with leading spaces."
        offset = text.find("two")
        out = resume_at_clean_boundary(text, offset)
        # Rewinds to start of "Para two"
        assert out.startswith("Para two")

    def test_handles_french_punctuation(self):
        text = "Première phrase. Deuxième phrase. Troisième phrase."
        offset = text.find("Deuxième") + 4  # mid "Deuxième"
        out = resume_at_clean_boundary(text, offset, language="fr")
        # Rewinds to start of "Deuxième phrase."
        assert out.startswith("Deuxième")
        assert "Troisième" in out

    def test_negative_offset_treated_as_zero(self):
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, -1)
        assert out == SAMPLE_PARAGRAPHS

    def test_very_early_offset_no_terminator_before(self):
        # Pause within the very first sentence (no terminator yet seen)
        offset = 10
        out = resume_at_clean_boundary(SAMPLE_PARAGRAPHS, offset)
        # No earlier terminator → fall back to raw offset slice
        assert out == SAMPLE_PARAGRAPHS[10:].lstrip()


class TestBackwardsCompatAlias:
    """The old function name `resume_at_next_idea` must keep working."""

    def test_alias_resolves_to_same_function(self):
        assert resume_at_next_idea is resume_at_clean_boundary

    def test_alias_call_works(self):
        out = resume_at_next_idea(SAMPLE_PARAGRAPHS, 50)
        assert out  # non-empty

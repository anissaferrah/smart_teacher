"""Tests that the sentence-boundary helpers handle non-Latin scripts.

Scripts covered :
  - Latin     : .  !  ?  …
  - CJK       : 。 ！ ？  (Chinese/Japanese full-width)
  - Arabic    : ؟  (Arabic question mark)

Spanish inverted ¿¡ are NOT terminators — they only OPEN sentences,
the actual end is still ``?`` / ``!``.
"""
from __future__ import annotations

import pytest

from services.presentation import (
    rewind_to_current_sentence_start,
    rewind_to_last_sentence_start,
    split_sentences_with_spans,
)


# ════════════════════════════════════════════════════════════════════
# split_sentences_with_spans — multi-script
# ════════════════════════════════════════════════════════════════════

class TestSplitSentencesMultiScript:

    def test_latin_french(self):
        text = "Première phrase. Deuxième phrase ! Troisième phrase ?"
        out = split_sentences_with_spans(text)
        assert len(out) == 3

    def test_chinese_full_width(self):
        text = "这是第一句。这是第二句！这是第三句？"
        out = split_sentences_with_spans(text)
        assert len(out) == 3

    def test_japanese_full_width(self):
        text = "最初の文。次の文！最後の文？"
        out = split_sentences_with_spans(text)
        assert len(out) == 3

    def test_arabic_with_arabic_qmark(self):
        text = "هذه جملة. ما هذا؟ نعم!"
        out = split_sentences_with_spans(text)
        # 3 sentences, each ending with a recognised terminator
        assert len(out) == 3

    def test_mixed_scripts(self):
        text = "Hello world. 你好世界。 مرحبا بالعالم؟"
        out = split_sentences_with_spans(text)
        assert len(out) == 3


# ════════════════════════════════════════════════════════════════════
# rewind_to_last_sentence_start — multi-script
# ════════════════════════════════════════════════════════════════════

class TestRewindLastSentenceMultiScript:

    def test_chinese(self):
        text = "第一句。第二句。第三句。"
        offset = rewind_to_last_sentence_start(text)
        assert text[offset:].startswith("第三句")

    def test_japanese(self):
        text = "最初の文！次の文！最後の文！"
        offset = rewind_to_last_sentence_start(text)
        assert text[offset:].startswith("最後の文")

    def test_arabic(self):
        text = "هذه جملة أولى. هذه جملة ثانية. هذه جملة ثالثة."
        offset = rewind_to_last_sentence_start(text)
        assert "ثالثة" in text[offset:]


# ════════════════════════════════════════════════════════════════════
# rewind_to_current_sentence_start — mid-sentence in non-Latin script
# ════════════════════════════════════════════════════════════════════

class TestRewindCurrentSentenceMultiScript:

    def test_pause_mid_chinese_second_sentence(self):
        text = "第一句话很重要。第二句话也很重要。第三句话最后。"
        # cursor inside "第二句话也很重要"
        cursor = text.find("第二") + 3
        start = rewind_to_current_sentence_start(text, cursor)
        assert text[start:].startswith("第二句话也很重要")

    def test_pause_mid_arabic_second_sentence(self):
        text = "الجملة الأولى كاملة. الجملة الثانية كاملة. الجملة الثالثة."
        cursor = text.find("الثانية") + 3
        start = rewind_to_current_sentence_start(text, cursor)
        # The current-sentence start should land on "الجملة الثانية..."
        replayed = text[start:]
        assert "الثانية" in replayed[:50]

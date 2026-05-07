import asyncio

import services.presentation as presentation


def test_split_sentences_with_spans():
    text = "This is sentence one. Here is sentence two! And sentence three?"
    spans = presentation.split_sentences_with_spans(text)
    assert isinstance(spans, list)
    assert len(spans) >= 3
    for s, (start, end) in spans:
        assert text[start:end] == s


def test_explain_slide_focused_monkeypatched(monkeypatch):
    # Monkeypatch get_brain to avoid heavy LLM init
    class DummyBrain:
        async def ask(self, prompt, **kwargs):
            return {"answer": "short explanation"}

    monkeypatch.setattr(presentation, "get_brain", lambda: DummyBrain())

    text = "Slide about variables and functions."

    result = asyncio.run(
        presentation.explain_slide_focused(text, max_sentences=2)
    )
    assert isinstance(result, str)
    assert len(result) > 0

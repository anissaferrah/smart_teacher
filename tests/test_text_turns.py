import asyncio

import services.text_turns as text_turns


def test_persist_learning_turn_monkeypatched(monkeypatch):
    # Provide dummy transcript_searcher and analytics engine
    class DummyTranscript:
        def index_interaction(self, *a, **k):
            return True

    class DummyAnalytics:
        def record_interaction(self, *a, **k):
            return True

    monkeypatch.setattr(text_turns, "get_transcript_searcher", lambda: DummyTranscript())
    monkeypatch.setattr(text_turns, "get_analytics_engine", lambda: DummyAnalytics())

    async def _run():
        await text_turns.persist_learning_turn(
            session_id="s1",
            user_id="u1",
            question_text="What is Python?",
            answer_text="A programming language",
            metadata={}
        )

    asyncio.run(_run())

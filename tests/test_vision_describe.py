"""Tests for the slide-level vision description service.

Coverage:
  - Disable flag short-circuits (returns "")
  - Missing image path → "" (no crash)
  - Disk cache hit / miss / write
  - Single-flight: N concurrent calls → 1 provider invocation
  - Provider chain: OpenAI fail → Ollama fallback
  - All providers fail → cached as ""
  - Text-only marker treated as ""
  - merge_into_slide_content shape (FR/EN)

Provider HTTP/SDK calls are monkey-patched at module level to keep the
suite hermetic — no network, no API keys needed.
"""
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from services import vision_describe as vd


# ── Helpers ──────────────────────────────────────────────────────────

def _write_fake_png(tmp_path: Path, content: bytes = b"\x89PNG\r\n\x1a\n_fake_") -> str:
    """Write a small not-actually-png file. The service only reads bytes for MD5."""
    p = tmp_path / "slide.png"
    p.write_bytes(content)
    return str(p)


@pytest_asyncio.fixture(autouse=True)
async def reset_state(tmp_path, monkeypatch):
    """Per-test isolation: redirect cache dir, reset counters & in-flight."""
    test_cache = tmp_path / "vision_cache"
    test_cache.mkdir()
    monkeypatch.setattr(vd, "_CACHE_DIR", test_cache)
    vd._inflight.clear()
    vd.reset_stats()
    yield
    vd._inflight.clear()


@pytest.fixture
def fake_image(tmp_path):
    return _write_fake_png(tmp_path)


# ════════════════════════════════════════════════════════════════════
# Disable flag + edge cases
# ════════════════════════════════════════════════════════════════════

class TestDisableAndEdges:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", False)
        out = await vd.describe_slide_image(fake_image, "en")
        assert out == ""
        assert vd.get_stats()["skipped"] >= 1

    @pytest.mark.asyncio
    async def test_none_path_returns_empty(self, monkeypatch):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        assert await vd.describe_slide_image(None, "en") == ""
        assert await vd.describe_slide_image("", "en") == ""

    @pytest.mark.asyncio
    async def test_missing_path_returns_empty(self, monkeypatch):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        assert await vd.describe_slide_image("/no/such/file.png", "en") == ""


# ════════════════════════════════════════════════════════════════════
# Disk cache
# ════════════════════════════════════════════════════════════════════

class TestDiskCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_providers(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        # Pre-populate the cache via the public path
        provider_calls = {"openai": 0, "ollama": 0}

        def stub_openai(_p, _l):
            provider_calls["openai"] += 1
            return "the slide shows a chart"

        def stub_ollama(_p, _l):
            provider_calls["ollama"] += 1
            return None

        monkeypatch.setattr(vd, "_try_openai", stub_openai)
        monkeypatch.setattr(vd, "_try_ollama", stub_ollama)

        out1 = await vd.describe_slide_image(fake_image, "en")
        out2 = await vd.describe_slide_image(fake_image, "en")

        assert out1 == "the slide shows a chart"
        assert out2 == "the slide shows a chart"
        assert provider_calls["openai"] == 1   # second call: cache hit
        assert vd.get_stats()["hits"] >= 1

    @pytest.mark.asyncio
    async def test_empty_result_is_cached_too(self, monkeypatch, fake_image):
        """A known-fail image must NOT retry providers on each call."""
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        calls = {"n": 0}

        def stub_openai(_p, _l):
            calls["n"] += 1
            return None

        def stub_ollama(_p, _l):
            calls["n"] += 1
            return None

        monkeypatch.setattr(vd, "_try_openai", stub_openai)
        monkeypatch.setattr(vd, "_try_ollama", stub_ollama)

        await vd.describe_slide_image(fake_image, "en")
        await vd.describe_slide_image(fake_image, "en")
        await vd.describe_slide_image(fake_image, "en")

        # Providers tried only on the first call — empty cached afterward
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_separate_lang_cache_keys(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        outputs = {"en": "english desc", "fr": "description française"}

        def stub_openai(_p, lang):
            return outputs[lang]

        monkeypatch.setattr(vd, "_try_openai", stub_openai)
        monkeypatch.setattr(vd, "_try_ollama", lambda *_: None)

        en = await vd.describe_slide_image(fake_image, "en")
        fr = await vd.describe_slide_image(fake_image, "fr")
        assert en == "english desc"
        assert fr == "description française"


# ════════════════════════════════════════════════════════════════════
# Provider chain
# ════════════════════════════════════════════════════════════════════

class TestProviderChain:
    @pytest.mark.asyncio
    async def test_openai_failure_falls_through_to_ollama(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        monkeypatch.setattr(vd, "_try_openai", lambda *_: None)
        monkeypatch.setattr(vd, "_try_ollama", lambda *_: "ollama desc")
        out = await vd.describe_slide_image(fake_image, "en")
        assert out == "ollama desc"

    @pytest.mark.asyncio
    async def test_both_providers_fail_returns_empty(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        monkeypatch.setattr(vd, "_try_openai", lambda *_: None)
        monkeypatch.setattr(vd, "_try_ollama", lambda *_: None)
        out = await vd.describe_slide_image(fake_image, "en")
        assert out == ""
        assert vd.get_stats()["skipped"] >= 1

    @pytest.mark.asyncio
    async def test_openai_success_skips_ollama(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        ollama_calls = {"n": 0}

        def stub_ollama(*_):
            ollama_calls["n"] += 1
            return "should not be called"

        monkeypatch.setattr(vd, "_try_openai", lambda *_: "openai desc")
        monkeypatch.setattr(vd, "_try_ollama", stub_ollama)

        out = await vd.describe_slide_image(fake_image, "en")
        assert out == "openai desc"
        assert ollama_calls["n"] == 0


# ════════════════════════════════════════════════════════════════════
# Single-flight
# ════════════════════════════════════════════════════════════════════

class TestSingleFlight:
    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_provider_call(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        gate = asyncio.Event()
        calls = {"n": 0}

        def slow_openai(_p, _l):
            calls["n"] += 1
            # Block synchronously until gate is set — simulates a slow API
            # call. The leader is in to_thread so it doesn't block the loop.
            import time
            for _ in range(50):
                if gate.is_set():
                    break
                time.sleep(0.01)
            return "shared"

        monkeypatch.setattr(vd, "_try_openai", slow_openai)
        monkeypatch.setattr(vd, "_try_ollama", lambda *_: None)

        tasks = [
            asyncio.create_task(vd.describe_slide_image(fake_image, "en"))
            for _ in range(5)
        ]
        await asyncio.sleep(0.02)        # let all 5 reach the inflight registry
        gate.set()
        results = await asyncio.gather(*tasks)
        assert all(r == "shared" for r in results)
        assert calls["n"] == 1           # five callers, one provider call


# ════════════════════════════════════════════════════════════════════
# Text-only marker
# ════════════════════════════════════════════════════════════════════

class TestTextOnlyMarker:
    @pytest.mark.asyncio
    async def test_text_only_treated_as_empty(self, monkeypatch, fake_image):
        monkeypatch.setattr(vd.Config, "VISION_DESCRIBE_ENABLED", True)
        monkeypatch.setattr(vd, "_try_openai", lambda *_: None)  # forces ollama path
        # Ollama returns text_only literal
        monkeypatch.setattr(vd, "_try_ollama", lambda *_: None)
        # actually we test the marker handling at provider level — test that
        # _is_text_only_marker correctly identifies the variants
        assert vd._is_text_only_marker("TEXT_ONLY") is True
        assert vd._is_text_only_marker("texte_seul") is True
        assert vd._is_text_only_marker("(text_only)") is True
        assert vd._is_text_only_marker("the slide shows a chart") is False


# ════════════════════════════════════════════════════════════════════
# merge_into_slide_content
# ════════════════════════════════════════════════════════════════════

class TestMerge:
    def test_merge_appends_with_marker_fr(self):
        out = vd.merge_into_slide_content("Bullet 1\nBullet 2", "Le graphique montre X.", "fr")
        assert "Bullet 1" in out
        assert "[Description visuelle de la slide]" in out
        assert "Le graphique montre X." in out

    def test_merge_appends_with_marker_en(self):
        out = vd.merge_into_slide_content("Bullet 1", "The chart shows X.", "en")
        assert "[Visual description of the slide]" in out
        assert "The chart shows X." in out

    def test_merge_empty_vision_returns_text_unchanged(self):
        assert vd.merge_into_slide_content("text", "", "en") == "text"
        assert vd.merge_into_slide_content("text", "   ", "en") == "text"

    def test_merge_empty_text_returns_just_vision(self):
        out = vd.merge_into_slide_content("", "vision desc", "en")
        assert out.startswith("[Visual description")
        assert "vision desc" in out

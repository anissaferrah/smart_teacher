"""Tests for the server-side auth gate that protects /static/*.html.

The middleware logic is tested in isolation against a minimal FastAPI
app that mounts ONLY the gate function from main.py — booting the
full main.py (LLM, RAG, Redis…) is too slow for a unit test.

We rebuild the same logic in a small fixture and verify the
behavioural contract :
  - login.html accessible without token
  - other .html pages redirect when no/invalid token
  - valid token (cookie or Bearer) grants access
  - non-HTML paths are not gated
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


_PUBLIC_HTML_PAGES = {"/static/login.html"}


def _build_test_app(static_dir: pathlib.Path) -> FastAPI:
    """Replicates the gate from main.py on a fresh FastAPI app."""
    app = FastAPI()

    @app.middleware("http")
    async def static_html_auth_gate(request, call_next):
        path = request.url.path
        if (
            request.method == "GET"
            and path.startswith("/static/")
            and path.endswith(".html")
            and path not in _PUBLIC_HTML_PAGES
        ):
            token = request.cookies.get("smart_teacher_token", "")
            if not token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
            if not token:
                return RedirectResponse(url="/static/login.html", status_code=302)
            try:
                from handlers.auth import decode_access_token
                decode_access_token(token)
            except Exception:                                              # noqa: BLE001
                return RedirectResponse(url="/static/login.html", status_code=302)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Build a minimal app with stub HTML files in a temp dir."""
    static_dir = tmp_path_factory.mktemp("static")
    # Stub files — the gate doesn't care about content
    for name in ("login.html", "index.html", "admin.html",
                 "profile.html", "styles.css"):
        (static_dir / name).write_text(f"<!-- {name} -->", encoding="utf-8")
    app = _build_test_app(static_dir)
    return TestClient(app)


def _valid_token() -> str:
    """Mint a valid JWT mirroring routes/auth.create_access_token."""
    from handlers.auth import create_access_token
    return create_access_token(
        "00000000-0000-0000-0000-000000000001",
        "test@example.com", "student",
    )


# ════════════════════════════════════════════════════════════════════
# Public access (no token needed)
# ════════════════════════════════════════════════════════════════════

class TestPublicAccess:
    def test_login_page_accessible_without_token(self, client):
        r = client.get("/static/login.html", follow_redirects=False)
        assert r.status_code == 200, "login.html must be public"

    def test_static_css_not_gated(self, client):
        """CSS / JS / images must NOT trigger the redirect, even if
        anonymous — login.html itself depends on them."""
        r = client.get("/static/styles.css", follow_redirects=False)
        assert r.status_code != 302


# ════════════════════════════════════════════════════════════════════
# Unauth → redirect
# ════════════════════════════════════════════════════════════════════

class TestUnauthRedirect:
    @pytest.mark.parametrize("page", [
        "/static/index.html",
        "/static/admin.html",
        "/static/profile.html",
    ])
    def test_protected_html_redirects_without_token(self, client, page):
        r = client.get(page, follow_redirects=False)
        assert r.status_code == 302, (
            f"{page} should redirect with no token, got {r.status_code}"
        )
        assert "login.html" in r.headers.get("location", "")

    def test_invalid_token_redirects(self, client):
        r = client.get(
            "/static/index.html",
            cookies={"smart_teacher_token": "not-a-real-jwt"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "login.html" in r.headers.get("location", "")


# ════════════════════════════════════════════════════════════════════
# Auth → access
# ════════════════════════════════════════════════════════════════════

class TestAuthAccess:
    def test_valid_cookie_grants_access(self, client):
        token = _valid_token()
        r = client.get(
            "/static/index.html",
            cookies={"smart_teacher_token": token},
            follow_redirects=False,
        )
        assert r.status_code == 200, (
            f"Expected 200 with valid cookie, got {r.status_code}"
        )

    def test_valid_bearer_header_grants_access(self, client):
        token = _valid_token()
        r = client.get(
            "/static/index.html",
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=False,
        )
        assert r.status_code == 200


# ════════════════════════════════════════════════════════════════════
# Method scope
# ════════════════════════════════════════════════════════════════════

class TestMethodScope:
    def test_post_to_html_not_redirected(self, client):
        r = client.post("/static/index.html", follow_redirects=False)
        # POST to a static .html → 405 Method Not Allowed, NOT a redirect
        assert r.status_code != 302

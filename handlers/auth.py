"""
JWT authentication helpers + FastAPI dependency + security hardening.

Includes :
  - bcrypt password hashing (rounds=12)
  - JWT signing/verification (HS256)
  - Password strength check (RFC 8907 inspired)
  - Rate limiting on login (Redis-backed sliding window)
  - Audit logging (sec_audit.csv)
  - Role-based dependencies (get_current_user, require_admin)
"""
from __future__ import annotations

import csv
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import Config

log = logging.getLogger("SmartTeacher.Auth")


# ── Password strength check ──────────────────────────────────────────

class PasswordStrengthError(ValueError):
    pass


_COMMON_PASSWORDS = {
    "password", "12345678", "qwerty123", "admin123", "password1",
    "letmein", "iloveyou", "azerty123", "motdepasse", "12345abc",
}


def check_password_strength(password: str) -> None:
    """Validate password meets minimum security requirements.

    Rules :
      - 8+ chars
      - Mixed alpha + digit
      - Not in common passwords list
    Raises PasswordStrengthError if too weak.
    """
    pw_len = len(password) if password else 0
    has_alpha = bool(re.search(r"[a-zA-Z]", password)) if password else False
    has_digit = bool(re.search(r"\d", password)) if password else False
    is_common = (password or "").lower() in _COMMON_PASSWORDS
    log.info(
        "🔐 password_strength | len=%d alpha=%s digit=%s common=%s",
        pw_len, has_alpha, has_digit, is_common,
    )
    if not password or pw_len < 8:
        log.warning("🔐 password_strength REJECT | reason=too_short (%d<8)", pw_len)
        raise PasswordStrengthError("Mot de passe trop court (minimum 8 caractères)")
    if is_common:
        log.warning("🔐 password_strength REJECT | reason=common_password")
        raise PasswordStrengthError("Mot de passe trop commun, choisis-en un autre")
    if not (has_alpha and has_digit):
        log.warning("🔐 password_strength REJECT | reason=missing_alpha_or_digit (alpha=%s digit=%s)", has_alpha, has_digit)
        raise PasswordStrengthError("Le mot de passe doit contenir lettres et chiffres")
    log.info("🔐 password_strength OK")

# ── Password hashing (bcrypt) ─────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt (work-factor 12)."""
    t0 = time.perf_counter()
    h = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
    log.info(
        "🔐 hash_password | bcrypt rounds=12 | took=%.0fms | hash_prefix=%s",
        (time.perf_counter() - t0) * 1000.0, h[:7],
    )
    return h


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time password verify."""
    if not plain or not hashed:
        log.info("🔐 verify_password | missing_input plain=%s hashed=%s",
                 bool(plain), bool(hashed))
        return False
    try:
        t0 = time.perf_counter()
        ok = bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        log.info(
            "🔐 verify_password | match=%s | bcrypt compare took=%.0fms",
            ok, (time.perf_counter() - t0) * 1000.0,
        )
        return ok
    except Exception as exc:
        log.warning(f"🔐 verify_password ERROR: {exc}")
        return False


# ── JWT tokens ────────────────────────────────────────────────────────

def create_access_token(
    student_id: str,
    email: str,
    account_level: str = "student",
    expires_hours: Optional[int] = None,
) -> str:
    """Create a signed JWT carrying user identity + role."""
    hours = expires_hours or Config.JWT_EXPIRATION_HOURS
    exp = datetime.utcnow() + timedelta(hours=hours)
    payload = {
        "sub":           str(student_id),
        "email":         email,
        "account_level": account_level,
        "iat":           datetime.utcnow(),
        "exp":           exp,
    }
    token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
    log.info(
        "🔐 JWT CREATE | sub=%s email=%s role=%s | algo=%s ttl=%dh exp=%s | token_prefix=%s...",
        str(student_id)[:8], email, account_level,
        Config.JWT_ALGORITHM, hours, exp.isoformat(),
        token[:20],
    )
    return token


def decode_access_token(token: str) -> dict:
    """Verify + decode a JWT. Raises HTTPException 401 on invalid/expired."""
    try:
        claims = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
        log.info(
            "🔐 JWT DECODE OK | sub=%s email=%s role=%s exp=%s",
            str(claims.get("sub", ""))[:8],
            claims.get("email", ""),
            claims.get("account_level", ""),
            datetime.utcfromtimestamp(claims["exp"]).isoformat() if claims.get("exp") else "?",
        )
        return claims
    except jwt.ExpiredSignatureError:
        log.warning("🔐 JWT DECODE FAIL | reason=expired_signature | token_prefix=%s...", token[:20])
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError as exc:
        log.warning("🔐 JWT DECODE FAIL | reason=invalid_token (%s) | token_prefix=%s...", exc, token[:20])
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}")


# ── FastAPI dependency: extract authenticated user ────────────────────

_security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> dict:
    """Returns the JWT claims dict (sub=student_id, email, account_level).

    Raises 401 if no/invalid token. Use as `Depends(get_current_user)` on
    any protected endpoint.
    """
    # Allow token via Authorization header OR cookie (for easier browser flows)
    token: Optional[str] = None
    source = "none"
    if credentials:
        token = credentials.credentials
        source = "header_bearer"
    if not token:
        token = request.cookies.get("smart_teacher_token")
        if token:
            source = "cookie"
    log.info(
        "🔐 get_current_user | path=%s method=%s | token_source=%s present=%s",
        request.url.path, request.method, source, bool(token),
    )
    if not token:
        log.warning("🔐 get_current_user REJECT | reason=no_token | path=%s", request.url.path)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return decode_access_token(token)


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Same as get_current_user but enforces role in {teacher, admin}."""
    role = user.get("account_level", "student")
    log.info(
        "🔐 require_admin | sub=%s role=%s | allowed=%s",
        str(user.get("sub", ""))[:8], role, role in ("teacher", "admin"),
    )
    if role not in ("teacher", "admin"):
        log.warning(
            "🔐 require_admin REJECT | sub=%s role=%s | reason=insufficient_privileges",
            str(user.get("sub", ""))[:8], role,
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin/teacher role required")
    return user


# ── Rate limiting (Redis sliding window) ──────────────────────────────

async def check_login_rate_limit(
    identifier: str, max_attempts: int = 5, window_seconds: int = 300,
) -> tuple[bool, int]:
    """Return (allowed, remaining_attempts).

    Compteur Redis : login:fails:{identifier} — TTL 5 min.
    Bloque après 5 échecs en 5 minutes.
    """
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        key = f"login:fails:{identifier.lower()}"
        n = int(await r.get(key) or 0)
        allowed = n < max_attempts
        remaining = max(0, max_attempts - n)
        log.info(
            "🔐 rate_limit CHECK | id=%s | failures=%d/%d window=%ds | allowed=%s remaining=%d",
            identifier.lower(), n, max_attempts, window_seconds, allowed, remaining,
        )
        if not allowed:
            return (False, 0)
        return (True, remaining)
    except Exception as exc:
        log.warning(f"🔐 rate_limit CHECK skipped (Redis down?): {exc} — failing OPEN")
        return (True, max_attempts)


async def record_login_failure(identifier: str, window_seconds: int = 300) -> int:
    """Incremente le compteur d'échecs. Returns new count."""
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        key = f"login:fails:{identifier.lower()}"
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, window_seconds)
        log.info(
            "🔐 rate_limit INCR | id=%s | failures=%d (TTL=%ds %s)",
            identifier.lower(), int(n), window_seconds,
            "set" if n == 1 else "carry",
        )
        return int(n)
    except Exception as exc:
        log.warning(f"🔐 rate_limit INCR skipped: {exc}")
        return 0


async def reset_login_failures(identifier: str) -> None:
    """Clear failure counter on successful login."""
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        await r.delete(f"login:fails:{identifier.lower()}")
        log.info("🔐 rate_limit RESET | id=%s | counter cleared after success", identifier.lower())
    except Exception as exc:
        log.debug(f"rate_limit RESET skipped: {exc}")


# ── Audit log (CSV) ──────────────────────────────────────────────────

_AUDIT_PATH = os.path.join(Config.LOGS_DIR, "sec_audit.csv")


def _ensure_audit_file():
    Path(_AUDIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(_AUDIT_PATH):
        with open(_AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "event", "actor", "target",
                "ip", "user_agent", "outcome", "details",
            ])


def audit_log(
    event: str, actor: str = "", target: str = "",
    request: Optional[Request] = None, outcome: str = "success",
    details: str = "",
) -> None:
    """Append-only security audit log (login, register, role change, etc.)."""
    try:
        _ensure_audit_file()
        ip = (request.client.host if request and request.client else "") if request else ""
        ua = (request.headers.get("user-agent", "") or "")[:120] if request else ""
        with open(_AUDIT_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.utcnow().isoformat(), event, actor[:100], target[:100],
                ip, ua, outcome, details[:200],
            ])
        log.info(
            "🔐 AUDIT | event=%s actor=%s target=%s ip=%s outcome=%s | %s",
            event, actor[:40] or "-", target[:60] or "-", ip or "-", outcome,
            details[:80] or "-",
        )
    except Exception as exc:
        log.warning(f"🔐 audit_log write FAILED: {exc}")

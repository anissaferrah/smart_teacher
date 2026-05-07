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
    if not password or len(password) < 8:
        raise PasswordStrengthError("Mot de passe trop court (minimum 8 caractères)")
    if password.lower() in _COMMON_PASSWORDS:
        raise PasswordStrengthError("Mot de passe trop commun, choisis-en un autre")
    has_alpha = bool(re.search(r"[a-zA-Z]", password))
    has_digit = bool(re.search(r"\d", password))
    if not (has_alpha and has_digit):
        raise PasswordStrengthError("Le mot de passe doit contenir lettres et chiffres")

# ── Password hashing (bcrypt) ─────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt (work-factor 12)."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time password verify."""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as exc:
        log.debug(f"verify_password error: {exc}")
        return False


# ── JWT tokens ────────────────────────────────────────────────────────

def create_access_token(
    student_id: str,
    email: str,
    account_level: str = "student",
    expires_hours: Optional[int] = None,
) -> str:
    """Create a signed JWT carrying user identity + role."""
    exp = datetime.utcnow() + timedelta(hours=expires_hours or Config.JWT_EXPIRATION_HOURS)
    payload = {
        "sub":           str(student_id),
        "email":         email,
        "account_level": account_level,
        "iat":           datetime.utcnow(),
        "exp":           exp,
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Verify + decode a JWT. Raises HTTPException 401 on invalid/expired."""
    try:
        return jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError as exc:
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
    if credentials:
        token = credentials.credentials
    if not token:
        token = request.cookies.get("smart_teacher_token")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return decode_access_token(token)


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Same as get_current_user but enforces role in {teacher, admin}."""
    role = user.get("account_level", "student")
    if role not in ("teacher", "admin"):
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
        if n >= max_attempts:
            return (False, 0)
        return (True, max_attempts - n)
    except Exception as exc:
        log.debug(f"rate-limit check skipped (Redis down?): {exc}")
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
        return int(n)
    except Exception as exc:
        log.debug(f"rate-limit incr skipped: {exc}")
        return 0


async def reset_login_failures(identifier: str) -> None:
    """Clear failure counter on successful login."""
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        await r.delete(f"login:fails:{identifier.lower()}")
    except Exception:
        pass


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
    except Exception as exc:
        log.debug(f"audit_log write skipped: {exc}")

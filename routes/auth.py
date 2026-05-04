"""Authentication routes — register, login, current user info."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import Student
from handlers.auth import (
    PasswordStrengthError, audit_log, check_login_rate_limit,
    check_password_strength, create_access_token,
    get_current_user, hash_password, record_login_failure,
    reset_login_failures, verify_password,
)

router = APIRouter(prefix="/auth")
log = logging.getLogger("SmartTeacher.routes.auth")


# ── Request / response schemas ─────────────────────────────────────────

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    first_name: str = "Étudiant"
    last_name: str = ""
    # Cours preference : language for narrations + Q&A (fr / en / ar)
    preferred_language: str = "fr"
    # Academic level used to adapt vocabulary and explanation depth.
    # Allowed : "collège" | "lycée" | "université". Defaults to lycée.
    student_level: str = "lycée"
    # account_level always defaults to "student" — admin must promote manually


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    student_id: str
    email: str
    account_level: str
    first_name: str


class MeOut(BaseModel):
    student_id: str
    email: str
    account_level: str
    first_name: str
    last_name: Optional[str]
    preferred_language: str
    student_level: str = "lycée"
    is_active: bool


# ── POST /auth/register ────────────────────────────────────────────────

@router.post("/register", response_model=TokenOut)
async def register(payload: RegisterIn, request: Request, response: Response):
    """Create a new student account + issue JWT.

    Hardening :
      - Password strength check (min 8 chars, alpha+digit, not in common list)
      - Email lowercase + dedup
      - Audit log
    """
    log.info(
        "🔐 REGISTER START | email=%s lang=%s level=%s | ip=%s",
        payload.email, payload.preferred_language, payload.student_level,
        request.client.host if request.client else "?",
    )
    try:
        check_password_strength(payload.password)
    except PasswordStrengthError as exc:
        audit_log("register_failed", target=payload.email, request=request,
                  outcome="failure", details=str(exc))
        log.warning("🔐 REGISTER REJECT | email=%s | weak_password: %s", payload.email, exc)
        raise HTTPException(400, str(exc))

    async with AsyncSessionLocal() as db:
        existing = (await db.execute(
            select(Student).where(Student.email == payload.email.lower())
        )).scalar_one_or_none()
        if existing is not None:
            log.warning("🔐 REGISTER REJECT | email=%s | reason=already_exists", payload.email)
            raise HTTPException(409, "Email already registered")

        # Validate student_level whitelist — block typos that would
        # later cause prompt adaptation to silently fall back.
        _allowed_levels = {"collège", "lycée", "université"}
        student_level = payload.student_level if payload.student_level in _allowed_levels else "lycée"
        # Validate preferred_language to a small set we actually support
        _allowed_langs = {"fr", "en", "ar"}
        preferred_language = payload.preferred_language if payload.preferred_language in _allowed_langs else "fr"
        log.info(
            "🔐 REGISTER VALIDATE | email=%s | level_input=%s→%s lang_input=%s→%s",
            payload.email,
            payload.student_level, student_level,
            payload.preferred_language, preferred_language,
        )

        student = Student(
            id=uuid.uuid4(),
            email=payload.email.lower(),
            password_hash=hash_password(payload.password),
            first_name=payload.first_name,
            last_name=payload.last_name,
            preferred_language=preferred_language,
            student_level=student_level,
            account_level="student",
            is_active=1,
        )
        db.add(student)
        await db.commit()
        await db.refresh(student)
        log.info(
            "🔐 REGISTER DB INSERT | sub=%s email=%s role=student",
            str(student.id)[:8], student.email,
        )

    token = create_access_token(str(student.id), student.email, student.account_level)
    response.set_cookie(
        "smart_teacher_token", token,
        httponly=True, samesite="lax", max_age=24 * 3600,
    )
    log.info("🔐 REGISTER SET-COOKIE | name=smart_teacher_token httponly=True samesite=lax max_age=86400s")
    audit_log("register", actor=str(student.id), target=student.email,
              request=request, outcome="success")
    log.info("🔐 REGISTER OK | sub=%s email=%s", str(student.id)[:8], student.email)
    return TokenOut(
        access_token=token, student_id=str(student.id), email=student.email,
        account_level=student.account_level, first_name=student.first_name,
    )


# ── POST /auth/login ───────────────────────────────────────────────────

@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn, request: Request, response: Response):
    """Verify password, issue JWT.

    Hardening :
      - Rate limiting (5 échecs / 5min par email) — Redis sliding window
      - Same error msg for unknown email/wrong password (no enumeration)
      - Audit log
    """
    email_lower = payload.email.lower()
    log.info(
        "🔐 LOGIN START | email=%s | ip=%s ua=%s",
        email_lower,
        request.client.host if request.client else "?",
        (request.headers.get("user-agent", "") or "")[:60],
    )

    # Rate limit check BEFORE password check (prevents timing-based enumeration)
    allowed, remaining = await check_login_rate_limit(email_lower)
    if not allowed:
        audit_log("login_rate_limited", target=email_lower, request=request,
                  outcome="failure", details="too many attempts")
        log.warning("🔐 LOGIN BLOCKED | email=%s | reason=rate_limited", email_lower)
        raise HTTPException(
            429,
            f"Trop de tentatives échouées. Réessaie dans 5 minutes.",
        )

    async with AsyncSessionLocal() as db:
        student = (await db.execute(
            select(Student).where(Student.email == email_lower)
        )).scalar_one_or_none()
        log.info(
            "🔐 LOGIN DB LOOKUP | email=%s | student_found=%s",
            email_lower, bool(student),
        )
        if student is None or not verify_password(payload.password, student.password_hash or ""):
            await record_login_failure(email_lower)
            audit_log("login_failed", target=email_lower, request=request,
                      outcome="failure",
                      details=f"remaining_attempts={max(0, remaining - 1)}")
            log.warning(
                "🔐 LOGIN REJECT | email=%s | reason=%s | remaining_attempts=%d",
                email_lower,
                "unknown_email" if student is None else "wrong_password",
                max(0, remaining - 1),
            )
            raise HTTPException(401, "Invalid email or password")
        if not student.is_active:
            audit_log("login_disabled", actor=str(student.id), target=email_lower,
                      request=request, outcome="failure")
            log.warning("🔐 LOGIN REJECT | email=%s | reason=account_disabled", email_lower)
            raise HTTPException(403, "Account disabled")

    # Successful login — clear failure counter
    await reset_login_failures(email_lower)
    token = create_access_token(str(student.id), student.email, student.account_level)
    response.set_cookie(
        "smart_teacher_token", token,
        httponly=True, samesite="lax", max_age=24 * 3600,
    )
    log.info("🔐 LOGIN SET-COOKIE | name=smart_teacher_token httponly=True samesite=lax max_age=86400s")
    audit_log("login", actor=str(student.id), target=email_lower,
              request=request, outcome="success",
              details=f"role={student.account_level}")
    log.info(
        "🔐 LOGIN OK | sub=%s email=%s role=%s",
        str(student.id)[:8], student.email, student.account_level,
    )
    return TokenOut(
        access_token=token, student_id=str(student.id), email=student.email,
        account_level=student.account_level, first_name=student.first_name,
    )


# ── POST /auth/logout ──────────────────────────────────────────────────

@router.post("/logout")
async def logout(response: Response, request: Request):
    """Clear the auth cookie."""
    response.delete_cookie("smart_teacher_token")
    log.info(
        "🔐 LOGOUT | cookie cleared | ip=%s",
        request.client.host if request.client else "?",
    )
    return {"status": "ok"}


# ── GET /auth/me ───────────────────────────────────────────────────────

@router.get("/me", response_model=MeOut)
async def me(user: dict = Depends(get_current_user)):
    """Return the currently-authenticated student's profile fields."""
    log.info("🔐 GET /me | sub=%s email=%s", str(user.get("sub", ""))[:8], user.get("email", ""))
    async with AsyncSessionLocal() as db:
        student = (await db.execute(
            select(Student).where(Student.id == uuid.UUID(user["sub"]))
        )).scalar_one_or_none()
        if student is None:
            log.warning("🔐 GET /me REJECT | sub=%s | reason=account_not_found_in_db", str(user.get("sub", ""))[:8])
            raise HTTPException(404, "Account not found")
    log.info(
        "🔐 GET /me OK | sub=%s lang=%s level=%s active=%s",
        str(student.id)[:8], student.preferred_language,
        getattr(student, "student_level", "?"), bool(student.is_active),
    )
    return MeOut(
        student_id=str(student.id),
        email=student.email,
        account_level=student.account_level,
        first_name=student.first_name,
        last_name=student.last_name,
        preferred_language=student.preferred_language,
        student_level=getattr(student, "student_level", None) or "lycée",
        is_active=bool(student.is_active),
    )

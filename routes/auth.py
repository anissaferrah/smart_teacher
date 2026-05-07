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
    preferred_language: str = "fr"
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
    try:
        check_password_strength(payload.password)
    except PasswordStrengthError as exc:
        audit_log("register_failed", target=payload.email, request=request,
                  outcome="failure", details=str(exc))
        raise HTTPException(400, str(exc))

    async with AsyncSessionLocal() as db:
        existing = (await db.execute(
            select(Student).where(Student.email == payload.email.lower())
        )).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(409, "Email already registered")

        student = Student(
            id=uuid.uuid4(),
            email=payload.email.lower(),
            password_hash=hash_password(payload.password),
            first_name=payload.first_name,
            last_name=payload.last_name,
            preferred_language=payload.preferred_language,
            account_level="student",
            is_active=1,
        )
        db.add(student)
        await db.commit()
        await db.refresh(student)

    token = create_access_token(str(student.id), student.email, student.account_level)
    response.set_cookie(
        "smart_teacher_token", token,
        httponly=True, samesite="lax", max_age=24 * 3600,
    )
    audit_log("register", actor=str(student.id), target=student.email,
              request=request, outcome="success")
    log.info(f"✅ Student registered: {student.email}")
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

    # Rate limit check BEFORE password check (prevents timing-based enumeration)
    allowed, remaining = await check_login_rate_limit(email_lower)
    if not allowed:
        audit_log("login_rate_limited", target=email_lower, request=request,
                  outcome="failure", details="too many attempts")
        raise HTTPException(
            429,
            f"Trop de tentatives échouées. Réessaie dans 5 minutes.",
        )

    async with AsyncSessionLocal() as db:
        student = (await db.execute(
            select(Student).where(Student.email == email_lower)
        )).scalar_one_or_none()
        if student is None or not verify_password(payload.password, student.password_hash or ""):
            await record_login_failure(email_lower)
            audit_log("login_failed", target=email_lower, request=request,
                      outcome="failure",
                      details=f"remaining_attempts={max(0, remaining - 1)}")
            raise HTTPException(401, "Invalid email or password")
        if not student.is_active:
            audit_log("login_disabled", actor=str(student.id), target=email_lower,
                      request=request, outcome="failure")
            raise HTTPException(403, "Account disabled")

    # Successful login — clear failure counter
    await reset_login_failures(email_lower)
    token = create_access_token(str(student.id), student.email, student.account_level)
    response.set_cookie(
        "smart_teacher_token", token,
        httponly=True, samesite="lax", max_age=24 * 3600,
    )
    audit_log("login", actor=str(student.id), target=email_lower,
              request=request, outcome="success",
              details=f"role={student.account_level}")
    log.info(f"✅ Login: {student.email} ({student.account_level})")
    return TokenOut(
        access_token=token, student_id=str(student.id), email=student.email,
        account_level=student.account_level, first_name=student.first_name,
    )


# ── POST /auth/logout ──────────────────────────────────────────────────

@router.post("/logout")
async def logout(response: Response):
    """Clear the auth cookie."""
    response.delete_cookie("smart_teacher_token")
    return {"status": "ok"}


# ── GET /auth/me ───────────────────────────────────────────────────────

@router.get("/me", response_model=MeOut)
async def me(user: dict = Depends(get_current_user)):
    """Return the currently-authenticated student's profile fields."""
    async with AsyncSessionLocal() as db:
        student = (await db.execute(
            select(Student).where(Student.id == uuid.UUID(user["sub"]))
        )).scalar_one_or_none()
        if student is None:
            raise HTTPException(404, "Account not found")
    return MeOut(
        student_id=str(student.id),
        email=student.email,
        account_level=student.account_level,
        first_name=student.first_name,
        last_name=student.last_name,
        preferred_language=student.preferred_language,
        is_active=bool(student.is_active),
    )

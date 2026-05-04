# database/crud.py
"""
Smart Teacher — CRUD Operations for PostgreSQL.

Provides async database operations for managing:
    - Courses with hierarchical structure (chapters → sections → concepts)
    - Learning sessions and student progress
    - Interaction logs for analytics and KPI tracking

All operations are async-compatible with SQLAlchemy AsyncSession.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Course, Chapter, Section, Concept, LearningSession, Interaction, LearningEvent, Student

log = logging.getLogger("SmartTeacher.CRUD")


# ════════════════════════════════════════════════════════════════════════
# AUTHENTICATION OPERATIONS
# ════════════════════════════════════════════════════════════════════════


async def create_student(
    db: AsyncSession,
    email: str,
    password_hash: str,
    first_name: str = "Utilisateur",
    last_name: str = "",
    preferred_language: str = "fr"
) -> Student:
    """
    Create a new student account.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    email : str
        Student email (must be unique)
    password_hash : str
        Hashed password
    first_name : str
        Student's first name
    last_name : str
        Student's last name
    preferred_language : str
        Preferred language code (default: "fr")
    
    Returns
    -------
    Student
        Created student object
    """
    student = Student(
        email=email,
        password_hash=password_hash,
        first_name=first_name,
        last_name=last_name,
        preferred_language=preferred_language
    )
    db.add(student)
    await db.flush()
    log.debug(f"Created student: {student.id} ({email})")
    return student


# ════════════════════════════════════════════════════════════════════════
# COURSE OPERATIONS
# ════════════════════════════════════════════════════════════════════════


async def create_course(
    db: AsyncSession,
    course_data: dict
) -> Course:
    """
    Create a new course.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    course_data : dict
        Course metadata with keys: title, subject, language, level, description, file_path
    
    Returns
    -------
    Course
        Created course object (with ID populated after flush)
    
    Raises
    ------
    sqlalchemy.exc.IntegrityError
        If required fields are missing
    """
    course = Course(**course_data)
    db.add(course)
    await db.flush()
    log.debug(f"Created course: {course.id}")
    return course


async def create_course_with_structure(
    db: AsyncSession,
    course_data: dict
) -> uuid.UUID:
    """
    Create a complete course with hierarchical structure in one transaction.
    
    Recursively creates Course → Chapters → Sections → Concepts
    and ensures referential integrity within a single transaction.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    course_data : dict
        Nested structure with keys:
        - title (str): Course title
        - subject (str, optional): Subject area
        - language (str, optional): ISO 639-1 code (default: "fr")
        - level (str, optional): Course level (default: "lycée")
        - description (str, optional): Course description
        - file_path (str, optional): Source file path
        - chapters (list[dict]): Chapter definitions, each with:
            - title (str): Chapter title
            - order (int): Order within course
            - summary (str, optional): Chapter summary
            - sections (list[dict]): Section definitions, each with:
                - title (str): Section title
                - order (int): Order within chapter
                - content (str, optional): Original course content
                - duration_s (int, optional): Duration in seconds (default: 120)
                - image_urls (list[str], optional): Associated slide URLs
                - concepts (list[dict]): Concept definitions, each with:
                    - term (str): Concept term
                    - definition (str, optional): Definition
                    - example (str, optional): Example or use case
                    - type (str, optional): Concept type (default: "definition")
    
    Returns
    -------
    uuid.UUID
        ID of created course
    
    Raises
    ------
    ValueError
        If course_data is malformed
    sqlalchemy.exc.IntegrityError
        If database constraints violated
    
    Examples
    --------
    >>> course_id = await create_course_with_structure(db, {
    ...     "title": "Statistics 101",
    ...     "language": "fr",
    ...     "chapters": [
    ...         {
    ...             "title": "Chapter 1: Introduction",
    ...             "order": 1,
    ...             "sections": [
    ...                 {
    ...                     "title": "What is this course about?",
    ...                     "order": 1,
    ...                     "content": "This course is about...",
    ...                     "concepts": [
    ...                         {"term": "k-means", "definition": "..."}
    ...                     ]
    ...                 }
    ...             ]
    ...         }
    ...     ]
    ... })
    """
    # Create course
    course = Course(
        title=course_data["title"],
        subject=course_data.get("subject", "general"),
        language=course_data.get("language", "fr"),
        level=course_data.get("level", "lycée"),
        description=course_data.get("description", ""),
        file_path=course_data.get("file_path", ""),
    )
    db.add(course)
    await db.flush()
    log.info(f"Created course: {course.id} - {course.title}")

    # Create chapters
    for ch_idx, ch_data in enumerate(course_data.get("chapters", [])):
        chapter = Chapter(
            course_id=course.id,
            title=ch_data["title"],
            order=ch_data.get("order", ch_idx + 1),
            summary=ch_data.get("summary", ""),
        )
        db.add(chapter)
        await db.flush()

        # Create sections
        for sec_idx, sec_data in enumerate(ch_data.get("sections", [])):
            image_url = sec_data.get("image_url", "")
            image_urls = sec_data.get("image_urls", [])
            if not image_url and image_urls:
                image_url = image_urls[0]

            section = Section(
                chapter_id=chapter.id,
                title=sec_data["title"],
                order=sec_data.get("order", sec_idx + 1),
                content=sec_data.get("content", ""),
                duration_s=sec_data.get("duration_s", 120),
                image_url=image_url,
                image_urls=sec_data.get("image_urls", []),
            )
            db.add(section)
            await db.flush()

            # Create concepts
            for c_data in sec_data.get("concepts", []):
                concept = Concept(
                    section_id=section.id,
                    term=c_data["term"],
                    definition=c_data.get("definition", ""),
                    example=c_data.get("example", ""),
                    concept_type=c_data.get("type", "definition"),
                )
                db.add(concept)

    await db.commit()
    log.info(f"✅ Course structure committed: {course.id}")
    return course.id


async def get_course(
    db: AsyncSession,
    course_id: uuid.UUID
) -> Optional[Course]:
    """
    Retrieve a course by ID.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    course_id : uuid.UUID
        Course ID
    
    Returns
    -------
    Course or None
        Course object if found, otherwise None
    """
    result = await db.execute(select(Course).where(Course.id == course_id))
    return result.scalar_one_or_none()


async def get_course_with_structure(
    db: AsyncSession,
    course_id: uuid.UUID
) -> Optional[Course]:
    """
    Retrieve a course with complete hierarchical structure eagerly loaded.
    
    Loads Course → Chapters → Sections → Concepts in a single query
    using selectinload to avoid N+1 queries.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    course_id : uuid.UUID
        Course ID
    
    Returns
    -------
    Course or None
        Course with all chapters, sections, and concepts populated,
        or None if course not found
    """
    result = await db.execute(
        select(Course)
        .where(Course.id == course_id)
        .options(
            selectinload(Course.chapters)
            .selectinload(Chapter.sections)
            .selectinload(Section.concepts)
        )
    )
    return result.scalar_one_or_none()


async def get_all_courses(
    db: AsyncSession
) -> List[Course]:
    """
    Retrieve all courses ordered by creation date (newest first).
    Eager-loads relationships to avoid lazy-loading errors.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    
    Returns
    -------
    list[Course]
        All courses in database (may be empty list)
    """
    from sqlalchemy.orm import selectinload
    
    from database.models import Chapter
    
    result = await db.execute(
        select(Course)
        .options(selectinload(Course.chapters).selectinload(Chapter.sections))
        .order_by(Course.created_at.desc())
    )
    courses = result.unique().scalars().all()
    log.debug(f"Retrieved {len(courses)} courses")
    return courses


async def delete_course(
    db: AsyncSession,
    course_id: uuid.UUID
) -> bool:
    """
    Delete a course and all associated data (cascade).
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    course_id : uuid.UUID
        Course ID
    
    Returns
    -------
    bool
        True if course was deleted, False if course not found
    """
    course = await get_course(db, course_id)
    if course:
        await db.delete(course)
        await db.commit()
        log.info(f"Deleted course: {course_id}")
        return True
    return False


# ════════════════════════════════════════════════════════════════════════
# LEARNING SESSION OPERATIONS
# ════════════════════════════════════════════════════════════════════════


async def create_learning_session( 
    db: AsyncSession,
    student_id: str,
    course_id: Optional[uuid.UUID] = None,
    language: str = "fr",
    level: str = "lycée"
) -> LearningSession:
    """
    Create a new learning session for a student.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    student_id : str
        Anonymous student identifier (hash or UUID format)
    course_id : uuid.UUID, optional
        Course the student is working on
    language : str, optional
        ISO 639-1 language code (default: "fr")
    level : str, optional
        Student level (default: "lycée")
    
    Returns
    -------
    LearningSession
        Created session with ID populated after flush
    """
    session = LearningSession(
        student_id=student_id,
        course_id=course_id,
        language=language,
        level=level,
    )
    db.add(session)
    await db.flush()
    log.info(f"Created learning session: {session.id} for student {student_id}")
    return session


async def get_session(
    db: AsyncSession,
    session_id: uuid.UUID
) -> Optional[LearningSession]:
    """
    Retrieve a learning session by ID.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    session_id : uuid.UUID
        Session ID
    
    Returns
    -------
    LearningSession or None
        Session object if found, otherwise None
    """
    result = await db.execute(select(LearningSession).where(LearningSession.id == session_id))
    return result.scalar_one_or_none()


async def update_session_state(
    db: AsyncSession,
    session_id: uuid.UUID,
    state: str,
    chapter_index: Optional[int] = None,
    section_index: Optional[int] = None,
    char_position: Optional[int] = None
) -> Optional[LearningSession]:
    """
    Update a session's state and optionally its position within course.
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    session_id : uuid.UUID
        Session ID
    state : str
        New state: "IDLE", "PRESENTING", "LISTENING", "PROCESSING", "RESPONDING"
    chapter_index : int, optional
        Zero-indexed chapter being studied
    section_index : int, optional
        Zero-indexed section within chapter
    char_position : int, optional
        Character position in section content (for resume)
    
    Returns
    -------
    LearningSession or None
        Updated session, or None if not found
    """
    updates = {"state": state, "updated_at": datetime.utcnow()}
    if chapter_index is not None:
        updates["chapter_index"] = chapter_index
    if section_index is not None:
        updates["section_index"] = section_index
    if char_position is not None:
        updates["char_position"] = char_position
    
    stmt = update(LearningSession).where(LearningSession.id == session_id).values(**updates)
    await db.execute(stmt)
    await db.commit()
    
    log.debug(f"Updated session {session_id} state={state}")
    return await get_session(db, session_id)


async def end_session(
    db: AsyncSession,
    session_id: uuid.UUID
) -> None:
    """
    End a learning session (set end_at timestamp and state to IDLE).
    
    Parameters
    ----------
    db : AsyncSession
        Database session
    session_id : uuid.UUID
        Session ID
    """
    await db.execute(
        update(LearningSession)
        .where(LearningSession.id == session_id)
        .values(ended_at=datetime.utcnow(), state="IDLE", updated_at=datetime.utcnow())
    )
    await db.commit()
    log.info(f"Ended session: {session_id}")


# ════════════════════════════════════════════════════════════════════════
# INTERACTION LOGGING
# ════════════════════════════════════════════════════════════════════════


async def log_learning_event(
    db: AsyncSession,
    session_id: uuid.UUID,
    student_id: str,
    course_id: Optional[uuid.UUID],
    event_type: str = "qa",
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    concept: Optional[str] = None,
    action_taken: Optional[str] = None,
    confusion_score: float = 0.0,
    reward: float = 0.0,
    stt_time: float = 0.0,
    llm_time: float = 0.0,
    tts_time: float = 0.0,
    total_time: float = 0.0,
    student_state: Optional[dict] = None,
    event_payload: Optional[dict] = None,
) -> LearningEvent:
    """Persist a rich learning event for later student modeling and analytics."""
    event = LearningEvent(
        session_id=session_id,
        student_id=student_id,
        course_id=course_id,
        event_type=event_type,
        input_text=input_text,
        output_text=output_text,
        concept=concept,
        action_taken=action_taken,
        confusion_score=float(confusion_score or 0.0),
        reward=float(reward or 0.0),
        stt_time=float(stt_time or 0.0),
        llm_time=float(llm_time or 0.0),
        tts_time=float(tts_time or 0.0),
        total_time=float(total_time or 0.0),
        student_state=dict(student_state or {}),
        event_payload=dict(event_payload or {}),
    )
    db.add(event)
    await db.flush()
    log.debug(
        "Logged learning event: %s (confusion=%.2f, reward=%.2f, total=%.2fs)",
        event_type,
        event.confusion_score,
        event.reward,
        event.total_time,
    )
    return event


# ══════════════════════════════════════════════════════════════════════
# INTERACTIONS
# ══════════════════════════════════════════════════════════════════════

async def log_interaction(
    db: AsyncSession,
    session_id: uuid.UUID,
    student_id: str,
    question: str,
    answer: str,
    language: str = "fr",
    stt_time: float = 0.0,
    llm_time: float = 0.0,
    tts_time: float = 0.0,
    total_time: float = 0.0,
    kpi_ok: bool = False,
    course_id: Optional[uuid.UUID] = None,
    interaction_type: str = "qa"
) -> Interaction:
    """Enregistre une interaction."""
    interaction = Interaction(
        session_id=session_id,
        student_id=student_id,
        course_id=course_id,
        type=interaction_type,
        question=question,
        answer=answer,
        language=language,
        stt_time=stt_time,
        llm_time=llm_time,
        tts_time=tts_time,
        total_time=total_time,
        kpi_ok=1 if kpi_ok else 0,
    )
    db.add(interaction)
    await db.flush()
    return interaction


async def get_session_stats(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """Retourne les statistiques d'une session."""
    result = await db.execute(
        select(func.count(Interaction.id))
        .where(Interaction.session_id == session_id)
    )
    total_interactions = result.scalar() or 0

    result = await db.execute(
        select(func.avg(Interaction.total_time))
        .where(Interaction.session_id == session_id)
    )
    avg_time = result.scalar() or 0.0

    result = await db.execute(
        select(func.sum(Interaction.kpi_ok))
        .where(Interaction.session_id == session_id)
    )
    kpi_ok_count = result.scalar() or 0

    return {
        "total_interactions": total_interactions,
        "avg_response_time": round(avg_time, 2),
        "kpi_rate": round(kpi_ok_count / total_interactions * 100, 1) if total_interactions else 0,
    }
# database/models.py
"""
Smart Teacher — SQLAlchemy Models for PostgreSQL.

Database schema with hierarchical structure:
    Course → Chapter → Section → Concept

Supports:
    - Course management with metadata (subject, language, level)
    - Chapter organization with ordering
    - Sections with original course content
    - Key concepts tied to sections
    - Student learning sessions with state tracking
    - Interaction logging for analytics

All models use UUID primary keys and timestamps for audit trails.
"""

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey, DateTime, JSON, UniqueConstraint, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, Mapped

# ════════════════════════════════════════════════════════════════════════
# BASE DECLARATION
# ════════════════════════════════════════════════════════════════════════

Base = declarative_base()


# ════════════════════════════════════════════════════════════════════════
# AUTHENTICATION MODEL
# ════════════════════════════════════════════════════════════════════════


class Student(Base):
    """
    Student model — user account for authentication.
    
    Attributes
    ----------
    id : UUID
        Unique student identifier (auto-generated)
    email : str
        Unique email address
    password_hash : str
        Hashed password (bcrypt)
    first_name : str
        Student's first name
    last_name : str
        Student's last name
    preferred_language : str
        ISO 639-1 language code (e.g., "fr", "en")
    account_level : str
        Account level (e.g., "student", "teacher", "admin")
    is_active : bool
        Account active status
    created_at : datetime
        Account creation timestamp (UTC)
    updated_at : datetime
        Last update timestamp (UTC)
    """

    __tablename__ = "students"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=True)
    first_name = Column(String(100), nullable=False, default="Utilisateur")
    last_name = Column(String(100), nullable=True)
    preferred_language = Column(String(5), nullable=False, default="fr")
    # Academic level used to adapt LLM prompts (collège / lycée / université).
    # Drives vocabulary level and explanation depth.
    student_level = Column(String(20), nullable=False, default="lycée")
    account_level = Column(String(20), nullable=False, default="student")
    is_active = Column(Integer, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# ════════════════════════════════════════════════════════════════════════
# COURSE HIERARCHY MODELS
# ════════════════════════════════════════════════════════════════════════


class Course(Base):
    """
    Course model — represents a complete course offering.
    
    Attributes
    ----------
    id : UUID
        Unique identifier (auto-generated)
    title : str
        Course title (e.g., "Fundamentals of Statistics")
    subject : str
        Subject area (e.g., "data_science", "mathematics", "general")
    language : str
        ISO 639-1 language code (e.g., "fr", "en")
    level : str
        Course level (e.g., "lycée", "master", "phd")
    description : str, optional
        Long-form description of course content
    file_path : str, optional
        Path to source PDF or course file
    created_at : datetime
        Course creation timestamp (UTC)
    updated_at : datetime
        Last update timestamp (UTC)
    chapters : List[Chapter]
        One-to-many relationship with Chapter
    """

    __tablename__ = "courses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    domain = Column(String(100), nullable=False, default="general", index=True)  # 🎯 Domaine (informatique, general, etc.)
    subject = Column(String(255), nullable=False, default="general", index=True)
    language = Column(String(5), nullable=False, default="fr")
    level = Column(String(20), nullable=False, default="lycée")
    description = Column(Text)
    file_path = Column(String(500))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relations
    chapters: Mapped[List["Chapter"]] = relationship(
        "Chapter",
        back_populates="course",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Chapter.order"
    )


class Chapter(Base):
    """
    Chapter model — represents a chapter within a course.
    
    Attributes
    ----------
    id : UUID
        Unique identifier (auto-generated)
    course_id : UUID
        Foreign key to parent Course
    title : str
        Chapter title (e.g., "Chapter 1: Introduction")
    order : int
        Sequential order within course (0-indexed)
    summary : str, optional
        Brief summary or learning objectives
    created_at : datetime
        Creation timestamp (UTC)
    course : Course
        Back-reference to parent Course
    sections : List[Section]
        One-to-many relationship with Section
    """

    __tablename__ = "chapters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(
        UUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title = Column(String(255), nullable=False, index=True)
    order = Column(Integer, nullable=False, default=0)
    summary = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relations
    course: Mapped[Course] = relationship("Course", back_populates="chapters")
    sections: Mapped[List["Section"]] = relationship(
        "Section",
        back_populates="chapter",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Section.order"
    )


class Section(Base):
    """
    Section model — represents a section within a chapter.
    
    Attributes
    ----------
    id : UUID
        Unique identifier (auto-generated)
    chapter_id : UUID
        Foreign key to parent Chapter
    title : str
        Section title
    order : int
        Sequential order within chapter (0-indexed)
    content : str
        Original course text (exact PDF content)
    duration_s : int
        Estimated reading/teaching time in seconds
    image_urls : list[str]
        URLs to associated slide images (JSON array)
    created_at : datetime
        Creation timestamp (UTC)
    chapter : Chapter
        Back-reference to parent Chapter
    concepts : List[Concept]
        One-to-many relationship with Concept
    """

    __tablename__ = "sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title = Column(String(255), nullable=False, index=True)
    order = Column(Integer, nullable=False, default=0)
    content = Column(Text)  # Original course content from PDF
    image_url = Column(String(500))  # 🎨 PNG slide path (e.g., /media/slides/general/mon_cours/chapter_1/page_001.png)
    duration_s = Column(Integer, nullable=False, default=120)
    image_urls = Column(JSON, nullable=False, default=list)  # Array of slide image URLs
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relations
    chapter: Mapped[Chapter] = relationship("Chapter", back_populates="sections")
    concepts: Mapped[List["Concept"]] = relationship(
        "Concept",
        back_populates="section",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Concept.created_at"
    )


class Concept(Base):
    """
    Concept model — represents a key concept within a section.
    
    Attributes
    ----------
    id : UUID
        Unique identifier (auto-generated)
    section_id : UUID
        Foreign key to parent Section
    term : str
        The concept term or name (e.g., "k-means clustering")
    definition : str, optional
        Formal definition or explanation
    example : str, optional
        Concrete example or use case
    concept_type : str
        Type of concept: "definition", "formula", "theorem", "example"
    created_at : datetime
        Creation timestamp (UTC)
    section : Section
        Back-reference to parent Section
    """

    __tablename__ = "concepts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    section_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sections.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    term = Column(String(100), nullable=False, index=True)
    definition = Column(Text)
    example = Column(Text)
    concept_type = Column(String(20), nullable=False, default="definition")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relations
    section: Mapped[Section] = relationship("Section", back_populates="concepts")


# ════════════════════════════════════════════════════════════════════════
# SESSION & INTERACTION MODELS
# ════════════════════════════════════════════════════════════════════════


class LearningSession(Base):
    """
    Learning session model — tracks a student's learning session.
    
    Attributes
    ----------
    id : UUID
        Unique session identifier (auto-generated)
    student_id : str
        Anonymous student identifier (hash or UUID)
    course_id : UUID, optional
        Foreign key to enrolled Course
    language : str
        ISO 639-1 language code (e.g., "fr", "en")
    level : str
        Student level (e.g., "lycée", "master")
    state : str
        Current state: "IDLE", "PRESENTING", "LISTENING", "PROCESSING", "RESPONDING"
    chapter_index : int
        Zero-indexed chapter currently being studied
    section_index : int
        Zero-indexed section within current chapter
    char_position : int
        Character position within current section (for resume capability)
    started_at : datetime
        Session start time (UTC)
    ended_at : datetime, optional
        Session end time (UTC) — null if still active
    updated_at : datetime
        Last activity timestamp (UTC)
    """

    __tablename__ = "learning_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    course_id = Column(
        UUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="SET NULL"),
        index=True
    )
    language = Column(String(5), nullable=False, default="fr")
    level = Column(String(20), nullable=False, default="lycée")
    state = Column(String(20), nullable=False, default="IDLE")
    chapter_index = Column(Integer, nullable=False, default=0)
    section_index = Column(Integer, nullable=False, default=0)
    char_position = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Interaction(Base):
    """
    Interaction model — logs each student-teacher exchange.
    
    Attributes
    ----------
    id : UUID
        Unique interaction identifier (auto-generated)
    session_id : UUID
        Foreign key to parent LearningSession
    student_id : str
        Anonymous student identifier (hash or UUID)
    course_id : UUID, optional
        Foreign key to Course
    type : str
        Interaction type: "qa" (question/answer), "interrupt", "navigation"
    question : str, optional
        Student's question (from STT)
    answer : str, optional
        Teacher's response (LLM output)
    language : str
        Language used in interaction (ISO 639-1)
    stt_time : float
        Speech-to-text processing time in seconds
    llm_time : float
        Language model inference time in seconds
    tts_time : float
        Text-to-speech generation time in seconds
    total_time : float
        Total wall-clock time in seconds
    kpi_ok : int
        Binary flag: 1 if response met performance KPI, 0 otherwise
    created_at : datetime
        Interaction timestamp (UTC)
    """

    __tablename__ = "interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    student_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    course_id = Column(
        UUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="SET NULL"),
        index=True
    )
    type = Column(String(20), nullable=False, default="qa", index=True)
    question = Column(Text)
    answer = Column(Text)
    language = Column(String(5), nullable=False, default="fr")
    stt_time = Column(Float, nullable=False, default=0.0)
    llm_time = Column(Float, nullable=False, default=0.0)
    tts_time = Column(Float, nullable=False, default=0.0)
    total_time = Column(Float, nullable=False, default=0.0)
    kpi_ok = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class LearningEvent(Base):
    """Detailed pedagogical event log for model readiness and offline training."""

    __tablename__ = "learning_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    student_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    course_id = Column(
        UUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="SET NULL"),
        index=True,
    )
    event_type = Column(String(30), nullable=False, default="qa", index=True)
    input_text = Column(Text)
    output_text = Column(Text)
    concept = Column(String(255), index=True)
    action_taken = Column(String(100), index=True)
    confusion_score = Column(Float, nullable=False, default=0.0)
    reward = Column(Float, nullable=False, default=0.0)
    stt_time = Column(Float, nullable=False, default=0.0)
    llm_time = Column(Float, nullable=False, default=0.0)
    tts_time = Column(Float, nullable=False, default=0.0)
    total_time = Column(Float, nullable=False, default=0.0)
    student_state = Column(JSON, nullable=False, default=dict)
    event_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


# ════════════════════════════════════════════════════════════════════════
# ADVANCED FEATURES MODELS
# ════════════════════════════════════════════════════════════════════════


class StudentProfile(Base):
    """Student profile with preferences and learning metadata.

    Scoped per (student, course): the same student can have very different
    pace / confusion rate / preferred depth across subjects (e.g. fast on
    a CS course they already know, slow on a new linguistics course).
    A row with ``course_id IS NULL`` is the cross-course default — used
    as a fallback before any course-specific signal has been observed.
    """
    __tablename__ = "student_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    # Nullable on purpose: the row with course_id NULL is the per-student
    # default, and rows with a real course_id override it for that course.
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"), nullable=True, index=True)
    learning_style = Column(String(50), default="visual")          # visual | auditory | kinesthetic | reading
    preferred_difficulty = Column(String(20), default="intermediate")
    topics_of_interest = Column(JSON, default=list)
    total_xp = Column(Integer, default=0)
    streak_days = Column(Integer, default=0)
    last_activity = Column(DateTime, default=datetime.utcnow)
    # Personnalisation cognitive (Sprint 6)
    pace = Column(String(20), default="normal")                   # slow | normal | fast (auto-detecte sur historique)
    avg_response_time_s = Column(Float, default=0.0)               # moyenne mobile temps de reponse practice
    confusion_rate = Column(Float, default=0.0)                    # ratio confusion / total turns derniers 30j
    preferred_explanation_depth = Column(String(20), default="balanced")  # concise | balanced | detailed
    # Free-form JSON bag for personalization sub-systems that don't warrant
    # their own column: VARK self-report, Bayesian learning-style posterior
    # (alpha, n_observations), and any future signal that's read/written as
    # a single dict. Producers of new keys must namespace them so they don't
    # collide. Read by pedagogy.personalization.learning_style.bayes and
    # routes/student.py:submit_vark_responses.
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("student_id", "course_id", name="uq_student_course_profile"),
    )


class StudentMistake(Base):
    """Track student mistakes for adaptive learning."""
    __tablename__ = "student_mistakes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    concept_id = Column(UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=True)
    mistake_type = Column(String(50), nullable=False)  # typo, logic, understanding, etc.
    context = Column(Text)
    frequency = Column(Integer, default=1)
    last_occurred = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class RAGChunk(Base):
    """Vector database chunks for RAG retrieval."""
    __tablename__ = "rag_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"), nullable=True)
    section_id = Column(UUID(as_uuid=True), ForeignKey("sections.id", ondelete="CASCADE"), nullable=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    vector_id = Column(String(100))  # Qdrant vector ID
    chunk_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """System-level logging and monitoring."""
    __tablename__ = "system_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    level = Column(String(20), default="INFO")  # INFO, WARNING, ERROR, DEBUG
    module = Column(String(100))
    message = Column(Text)
    context = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class PerformanceMetric(Base):
    """Performance metrics and KPIs."""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float)
    student_id = Column(UUID(as_uuid=True), index=True)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="SET NULL"))
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)


class IngestedAssetDB(Base):
    """Asset extrait d'un PDF lors de l'ingestion intelligente.

    Persiste : images embarquees + OCR, tables, figure captions, titres.
    Permet de retrouver les schemas/diagrammes d'un cours et leur contexte.
    """
    __tablename__ = "ingested_asset"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id    = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"), nullable=True, index=True)
    asset_type   = Column(String(20), nullable=False, index=True)   # text|image|table|caption|title
    page_num     = Column(Integer, default=0, index=True)
    text         = Column(Text, default="")                          # contenu textuel ou OCR
    image_path   = Column(String(500), default="")                   # chemin disque relatif
    image_index_in_page = Column(Integer, default=0)
    asset_metadata = Column(JSON, default=dict)                      # metadata libre
    created_at   = Column(DateTime, default=datetime.utcnow)


# ── ConceptKG / ConceptChunkLink / ConceptCooccurrence retires ──────────
# Stage 3 unification : les concepts vivent maintenant dans le KnowledgeGraph
# en memoire (pedagogy/knowledge_graph/graph.py). Plus de persistance
# concept-level dedie. PracticeQuestion / ReviewQueue identifient les
# concepts par leur `name` (string snake_case = ConceptInfo.name) au lieu
# d'un FK UUID.


class ReviewQueue(Base):
    """File de revisions FSRS par (student, concept_name).

    Stage 3 : `concept_id` (UUID FK -> concept_kg.id) est devenu `concept_name`
    (String, = ConceptInfo.name dans le KnowledgeGraph). Plus de FK : les
    concepts sont in-memory, pas en DB.
    """
    __tablename__ = "review_queue"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id   = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    concept_name = Column(String(80), nullable=False, index=True)
    # FSRS state (encoded JSON pour eviter explosion de colonnes)
    fsrs_state   = Column(JSON, default=dict)              # serialise du Card FSRS
    due          = Column(DateTime, default=datetime.utcnow, index=True)
    last_review  = Column(DateTime, default=datetime.utcnow)
    review_count = Column(Integer, default=0)
    lapse_count  = Column(Integer, default=0)               # combien de "Again"
    state        = Column(String(20), default="learning")   # learning | review | relearning
    stability    = Column(Float, default=0.0)
    difficulty   = Column(Float, default=0.5)               # FSRS difficulty [0..1]
    created_at   = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("student_id", "concept_name", name="uq_review_student_concept"),
        Index("ix_review_due", "student_id", "due"),
    )


class PracticeQuestion(Base):
    """Question d'entrainement auto-generee par concept_name (Stage 3 schema)."""
    __tablename__ = "practice_question"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    concept_name  = Column(String(80), nullable=False, index=True)
    question      = Column(Text, nullable=False)
    answer        = Column(Text, nullable=False)
    difficulty    = Column(String(10), default="medium")   # easy | medium | hard
    hints         = Column(JSON, default=list)             # list[str] : 3-tier hints
    language      = Column(String(5), default="fr")
    created_at    = Column(DateTime, default=datetime.utcnow)


class PracticeAttempt(Base):
    """Journal des tentatives d'un eleve sur une question (Sprint 1)."""
    __tablename__ = "practice_attempt"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id    = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id   = Column(UUID(as_uuid=True), ForeignKey("practice_question.id", ondelete="CASCADE"), nullable=False, index=True)
    answer_text   = Column(Text)
    is_correct    = Column(Boolean, default=False)
    hints_used    = Column(Integer, default=0)             # 0..3
    time_taken_s  = Column(Float, default=0.0)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)


class StudentMastery(Base):
    """Persistent mastery tracking per (student, course, idea_id).

    idea_id correspond a la metadata RAG : md5[:12] stable cross-ingestion.
    Score continu [0, 1] :
      - +0.10 sur reponse propre (pas de confusion)
      - -0.15 sur confusion detectee
      - mastered_at est set quand score >= 0.85 (et attempts >= 3)
    """
    __tablename__ = "student_mastery"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id   = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id    = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"), nullable=True, index=True)
    idea_id      = Column(String(16), nullable=False, index=True)
    score        = Column(Float, default=0.5, nullable=False)
    attempts     = Column(Integer, default=0, nullable=False)
    confusions   = Column(Integer, default=0, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    mastered_at  = Column(DateTime, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("student_id", "course_id", "idea_id", name="uq_student_course_idea"),
        Index("ix_student_mastery_score", "student_id", "score"),
    )


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 PERSISTENCE — long-term storage for analysis & defense
# ════════════════════════════════════════════════════════════════════════
# These three tables capture data that previously lived only in Redis (TTL
# 24h-90j) or wasn't recorded at all. Without DB persistence, a thesis-scale
# study could lose its raw observations before defense — and every
# retroactive "what really happened" question becomes unanswerable. Each
# model below is the canonical store ; Redis is now an "in-flight buffer"
# that gets flushed to these tables.


class LearningGainTest(Base):
    """Pre-test / post-test result for a (student, course) couple.

    This is THE table that lets us compute Hake's normalized learning
    gain ``g = (post - pre) / (1 - pre)`` — the standard pedagogy KPI
    for "did this teaching method actually work ?" (Hake 1998, *Am. J.
    Phys.*). Without this, no defensible claim can be made about Smart
    Teacher's pedagogical efficacy.

    ``test_type`` is "pretest" or "posttest". A complete experiment is
    one of each per (student, course) ; the analysis pairs them up by
    that key.

    Questions and responses are stored as JSON to stay schema-flexible :
    a 5-question MCQ pre-test and a 10-question short-answer post-test
    can coexist without table migration.
    """
    __tablename__ = "learning_gain_tests"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id   = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    course_id    = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    test_type    = Column(String(20), nullable=False)   # "pretest" | "posttest"
    questions    = Column(JSON, nullable=False)          # [{q, choices, correct_idx}, …]
    responses    = Column(JSON, nullable=False)          # [{q_idx, chosen_idx, correct}, …]
    score        = Column(Float, nullable=False)        # ratio in [0, 1]
    duration_s   = Column(Integer, nullable=True)
    taken_at     = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_lgt_student_course_type", "student_id", "course_id", "test_type"),
    )


class ConfusionEvent(Base):
    """One detailed confusion event with context.

    Complements ``student_mistakes.confusions`` (a counter) by recording
    *when, on what slide, with what trigger text, by which detector*
    each confusion fired. This unlocks analyses like "concept X causes
    80 % of confusions on this course" or "the prosody-based detector
    has a 30 % false-positive rate vs the SIGHT model".

    ``trigger_text`` is the student utterance that fired the detector
    (truncated to 500 chars for storage hygiene).
    """
    __tablename__ = "confusion_events"

    id                    = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id            = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
                                   nullable=False, index=True)
    course_id             = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"),
                                   nullable=True, index=True)
    session_id            = Column(String(120), nullable=True, index=True)
    concept_id            = Column(UUID(as_uuid=True), nullable=True, index=True)
    slide_idx             = Column(Integer, nullable=True)
    trigger_text          = Column(Text, nullable=True)
    source                = Column(String(30), nullable=False)  # "sight_model" | "prosody" | "keyword" | "llm"
    score                 = Column(Float, nullable=False)
    resolved              = Column(Boolean, default=False, nullable=False)
    resolution_strategy   = Column(String(60), nullable=True)
    created_at            = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_confusion_concept_resolved", "concept_id", "resolved"),
    )


class VarkResponse(Base):
    """Raw VARK questionnaire submissions (one row per submission).

    The Bayesian posterior in ``student_profiles.preferences['vark.*']``
    is *derived* from these rows. Storing the raw responses lets us :
      - recompute posteriors retroactively if the scoring rule changes,
      - audit "did this student's VARK score really reflect their
        questionnaire answers ?" (rare but required for IRB compliance),
      - build aggregate analyses across cohorts (e.g. "VARK distribution
        for engineering students").

    ``responses`` is a list of items shaped like
    ``[{"question_id": int, "choices_picked": ["V", "K"]}, ...]``.
    """
    __tablename__ = "vark_responses"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id  = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    course_id   = Column(UUID(as_uuid=True), ForeignKey("courses.id", ondelete="CASCADE"),
                         nullable=True, index=True)
    responses   = Column(JSON, nullable=False)
    taken_at    = Column(DateTime, default=datetime.utcnow, nullable=False)
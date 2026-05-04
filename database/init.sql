-- ============================================================================
-- SMART TEACHER - SCHÉMA BASE DE DONNÉES PostgreSQL
-- ============================================================================
-- Créé: 2026-03-30
-- Auteur: Smart Teacher Team
-- Description: Structure complète de la base de données pour la plateforme
-- ============================================================================

-- ============================================================================
-- 1. TABLES PRINCIPAUX - COURS ET CONTENU
-- ============================================================================

-- Table: courses
-- Description: Liste des cours disponibles
CREATE TABLE IF NOT EXISTS courses (
    course_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    subject VARCHAR(100),
    description TEXT,
    language VARCHAR(10) DEFAULT 'fr',  -- fr, ar, en
    level INTEGER DEFAULT 1,  -- 1 (beginner), 2 (intermediate), 3 (advanced)
    file_path VARCHAR(500),
    total_duration_minutes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_courses_subject ON courses(subject);
CREATE INDEX idx_courses_language ON courses(language);
CREATE INDEX idx_courses_is_active ON courses(is_active);


-- Table: chapters
-- Description: Chapitres au sein d'un cours
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id SERIAL PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    summary TEXT,
    estimated_duration_minutes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chapters_course_id ON chapters(course_id);
CREATE INDEX idx_chapters_chapter_number ON chapters(chapter_number);

-- Constraint: Unicité course_id + chapter_number
ALTER TABLE chapters ADD CONSTRAINT unique_course_chapter 
    UNIQUE (course_id, chapter_number);


-- Table: sections
-- Description: Sections au sein d'un chapitre
CREATE TABLE IF NOT EXISTS sections (
    section_id SERIAL PRIMARY KEY,
    chapter_id INTEGER NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    section_number INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    original_pdf_content TEXT,
    estimated_duration_minutes INTEGER,
    slide_image_url VARCHAR(500),
    character_offset INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sections_chapter_id ON sections(chapter_id);
CREATE INDEX idx_sections_section_number ON sections(section_number);


-- Table: concepts
-- Description: Termes clés, définitions, concepts au sein d'une section
CREATE TABLE IF NOT EXISTS concepts (
    concept_id SERIAL PRIMARY KEY,
    section_id INTEGER NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    term VARCHAR(255) NOT NULL,
    definition TEXT NOT NULL,
    example TEXT,
    concept_type VARCHAR(50),  -- definition, algorithm, metric, theorem, etc.
    difficulty_level INTEGER DEFAULT 1,  -- 1-5
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_concepts_section_id ON concepts(section_id);
CREATE INDEX idx_concepts_term ON concepts(term);
CREATE INDEX idx_concepts_concept_type ON concepts(concept_type);


-- ============================================================================
-- 2. TABLES ÉTUDIANT ET PROFIL
-- ============================================================================

-- Table: students
-- Description: Informations de base des étudiants
CREATE TABLE IF NOT EXISTS students (
    student_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'fr',
    timezone VARCHAR(50) DEFAULT 'UTC',
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_students_email ON students(email);
CREATE INDEX idx_students_is_active ON students(is_active);


-- Table: student_profiles
-- Description: Profil d'apprentissage personnalisé de chaque étudiant
CREATE TABLE IF NOT EXISTS student_profiles (
    profile_id SERIAL PRIMARY KEY,
    student_id INTEGER UNIQUE NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    learning_level VARCHAR(50) DEFAULT 'beginner',  -- beginner, intermediate, advanced
    accuracy FLOAT DEFAULT 0.0,  -- 0-1 (percentage de réponses correctes)
    total_answers INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    learning_style VARCHAR(100),  -- visual, auditory, kinesthetic, etc.
    preferred_difficulty INTEGER DEFAULT 1,  -- 1-5
    learning_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_student_profiles_student_id ON student_profiles(student_id);
CREATE INDEX idx_student_profiles_learning_level ON student_profiles(learning_level);


-- Table: student_mistakes
-- Description: Suivi des erreurs par sujet pour identifier les faiblesses
CREATE TABLE IF NOT EXISTS student_mistakes (
    mistake_id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    concept_id INTEGER REFERENCES concepts(concept_id) ON DELETE SET NULL,
    topic VARCHAR(255),
    error_count INTEGER DEFAULT 1,
    last_error TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_student_mistakes_student_id ON student_mistakes(student_id);
CREATE INDEX idx_student_mistakes_topic ON student_mistakes(topic);


-- ============================================================================
-- 3. TABLES SESSION ET INTERACTION
-- ============================================================================

-- Table: learning_sessions
-- Description: Sessions d'apprentissage de chaque étudiant
CREATE TABLE IF NOT EXISTS learning_sessions (
    session_id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(course_id),
    chapter_id INTEGER REFERENCES chapters(chapter_id),
    section_id INTEGER REFERENCES sections(section_id),
    session_state VARCHAR(50) DEFAULT 'IDLE',  -- IDLE, PRESENTING, LISTENING, PROCESSING, RESPONDING
    character_offset INTEGER DEFAULT 0,
    total_duration_seconds INTEGER DEFAULT 0,
    total_interactions INTEGER DEFAULT 0,
    session_accuracy FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed BOOLEAN DEFAULT FALSE,
    ended_at TIMESTAMP
);

CREATE INDEX idx_learning_sessions_student_id ON learning_sessions(student_id);
CREATE INDEX idx_learning_sessions_course_id ON learning_sessions(course_id);
CREATE INDEX idx_learning_sessions_session_state ON learning_sessions(session_state);
CREATE INDEX idx_learning_sessions_created_at ON learning_sessions(created_at);


-- Table: interactions
-- Description: Chaque échange étudiant-IA détaillé
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES learning_sessions(session_id) ON DELETE CASCADE,
    student_id INTEGER NOT NULL REFERENCES students(student_id),
    concept_id INTEGER REFERENCES concepts(concept_id),
    stt_input TEXT,  -- Texte transcrit (parole de l'étudiant)
    llm_output TEXT,  -- Réponse du LLM
    stt_confidence FLOAT,  -- 0-1 confiance Whisper
    is_correct BOOLEAN,  -- L'étudiant a-t-il bien répondu?
    response_time_ms INTEGER,  -- Temps entre question et réponse
    latency_stt_ms INTEGER,  -- Durée STT
    latency_llm_ms INTEGER,  -- Durée LLM
    latency_tts_ms INTEGER,  -- Durée TTS
    latency_total_ms INTEGER,  -- Latence totale
    language VARCHAR(10),
    subject_tag VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_interactions_session_id ON interactions(session_id);
CREATE INDEX idx_interactions_student_id ON interactions(student_id);
CREATE INDEX idx_interactions_concept_id ON interactions(concept_id);
CREATE INDEX idx_interactions_created_at ON interactions(created_at);
CREATE INDEX idx_interactions_is_correct ON interactions(is_correct);


-- ============================================================================
-- 4. TABLES RAG ET INDEXATION
-- ============================================================================

-- Table: rag_chunks
-- Description: Chunks indexés pour RAG (Qdrant)
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id SERIAL PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(course_id),
    section_id INTEGER REFERENCES sections(section_id),
    concept_id INTEGER REFERENCES concepts(concept_id),
    content TEXT NOT NULL,
    embedding_vector_id VARCHAR(255),  -- ID dans Qdrant
    chunk_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rag_chunks_course_id ON rag_chunks(course_id);
CREATE INDEX idx_rag_chunks_section_id ON rag_chunks(section_id);


-- ============================================================================
-- 5. TABLES LOGS ET MÉTRIQUES
-- ============================================================================

-- Table: system_logs
-- Description: Logs structurés du système
CREATE TABLE IF NOT EXISTS system_logs (
    log_id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(student_id),
    session_id INTEGER REFERENCES learning_sessions(session_id),
    log_level VARCHAR(20),  -- INFO, WARNING, ERROR, DEBUG
    component VARCHAR(100),  -- STT, LLM, TTS, RAG, etc.
    message TEXT,
    metadata JSONB,  -- Données additionnelles JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_system_logs_student_id ON system_logs(student_id);
CREATE INDEX idx_system_logs_log_level ON system_logs(log_level);
CREATE INDEX idx_system_logs_component ON system_logs(component);
CREATE INDEX idx_system_logs_created_at ON system_logs(created_at);


-- Table: performance_metrics
-- Description: Métriques de performance globales
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(100),  -- avg_latency, error_rate, cache_hit_rate, etc.
    metric_value FLOAT,
    unit VARCHAR(50),  -- ms, %, count, etc.
    details JSONB
);

CREATE INDEX idx_performance_metrics_metric_name ON performance_metrics(metric_name);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);


-- ============================================================================
-- 6. TABLES CACHE ET ÉTAT
-- ============================================================================

-- Table: llm_cache
-- Description: Cache des réponses LLM fréquentes
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_id SERIAL PRIMARY KEY,
    question_hash VARCHAR(64) UNIQUE,  -- MD5 hash de (question + context + level)
    question_text TEXT,
    context VARCHAR(255),  -- Contexte RAG (section)
    level VARCHAR(50),  -- beginner, intermediate, advanced
    response_text TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    hit_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP
);

CREATE INDEX idx_llm_cache_question_hash ON llm_cache(question_hash);
CREATE INDEX idx_llm_cache_expires_at ON llm_cache(expires_at);


-- ============================================================================
-- 9. VIEWS UTILES
-- ============================================================================

-- Vue: Student Statistics
CREATE OR REPLACE VIEW student_statistics AS
SELECT 
    s.student_id,
    s.email,
    sp.learning_level,
    sp.accuracy,
    sp.total_answers,
    sp.correct_answers,
    g.total_points,
    g.level,
    g.daily_streak,
    COUNT(DISTINCT ls.session_id) as total_sessions,
    AVG(i.latency_total_ms) as avg_response_time_ms,
    COUNT(CASE WHEN i.is_correct THEN 1 END) as correct_interactions,
    s.last_login
FROM students s
LEFT JOIN student_profiles sp ON s.student_id = sp.student_id
LEFT JOIN gamification g ON s.student_id = g.student_id
LEFT JOIN learning_sessions ls ON s.student_id = ls.student_id
LEFT JOIN interactions i ON ls.session_id = i.session_id
GROUP BY s.student_id, s.email, sp.learning_level, sp.accuracy, sp.total_answers, 
         sp.correct_answers, g.total_points, g.level, g.daily_streak, s.last_login;


-- Vue: Course Performance
CREATE OR REPLACE VIEW course_performance AS
SELECT 
    c.course_id,
    c.title,
    c.subject,
    COUNT(DISTINCT ls.student_id) as students_enrolled,
    COUNT(DISTINCT ls.session_id) as total_sessions,
    AVG(ls.session_accuracy) as avg_accuracy,
    AVG(ls.total_duration_seconds) as avg_session_duration,
    COUNT(DISTINCT CASE WHEN ls.completed THEN ls.session_id END) as completed_sessions
FROM courses c
LEFT JOIN learning_sessions ls ON c.course_id = ls.course_id
GROUP BY c.course_id, c.title, c.subject;


-- Vue: Concept Difficulty
CREATE OR REPLACE VIEW concept_difficulty AS
SELECT 
    c.concept_id,
    c.term,
    c.concept_type,
    COUNT(i.interaction_id) as total_attempts,
    COUNT(CASE WHEN i.is_correct THEN 1 END) as correct_attempts,
    ROUND(100.0 * COUNT(CASE WHEN i.is_correct THEN 1 END) / 
          NULLIF(COUNT(i.interaction_id), 0), 2) as success_rate,
    AVG(i.response_time_ms) as avg_response_time_ms
FROM concepts c
LEFT JOIN interactions i ON c.concept_id = i.concept_id
GROUP BY c.concept_id, c.term, c.concept_type;


-- ============================================================================
-- 10. FONCTIONS UTILES
-- ============================================================================

-- Fonction: Update student level based on accuracy
CREATE OR REPLACE FUNCTION update_student_level()
RETURNS TRIGGER AS $$
BEGIN
    -- Met à jour le level d'apprentissage basé sur la précision
    IF NEW.accuracy >= 0.85 THEN
        NEW.learning_level = 'advanced';
    ELSIF NEW.accuracy >= 0.70 THEN
        NEW.learning_level = 'intermediate';
    ELSE
        NEW.learning_level = 'beginner';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_student_level
BEFORE UPDATE ON student_profiles
FOR EACH ROW
EXECUTE FUNCTION update_student_level();


-- Fonction: Update gamification level
CREATE OR REPLACE FUNCTION update_gamification_level()
RETURNS TRIGGER AS $$
BEGIN
    NEW.level = (NEW.total_points / 100) + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_gamification_level
BEFORE UPDATE ON gamification
FOR EACH ROW
EXECUTE FUNCTION update_gamification_level();


-- ============================================================================
-- 11. EXTENSION POUR JSON (optionnel mais recommandé)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- ============================================================================
-- DONNÉES INITIALES (Optionnel)
-- ============================================================================

-- Insérer un cours de démonstration
INSERT INTO courses (title, subject, description, language, level)
VALUES (
    'Data Mining Fundamentals',
    'Data Mining',
    'Introduction à l''exploitation des données et la fouille de données',
    'fr',
    1
) ON CONFLICT DO NOTHING;


-- ============================================================================
-- FIN DU SCHÉMA
-- ============================================================================
-- Créated on: 2026-03-30
-- Database: smart_teacher_db
-- PostgreSQL Version: 15+
-- ============================================================================

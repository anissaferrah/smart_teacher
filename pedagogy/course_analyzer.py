"""
Smart Teacher — Course Analyzer
Analyse le cours à l'ingestion pour comprendre sa structure AVANT le LLM.
"""

import logging
from typing import Optional
from langdetect import detect

log = logging.getLogger("SmartTeacher.CourseAnalyzer")


class CourseAnalyzer:
    """
    Analyse un cours AVANT de le servir aux étudiants.
    Détecte : langue, niveau, structure, topics.

    In-memory cache: ``analyze()`` is called every time a student
    opens the first slide of a course (see services/course_slides.py),
    so without caching it re-runs full text extraction + langdetect +
    topic extraction on every navigation. The cache key is a stable
    hash of the course's title + chapter titles, so re-uploading the
    same content key reuses the analysis.
    """

    def __init__(self) -> None:
        # course_hash → analysis dict
        self._cache: dict[str, dict] = {}

    @staticmethod
    def _hash_course(course_dict: dict) -> str:
        import hashlib
        title = str(course_dict.get("title", ""))
        domain = str(course_dict.get("domain", ""))
        chapters = course_dict.get("chapters") or []
        ch_titles = "|".join(str(ch.get("title", "")) for ch in chapters)
        # Number of sections per chapter — cheap proxy for "did the
        # course content change?". Avoids hashing the full content
        # which would defeat the cache on minor edits.
        sec_counts = "|".join(
            str(len(ch.get("sections") or [])) for ch in chapters
        )
        raw = f"{title}::{domain}::{ch_titles}::{sec_counts}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def analyze(self, course_dict: dict) -> dict:
        """
        Analyse complète du cours.
        
        Args:
            course_dict: {
                "title": "...",
                "domain": "informatique",
                "chapters": [
                    {"title": "Ch1", "sections": [{"content": "..."}]}
                ]
            }
        
        Returns:
            {
                "language": "fr",
                "level": "lycée",  # détecté automatiquement
                "domain": "informatique",
                "topics": ["TALN", "NLP", "..."],
                "structure_analysis": {...},
                "summary": "..."  # résumé court du cours
            }
        """
        # ── In-memory cache check ──
        # Stops the analyzer from re-running its full pipeline (langdetect,
        # topic extraction, summary) on every navigation back to slide 1.
        course_hash = self._hash_course(course_dict)
        cached = self._cache.get(course_hash)
        if cached is not None:
            return cached

        try:
            # 🔍 1. Détecter la langue
            all_text = self._extract_all_text(course_dict)
            language = self._detect_language(all_text)
            
            # 📊 2. Analyser la structure
            structure = self._analyze_structure(course_dict)
            
            # 🎓 3. Détecter le niveau
            level = self._detect_level(all_text, structure)
            
            # 🏷️ 4. Extraire les topics
            topics = self._extract_topics(all_text)
            
            # 📝 5. Générer un résumé court
            summary = self._generate_summary(course_dict, topics, level)
            
            result = {
                "language": language,
                "level": level,
                "domain": course_dict.get("domain", "general"),
                "topics": topics,
                "structure": {
                    "num_chapters": structure["num_chapters"],
                    "total_sections": structure["total_sections"],
                    "avg_content_length": structure["avg_content_length"],
                },
                "summary": summary,
            }
            
            log.info(
                f"✅ Course analyzed: lang={language} level={level} "
                f"topics={len(topics)} chapters={structure['num_chapters']}"
            )
            # Cache the result so subsequent calls for the same course
            # return instantly. Cache is unbounded but keys are 16-char
            # hashes so memory cost is negligible (a typical Smart
            # Teacher install has 1-100 courses).
            self._cache[course_hash] = result
            return result
            
        except Exception as e:
            log.error(f"❌ Course analysis error: {e}")
            return {
                "language": "en",
                "level": "lycée",
                "domain": "general",
                "topics": [],
                "structure": {"num_chapters": 0, "total_sections": 0},
                "summary": "Cours indéterminé",
            }

    def _extract_all_text(self, course_dict: dict) -> str:
        """Extraire tout le texte du cours."""
        texts = []
        
        if "title" in course_dict:
            texts.append(course_dict["title"])
        
        if "chapters" in course_dict:
            for chapter in course_dict["chapters"]:
                if "title" in chapter:
                    texts.append(chapter["title"])
                if "sections" in chapter:
                    for section in chapter["sections"]:
                        if "title" in section:
                            texts.append(section["title"])
                        if "content" in section:
                            texts.append(section["content"][:500])  # Limiter
        
        return "\n".join(texts)

    def _detect_language(self, text: str) -> str:
        """Détecter la langue du cours."""
        try:
            if not text or len(text) < 10:
                return "en"
            lang = detect(text)
            return "fr" if lang.startswith("fr") else "en"
        except Exception:
            return "en"

    def _analyze_structure(self, course_dict: dict) -> dict:
        """Analyser la structure (nombre chapitres, sections, etc)."""
        num_chapters = 0
        total_sections = 0
        total_chars = 0
        
        if "chapters" in course_dict:
            num_chapters = len(course_dict["chapters"])
            for chapter in course_dict["chapters"]:
                if "sections" in chapter:
                    total_sections += len(chapter["sections"])
                    for sec in chapter["sections"]:
                        if "content" in sec:
                            total_chars += len(sec["content"])
        
        avg_content = (
            total_chars // max(total_sections, 1)
            if total_sections > 0
            else 0
        )
        
        return {
            "num_chapters": num_chapters,
            "total_sections": total_sections,
            "avg_content_length": avg_content,
        }

    def _detect_level(self, text: str, structure: dict) -> str:
        """
        Détecter le niveau du cours.
        Heuristiques : vocabulaire, profondeur, structure complexe.
        """
        text_lower = text.lower()
        
        # 🔍 Mots-clés par niveau
        master_keywords = [
            "théorie", "recherche", "hypothèse", "méthodologie",
            "complexité", "optimisation", "algorithme avancé",
        ]
        licence_keywords = [
            "concepts", "implémentation", "projet",
            "bibliothèque", "framework", "architecture",
        ]
        lycee_keywords = [
            "concept", "exemple", "basique", "introduction",
            "simple", "facile", "débutant",
        ]
        
        master_score = sum(1 for kw in master_keywords if kw in text_lower)
        licence_score = sum(1 for kw in licence_keywords if kw in text_lower)
        lycee_score = sum(1 for kw in lycee_keywords if kw in text_lower)
        
        # Structure complexe = plus haut niveau
        if structure["num_chapters"] > 5:
            master_score += 2
        
        # Décision
        if master_score >= licence_score and master_score >= lycee_score:
            return "master"
        elif licence_score >= lycee_score:
            return "licence"
        else:
            return "lycée"

    def _extract_topics(self, text: str) -> list:
        """Extraire les topics/concepts clés du cours."""
        # Heuristique simple : mots en majuscules + contexte
        topics = []
        words = text.split()
        
        for word in words:
            # Cherche mots en MAJUSCULES (acronymes, concepts)
            if word.isupper() and len(word) > 2 and len(word) < 15:
                if word not in topics:
                    topics.append(word)
        
        # Limiter à 10 topics
        return topics[:10]

    def _generate_summary(
        self, course_dict: dict, topics: list, level: str
    ) -> str:
        """Générer un résumé court du cours."""
        title = course_dict.get("title", "Untitled Course")
        domain = course_dict.get("domain", "general")
        num_chapters = (
            len(course_dict.get("chapters", [])) or 0
        )
        
        topics_str = ", ".join(topics[:5]) if topics else "concepts variés"
        
        return (
            f"Cours '{title}' ({domain}) | "
            f"Niveau: {level} | "
            f"{num_chapters} chapitres | "
            f"Topics: {topics_str}"
        )


# Singleton
_analyzer: Optional[CourseAnalyzer] = None


def get_analyzer() -> CourseAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = CourseAnalyzer()
    return _analyzer

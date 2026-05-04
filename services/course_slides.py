"""Course slide context loader — used by WS handler & quiz routes.

Loads a single slide's full context from PostgreSQL given (course_id, chapter, section).
Returns the slide payload (title, content, image, concepts, progress_pct) or None.
"""

import logging
import uuid

log = logging.getLogger("SmartTeacher.services.course_slides")


async def load_course_slide_context(
    course_id: str,
    chapter_index: int,
    section_index: int,
) -> dict | None:
    """Charge la slide courante d'un cours depuis PostgreSQL."""
    if not course_id:
        return None

    try:
        from database.init_db import AsyncSessionLocal
        from database.crud import get_course_with_structure
        from pedagogy.course_analyzer import get_analyzer

        async with AsyncSessionLocal() as db:
            course = await get_course_with_structure(db, uuid.UUID(course_id))
            if not course:
                return None

            chapters = sorted(course.chapters, key=lambda ch: ch.order or 0)
            if chapter_index < 0 or chapter_index >= len(chapters):
                return None

            chapter = chapters[chapter_index]
            sections = sorted(chapter.sections, key=lambda sec: sec.order or 0)
            if section_index < 0 or section_index >= len(sections):
                return None

            section = sections[section_index]

            # Analyser le cours (1 fois par cours, à chapter_index=0)
            analysis = None
            if chapter_index == 0:
                try:
                    analyzer = get_analyzer()
                    course_data = {
                        "title": course.title,
                        "domain": course.domain or "general",
                        "chapters": [
                            {
                                "title": ch.title,
                                "sections": [
                                    {"title": sec.title, "content": sec.content or ""}
                                    for sec in sorted(ch.sections, key=lambda s: s.order or 0)
                                ],
                            }
                            for ch in chapters
                        ],
                    }
                    analysis = analyzer.analyze(course_data)
                except Exception as e:
                    log.debug(f"Course analysis error: {e}")
                    analysis = None

            slide_path = section.image_url or (
                section.image_urls[0]
                if getattr(section, "image_urls", None)
                else ""
            )

            total_sections = sum(
                len(sorted(ch.sections, key=lambda sec: sec.order or 0))
                for ch in chapters
            )
            global_slide_index = sum(
                len(sorted(ch.sections, key=lambda sec: sec.order or 0))
                for ch in chapters[:chapter_index]
            ) + section_index
            progress_pct = 0
            if total_sections > 1:
                progress_pct = round(global_slide_index / max(total_sections - 1, 1) * 100)

            return {
                "course_id": str(course.id),
                "course_title": course.title,
                "course_subject": course.subject,
                "course_domain": course.domain or "general",
                "language": course.language,
                "level": course.level,
                "chapter_index": chapter_index,
                "chapter_order": chapter.order or chapter_index + 1,
                "chapter_title": chapter.title,
                "section_index": section_index,
                "section_order": section.order or section_index + 1,
                "section_title": section.title,
                "content": section.content or "",
                "slide_path": slide_path,
                "image_url": slide_path,
                "slide_index": global_slide_index,
                "slide_type": "image" if slide_path else "section",
                "keywords": [c.term for c in section.concepts if c.term],
                "concepts": [
                    {
                        "term": c.term,
                        "definition": c.definition,
                        "example": c.example,
                        "type": c.concept_type,
                    }
                    for c in section.concepts
                ],
                "progress_pct": progress_pct,
                "course_summary": analysis.get("summary", "") if analysis else "",
                "course_analysis": analysis or {},
            }
    except Exception as exc:
        log.debug(
            "Unable to load slide context for course %s ch=%s sec=%s: %s",
            course_id, chapter_index, section_index, exc,
        )
        return None

"""Course endpoints — build, list, structure, concept-graph, assets."""

import logging
import shutil
import tempfile
import uuid
import uuid as _uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import deps

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.course")


# ── Knowledge graph cache (module-level, shared across requests) ──────
_concept_graph_cache: dict[str, dict] = {}


# ── /course/build ─────────────────────────────────────────────────────

@router.post("/course/build")
async def build_course(
    files:                list[UploadFile] = File(...),
    language:             str              = Form("fr"),
    level:                str              = Form("lycée"),
    domain:               str              = Form("general"),
    append_to_course_id:  str | None       = Form(None),
    course_title:         str | None       = Form(None),
    course_slug:          str | None       = Form(None),
    auto_group:           bool             = Form(True),
):
    """Upload PDF/DOCX/PPTX → structure en cours présentable. PostgreSQL + RAG + slides PNG.

    ``language`` and ``level`` are properties of the COURSE itself
    (set by the teacher at upload time) — they drive the narration's
    language and the depth/vocabulary of explanations. The same PDF
    can be uploaded twice with different (language, level) tuples
    if different audiences need different versions.

    Multi-chapter courses :
      - ``append_to_course_id`` (UUID, optional) : explicitly add this PDF
        as a new chapter to an existing course. Skips course creation.
      - ``course_title`` (optional) : override the auto-detected title
        when CREATING a new course (ignored when appending).
      - ``auto_group`` (default True) : when no ``append_to_course_id`` is
        given, look for an existing course with the same (domain, subject)
        slug and append to it instead of creating a duplicate. Set to
        False to force a new course.
    """
    from pedagogy.course_builder import CourseBuilder
    from database.init_db import AsyncSessionLocal
    from sqlalchemy import select
    from database.models import Course

    rag = deps.get_rag()

    if not files:
        raise HTTPException(status_code=400, detail="No file provided")

    builder = CourseBuilder()

    # ── Resolve target_domain / target_course (the slug pair used both
    # for storage layout and DB metadata). The caller MUST provide the
    # context — there is no LLM-classifier or file-path heuristic anymore.
    #
    # 2 valid input shapes :
    #
    #   A. APPEND : caller supplies ``append_to_course_id`` → we look
    #      up that Course row and reuse its (domain, subject).
    #
    #   B. NEW    : caller supplies ``domain`` + (``course_slug`` or
    #      ``course_title``). The slug is built from whichever of those
    #      two is present.
    #
    # Anything else is rejected with 400 — we no longer guess.

    target_domain: str
    target_course: str

    # ``next_chapter_idx`` = order of the NEXT course to add (each uploaded
    # PDF is, in the user-facing vocabulary, a "course" inside a subject).
    # For NEW subjects this starts at 1; for APPEND we look up
    # max(Chapter.order)+1 so every PDF lands in its own course_N/ folder
    # (no PNG collisions between PDFs sharing the same logical subject).
    next_chapter_idx: int = 1

    if append_to_course_id:
        from sqlalchemy import func
        from database.models import Chapter
        try:
            async with AsyncSessionLocal() as db:
                row = (await db.execute(
                    select(Course.domain, Course.subject).where(Course.id == uuid.UUID(append_to_course_id))
                )).first()
                if row is not None:
                    next_chapter_idx = int((await db.execute(
                        select(func.coalesce(func.max(Chapter.order), 0))
                        .where(Chapter.course_id == uuid.UUID(append_to_course_id))
                    )).scalar() or 0) + 1
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid append_to_course_id: {exc}")
        if row is None:
            raise HTTPException(status_code=404, detail=f"Course not found: {append_to_course_id}")
        target_domain, target_course = row[0] or "general", row[1] or "generic"
    else:
        if not domain or domain == "general":
            raise HTTPException(status_code=400, detail="Missing 'domain' (must be created via the Course Manager UI)")
        if not (course_slug or course_title):
            raise HTTPException(status_code=400, detail="Missing 'course_slug' or 'course_title'")
        target_domain = builder._course_slug(domain, domain)
        slug_seed = course_slug or course_title
        target_course = builder._course_slug(slug_seed, slug_seed, domain=target_domain)

    log.info(f"🎯 Resolved context : domain={target_domain} subject={target_course} starting at course_{next_chapter_idx}")

    results = []
    files_to_index: list[dict] = []

    for f in files:
        # Each file in this batch lands in its OWN course_N/ folder so
        # PNG slides from one PDF never overwrite PNG slides from another.
        target_chapter = f"course_{next_chapter_idx}"
        next_chapter_idx += 1

        raw_upload_name = (f.filename or "upload.pdf").replace("\\", "/")
        upload_filename = Path(raw_upload_name).name or f"upload_{uuid.uuid4().hex[:8]}.pdf"
        payload = await f.read()
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload_filename).suffix or ".pdf") as tmp:
                tmp.write(payload)
                temp_path = Path(tmp.name)

            # All persistent course assets live under ``media/`` so the
            # FastAPI StaticFiles mount (``/media``) can serve them.
            #   - PDFs / DOCX / PPTX  → media/courses/<domain>/<course>/<chapter>/
            #   - Rendered slide PNGs → media/slides/<domain>/<course>/<chapter>/
            target_dir = Path("media/courses") / target_domain / target_course / target_chapter
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / upload_filename
            shutil.move(str(temp_path), str(dest))

            log.info(f"📁 Course file saved : {dest}")

            # 0. Ingestion intelligente multi-format
            from pedagogy.intelligent_ingester import IntelligentIngester
            from database.models import IngestedAssetDB
            try:
                ingester = IntelligentIngester(ocr_languages="fra+eng")
                ingestion = await ingester.ingest_file(
                    file_path=str(dest), media_root="media/courses", course_id="",
                )
                log.info(
                    f"🧠 Ingestion ({Path(dest).suffix}): {ingestion.total_pages} pages/slides, "
                    f"{ingestion.total_images} images, {ingestion.total_tables} tables, "
                    f"{ingestion.total_captions} captions ({ingestion.extraction_time_s}s)"
                )
            except Exception as exc:
                log.warning(f"intelligent ingester failed (continuing without): {exc}")
                ingestion = None

            # 1. Build course
            file_ext = Path(dest).suffix.lower()
            if file_ext == ".pdf":
                course_data = await builder.build_from_file_direct(
                    str(dest), language=language, level=level,
                    domain=target_domain, subject=target_course, chapter=target_chapter,
                )
            elif ingestion and ingestion.pages:
                aggregated_text = "\n\n".join(
                    f"[Page {p['page_num']}]\n{p['text']}"
                    for p in ingestion.pages if p.get("text", "").strip()
                )
                # No LLM classifier — fall back to the file stem only.
                title_hint = Path(dest).stem
                log.info(f"📚 Building from {file_ext} via build_from_text ({len(aggregated_text)} chars)")
                course_data = await builder.build_from_text(
                    text=aggregated_text, title=title_hint,
                    language=ingestion.language or language, level=level, subject=target_course,
                )
                course_data.setdefault("file_path", str(dest))
                course_data.setdefault("language", ingestion.language or language)
                course_data.setdefault("level", level)
            else:
                log.warning(f"No ingestion data + non-PDF ({file_ext}) → legacy fallback")
                course_data = await builder.build_from_file_direct(
                    str(dest), language=language, level=level,
                    domain=target_domain, subject=target_course, chapter=target_chapter,
                )

            # 1.5. Inject visual assets
            if ingestion and ingestion.assets:
                visual_assets_by_page: dict[int, list[dict]] = {}
                for asset in ingestion.assets:
                    if asset.asset_type in {"image", "caption"}:
                        visual_assets_by_page.setdefault(asset.page_num, []).append(asset.to_dict())
                for ch in (course_data.get("chapters") or []):
                    for sec in (ch.get("sections") or []):
                        page_idx = int(sec.get("page_index") or 0)
                        if page_idx and page_idx in visual_assets_by_page:
                            sec.setdefault("visual_assets", []).extend(visual_assets_by_page[page_idx])
                course_data["ingestion_summary"] = {
                    "total_pages": ingestion.total_pages,
                    "total_images": ingestion.total_images,
                    "total_tables": ingestion.total_tables,
                    "total_captions": ingestion.total_captions,
                    "language_detected": ingestion.language,
                }

            # Override course title only when CREATING (not when appending —
            # the existing course already has its own title and we shouldn't
            # silently rename it from a chapter upload).
            if course_title and not append_to_course_id:
                course_data["title"] = course_title

            course_id = None
            db_action = None      # "created" | "appended"
            db_error = None
            try:
                async with AsyncSessionLocal() as db:
                    course_id, db_action = await builder.save_or_append_smart(
                        course_data, db,
                        domain=target_domain,
                        append_to=append_to_course_id,
                        auto_group=auto_group,
                        rag=rag,    # enables embedding-similarity matching
                    )
                log.info(
                    f"💾 Course persisted ({db_action}) : id={course_id} "
                    f"domain={target_domain} subject={course_data.get('subject')}"
                )
            except Exception as exc:
                db_error = str(exc)
                log.info(f"ℹ️ PostgreSQL indisponible pour {f.filename}: {exc}")

            # 2.5. Persist assets in DB
            if ingestion and course_id:
                try:
                    cid_uuid = _uuid.UUID(str(course_id))
                    async with AsyncSessionLocal() as db:
                        for asset in ingestion.assets:
                            db.add(IngestedAssetDB(
                                course_id=cid_uuid,
                                asset_type=asset.asset_type,
                                page_num=asset.page_num,
                                text=asset.text[:5000],
                                image_path=asset.image_path,
                                image_index_in_page=asset.image_index_in_page,
                                asset_metadata=asset.metadata,
                            ))
                        await db.commit()
                        log.info(f"💾 {len(ingestion.assets)} assets persistes (course={course_id[:16]})")
                except Exception as exc:
                    log.warning(f"asset persistence failed (non-blocking): {exc}")

            files_to_index.append({
                "course_data": course_data, "course_id": course_id,
                "domain": target_domain, "course": target_course, "chapter": target_chapter,
                "storage_path": str(dest.resolve()),
            })

            chapters = len(course_data.get("chapters", []))
            sections = sum(len(ch.get("sections", [])) for ch in course_data.get("chapters", []))

            results.append({
                "file": upload_filename, "course_id": course_id,
                "title": course_data.get("title"), "chapters": chapters, "sections": sections,
                "domain": target_domain, "course": target_course, "chapter": target_chapter,
                "storage_path": str(dest),
                "db_action": db_action,    # "created" | "appended" | None on error
                "status": "ok" if db_error is None else "partial", "db_error": db_error,
            })

        except Exception as exc:
            log.error(f"❌ Build course failed for {f.filename}: {exc}")
            results.append({"file": f.filename, "status": "error", "error": str(exc)})
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    if files_to_index:
        log.info(f"📤 Ingestion RAG batch lancée pour {len(files_to_index)} fichier(s)")
        for item in files_to_index:
            course_data = item["course_data"]
            course_id = item["course_id"]
            target_domain = item["domain"]
            target_course = item["course"]
            target_chapter = item.get("chapter", "chapter_1")
            log.info(f"   📚 Indexing {Path(course_data.get('file_path', 'unknown')).name} ({target_domain}/{target_course}/{target_chapter}) course_id={course_id}")
            rag_ok = rag.run_ingestion_pipeline_from_course_data(
                course_data, domain=target_domain, course=target_course,
                course_id=course_id, incremental=True,
            )
            if not rag_ok:
                log.info(f"ℹ️ Ingestion RAG terminée sans indexation pour {item['storage_path']}")

    return {"results": results, "rag_stats": rag.get_stats()}


# ── /course/list ──────────────────────────────────────────────────────

@router.get("/course/list")
async def list_courses():
    """Liste tous les cours disponibles dans PostgreSQL."""
    try:
        from database.init_db import AsyncSessionLocal
        from database.crud import get_all_courses
        async with AsyncSessionLocal() as db:
            courses = await get_all_courses(db)
            return {
                "courses": [{
                    "id": str(c.id), "title": c.title, "subject": c.subject,
                    "domain": c.domain, "language": c.language, "level": c.level,
                } for c in courses]
            }
    except Exception as exc:
        return {"courses": [], "error": str(exc)}


# ── /course/tree ──────────────────────────────────────────────────────

@router.get("/course/tree")
async def list_courses_tree():
    """Hierarchical view of all courses : Domain → Course (subject) → Chapters.

    Returned shape mirrors the courses.html admin tree :

        {
          "tree": [
            {
              "domain": "informatique",
              "courses": [
                {
                  "id": "uuid", "title": "Python Basics", "subject": "python_basics",
                  "language": "fr", "level": "lycée",
                  "chapters": [
                    {"id": "...", "title": "Intro Python", "order": 1, "section_count": 12},
                    ...
                  ]
                },
                ...
              ]
            },
            ...
          ]
        }
    """
    from sqlalchemy import select, func
    from database.init_db import AsyncSessionLocal
    from database.models import Course, Chapter, Section

    try:
        async with AsyncSessionLocal() as db:
            # Count sections per chapter in one shot
            section_count_q = (
                select(Section.chapter_id, func.count(Section.id).label("n"))
                .group_by(Section.chapter_id)
            )
            sec_count_map = {r[0]: r[1] for r in (await db.execute(section_count_q)).all()}

            chapters_q = (
                select(Chapter.id, Chapter.course_id, Chapter.title, Chapter.order)
                .order_by(Chapter.course_id, Chapter.order)
            )
            chapters_by_course: dict[str, list[dict]] = {}
            for cid, course_id, title, order in (await db.execute(chapters_q)).all():
                chapters_by_course.setdefault(str(course_id), []).append({
                    "id": str(cid),
                    "title": title,
                    "order": order,
                    "section_count": sec_count_map.get(cid, 0),
                })

            courses_q = (
                select(Course.id, Course.title, Course.subject, Course.domain,
                       Course.language, Course.level, Course.created_at)
                .order_by(Course.domain.asc(), Course.created_at.asc())
            )
            tree_by_domain: dict[str, list[dict]] = {}
            for cid, title, subject, domain, language, level, created in (
                await db.execute(courses_q)
            ).all():
                domain = domain or "general"
                tree_by_domain.setdefault(domain, []).append({
                    "id": str(cid),
                    "title": title,
                    "subject": subject,
                    "language": language,
                    "level": level,
                    "created_at": created.isoformat() if created else None,
                    "chapters": chapters_by_course.get(str(cid), []),
                })

            return {
                "tree": [
                    {"domain": d, "courses": cs}
                    for d, cs in sorted(tree_by_domain.items())
                ],
            }
    except Exception as exc:
        log.exception("course/tree failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── /course/{id} delete ───────────────────────────────────────────────

@router.delete("/course/{course_id}")
async def delete_course(course_id: str):
    """Delete a course and all its chapters/sections/concepts (cascade).

    Also clears its chunks from Qdrant (filtered by course_id) and the
    RAG in-memory cache. RAG-side cleanup is best-effort : if it fails
    the DB row is still gone but stale chunks may linger until the next
    full reset.
    """
    import uuid as _uuid_mod
    from sqlalchemy import select
    from database.init_db import AsyncSessionLocal
    from database.models import Course

    try:
        cid = _uuid_mod.UUID(course_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="invalid course_id (UUID)")

    rag = deps.get_rag()
    try:
        async with AsyncSessionLocal() as db:
            course = (await db.execute(select(Course).where(Course.id == cid))).scalar_one_or_none()
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
            await db.delete(course)
            await db.commit()

        # Best-effort RAG cleanup
        try:
            if hasattr(rag, "delete_by_course_id"):
                rag.delete_by_course_id(course_id)
            elif getattr(rag, "all_docs", None):
                rag.all_docs = [
                    d for d in rag.all_docs
                    if (d.metadata or {}).get("course") != course_id
                ]
        except Exception as exc:    # noqa: BLE001
            log.warning("RAG cleanup failed for %s : %s", course_id, exc)

        return {"deleted": course_id}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("course delete failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── /course/{course_id}/chapter/{chapter_id} delete ──────────────────

@router.delete("/course/{course_id}/chapter/{chapter_id}")
async def delete_chapter(course_id: str, chapter_id: str):
    """Delete a single chapter (its sections cascade via FK).

    Also clears the chapter's chunks from Qdrant (filtered by
    ``course`` + ``chapter_idx``) and the RAG in-memory cache.
    RAG-side cleanup is best-effort: the DB row is gone first, so a
    Qdrant failure only leaves stale chunks until the next full reset.

    The chapter must belong to the given course — cross-course
    deletion via URL forgery returns 404.
    """
    import uuid as _uuid_mod
    from sqlalchemy import select
    from database.init_db import AsyncSessionLocal
    from database.models import Chapter

    try:
        cid = _uuid_mod.UUID(course_id)
        chid = _uuid_mod.UUID(chapter_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="invalid UUID(s)")

    rag = deps.get_rag()
    try:
        async with AsyncSessionLocal() as db:
            chapter = (await db.execute(
                select(Chapter).where(Chapter.id == chid, Chapter.course_id == cid)
            )).scalar_one_or_none()
            if chapter is None:
                raise HTTPException(status_code=404, detail="chapter not found in this course")
            # Capture ``order`` before delete — needed to filter Qdrant
            # chunks, which were stamped with this value as
            # ``chapter_idx`` at ingestion time.
            chapter_order = int(chapter.order)
            chapter_title = chapter.title
            await db.delete(chapter)
            await db.commit()

        # Best-effort RAG cleanup
        try:
            if hasattr(rag, "delete_by_course_and_chapter"):
                rag.delete_by_course_and_chapter(
                    course_id,
                    chapter_idx=chapter_order,
                    chapter_title=chapter_title,
                )
            elif getattr(rag, "all_docs", None):
                # Fallback path: old RAG instance without the method.
                # Mirror the strict-AND logic of the canonical method —
                # prefer title (reliably stamped), fall back to idx only
                # when the title is missing. Never wildcard within a
                # course (that's delete_by_course_id's job).
                title_norm = (chapter_title or "").strip()
                use_title = bool(title_norm)
                def _keep(d):
                    meta = d.metadata or {}
                    if meta.get("course") != course_id:
                        return True
                    if use_title:
                        return (meta.get("chapter_title") or "").strip() != title_norm
                    try:
                        return int(meta.get("chapter_idx")) != chapter_order
                    except (TypeError, ValueError):
                        return True
                rag.all_docs = [d for d in rag.all_docs if _keep(d)]
                # Persist + rebuild BM25 (use the private helpers; the
                # canonical method does the same work).
                if hasattr(rag, "_save_docs_cache"):
                    rag._save_docs_cache()
                if hasattr(rag, "_build_hybrid_retriever"):
                    if rag.all_docs:
                        rag._build_hybrid_retriever()
                    else:
                        rag.bm25_retriever = None
        except Exception as exc:    # noqa: BLE001
            log.warning(
                "RAG chapter cleanup failed for %s/%s : %s",
                course_id, chapter_id, exc,
            )

        return {
            "deleted_chapter_id": chapter_id,
            "course_id": course_id,
            "chapter_order": chapter_order,
            "chapter_title": chapter_title,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("chapter delete failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── /course/{id}/structure ────────────────────────────────────────────

@router.get("/course/{course_id}/structure")
async def get_course_structure(course_id: str):
    """Retourne la structure complète d'un cours (chapitres + sections + PNG slides)."""
    try:
        from database.init_db import AsyncSessionLocal
        from database.crud import get_course_with_structure

        async with AsyncSessionLocal() as db:
            course = await get_course_with_structure(db, _uuid.UUID(course_id))
            if not course:
                raise HTTPException(status_code=404, detail="Cours introuvable")

            slides = []
            chapters_data = []

            for ch in course.chapters:
                sections_data = []
                for sec in ch.sections:
                    slide_path = sec.image_url or (sec.image_urls[0] if getattr(sec, "image_urls", None) else "")
                    if slide_path:
                        slides.append(slide_path)

                    sections_data.append({
                        "title": sec.title, "order": sec.order, "duration_s": sec.duration_s,
                        "image_url": slide_path, "content": sec.content or "",
                        "concepts": [{"term": c.term, "definition": c.definition} for c in sec.concepts],
                    })

                chapters_data.append({
                    "title": ch.title, "order": ch.order, "sections": sections_data,
                })

            return {
                "id": str(course.id), "title": course.title, "subject": course.subject,
                "domain": course.domain or "general",
                "language": course.language, "level": course.level,
                "file_path": course.file_path or "",
                "slides": slides, "chapters": chapters_data,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /course/{id}/concept-graph ────────────────────────────────────────

@router.get("/course/{course_id}/concept-graph")
async def get_course_concept_graph(
    course_id: str,
    refresh: bool = False,
    max_concepts: int = 50,
    enrich: bool = True,
):
    """Knowledge Graph unifie : KeyBERT extraction → ConceptInfo → attach
    au KnowledgeGraph singleton. Edges = prereq derive du graphe d'idees
    (depends_on), plus de cooccurrence heuristique.

    Le pipeline d'extraction est maintenant dans `ensure_concepts_loaded()`
    et partage avec les consumers internes (path_recommender, skill_tree,
    practice_engine) qui font du lazy bootstrap.
    """
    from pedagogy.knowledge_graph import get_or_build, ensure_concepts_loaded

    rag = deps.get_rag()

    # Layer cache mémoire
    if not refresh and course_id in _concept_graph_cache:
        log.info(f"📊 Concept graph cache HIT (memory) for course={course_id[:16]}")
        return _concept_graph_cache[course_id]

    kg = get_or_build(rag)

    # Si refresh demande, on vide les concepts du cours pour forcer re-extraction
    if refresh:
        existing_others = [
            ci for ci in kg.list_concepts() if ci.course_id != course_id
        ]
        kg.attach_concepts(existing_others)   # purge ce cours
        _concept_graph_cache.pop(course_id, None)

    # Si le KG a deja ce cours attache (et pas de refresh), servir depuis le KG
    if not refresh and kg.list_concepts(course_id):
        graph = _build_cytoscape_from_kg(kg, course_id, rag)
        _concept_graph_cache[course_id] = graph
        log.info(
            f"📊 Concept graph from KG (already attached) for course={course_id[:16]} "
            f"({len(graph['nodes'])} concepts)"
        )
        return graph

    if not rag.all_docs:
        raise HTTPException(status_code=404, detail="RAG cache empty — ingest courses first")

    log.info(f"🧠 Extracting concepts for course={course_id[:16]}… (enrich={enrich})")
    try:
        concepts = ensure_concepts_loaded(
            rag, course_id, max_concepts=max_concepts, enrich=enrich,
        )
        if not concepts:
            raise HTTPException(
                status_code=404,
                detail=f"No concepts extracted for course {course_id} (RAG empty or extraction failed)",
            )
        graph = _build_cytoscape_from_kg(kg, course_id, rag)
        _concept_graph_cache[course_id] = graph
        log.info(
            f"✅ Concept graph built : {len(graph['nodes'])} nodes, "
            f"{len(graph['edges'])} edges (course={course_id[:16]})"
        )
        return graph
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("concept-graph extraction failed")
        raise HTTPException(status_code=500, detail=f"Concept extraction error: {exc}")


def _build_cytoscape_from_kg(kg, course_id: str, rag) -> dict:
    """Build Cytoscape JSON depuis le KnowledgeGraph (replace l'ancien
    ConceptExtractor.build_cytoscape_graph qui utilisait la cooccurrence).

    Edges = prereq semantique. Direction prereq → dependent.
    """
    concepts = kg.list_concepts(course_id)
    nodes = []
    for c in concepts:
        nodes.append({
            "data": {
                "id":             c.name,
                "label":          c.canonical_name or c.display_name or c.name,
                "score":          round(c.score, 3),
                "chapters":       sorted(c.chapter_idxs),
                "bloom":          c.bloom_level or "",
                "description":    c.description or "",
                "idea_count":     len(c.idea_ids),
            }
        })

    edges = []
    seen: set[tuple[str, str]] = set()
    for c in concepts:
        for p in kg.prereq_concepts(c.name):
            key = (p.name, c.name)
            if key in seen:
                continue
            seen.add(key)
            # Poids = nombre d'edges idee-level entre les 2 concepts (pour
            # Cytoscape mapData(weight, 1, 10, 1, 6) → epaisseur visuelle).
            weight = max(1, kg.count_idea_edges_between(p.name, c.name))
            edges.append({
                "data": {
                    "id":     f"{p.name}__{c.name}",
                    "source": p.name,
                    "target": c.name,
                    "weight": weight,
                }
            })

    return {
        "nodes":           nodes,
        "edges":           edges,
        "concepts_detail": [c.to_dict() for c in concepts],
        "course_id":       course_id,
        "total_chunks":    sum(
            1 for d in rag.all_docs if (d.metadata or {}).get("course") == course_id
        ),
    }


# ── /course/{id}/assets ───────────────────────────────────────────────

@router.get("/course/{course_id}/assets")
async def get_course_assets(course_id: str, asset_type: str | None = None, page_num: int | None = None):
    """Retourne les assets ingérés d'un cours (images, tables, captions, titres)."""
    from sqlalchemy import select
    from database.init_db import AsyncSessionLocal
    from database.models import IngestedAssetDB

    try:
        cid = _uuid.UUID(course_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="invalid course_id (UUID)")
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(IngestedAssetDB).where(IngestedAssetDB.course_id == cid)
            if asset_type:
                stmt = stmt.where(IngestedAssetDB.asset_type == asset_type)
            if page_num is not None:
                stmt = stmt.where(IngestedAssetDB.page_num == page_num)
            stmt = stmt.order_by(IngestedAssetDB.page_num.asc(), IngestedAssetDB.image_index_in_page.asc())
            rows = (await db.execute(stmt)).scalars().all()
            return {
                "course_id": course_id, "count": len(rows),
                "assets": [{
                    "id": str(a.id), "asset_type": a.asset_type, "page_num": a.page_num,
                    "text": (a.text or "")[:500], "image_path": a.image_path or "",
                    "image_index_in_page": a.image_index_in_page,
                    "metadata": a.asset_metadata or {},
                } for a in rows],
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Asset query error: {exc}")

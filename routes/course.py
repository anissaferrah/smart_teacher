"""Course endpoints — build, list, structure, concept-graph, assets."""

import logging
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
    files:    list[UploadFile] = File(...),
    language: str              = Form("fr"),
    level:    str              = Form("lycée"),
    domain:   str              = Form("general"),
):
    """Upload PDF/DOCX/PPTX → structure en cours présentable. PostgreSQL + RAG + slides PNG.

    ``language`` and ``level`` are properties of the COURSE itself
    (set by the teacher at upload time) — they drive the narration's
    language and the depth/vocabulary of explanations. The same PDF
    can be uploaded twice with different (language, level) tuples
    if different audiences need different versions. The student
    profile's ``preferred_language`` is informational and does not
    override the course's narration language.
    """
    from pedagogy.course_builder import CourseBuilder
    from database.init_db import AsyncSessionLocal
    from core.domains_config import auto_detect_course, classify_course_via_llm

    rag = deps.get_rag()

    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")

    results = []
    files_to_index: list[dict] = []
    builder = CourseBuilder()

    for f in files:
        raw_upload_name = (f.filename or "upload.pdf").replace("\\", "/")
        upload_filename = Path(raw_upload_name).name or f"upload_{uuid.uuid4().hex[:8]}.pdf"
        payload = await f.read()
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload_filename).suffix or ".pdf") as tmp:
                tmp.write(payload)
                temp_path = Path(tmp.name)

            detected_domain, detected_course = auto_detect_course(str(temp_path))

            llm_title_hint: str | None = None
            if detected_domain == "general" and detected_course == "generic" and domain == "general":
                llm_classification = classify_course_via_llm(str(temp_path), max_pages=2)
                if llm_classification:
                    detected_domain = llm_classification["domain"]
                    detected_course = llm_classification["course"]
                    llm_title_hint = llm_classification["title"]
                    log.info(
                        f"🤖 Domaine/cours auto-classifies par LLM : "
                        f"{detected_domain}/{detected_course} ('{llm_title_hint}')"
                    )

            target_domain = detected_domain if detected_domain != "general" else domain
            fallback_course = detected_course if detected_course != "generic" else None
            if fallback_course is None and upload_filename:
                stem_hint = Path(upload_filename).stem
                if not builder._looks_like_chapter(stem_hint):
                    fallback_course = stem_hint

            target_domain, target_course, target_chapter = builder.infer_upload_context(
                raw_upload_name,
                fallback_domain=target_domain,
                fallback_course=fallback_course,
                fallback_chapter="chapter_1",
            )

            target_dir = Path("courses") / target_domain / target_course / target_chapter
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / upload_filename
            temp_path.replace(dest)

            log.info(f"📁 Sauvegarde cours : {dest}")

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
                title_hint = llm_title_hint or Path(dest).stem
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

            course_id = None
            db_error = None
            try:
                async with AsyncSessionLocal() as db:
                    course_id = await builder.save_to_database(course_data, db, domain=target_domain)
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
                    "language": c.language, "level": c.level,
                } for c in courses]
            }
    except Exception as exc:
        return {"courses": [], "error": str(exc)}


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

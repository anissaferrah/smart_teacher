"""
╔══════════════════════════════════════════════════════════════════════╗
║        SMART TEACHER — Constructeur de Cours v3                    ║
║                                                                      ║
║  AMÉLIORATIONS v3 :                                                  ║
║    ✅ Structure hiérarchique : Domain → Courses → Chapters          ║
║    ✅ Support multi-domaines                                     ║
║    ✅ Chargement depuis courses/{domain}/{course}/                  ║
║    ✅ Préservation de la structure PPTX slide par slide              ║
║    ✅ Subject automatique selon le domaine & cours                  ║
║    ✅ Pipeline complet : build_course_chapters()                    ║
║    ✅ Détection automatique du niveau (université)                  ║
║    ✅ Concepts extraits intelligemment                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import logging
import re
import warnings
from pathlib import Path

# Import configuration domaines & cours
from core.config import Config
from core.domains_config import DEFAULT_DOMAIN, DEFAULT_COURSE, get_chapters, get_courses, get_domains
from ai.llm import Brain

log = logging.getLogger("SmartTeacher.CourseBuilder")


# ══════════════════════════════════════════════════════════════════════
#  EXTRACTEUR DE TEXTE PDF/DOCX/PPTX
# ══════════════════════════════════════════════════════════════════════

class TextExtractor:
    """Extrait le texte structuré depuis PDF, DOCX, PPTX, TXT."""

    def extract(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        log.info(f"📄 Extraction : {Path(file_path).name}")
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext == ".docx":
            return self._extract_docx(file_path)
        elif ext == ".pptx":
            return self._extract_pptx(file_path)
        elif ext in (".txt", ".md"):
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        else:
            raise ValueError(f"Format non supporté : {ext}")

    def extract_structured_pptx(self, path: str) -> list[dict]:
        """
        Extrait le PPTX slide par slide.
        Retourne une liste de dicts {slide_idx, title, bullets, content}.
        Préserve la structure des slides pour le SlideSync.
        """
        try:
            from pptx import Presentation
            prs = Presentation(path)
            slides = []
            for i, slide in enumerate(prs.slides):
                title_text   = ""
                bullet_texts = []

                for shape in slide.shapes:
                    if not hasattr(shape, "text") or not shape.text.strip():
                        continue
                    text = shape.text.strip()
                    is_title_placeholder = shape.shape_type == 13
                    if not is_title_placeholder:
                        try:
                            placeholder_format = shape.placeholder_format
                        except (AttributeError, ValueError):
                            placeholder_format = None
                        is_title_placeholder = bool(placeholder_format and placeholder_format.idx == 0)

                    # Le premier texte grand = titre (souvent placeholder title)
                    if is_title_placeholder:
                        title_text = text
                    else:
                        bullet_texts.append(text)

                if not title_text and bullet_texts:
                    title_text = bullet_texts.pop(0)

                content = "\n".join(bullet_texts)
                if title_text or content:
                    slides.append({
                        "slide_idx": i + 1,
                        "title":     title_text,
                        "bullets":   bullet_texts,
                        "content":   f"{title_text}\n{content}".strip(),
                    })
            log.info(f"  ✅ {len(slides)} slides extraites")
            return slides
        except ImportError:
            raise ImportError("Installez python-pptx : pip install python-pptx")

    def _extract_pdf(self, path: str) -> str:
        try:
            import pypdf
            text = ""
            with open(path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module=r"pypdf\.generic\._base")
                    reader = pypdf.PdfReader(f, strict=False)
                    for page in reader.pages:
                        text += (page.extract_text() or "") + "\n\n"
            return text.strip()
        except ImportError:
            pass
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            raise ImportError("Installez pypdf : pip install pypdf")

    def _extract_docx(self, path: str) -> str:
        try:
            import docx
            doc = docx.Document(path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError("Installez python-docx : pip install python-docx")

    def _extract_pptx(self, path: str) -> str:
        slides = self.extract_structured_pptx(path)
        return "\n\n".join(
            f"[Slide {s['slide_idx']}]\n{s['content']}"
            for s in slides if s['content']
        )


# ══════════════════════════════════════════════════════════════════════
#  STRUCTUREUR LOCAL — TESSERACT OCR (sans API OpenAI)
# ══════════════════════════════════════════════════════════════════════

class LocalStructurer:
    """
    ✅ Structure le texte SANS modifier le contenu original.
    ✅ Utilise Tesseract-OCR local (gratuit, rapide).
    ✅ Génère les PNG pour l'affichage.
    ✅ Indexe le texte brut directement sans appel OpenAI.
    """

    def __init__(self, course_builder=None):
        """Initialise Tesseract (déjà dans le Dockerfile)"""
        self.course_builder = course_builder  # Référence pour génération de résumés
        try:
            import pytesseract
            from pdf2image import convert_from_path
            self.pytesseract = pytesseract
            self.convert_from_path = convert_from_path
            self.available_ocr_langs = set()
            try:
                self.available_ocr_langs = set(pytesseract.get_languages(config="") or [])
            except Exception:
                self.available_ocr_langs = set()
            log.info("✅ Tesseract-OCR en local activé")
        except ImportError as e:
            log.info(f"ℹ️ Tesseract non disponible: {e}")
            self.pytesseract = None
            self.convert_from_path = None
            self.available_ocr_langs = set()

    def _select_ocr_language(self, requested_lang: str) -> str:
        """Choisit une langue Tesseract réellement installée pour éviter le fallback English par défaut."""
        if not self.pytesseract or not self.available_ocr_langs:
            return ""

        requested_parts = [part.strip() for part in requested_lang.split("+") if part.strip()]
        selected = [part for part in requested_parts if part in self.available_ocr_langs]
        if selected:
            return "+".join(selected)

        for fallback in ("fra", "ara", "eng"):
            if fallback in self.available_ocr_langs:
                return fallback

        return next(iter(sorted(self.available_ocr_langs)), "")

    def _pdf_to_images(self, pdf_path: str, output_dir: str) -> list[str]:
        """Convertit PDF en images PNG."""
        if not self.convert_from_path:
            return []
        try:
            images = self.convert_from_path(pdf_path, dpi=150)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            paths = []
            for i, img in enumerate(images, 1):
                png_path = Path(output_dir) / f"page_{i:03d}.png"
                img.save(str(png_path), "PNG")
                paths.append(str(png_path))
            log.info(f"  ✅ {len(paths)} PNG générées")
            return paths
        except Exception as e:
            log.info(f"  ℹ️ Conversion PDF→PNG échouée: {e}")
            return []

    def _extract_text_with_ocr(self, pdf_path: str, lang: str = "fra+ara+eng") -> str:
        """Extrait le texte brut avec Tesseract-OCR."""
        if not self.pytesseract or not self.convert_from_path:
            return ""
        try:
            resolved_lang = self._select_ocr_language(lang)
            if not resolved_lang:
                log.info("ℹ️ OCR ignoré: aucun pack de langue Tesseract disponible")
                return ""

            images = self.convert_from_path(pdf_path, dpi=150)
            full_text = ""
            for img in images:
                # Utiliser Tesseract pour extraire le texte
                text = self.pytesseract.image_to_string(img, lang=resolved_lang)
                full_text += text + "\n\n"
            return full_text.strip()
        except Exception as e:
            log.info(f"ℹ️ OCR échouée: {e}")
            return ""

    @staticmethod
    def _derive_section_title(content: str, parent_title: str, fallback_index: int) -> str:
        """Pick a real section title from a chunk of raw text.

        Replaces the previous heuristic ``content[:60] + '...'`` which
        produced useless titles like ``"n is sample size and N is..."``
        when the chunk happened to start with a notation legend or a body
        sentence. Those bad titles then leaked into the planner as
        ``section_title``, polluting its context-stop derivation and
        causing real concept words to be rejected as "parent words" on
        later slides.

        Strategy:
          1. Reuse the structural concept extractor from the planner —
             same heuristic that finds "Trimmed mean", "Median", "Newton
             second law" inside flat-extracted slide text. Course- and
             language-agnostic.
          2. If nothing extracts cleanly, use a numbered placeholder
             tied to ``parent_title`` (e.g. "Cours importé — Section 3")
             so the title is at least useful for navigation.
        """
        try:
            from agentic.teaching.planner import _extract_main_concept
            concept = _extract_main_concept(content, section_title=parent_title)
            if concept:
                return concept
        except Exception as exc:
            log.debug(f"section-title extraction skipped: {exc}")
        suffix = parent_title.strip() or "Cours"
        return f"{suffix} — Section {fallback_index}"

    async def structure(
        self,
        raw_text: str,
        language: str = "en",
        level: str = "université",
        title: str = "",
        subject: str = DEFAULT_COURSE,
        chapter_idx: int | None = None,
    ) -> dict:
        """
        Structure le texte SANS le modifier (100% local).

        Retourne une structure simple pour l'indexation RAG.
        Le contenu est gardé intégralement.
        """
        log.info(f"📚 Structuration locale ({len(raw_text)} chars, tesseract)…")

        # Diviser le texte en sections (~ 500 mots chacune)
        sentences = [s.strip() for s in raw_text.replace("\n\n", ". ").split(". ") if s.strip()]

        sections = []
        current_section = []
        word_count = 0
        parent_title = title or f"Chapitre {chapter_idx or 1}"

        for sentence in sentences:
            words = sentence.split()
            current_section.append(sentence)
            word_count += len(words)

            if word_count >= 150:  # ~1 minute orale
                content = ". ".join(current_section)
                if content.strip():
                    section_title = self._derive_section_title(
                        content, parent_title, len(sections) + 1,
                    )
                    summary = ""
                    if self.course_builder:
                        try:
                            summary = await self.course_builder._generate_section_summary(
                                content, section_title, language
                            )
                        except Exception as e:
                            log.warning(f"Échec résumé section: {e}")

                    sections.append({
                        "title": section_title,
                        "order": len(sections) + 1,
                        "content": content,
                        "summary": summary,
                        "duration_s": max(60, word_count // 3),  # ~3 mots/seconde
                        "concepts": [],  # see note on `_extract_concepts` below
                    })
                current_section = []
                word_count = 0

        # Dernière section
        if current_section:
            content = ". ".join(current_section)
            section_title = self._derive_section_title(
                content, parent_title, len(sections) + 1,
            )
            summary = ""
            if self.course_builder:
                try:
                    summary = await self.course_builder._generate_section_summary(
                        content, section_title, language
                    )
                except Exception as e:
                    log.warning(f"Échec résumé section finale: {e}")

            sections.append({
                "title": section_title,
                "order": len(sections) + 1,
                "content": content,
                "summary": summary,
                "duration_s": max(60, word_count // 3),
                "concepts": [],  # see note on `_extract_concepts` below
            })

        return {
            "title": title or "Cours importé",
            "subject": subject,
            "level": level,
            "language": language,
            "chapters": [
                {
                    "title": f"Chapitre {chapter_idx or 1}",
                    "order": chapter_idx or 1,
                    "sections": sections,
                }
            ],
        }

    def _extract_concepts(self, text: str, subject: str = DEFAULT_COURSE) -> list[dict]:
        """Deprecated. Returns ``[]``.

        The previous implementation matched a hardcoded English keyword
        list per ``subject`` ("algorithm", "theorem", "vocabulary", …)
        and emitted stub entries with the placeholder definition
        ``"Terme trouvé dans le contenu"``. Three problems made it
        actively harmful:

          - English-only — French ("théorème", "algorithme"), Arabic,
            and any other language got zero matches.
          - Subject-specific — required maintaining keyword lists per
            domain. New domains needed code edits.
          - The stub entries ("Terme trouvé dans le contenu") leaked
            into downstream consumers as fake concepts with no real
            information.

        Concept identification now happens at slide-presentation time
        via the structural extractor in
        ``agentic.teaching.planner._extract_main_concept`` — which works
        for any course in any language because it relies on punctuation
        and structure, not vocabulary. Sections are titled the same way
        in ``_derive_section_title`` above. The empty list is returned
        only to keep the dict shape stable for ``slide_sync`` and
        ``services.course_slides``, both of which tolerate empty.
        """
        return []


# ══════════════════════════════════════════════════════════════════════
#  CONSTRUCTEUR DE COURS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class CourseBuilder:
    """
    Pipeline complet : fichier/dossier → cours présentable par l'IA.

    Exemple d'usage :
        builder = CourseBuilder()
        course_data = await builder.build_from_file("mon_fichier.pdf", language="en")
    """

    def __init__(self):
        self.extractor  = TextExtractor()
        self.structurer = LocalStructurer(self)  # Passer self pour accès à LLM
        self.llm = Brain()  # Pour génération de résumés

    async def _resolve_section_title(
        self,
        content: str,
        parent_title: str,
        fallback_index: int,
        image_path: str = "",
        language: str = "fr",
    ) -> str:
        """Vision-first title resolution for a slide/section.

        Resolution order:
          1. **Vision LLM** (gpt-4o-mini → Ollama LLaVA) — *looks at* the
             rendered slide PNG and identifies the main concept by reading
             the visual hierarchy (largest/coloured/boxed text). Handles
             arbitrary layouts, image labels, hand-drawn boxes, tables,
             without any rules. Cached on disk by image MD5 — first run
             costs ~$0.0002/slide on OpenAI; re-builds are free.
          2. **Structural fallback** (text-only) — used when no image is
             available, when vision providers are unreachable, or when
             they return UNKNOWN. Same heuristic the planner uses at
             presentation time.
          3. **Numbered placeholder** — used when both above fail, so
             the title is still useful for navigation.

        ``image_path`` is the on-disk path to the slide PNG (or "" when
        unavailable, e.g. for raw-text imports).
        """
        verbose = getattr(Config, "INGESTION_VERBOSE_LOGS", False)
        # Whole-pipeline kill-switch : when DISABLE_VISION_TITLES=true,
        # we skip the vision call entirely and go straight to the
        # structural extractor. ``vision_describe.extract_slide_concept``
        # also enforces this internally; checking here saves a function
        # call and keeps the operator log clean.
        skip_vision = getattr(Config, "DISABLE_VISION_TITLES", False)

        # Always-on per-slide banner. Even without verbose logs, the
        # operator wants to see "page N → title X (source)" for every
        # slide so they can spot bad detections at a glance.
        log.info(
            "─── 📄 Page %d ───────────────────────────────────────────",
            fallback_index,
        )
        log.info("    parent_title : %r", parent_title)
        log.info("    image_path   : %s",
                 Path(image_path).name if image_path else "(none)")

        if verbose:
            log.info("    language     : %s", language)
            log.info("    content excerpt (%d chars total):", len(content))
            for i, line in enumerate((content or "").splitlines()[:10], 1):
                log.info("      %02d │ %s", i, line[:120])
            log.info("    vision_disabled : %s", skip_vision)

        if image_path and not skip_vision:
            try:
                from services.vision_describe import extract_slide_concept
                # Pass parent_title as section_title context — the vision
                # model uses it to ignore the repeating section header
                # at the top of every slide and focus on the slide-
                # specific topic.
                vision_concept = (
                    await extract_slide_concept(
                        image_path,
                        language,
                        section_title=parent_title,
                        slide_text=content,
                    )
                ).strip()
                if vision_concept:
                    log.info(
                        "    ✅ FINAL TITLE (page %d) : %r  [source=vision]",
                        fallback_index, vision_concept,
                    )
                    return vision_concept
                else:
                    log.info(
                        "    ⚠️  vision empty (image=%s) → structural fallback",
                        Path(image_path).name,
                    )
            except Exception as exc:
                # Surface failures at INFO so the operator sees them.
                log.info(
                    "    ⚠️  vision raised: %s → structural fallback",
                    str(exc)[:160],
                )
        elif skip_vision:
            log.info("    ⏭️  vision skipped (DISABLE_VISION_TITLES=true)")
        else:
            log.info("    ℹ️  no image_path → text/structural fallback")

        # ── Step 2 : LLM-text title extraction ──────────────────────
        # Vision saw the image (when enabled) ; this stage looks at the
        # OCR text directly and asks the LLM "what is the title?". It
        # handles cases the regex-based structural extractor fails on:
        # multi-line titles, sub-section bullets being preferred over
        # the page title, repeated chapter headers polluting the output.
        # Routed via LLMRouter, so DISABLE_OPENAI=true sends straight to
        # Ollama. Cached on disk by content MD5.
        if content and content.strip():
            try:
                from services.title_extractor import extract_title_via_llm_async
                llm_title = (await extract_title_via_llm_async(content, language, chapter_title=parent_title)).strip()
                if llm_title:
                    log.info(
                        "    ✅ FINAL TITLE (page %d) : %r  [source=llm]",
                        fallback_index, llm_title,
                    )
                    return llm_title
                else:
                    log.info(
                        "    ⚠️  LLM title empty → structural fallback"
                    )
            except Exception as exc:
                log.info(
                    "    ⚠️  LLM title raised: %s → structural fallback",
                    str(exc)[:160],
                )

        # ── Step 3 : Structural extraction (last-ditch fallback) ────
        title = self.structurer._derive_section_title(
            content, parent_title, fallback_index,
        )
        # Identify which heuristic level fired so the log is meaningful:
        # a real concept extraction vs a numbered placeholder ("Cours —
        # Section N") fallback. The structural extractor uses the latter
        # only when nothing extractable exists in the slide.
        is_placeholder = (
            title.endswith(f" — Section {fallback_index}")
            or title == f"Cours — Section {fallback_index}"
        )
        source = "structural-placeholder" if is_placeholder else "structural"
        log.info(
            "    ✅ FINAL TITLE (page %d) : %r  [source=%s]",
            fallback_index, title, source,
        )
        return title

    async def _generate_section_summary(self, content: str, title: str, language: str = "fr") -> str:
        """
        Génère un résumé concis de la section en utilisant l'LLM.
        """
        try:
            prompt = f"""Résume cette section de cours en 2-3 phrases maximum.
Titre de la section: {title}

Contenu:
{content[:2000]}...  # Tronqué pour éviter dépassement token

Résumé concis:"""

            summary, _ = self.llm.ask(
                question=prompt,
                reply_language=language,
            )
            return summary.strip()
        except Exception as e:
            log.warning(f"Échec génération résumé pour '{title}': {e}")
            # Fallback: extraire première phrase
            sentences = content.split('.')
            return sentences[0].strip() + '.' if sentences else "Section sans résumé disponible."

    @staticmethod
    def _course_slug(value: str | None, fallback: str = DEFAULT_COURSE, domain: str | None = None) -> str:
        candidate = (value or fallback).strip()

        if domain:
            try:
                for existing_course in get_courses(domain):
                    if candidate == existing_course or candidate.lower() == existing_course.lower():
                        return existing_course
            except Exception:
                pass

        candidate = candidate.lower()
        candidate = re.sub(r"[^a-z0-9]+", "_", candidate)
        candidate = candidate.strip("_")
        return candidate or fallback

    @staticmethod
    def _looks_like_chapter(value: str | None) -> bool:
        if not value:
            return False

        normalized = value.strip().lower().replace("\\", "/").split("/")[-1]
        return bool(
            re.search(r"(?:chapter|chapitre|chap)\s*[_\-\s]*\d+", normalized)
            or re.fullmatch(r"chapter[_\-\s]*\d+", normalized)
            or re.fullmatch(r"chapitre[_\-\s]*\d+", normalized)
            or re.fullmatch(r"ch\d+", normalized)
        )

    @staticmethod
    def _chapter_slug(value: str | None, fallback: str = "chapter_1") -> str:
        candidate = (value or fallback).strip().replace("\\", "/").split("/")[-1]
        candidate = candidate.lower()
        candidate = re.sub(r"[^a-z0-9]+", "_", candidate)
        candidate = candidate.strip("_")
        return candidate or fallback

    @staticmethod
    def _display_label(value: str | None, fallback: str) -> str:
        candidate = (value or "").replace("_", " ").replace("-", " ").strip()
        return candidate.title() if candidate else fallback

    def infer_upload_context(
        self,
        upload_name: str | None,
        fallback_domain: str = DEFAULT_DOMAIN,
        fallback_course: str | None = None,
        fallback_chapter: str = "chapter_1",
    ) -> tuple[str, str, str]:
        """Déduit domaine / cours / chapitre depuis le chemin uploadé."""
        raw_path = (upload_name or "").replace("\\", "/").strip()
        parts = [part for part in Path(raw_path).parts if part not in ("", ".", "..")]
        if parts and len(parts[0]) == 2 and parts[0][1] == ":":
            parts = parts[1:]

        folders = parts[:-1]
        file_stem = Path(parts[-1]).stem if parts else ""
        known_domains = {name.lower(): name for name in get_domains()}

        domain = fallback_domain or DEFAULT_DOMAIN
        course = fallback_course if fallback_course and not self._looks_like_chapter(fallback_course) else None
        if course is None and file_stem and not self._looks_like_chapter(file_stem):
            course = file_stem
        if course is None:
            course = DEFAULT_COURSE
        chapter = fallback_chapter or "chapter_1"
        chapter_from_folder = False

        if folders:
            first_folder = folders[0]
            matched_domain = known_domains.get(first_folder.lower())
            if matched_domain:
                domain = matched_domain

            chapter_idx = next(
                (idx for idx in range(len(folders) - 1, -1, -1) if self._looks_like_chapter(folders[idx])),
                None,
            )
            if chapter_idx is not None:
                chapter = folders[chapter_idx]
                chapter_from_folder = True
                if chapter_idx >= 1:
                    course = folders[chapter_idx - 1]
                if chapter_idx >= 2:
                    domain = folders[chapter_idx - 2]
            elif len(folders) >= 3:
                domain, course, chapter = folders[-3], folders[-2], folders[-1]
                chapter_from_folder = True
            elif len(folders) == 2:
                first, second = folders
                if self._looks_like_chapter(second):
                    if not matched_domain:
                        course = first
                    chapter = second
                    chapter_from_folder = True
                else:
                    domain, course = first, second
            elif len(folders) == 1:
                if matched_domain:
                    domain = matched_domain
                elif self._looks_like_chapter(folders[0]):
                    chapter = folders[0]
                    chapter_from_folder = True
                else:
                    course = folders[0]

        if file_stem and self._looks_like_chapter(file_stem) and not chapter_from_folder:
            chapter = file_stem

        if not domain:
            domain = fallback_domain or DEFAULT_DOMAIN
        if not course or self._looks_like_chapter(course):
            if fallback_course and not self._looks_like_chapter(fallback_course):
                course = fallback_course
            elif file_stem and not self._looks_like_chapter(file_stem):
                course = file_stem
            else:
                course = DEFAULT_COURSE

        return (
            domain,
            self._course_slug(course, fallback_course or course or DEFAULT_COURSE, domain=domain),
            self._chapter_slug(chapter, fallback_chapter),
        )

    # ── Pipeline de cours complet ─────────────────────────────────────
    async def build_course_chapters(
        self,
        domain: str = DEFAULT_DOMAIN,
        course: str = DEFAULT_COURSE,
        language: str = "en",
        auto_detect: bool = False,
        sample_file: str | None = None,
    ) -> dict[int, dict]:
        """
        Charge tous les chapitres d'un cours depuis courses/{domain}/{course}/.
        Retourne un dict {chapter_idx: course_data}.

        Args:
            domain: Nom du domaine (ex: "informatique")
            course: Nom du cours (ex: "mathematiques")
            language: Langue (ex: "fr", "en")
            auto_detect: Si True, détecte automatiquement le domaine/cours depuis sample_file
            sample_file: Chemin vers un fichier PDF pour auto-détection

        Structure attendue :
            courses/informatique/mathematiques/
              Chapter 1.pdf
              Chapter 2.pdf
              ...
              Chapter N.pdf

        Exemples :
            # Spécifique
            chapters = await builder.build_course_chapters("informatique", "mathematiques", "fr")

            # Auto-détection depuis un PDF
            chapters = await builder.build_course_chapters(
                auto_detect=True,
                sample_file="Chapter 1.pdf",
                language="fr"
            )
        """
        if auto_detect and sample_file:
            log.info(f"🔍 Auto-détection du domaine/cours depuis : {sample_file}")
            try:
                from core.domains_config import auto_detect_course
                domain, course = auto_detect_course(sample_file)
                log.info(f"✅ Détecté : {domain} / {course}")
            except Exception as e:
                log.info(f"ℹ️  Auto-détection échouée : {e}. Utilisation valeurs par défaut.")
                domain = DEFAULT_DOMAIN
                course = DEFAULT_COURSE

        available_courses = get_courses(domain)
        if course not in available_courses:
            raise ValueError(
                f"Cours '{course}' introuvable dans '{domain}'. "
                f"Disponibles : {available_courses}"
            )

        course_path = Path("courses") / domain / course
        if not course_path.exists():
            raise FileNotFoundError(f"Dossier introuvable : {course_path}")

        log.info(f"\n{'='*60}")
        log.info(f"📚 Construction {domain.upper()} / {course.upper()}")
        log.info(f"📂 Chemin : {course_path}")
        log.info(f"{'='*60}")

        results: dict[int, dict] = {}
        chapters = get_chapters(domain, course)

        for ch_idx, ch_title in chapters.items():
            file_path = self._find_chapter_file(course_path, ch_idx)
            if not file_path:
                log.info(f"ℹ️  Chapitre {ch_idx} introuvable, ignoré")
                continue

            log.info(f"\n📖 Ch{ch_idx} — {ch_title} : {file_path.name}")
            try:
                course_data = await self._build_chapter(
                    file_path,
                    ch_idx,
                    ch_title,
                    language,
                    domain=domain,
                    subject=course,
                )
                results[ch_idx] = course_data
                log.info(f"  ✅ Ch{ch_idx} structuré")
            except Exception as exc:
                log.error(f"  ❌ Ch{ch_idx} échoué : {exc}")

        log.info(f"\n✅ Cours {course} prêt : {len(results)}/{len(chapters)} chapitres chargés")
        return results

    def _infer_context_from_file(
        self,
        file_path: str,
        fallback_domain: str = DEFAULT_DOMAIN,
        fallback_course: str | None = None,
        fallback_chapter: str = "chapter_1",
    ) -> tuple[str, str, str]:
        """Infère la hiérarchie à partir du chemin local sauvegardé."""
        return self.infer_upload_context(
            file_path,
            fallback_domain=fallback_domain,
            fallback_course=fallback_course,
            fallback_chapter=fallback_chapter,
        )

    async def _build_chapter(
        self,
        file_path: Path,
        chapter_idx: int,
        chapter_title: str,
        language: str,
        domain: str = DEFAULT_DOMAIN,
        subject: str = DEFAULT_COURSE,
        chapter: str | None = None,
    ) -> dict:
        """Construit un chapitre depuis un fichier."""
        
        # 🎨 Générer les PNG (images visuelles du cours)
        chapter_slug = self._chapter_slug(chapter or f"chapter_{chapter_idx}", f"chapter_{chapter_idx}")
        course_dir = Path("media/slides") / domain / subject
        chapter_dir = course_dir / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        png_paths = self.structurer._pdf_to_images(
            str(file_path),
            str(chapter_dir)
        )
        log.info(f"  📸 {len(png_paths)} PNG générées → {chapter_dir}")
        
        # Extraction spéciale PPTX (slide par slide)
        if file_path.suffix.lower() == ".pptx":
            slides = self.extractor.extract_structured_pptx(str(file_path))
            raw_text = "\n\n".join(
                f"[Slide {s['slide_idx']}]\n{s['content']}"
                for s in slides
            )
        else:
            raw_text = self.extractor.extract(str(file_path))

        if len(raw_text.strip()) < 100:
            raise ValueError(f"Texte trop court : {len(raw_text)} chars")

        course_data = await self.structurer.structure(
            raw_text=raw_text,
            language=language,
            level="université",
            title=f"Chapter {chapter_idx}: {chapter_title}",
            subject=subject,
            chapter_idx=chapter_idx,
        )
        course_data["chapter_idx"]   = chapter_idx
        course_data["chapter_title"] = chapter_title
        course_data["chapter_slug"]  = chapter_slug
        course_data["file_path"]     = str(file_path)
        course_data["slides"]        = [f"/media/slides/{domain}/{subject}/{chapter_slug}/page_{i+1:03d}.png" for i in range(len(png_paths))]
        return course_data

    def _find_chapter_file(self, course_path: Path, ch_idx: int) -> Path | None:
        """Trouve le fichier d'un chapitre dans le dossier du cours."""
        patterns = [
            f"ch{ch_idx}.pdf", f"ch{ch_idx}.pptx", f"ch{ch_idx}.docx",
            f"CH{ch_idx}.pdf", f"CH{ch_idx}.pptx",
            f"chapter_{ch_idx}.pdf", f"chapter_{ch_idx}.pptx",
            f"Chapter_{ch_idx}.pdf", f"Chapter_{ch_idx}.pptx",
            f"Chapter{ch_idx}.pdf",
        ]
        # Fichiers directs
        for p in patterns:
            fp = course_path / p
            if fp.exists():
                return fp

        # Sous-dossier
        for sub in [f"ch{ch_idx}", f"CH{ch_idx}", f"chapter_{ch_idx}", f"Chapter_{ch_idx}"]:
            sub_path = course_path / sub
            if sub_path.is_dir():
                for ext in ["*.pdf", "*.pptx", "*.docx"]:
                    files = list(sub_path.glob(ext))
                    if files:
                        return files[0]

        # Chercher par numéro dans le nom
        for f in course_path.iterdir():
            name = f.name.lower()
            if (f"ch{ch_idx}" in name or f"chapter{ch_idx}" in name or
                f"_{ch_idx}." in name or f"{ch_idx}." in name) and \
               f.suffix.lower() in (".pdf", ".pptx", ".docx"):
                return f

        return None

    # ── Pipeline fichier unique ───────────────────────────────────────
    async def build_from_file(
        self,
        file_path: str,
        language:  str = "en",
        level:     str = "université",
        subject:   str = "",
        domain:    str = DEFAULT_DOMAIN,  # 🎯 Domaine (général, informatique, etc.)
        chapter:   str = "chapter_1",
    ) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        log.info(f"\n{'='*60}")
        log.info(f"📚 Construction : {path.name}")
        log.info(f"{'='*60}")

        inferred_domain, course_slug, chapter_slug = self._infer_context_from_file(
            str(path),
            fallback_domain=domain,
            fallback_course=subject or None,
            fallback_chapter=chapter,
        )
        domain = inferred_domain or domain

        # 🎨 Générer les PNG (images visuelles du cours)
        course_dir = Path("media/slides") / domain / course_slug
        chapter_dir = course_dir / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        png_paths = self.structurer._pdf_to_images(
            file_path,
            str(chapter_dir)
        )
        log.info(f"  📸 {len(png_paths)} PNG générées → {chapter_dir}")

        raw_text = self.extractor.extract(file_path)
        log.info(f"   ✅ {len(raw_text)} caractères extraits")

        if len(raw_text.strip()) < 100:
            raise ValueError(f"Texte trop court : {len(raw_text)} chars")

        course_data = await self.structurer.structure(
            raw_text=raw_text,
            language=language,
            level=level,
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            subject=course_slug,
        )
        course_data["file_path"] = str(file_path)
        course_data["subject"] = course_slug
        course_data["chapter_slug"] = chapter_slug
        course_data["chapters"][0]["title"] = self._display_label(chapter_slug, "Chapter 1")
        course_data["slides"] = [f"/media/slides/{domain}/{course_slug}/{chapter_slug}/page_{i+1:03d}.png" for i in range(len(png_paths))]
        self._print_summary(course_data)
        return course_data

    async def build_from_file_direct(
        self,
        file_path: str,
        language: str = "en",
        level: str = "université",
        subject: str = "",
        domain: str = DEFAULT_DOMAIN,
        chapter: str = "chapter_1",
    ) -> dict:
        """
        Construit un cours directement depuis le fichier, sans génération IA.
        - PPTX : 1 slide = 1 section (structure conservée)
        - PDF  : 1 page  = 1 section (ordre exact conservé)
        - DOCX/TXT/MD : sections extraites par heuristique sur les titres
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        inferred_domain, course_slug, chapter_slug = self._infer_context_from_file(
            str(path),
            fallback_domain=domain,
            fallback_course=subject or None,
            fallback_chapter=chapter,
        )
        domain = inferred_domain or domain
        chapter_title = self._display_label(chapter_slug, "Chapter 1")

        log.info(f"\n{'='*60}")
        log.info(f"📚 Construction directe (sans IA) : {path.name}")
        log.info(f"{'='*60}")

        sections: list[dict] = []
        ext = path.suffix.lower()
        slides: list[str] = []

        if ext == ".pdf":
            course_dir = Path("media/slides") / domain / course_slug
            chapter_dir = course_dir / chapter_slug
            chapter_dir.mkdir(parents=True, exist_ok=True)

            png_paths = self.structurer._pdf_to_images(str(path), str(chapter_dir))
            slides = [
                f"/media/slides/{domain}/{course_slug}/{chapter_slug}/page_{i+1:03d}.png"
                for i in range(len(png_paths))
            ]

            # Per-slide visual-content detection. Must run BEFORE vision
            # title resolution so the gate in extract_slide_concept can
            # read the cached has_visuals flag and bypass the OCR-based
            # heuristic on slides whose visuals (diagrams, schemas,
            # formula-as-image) the text gate can't see.
            try:
                from services.pdf_visuals import precompute_slide_visuals
                precompute_slide_visuals(str(path), [str(p) for p in png_paths])
            except Exception as exc:                                       # noqa: BLE001
                log.debug(f"pdf_visuals precompute failed: {exc}")

            pages = self._extract_pdf_pages(str(path))

            # Vision-first title resolution. Concurrency is bounded by a
            # semaphore (default 3) so we don't crush the local Ollama
            # vision provider — llama3.2-vision on CPU is single-stream
            # for one model, so >3 in flight queues up at the model
            # server. With OpenAI as the primary provider, 3 in flight
            # is also a polite rate (~$0.0006/s, well below TPM caps).
            # Tunable via env COURSE_BUILDER_VISION_PARALLELISM.
            non_empty = [(idx, txt) for idx, txt in pages if (txt or "").strip()]

            import asyncio
            import os as _os
            _max_parallel = int(_os.getenv("COURSE_BUILDER_VISION_PARALLELISM", "3"))
            _sem = asyncio.Semaphore(max(1, _max_parallel))
            log.info(
                "  🎯 vision title resolution: %d slides, max %d in parallel",
                len(non_empty), _max_parallel,
            )

            async def _resolve_one(page_idx: int, page_text: str) -> tuple[int, str, str]:
                async with _sem:
                    image_disk_path = (
                        str((chapter_dir / f"page_{page_idx:03d}.png").resolve())
                        if page_idx - 1 < len(png_paths)
                        else ""
                    )
                    title = await self._resolve_section_title(
                        content=page_text.strip(),
                        parent_title=chapter_title,
                        fallback_index=page_idx,
                        image_path=image_disk_path,
                        language=language,
                    ) or f"Page {page_idx}"
                    return page_idx, title, page_text.strip()

            resolved = await asyncio.gather(
                *(_resolve_one(idx, txt) for idx, txt in non_empty)
            )

            for page_idx, title, content in resolved:
                sections.append({
                    "title": title,
                    "order": page_idx,
                    "page_index": page_idx,
                    "content": content,
                    "duration_s": self._estimate_duration(content),
                    "concepts": [],
                    "image_url": slides[page_idx - 1] if page_idx - 1 < len(slides) else "",
                })
        elif ext == ".pptx":
            slides_data = self.extractor.extract_structured_pptx(str(path))
            for i, s in enumerate(slides_data, start=1):
                title = (s.get("title") or f"Slide {i}").strip()
                content = (s.get("content") or "").strip()
                if not content:
                    continue
                sections.append({
                    "title": title,
                    "order": i,
                    "page_index": i,
                    "content": content,
                    "duration_s": self._estimate_duration(content),
                    "concepts": [],
                    "image_url": "",
                })
        else:
            raw_text = self.extractor.extract(str(path)).strip()
            if len(raw_text) < 30:
                raise ValueError(f"Texte trop court : {len(raw_text)} chars")

            chunks = self._split_text_into_sections(raw_text)
            for i, chunk in enumerate(chunks, start=1):
                title, content = chunk
                if not content.strip():
                    continue
                sections.append({
                    "title": title or f"Section {i}",
                    "order": i,
                    "page_index": i,
                    "content": content.strip(),
                    "duration_s": self._estimate_duration(content),
                    "concepts": [],
                    "image_url": "",
                })

        if not sections:
            raise ValueError("Impossible d'extraire des sections depuis ce fichier")

        course_data = {
            "title": path.stem.replace("_", " ").replace("-", " ").title(),
            "subject": course_slug,
            "language": language,
            "level": level,
            "description": "Cours importé directement (sans génération IA)",
            "chapters": [
                {
                    "title": chapter_title,
                    "order": 1,
                    "summary": "Import direct du document original",
                    "sections": sections,
                }
            ],
            "file_path": str(path),
            "slides": slides,
            "chapter_slug": chapter_slug,
        }

        self._print_summary(course_data)
        return course_data

    def _extract_pdf_pages(self, path: str) -> list[tuple[int, str]]:
        """Extrait le texte page par page pour conserver l'ordre exact du PDF."""
        try:
            import pypdf
            out: list[tuple[int, str]] = []
            with open(path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module=r"pypdf\.generic\._base")
                    reader = pypdf.PdfReader(f, strict=False)
                    for i, page in enumerate(reader.pages, start=1):
                        out.append((i, (page.extract_text() or "").strip()))
            return out
        except ImportError as exc:
            raise ImportError("Installez pypdf : pip install pypdf") from exc

    async def build_from_text(
        self,
        text:     str,
        title:    str = "Imported Course",
        language: str = "en",
        level:    str = "université",
        subject:  str = "",
    ) -> dict:
        log.info(f"📝 Construction depuis texte : {len(text)} chars")
        course_slug = self._course_slug(subject, title)
        return await self.structurer.structure(
            raw_text=text,
            language=language,
            level=level,
            title=title,
            subject=course_slug,
        )

    async def save_to_database(self, course_data: dict, db, domain: str = DEFAULT_DOMAIN) -> str:
        from database.models import Course, Chapter, Section, Concept
        log.info("💾 Sauvegarde PostgreSQL…")

        course = Course(
            title=course_data.get("title", "Cours importé"),
            domain=domain,  # 🎯 Stocker le domaine
            subject=course_data.get("subject", DEFAULT_COURSE),
            language=course_data.get("language", "en"),
            level=course_data.get("level", "université"),
            description=course_data.get("description", ""),
            file_path=course_data.get("file_path", ""),
        )
        db.add(course)
        await db.flush()

        slides = course_data.get("slides", [])  # 🎨 Récupérer les PNG paths

        for ch_data in course_data.get("chapters", []):
            chapter = Chapter(
                course_id=course.id,
                title=ch_data["title"],
                order=ch_data.get("order", 0),
                summary=ch_data.get("summary", ""),
            )
            db.add(chapter)
            await db.flush()

            for i, sec_data in enumerate(ch_data.get("sections", [])):
                # 🎨 Préserver le lien exact entre la section et sa slide PNG.
                section_order = sec_data.get("page_index") or sec_data.get("order") or (i + 1)
                image_url = (sec_data.get("image_url") or "").strip()

                if not image_url and section_order:
                    slide_index = int(section_order) - 1
                    if 0 <= slide_index < len(slides):
                        image_url = slides[slide_index]

                if not image_url and i < len(slides):
                    image_url = slides[i]

                image_urls = sec_data.get("image_urls") or ([] if not image_url else [image_url])
                
                section = Section(
                    chapter_id=chapter.id,
                    title=sec_data["title"],
                    order=section_order,
                    content=sec_data.get("content", ""),
                    image_url=image_url,  # 🎨 PNG path de cette slide
                    image_urls=image_urls,
                    duration_s=sec_data.get("duration_s", 120),
                )
                db.add(section)
                await db.flush()

                for c_data in sec_data.get("concepts", []):
                    concept = Concept(
                        section_id=section.id,
                        term=c_data.get("term", ""),
                        definition=c_data.get("definition", ""),
                        example=c_data.get("example", ""),
                        concept_type=c_data.get("type", "definition"),
                    )
                    db.add(concept)

        await db.commit()
        course_id = str(course.id)
        log.info(f"✅ Cours sauvegardé : ID={course_id}")
        return course_id

    # ── Multi-chapter ingestion : append to an existing course ──────────
    #
    # Why this exists : a "course" in the pedagogical sense (e.g. "Recherche
    # d'Information") is normally split across MULTIPLE PDFs (Ch.1, Ch.2…).
    # Each PDF should NOT create its own course_id — otherwise RAG retrieval
    # filtered by course_id misses content from other chapters of the same
    # logical course. The methods below let callers either (a) explicitly
    # append to a known course_id, or (b) auto-group by (domain, subject)
    # so a re-upload of the same course slug joins the existing course row.

    # ── Title normalization (strip chapter markers) ─────────────────────
    #
    # Why : LLM classification is non-deterministic. Two PDFs of the same
    # logical course can come back with different slugs ("recherche_information"
    # vs "information_retrieval") or different titles ("Chapitre 1 — Recherche
    # d'Information" vs "Cours RI - Partie 2"). Exact slug match alone misses
    # these cases. We normalize the title (strip chapter markers, accents,
    # punctuation) and use it as a robust grouping signal alongside the slug.

    _CHAPTER_PREFIX_RE = re.compile(
        r"^\s*(?:chapitre|chapter|chapt|chap|ch|"
        r"part|partie|lecture|lec|section|sec|"
        r"module|mod|tp|td|tdtp|unite|unit|"
        r"cours|course|lesson|lecon)"
        r"\s*[\-:.\s]?\s*(?:\d+|[ivxlc]+)\s*[\-:.\s]?\s*",
        re.IGNORECASE,
    )
    _NUMERIC_PREFIX_RE = re.compile(
        r"^\s*(?:\d+|[ivxlc]+)\s*[\-:.\s)]+\s*",
        re.IGNORECASE,
    )

    @classmethod
    def _normalize_course_title(cls, title: str) -> str:
        """Strip chapter markers + accents + punctuation for fuzzy matching.

        Examples
        --------
        ``"Chapitre 1 — Recherche d'Information"`` → ``"recherche d information"``
        ``"Ch. 2 : Data Mining"``                  → ``"data mining"``
        ``"Part III - NLP Basics"``                → ``"nlp basics"``
        """
        import unicodedata as _ud

        if not title:
            return ""
        s = title.strip()

        # Strip up to 2 chapter prefix patterns (handles "Chapitre 1 - 2.1 - Title")
        for _ in range(2):
            stripped = cls._CHAPTER_PREFIX_RE.sub("", s).strip()
            if stripped == s:
                stripped = cls._NUMERIC_PREFIX_RE.sub("", s).strip()
            if stripped == s:
                break
            s = stripped

        s = _ud.normalize("NFKD", s)
        s = "".join(c for c in s if not _ud.combining(c))
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    async def find_existing_course_id(
        self,
        db,
        domain: str,
        subject: str | None = None,
        course_data: dict | None = None,
        rag=None,
        fuzzy_threshold: float = 0.70,
        embedding_threshold: float = 0.80,
    ) -> tuple[str | None, str]:
        """3-level smart matching to find an existing course this PDF
        should join. Returns ``(course_id, reason)`` — reason is for
        logging/debug ("slug-exact", "fuzzy-jaccard=0.83", etc.).

        Resolution order (cheapest → most expensive) :

          1. **Exact slug** : ``courses.subject == subject`` (same domain).
             Cheap, but fails when LLM gives different slugs to the same
             course across uploads.

          2. **Normalized-title token Jaccard** : titles stripped of
             chapter markers, accents, punctuation, then compared as
             bag-of-words. Catches "Recherche d'Information" vs
             "Recherche Information Cours" without any LLM call.

          3. **BGE-m3 embedding cosine** : multilingual semantic match
             via the RAG embedder (if provided). Catches cross-language
             duplicates like "Recherche d'Information" ↔ "Information
             Retrieval". Costs one embedding call per existing course
             title (negligible, titles are short and N is tiny).

        Returns ``(None, reason)`` if nothing matches above thresholds.
        """
        from sqlalchemy import select
        from database.models import Course

        # Level 1 : exact slug match
        if subject and subject not in (DEFAULT_COURSE, "generic", ""):
            stmt = (
                select(Course.id)
                .where(Course.domain == domain, Course.subject == subject)
                .order_by(Course.created_at.asc())
                .limit(1)
            )
            cid = (await db.execute(stmt)).scalar_one_or_none()
            if cid:
                return str(cid), f"slug-exact:{subject}"

        # Need a title for fuzzy / embedding levels
        new_title_raw = (course_data or {}).get("title") or ""
        new_title = self._normalize_course_title(new_title_raw)
        if not new_title:
            return None, "no-title"

        # Fetch all courses in this domain (small N — usually < 100)
        stmt = (
            select(Course.id, Course.title, Course.subject)
            .where(Course.domain == domain)
            .order_by(Course.created_at.asc())
        )
        candidates = list((await db.execute(stmt)).all())
        if not candidates:
            return None, "no-candidates"

        cand_norm = [self._normalize_course_title(t or "") for _, t, _ in candidates]

        # Level 2 : token Jaccard on normalized titles
        new_tokens = set(new_title.split())
        if new_tokens:
            best_score = 0.0
            best_idx = -1
            for i, ct in enumerate(cand_norm):
                ct_tokens = set(ct.split())
                if not ct_tokens:
                    continue
                jaccard = len(new_tokens & ct_tokens) / len(new_tokens | ct_tokens)
                if jaccard > best_score:
                    best_score = jaccard
                    best_idx = i
            if best_idx >= 0 and best_score >= fuzzy_threshold:
                cid = candidates[best_idx][0]
                return str(cid), f"fuzzy-jaccard={best_score:.2f}:{cand_norm[best_idx][:40]}"

        # Level 3 : BGE-m3 embedding cosine
        if rag is not None and getattr(rag, "embeddings", None) is not None:
            try:
                new_emb = rag.embeddings.embed_query(new_title)
                # Build a parallel list of (idx, embedding) skipping empty titles
                idx_with_titles = [(i, ct) for i, ct in enumerate(cand_norm) if ct]
                if idx_with_titles:
                    cand_embs = rag.embeddings.embed_documents(
                        [ct for _, ct in idx_with_titles]
                    )

                    def _cos(a, b):
                        import math
                        dp = sum(x * y for x, y in zip(a, b))
                        na = math.sqrt(sum(x * x for x in a))
                        nb = math.sqrt(sum(x * x for x in b))
                        return dp / (na * nb) if na and nb else 0.0

                    best_score = 0.0
                    best_idx = -1
                    for (i, ct), emb in zip(idx_with_titles, cand_embs):
                        sim = _cos(new_emb, emb)
                        if sim > best_score:
                            best_score = sim
                            best_idx = i
                    if best_idx >= 0 and best_score >= embedding_threshold:
                        cid = candidates[best_idx][0]
                        return str(cid), f"embedding-cos={best_score:.2f}:{cand_norm[best_idx][:40]}"
            except Exception as exc:    # noqa: BLE001
                log.warning("find_existing_course_id : embedding level failed (%s) — skipping", exc)

        return None, "no-match"

    async def append_chapters_to_course(self, course_data: dict, db, course_id: str) -> int:
        """Append chapters / sections / concepts from ``course_data`` into
        an existing Course row. No new Course() created.

        Chapter.order is auto-assigned to ``max(existing.order) + 1`` so new
        chapters always land at the end of the course outline. Returns the
        number of chapters added.
        """
        import uuid as _uuid_mod
        from sqlalchemy import select, func
        from database.models import Course, Chapter, Section, Concept

        course = await db.get(Course, _uuid_mod.UUID(course_id))
        if course is None:
            raise ValueError(f"append_chapters_to_course : course_id not found : {course_id}")

        res = await db.execute(
            select(func.coalesce(func.max(Chapter.order), 0))
            .where(Chapter.course_id == course.id)
        )
        next_order = int(res.scalar() or 0) + 1

        slides = course_data.get("slides", [])
        n_added = 0

        for ch_data in course_data.get("chapters", []):
            chapter = Chapter(
                course_id=course.id,
                title=ch_data["title"],
                order=next_order,
                summary=ch_data.get("summary", ""),
            )
            db.add(chapter)
            await db.flush()
            next_order += 1
            n_added += 1

            for i, sec_data in enumerate(ch_data.get("sections", [])):
                section_order = sec_data.get("page_index") or sec_data.get("order") or (i + 1)
                image_url = (sec_data.get("image_url") or "").strip()
                if not image_url and section_order:
                    slide_idx = int(section_order) - 1
                    if 0 <= slide_idx < len(slides):
                        image_url = slides[slide_idx]
                if not image_url and i < len(slides):
                    image_url = slides[i]
                image_urls = sec_data.get("image_urls") or ([] if not image_url else [image_url])

                section = Section(
                    chapter_id=chapter.id,
                    title=sec_data["title"],
                    order=section_order,
                    content=sec_data.get("content", ""),
                    image_url=image_url,
                    image_urls=image_urls,
                    duration_s=sec_data.get("duration_s", 120),
                )
                db.add(section)
                await db.flush()

                for c_data in sec_data.get("concepts", []):
                    db.add(Concept(
                        section_id=section.id,
                        term=c_data.get("term", ""),
                        definition=c_data.get("definition", ""),
                        example=c_data.get("example", ""),
                        concept_type=c_data.get("type", "definition"),
                    ))

        await db.commit()
        log.info(f"✅ Append : {n_added} chapter(s) added to course_id={course_id}")
        return n_added

    async def save_or_append_smart(
        self,
        course_data: dict,
        db,
        domain: str = DEFAULT_DOMAIN,
        append_to: str | None = None,
        auto_group: bool = True,
        rag=None,
    ) -> tuple[str, str]:
        """Smart persistence : either CREATE a new course or APPEND to an
        existing one. Returns ``(course_id, action)`` where action is one
        of ``"created"`` / ``"appended:<reason>"``.

        Resolution order :
          1. Explicit ``append_to`` → APPEND to that course_id.
          2. ``auto_group=True`` → call :meth:`find_existing_course_id`
             with the 3-level matcher (slug → title fuzzy → embedding).
             If a match is found, APPEND.
          3. Otherwise → CREATE a new course.

        ``rag`` is the optional RAG instance ; passing it enables the
        embedding-similarity level of the matcher (multilingual via
        BGE-m3). Without it, only slug + fuzzy levels are used.
        """
        if append_to:
            await self.append_chapters_to_course(course_data, db, append_to)
            return append_to, "appended:explicit"

        if auto_group:
            subject = course_data.get("subject") or ""
            existing, reason = await self.find_existing_course_id(
                db, domain,
                subject=subject,
                course_data=course_data,
                rag=rag,
            )
            if existing:
                log.info(f"🔗 Auto-group : appending to course_id={existing} (reason={reason})")
                await self.append_chapters_to_course(course_data, db, existing)
                return existing, f"appended:{reason}"

        course_id = await self.save_to_database(course_data, db, domain=domain)
        return course_id, "created"

    def _estimate_duration(self, text: str) -> int:
        """Estime une durée de lecture (en secondes) selon le nombre de mots."""
        words = len((text or "").split())
        # ~130 mots/minute avec bornes raisonnables
        return max(30, min(600, int((words / 130.0) * 60)))

    def _split_text_into_sections(self, raw_text: str) -> list[tuple[str, str]]:
        """
        Découpe un texte brut en sections à partir des lignes de titre probables.
        Retourne [(title, content), ...].

        Heading detection is purely structural — no vocabulary lists.
        A line is a section break if any of these hold:
          - It is short (≤ 120 chars) AND all-uppercase (a SHOUTED title).
          - It begins with a numeric outline marker (``1.``, ``2.3``,
            ``1.1.4)``, ``III.``) — universal section numbering shape.
          - It is short AND doesn't end with sentence punctuation
            (``.``, ``!``, ``?``) AND has ≤ 8 words — fits the noun-
            phrase shape of headings.
        """
        lines = [ln.rstrip() for ln in raw_text.splitlines()]
        cleaned = [ln for ln in lines if ln.strip()]
        if not cleaned:
            return []

        # Numeric outline marker: "1.", "1.2", "1.2.3)", "III." etc.
        numbered_re = re.compile(
            r"^(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+)[\)\.]?\s+.+$",
            re.IGNORECASE,
        )

        def _looks_like_heading(line: str) -> bool:
            stripped = line.strip()
            if not stripped or len(stripped) > 120:
                return False
            if stripped.isupper() and len(stripped) >= 4:
                return True
            if numbered_re.match(stripped):
                return True
            # Short noun-phrase heading: ≤ 8 words, no sentence terminator,
            # starts with capital. Matches "Photosynthesis", "Quantum
            # States", "Trimmed mean" — works in any language whose
            # alphabet has a notion of upper case.
            if (
                len(stripped) <= 80
                and stripped[0].isupper()
                and stripped[-1] not in ".!?"
                and len(stripped.split()) <= 8
            ):
                return True
            return False

        sections: list[tuple[str, list[str]]] = []
        current_title = "Introduction"
        current_lines: list[str] = []

        for ln in cleaned:
            if _looks_like_heading(ln) and current_lines:
                sections.append((current_title, current_lines))
                current_title = ln.strip()
                current_lines = []
            elif _looks_like_heading(ln) and not current_lines and current_title == "Introduction":
                current_title = ln.strip()
            else:
                current_lines.append(ln)

        if current_lines:
            sections.append((current_title, current_lines))

        if not sections:
            text = "\n".join(cleaned)
            return [("Contenu", text)]

        out: list[tuple[str, str]] = []
        for title, body_lines in sections:
            content = "\n".join(body_lines).strip()
            if content:
                out.append((title.strip() or "Section", content))

        if not out:
            text = "\n".join(cleaned)
            return [("Contenu", text)]
        return out

    def _print_summary(self, data: dict) -> None:
        chapters = data.get("chapters", [])
        total_sections = sum(len(ch.get("sections", [])) for ch in chapters)
        total_concepts = sum(
            len(sec.get("concepts", []))
            for ch in chapters
            for sec in ch.get("sections", [])
        )
        total_duration = sum(
            sec.get("duration_s", 120)
            for ch in chapters
            for sec in ch.get("sections", [])
        )
        log.info(f"\n{'='*60}")
        log.info(f"✅ COURS : {data.get('title')}")
        log.info(f"   Matière    : {data.get('subject')}")
        log.info(f"   Niveau     : {data.get('level')}")
        log.info(f"   Langue     : {data.get('language')}")
        log.info(f"   Chapitres  : {len(chapters)}")
        log.info(f"   Sections   : {total_sections}")
        log.info(f"   Concepts   : {total_concepts}")
        log.info(f"   Durée est. : {total_duration // 60} min")
        log.info(f"{'='*60}\n")
        for i, ch in enumerate(chapters):
            log.info(f"  📖 Ch{i+1} : {ch['title']}")
            for j, sec in enumerate(ch.get("sections", [])):
                log.info(f"     └─ §{j+1} : {sec['title']} ({sec.get('duration_s',120)}s)")
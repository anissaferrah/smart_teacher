"""Intelligent Ingester — extraction COMPLETE d'un PDF avant le course builder.

Pipeline d'ingestion :
  1. PDF → text par page (pypdf)
  2. PDF → PNG slides (pdf2image) — pour affichage frontend
  3. PDF → images embarquees (pypdf) — pour KG / OCR
  4. Images → OCR (pytesseract) — texte des schemas/figures
  5. PDF → unstructured elements (Tables, FigureCaptions) — structure semantique

Resultat : IngestionResult avec TOUTES les donnees, alimente :
  - RAG (texte + captions)
  - Concept Extractor (concepts + KG)
  - CourseBuilder (structure pedagogique)
  - Frontend (slides PNG + image references)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger("SmartTeacher.IntelligentIngester")


@dataclass
class IngestedAsset:
    """Un asset extrait du PDF : texte, image, table, ou caption."""
    asset_type: str               # "text" | "image" | "table" | "caption" | "title"
    page_num: int                 # 1-based
    text: str = ""                # contenu textuel (ou OCR pour images)
    image_path: str = ""          # chemin disque relatif si type=image
    image_index_in_page: int = 0  # ordre dans la page (pour images multiples)
    # Coordonnees bbox de l'image dans sa page PDF (PyMuPDF only).
    # Permet de recropper / re-rendre la figure pour un retrieval spatial
    # (ex: "show the figure for concept X"). None si extraction via pypdf
    # (qui n'expose pas les bbox).
    # Format : {"x0", "y0", "x1", "y1", "page_width", "page_height",
    #           "x0_norm", "y0_norm", "x1_norm", "y1_norm"} en points PDF.
    bbox: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionResult:
    """Output complet d'une ingestion PDF — feed le course_builder + RAG."""
    file_path: str                # chemin PDF source
    course_id: str = ""           # course_id (UUID si deja attribue)
    pages: list[dict[str, Any]] = field(default_factory=list)  # [{page, text, slide_path}]
    slide_pngs: list[str] = field(default_factory=list)        # paths PNG par page
    assets: list[IngestedAsset] = field(default_factory=list)  # tous les assets
    elements: list[Any] = field(default_factory=list)          # raw unstructured Elements
    language: str = "en"
    total_pages: int = 0
    total_images: int = 0
    total_tables: int = 0
    total_captions: int = 0
    extraction_time_s: float = 0.0

    def assets_by_type(self, kind: str) -> list[IngestedAsset]:
        return [a for a in self.assets if a.asset_type == kind]

    def assets_by_page(self, page: int) -> list[IngestedAsset]:
        return [a for a in self.assets if a.page_num == page]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "course_id": self.course_id,
            "language": self.language,
            "total_pages": self.total_pages,
            "total_images": self.total_images,
            "total_tables": self.total_tables,
            "total_captions": self.total_captions,
            "extraction_time_s": self.extraction_time_s,
            "slide_pngs": self.slide_pngs,
            "pages": self.pages,
            "assets": [a.to_dict() for a in self.assets],
        }


class IntelligentIngester:
    """Orchestre l'extraction multi-modale (PDF, PPTX, DOCX, TXT, MD, HTML, …)."""

    SUPPORTED_EXTS = {".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
    PARA_PER_PAGE = 30   # DOCX: regrouper N paragraphes par "page logique"

    def __init__(
        self,
        ocr_languages: str = "fra+eng",
        slide_dpi: int = 150,
        # Optimisations OCR
        min_image_size_bytes: int = 5_000,      # skip si image < 5KB (probable bruit)
        min_image_dim_px: int = 50,              # skip si <50px de cote (icones)
        # HTML download
        enable_html_image_download: bool = True,
        html_image_max_bytes: int = 5_000_000,   # 5 MB max par image HTML
        # Vision LLM (optional, costly)
        enable_vision_llm: bool | None = None,   # None=auto via env USE_VISION_LLM
        vision_llm_model: str = "gpt-4o-mini",
        # PPTX rendering
        libreoffice_bin: str = "libreoffice",   # ou "soffice" sur certains systemes
    ) -> None:
        from core.config import Config
        self.ocr_languages = ocr_languages
        self.slide_dpi = slide_dpi
        self.min_image_size_bytes = Config.OCR_MIN_IMAGE_BYTES
        self.min_image_dim_px = Config.OCR_MIN_IMAGE_DIM_PX
        self.enable_html_image_download = enable_html_image_download
        self.html_image_max_bytes = html_image_max_bytes
        if enable_vision_llm is None:
            enable_vision_llm = Config.USE_VISION_LLM
        self.enable_vision_llm = enable_vision_llm
        self.vision_llm_model = Config.VISION_LLM_MODEL
        self.libreoffice_bin = Config.LIBREOFFICE_BIN
        # Caches
        self._ocr_cache: dict[str, str] = {}      # hash(bytes) → text
        self._vision_cache: dict[str, str] = {}   # hash(bytes) → description

    # ── Public API : dispatcher ────────────────────────────────────────────

    async def ingest_file(
        self,
        file_path: str,
        media_root: str = "media/courses",
        course_id: str = "",
    ) -> IngestionResult:
        """Dispatcher : route vers le handler selon l'extension."""
        ext = Path(file_path).suffix.lower()
        log.info(f"📥 Ingest file ext={ext} path={file_path}")
        if ext == ".pdf":
            return await self.ingest_pdf(file_path, media_root, course_id)
        if ext in {".pptx", ".ppt"}:
            return await self.ingest_pptx(file_path, media_root, course_id)
        if ext in {".docx", ".doc"}:
            return await self.ingest_docx(file_path, media_root, course_id)
        if ext in {".txt", ".md"}:
            return await self.ingest_text(file_path, media_root, course_id)
        if ext in {".html", ".htm"}:
            return await self.ingest_html(file_path, media_root, course_id)
        # Fallback : try unstructured.partition.auto (handles many formats)
        log.info("  → fallback to ingest_generic (unstructured)")
        return await self.ingest_generic(file_path, media_root, course_id)

    # ── PDF (handler historique) ───────────────────────────────────────────

    async def ingest_pdf(
        self,
        pdf_path: str,
        media_root: str = "media/courses",
        course_id: str = "",
    ) -> IngestionResult:
        """Ingestion complete d'un PDF.

        Args:
            pdf_path   : chemin absolu vers le PDF
            media_root : repertoire racine pour les assets extraits
            course_id  : UUID du cours (si deja attribue) — sinon un dossier temp est cree

        Returns IngestionResult avec pages, slides PNG, images extraites + OCR, tables, captions.
        """
        import time
        start = time.time()
        pdf_path_obj = Path(pdf_path).resolve()
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Setup output dir : media/courses/{course_id_or_stem}/
        course_key = course_id or pdf_path_obj.stem.replace(" ", "_")
        out_dir = Path(media_root) / course_key
        slides_dir = out_dir / "slides"
        images_dir = out_dir / "images"
        slides_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        result = IngestionResult(file_path=str(pdf_path_obj), course_id=course_id)

        # Step 1 : pages text (pypdf, leger)
        pages_text = self._extract_pages_text(str(pdf_path_obj))
        result.total_pages = len(pages_text)
        log.info(f"📄 Pages text : {result.total_pages} pages extracted")

        # Step 2 : PNG slides (pdf2image) — pour affichage
        slide_paths = self._render_slides_to_png(str(pdf_path_obj), slides_dir)
        result.slide_pngs = [str(p) for p in slide_paths]
        log.info(f"🖼️  PNG slides : {len(slide_paths)} generated @ {self.slide_dpi}dpi")

        # Compose pages list (texte + slide path par page)
        for i, txt in enumerate(pages_text, start=1):
            slide_path = str(slide_paths[i - 1]) if i - 1 < len(slide_paths) else ""
            result.pages.append({
                "page_num": i,
                "text": txt,
                "slide_path": slide_path,
            })
            # Asset texte par page
            if txt.strip():
                result.assets.append(IngestedAsset(
                    asset_type="text", page_num=i, text=txt[:5000],
                ))

        # Step 3 : images embedded → save + OCR
        try:
            n_imgs, image_assets = self._extract_and_ocr_images(
                str(pdf_path_obj), images_dir,
            )
            result.total_images = n_imgs
            result.assets.extend(image_assets)
            log.info(f"🖼️  Embedded images : {n_imgs} extracted + OCR'd")
        except Exception as exc:
            log.warning(f"image extraction failed: {exc}")

        # Step 4 : unstructured.partition → Tables, FigureCaptions
        try:
            elements = self._run_unstructured(str(pdf_path_obj))
            result.elements = elements
            for el in elements:
                el_type = type(el).__name__
                page_num = self._safe_page_num(el)
                text = (getattr(el, "text", "") or "").strip()
                if not text:
                    continue
                if el_type == "Table":
                    result.assets.append(IngestedAsset(
                        asset_type="table", page_num=page_num, text=text[:3000],
                        metadata={"category": el_type},
                    ))
                    result.total_tables += 1
                elif el_type == "FigureCaption":
                    result.assets.append(IngestedAsset(
                        asset_type="caption", page_num=page_num, text=text[:1000],
                        metadata={"category": el_type},
                    ))
                    result.total_captions += 1
                elif el_type == "Title":
                    result.assets.append(IngestedAsset(
                        asset_type="title", page_num=page_num, text=text[:300],
                        metadata={"category": el_type},
                    ))
            log.info(
                f"📦 Unstructured elements : {len(elements)} "
                f"(tables={result.total_tables}, captions={result.total_captions})"
            )
        except Exception as exc:
            log.warning(f"unstructured partition failed: {exc}")

        # Step 5 : detect language sur le texte agrege (premier 3000 chars)
        all_text = "\n".join(p.get("text", "") for p in result.pages[:5])[:3000]
        result.language = self._detect_lang(all_text)

        result.extraction_time_s = round(time.time() - start, 2)
        log.info(
            f"✅ Ingestion complete : pages={result.total_pages} "
            f"images={result.total_images} tables={result.total_tables} "
            f"captions={result.total_captions} elapsed={result.extraction_time_s}s"
        )
        return result

    # ── PPTX (1 slide = 1 page) ────────────────────────────────────────────

    async def ingest_pptx(
        self, pptx_path: str, media_root: str = "media/courses", course_id: str = "",
    ) -> IngestionResult:
        import time
        start = time.time()
        path = Path(pptx_path).resolve()
        course_key = course_id or path.stem.replace(" ", "_")
        out_dir = Path(media_root) / course_key
        slides_dir = out_dir / "slides"
        images_dir = out_dir / "images"
        slides_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        result = IngestionResult(file_path=str(path), course_id=course_id)

        # 0. PNG slides : tente conversion LibreOffice headless → PDF intermediate → PNG
        try:
            slide_paths = self._render_pptx_slides_to_png(str(path), slides_dir)
            if slide_paths:
                result.slide_pngs = [str(p) for p in slide_paths]
                log.info(f"🖼️  PPTX slides PNG : {len(slide_paths)} via LibreOffice")
        except Exception as exc:
            log.debug(f"PPTX slide render skipped: {exc}")

        try:
            from pptx import Presentation
            prs = Presentation(str(path))
        except Exception as exc:
            log.warning(f"PPTX parse failed: {exc}")
            return result

        for slide_idx, slide in enumerate(prs.slides, start=1):
            page_texts: list[str] = []
            n_imgs_in_slide = 0
            for shape_idx, shape in enumerate(slide.shapes):
                # Texte
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text.strip():
                            page_texts.append(para.text.strip())
                # Image
                try:
                    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                        img = shape.image
                        ext = "." + (img.ext or "png").lower().lstrip(".")
                        out_path = images_dir / f"slide{slide_idx:03d}_img{n_imgs_in_slide:02d}{ext}"
                        with open(out_path, "wb") as fh:
                            fh.write(img.blob)
                        ocr_text = self._ocr_image(str(out_path))
                        vision_desc = self._describe_image_with_vision_llm(str(out_path))
                        combined = ocr_text + (f"\n[Vision] {vision_desc}" if vision_desc else "")
                        result.assets.append(IngestedAsset(
                            asset_type="image", page_num=slide_idx,
                            text=combined[:2000], image_path=str(out_path),
                            image_index_in_page=n_imgs_in_slide,
                            metadata={
                                "size_bytes": len(img.blob),
                                "ocr_chars": len(ocr_text),
                                "vision_description": vision_desc[:500],
                            },
                        ))
                        n_imgs_in_slide += 1
                        result.total_images += 1
                except Exception as exc:
                    log.debug(f"pptx shape image extract: {exc}")
                # Notes (texte des notes du speaker)
            try:
                notes_frame = slide.notes_slide.notes_text_frame if slide.has_notes_slide else None
                if notes_frame and notes_frame.text.strip():
                    page_texts.append(f"[Notes] {notes_frame.text.strip()}")
            except Exception:
                pass

            page_text = "\n".join(page_texts).strip()
            # Link au PNG slide rendered si disponible
            slide_png = result.slide_pngs[slide_idx - 1] if slide_idx - 1 < len(result.slide_pngs) else ""
            result.pages.append({
                "page_num": slide_idx, "text": page_text, "slide_path": slide_png,
            })
            if page_text:
                result.assets.append(IngestedAsset(
                    asset_type="text", page_num=slide_idx, text=page_text[:5000],
                ))
            result.total_pages += 1

        # Detect language sur les premiers slides
        all_text = "\n".join(p.get("text", "") for p in result.pages[:5])[:3000]
        result.language = self._detect_lang(all_text)
        result.extraction_time_s = round(time.time() - start, 2)
        log.info(
            f"✅ PPTX ingested : {result.total_pages} slides, {result.total_images} images, "
            f"slide PNGs: {len(result.slide_pngs)} ({result.extraction_time_s}s)"
        )
        return result

    # ── DOCX (paragraphes regroupes par "page logique") ────────────────────

    async def ingest_docx(
        self, docx_path: str, media_root: str = "media/courses", course_id: str = "",
    ) -> IngestionResult:
        import time
        start = time.time()
        path = Path(docx_path).resolve()
        course_key = course_id or path.stem.replace(" ", "_")
        out_dir = Path(media_root) / course_key
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        result = IngestionResult(file_path=str(path), course_id=course_id)
        try:
            from docx import Document
            doc = Document(str(path))
        except Exception as exc:
            log.warning(f"DOCX parse failed: {exc}")
            return result

        # Paragraphes (texte + niveau heading pour titles)
        paragraphs: list[tuple[str, str]] = []  # (text, style_name)
        for para in doc.paragraphs:
            txt = para.text.strip()
            if txt:
                style = (para.style.name if para.style else "") or ""
                paragraphs.append((txt, style))

        # Tables -> markdown
        for tbl_idx, table in enumerate(doc.tables, start=1):
            md_rows: list[str] = []
            for r_idx, row in enumerate(table.rows):
                cells = [c.text.strip() or " " for c in row.cells]
                md_rows.append("| " + " | ".join(cells) + " |")
                if r_idx == 0 and cells:
                    md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            if md_rows:
                # Affecte la table a sa "page logique" (estimation)
                result.assets.append(IngestedAsset(
                    asset_type="table", page_num=max(1, tbl_idx),
                    text="\n".join(md_rows)[:3000],
                    metadata={"category": "DocxTable"},
                ))
                result.total_tables += 1

        # Images : DOCX zip → word/media/*
        try:
            import zipfile
            with zipfile.ZipFile(str(path)) as zf:
                media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
                for img_idx, name in enumerate(media_files):
                    raw = zf.read(name)
                    ext = Path(name).suffix.lower() or ".png"
                    out_path = images_dir / f"docx_img{img_idx:03d}{ext}"
                    with open(out_path, "wb") as fh:
                        fh.write(raw)
                    ocr_text = self._ocr_image(str(out_path))
                    vision_desc = self._describe_image_with_vision_llm(str(out_path))
                    combined = ocr_text + (f"\n[Vision] {vision_desc}" if vision_desc else "")
                    result.assets.append(IngestedAsset(
                        asset_type="image", page_num=img_idx + 1,
                        text=combined[:2000], image_path=str(out_path),
                        metadata={
                            "size_bytes": len(raw),
                            "ocr_chars": len(ocr_text),
                            "vision_description": vision_desc[:500],
                        },
                    ))
                    result.total_images += 1
        except Exception as exc:
            log.debug(f"docx image extract failed: {exc}")

        # Build "pages" en regroupant N paragraphes
        for i in range(0, len(paragraphs), self.PARA_PER_PAGE):
            chunk = paragraphs[i: i + self.PARA_PER_PAGE]
            page_num = (i // self.PARA_PER_PAGE) + 1
            page_text = "\n".join(p[0] for p in chunk)
            result.pages.append({
                "page_num": page_num, "text": page_text, "slide_path": "",
            })
            result.assets.append(IngestedAsset(
                asset_type="text", page_num=page_num, text=page_text[:5000],
            ))
            # Heading-style paragraphs → asset "title"
            for txt, style in chunk:
                if style and "Heading" in style:
                    result.assets.append(IngestedAsset(
                        asset_type="title", page_num=page_num, text=txt[:300],
                        metadata={"docx_style": style},
                    ))
            result.total_pages += 1

        all_text = "\n".join(p.get("text", "") for p in result.pages[:3])[:3000]
        result.language = self._detect_lang(all_text)
        result.extraction_time_s = round(time.time() - start, 2)
        log.info(
            f"✅ DOCX ingested : {len(paragraphs)} paras → {result.total_pages} pages, "
            f"{result.total_tables} tables, {result.total_images} images "
            f"({result.extraction_time_s}s)"
        )
        return result

    # ── TXT / MD (1 page unique) ───────────────────────────────────────────

    async def ingest_text(
        self, file_path: str, media_root: str = "media/courses", course_id: str = "",
    ) -> IngestionResult:
        import time
        start = time.time()
        path = Path(file_path).resolve()
        result = IngestionResult(file_path=str(path), course_id=course_id)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.warning(f"text file read failed: {exc}")
            return result

        # Pour MD : extraire les # comme titles
        if path.suffix.lower() == ".md":
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    title_text = stripped.lstrip("#").strip()
                    if title_text:
                        result.assets.append(IngestedAsset(
                            asset_type="title", page_num=1, text=title_text[:300],
                            metadata={"md_level": stripped.count("#", 0, 6)},
                        ))

        result.pages.append({"page_num": 1, "text": content, "slide_path": ""})
        result.total_pages = 1
        if content.strip():
            result.assets.append(IngestedAsset(
                asset_type="text", page_num=1, text=content[:5000],
            ))
        result.language = self._detect_lang(content[:3000])
        result.extraction_time_s = round(time.time() - start, 2)
        log.info(f"✅ Text/MD ingested : {len(content)} chars ({result.extraction_time_s}s)")
        return result

    # ── HTML (BeautifulSoup) ───────────────────────────────────────────────

    async def ingest_html(
        self, file_path: str, media_root: str = "media/courses", course_id: str = "",
    ) -> IngestionResult:
        import time
        start = time.time()
        path = Path(file_path).resolve()
        course_key = course_id or path.stem.replace(" ", "_")
        out_dir = Path(media_root) / course_key
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        result = IngestionResult(file_path=str(path), course_id=course_id)
        try:
            from bs4 import BeautifulSoup
            html = path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            log.warning(f"HTML parse failed: {exc}")
            return result

        # Texte agrege
        text_content = soup.get_text(separator="\n", strip=True)
        result.pages.append({"page_num": 1, "text": text_content, "slide_path": ""})
        result.total_pages = 1
        if text_content:
            result.assets.append(IngestedAsset(
                asset_type="text", page_num=1, text=text_content[:5000],
            ))

        # Headings → titles
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                txt = heading.get_text(strip=True)
                if txt:
                    result.assets.append(IngestedAsset(
                        asset_type="title", page_num=1, text=txt[:300],
                        metadata={"html_level": level},
                    ))

        # Tables → markdown
        for tbl_idx, table in enumerate(soup.find_all("table")):
            rows = table.find_all("tr")
            if not rows:
                continue
            md_rows = []
            for r_idx, row in enumerate(rows):
                cells = [c.get_text(strip=True) or " " for c in row.find_all(["td", "th"])]
                md_rows.append("| " + " | ".join(cells) + " |")
                if r_idx == 0 and cells:
                    md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            result.assets.append(IngestedAsset(
                asset_type="table", page_num=1, text="\n".join(md_rows)[:3000],
                metadata={"category": "HtmlTable"},
            ))
            result.total_tables += 1

        # Images <img> : download + OCR + Vision LLM (toggle via enable_html_image_download)
        if self.enable_html_image_download:
            try:
                import requests
            except Exception:
                requests = None
            for img_idx, img_tag in enumerate(soup.find_all("img")):
                src = (img_tag.get("src") or "").strip()
                alt = img_tag.get("alt", "") or ""
                if not src or src.startswith(("data:", "javascript:")):
                    continue
                if src.startswith("//"):
                    src = "https:" + src
                # Si URL relative et pas de base href, on tente comme path local
                local_resolved = None
                if not src.startswith(("http://", "https://")):
                    candidate = (path.parent / src).resolve()
                    if candidate.exists():
                        local_resolved = candidate
                if local_resolved is None and not src.startswith(("http://", "https://")):
                    # URL relative impossible a resoudre → skip download mais garde URL
                    result.assets.append(IngestedAsset(
                        asset_type="image", page_num=1, text=alt[:300],
                        image_path=src,
                        metadata={"url_only": True, "alt": alt, "reason": "relative_no_base"},
                    ))
                    result.total_images += 1
                    continue
                # Download ou copie locale
                try:
                    if local_resolved is not None:
                        content = local_resolved.read_bytes()
                        ext = local_resolved.suffix.lower() or ".png"
                    else:
                        if requests is None:
                            raise RuntimeError("requests not available")
                        r = requests.get(src, timeout=10, stream=True)
                        if r.status_code != 200:
                            raise RuntimeError(f"HTTP {r.status_code}")
                        content = b""
                        for chunk in r.iter_content(8192):
                            content += chunk
                            if len(content) > self.html_image_max_bytes:
                                break
                        ext = self._guess_image_ext(src)
                    if len(content) < self.min_image_size_bytes:
                        continue
                    out_path = images_dir / f"html_img{img_idx:03d}{ext}"
                    with open(out_path, "wb") as fh:
                        fh.write(content)
                    ocr_text = self._ocr_image(str(out_path))
                    vision_desc = self._describe_image_with_vision_llm(str(out_path))
                    combined = ocr_text + (f"\n[Vision] {vision_desc}" if vision_desc else "")
                    if not combined and alt:
                        combined = alt
                    result.assets.append(IngestedAsset(
                        asset_type="image", page_num=1,
                        text=combined[:2000], image_path=str(out_path),
                        image_index_in_page=img_idx,
                        metadata={
                            "alt": alt, "src_url": src, "size_bytes": len(content),
                            "ocr_chars": len(ocr_text),
                            "vision_description": vision_desc[:500],
                        },
                    ))
                    result.total_images += 1
                except Exception as exc:
                    log.debug(f"HTML image download failed for {src}: {exc}")
                    # Fallback : juste l'URL
                    result.assets.append(IngestedAsset(
                        asset_type="image", page_num=1, text=alt[:300],
                        image_path=src,
                        metadata={"url_only": True, "alt": alt, "reason": str(exc)[:100]},
                    ))
                    result.total_images += 1
        else:
            # URL only mode (legacy)
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                alt = img_tag.get("alt", "")
                if src:
                    result.assets.append(IngestedAsset(
                        asset_type="image", page_num=1, text=alt[:300],
                        image_path=src, metadata={"url_only": True, "alt": alt},
                    ))
                    result.total_images += 1

        result.language = self._detect_lang(text_content[:3000])
        result.extraction_time_s = round(time.time() - start, 2)
        log.info(
            f"✅ HTML ingested : {len(text_content)} chars, "
            f"{result.total_tables} tables, {result.total_images} images "
            f"({result.extraction_time_s}s)"
        )
        return result

    # ── Generic fallback (unstructured.partition.auto) ─────────────────────

    async def ingest_generic(
        self, file_path: str, media_root: str = "media/courses", course_id: str = "",
    ) -> IngestionResult:
        """Fallback : tente unstructured.partition.auto qui supporte de nombreux formats
        (RTF, EPUB, EML, etc). Pas d'extraction d'images dediee."""
        import time
        start = time.time()
        path = Path(file_path).resolve()
        result = IngestionResult(file_path=str(path), course_id=course_id)
        try:
            elements = self._run_unstructured(str(path))
        except Exception as exc:
            log.warning(f"ingest_generic failed: {exc}")
            return result

        # Group by page_num
        text_per_page: dict[int, list[str]] = {}
        for el in elements:
            page_num = self._safe_page_num(el) or 1
            text = (getattr(el, "text", "") or "").strip()
            if text:
                text_per_page.setdefault(page_num, []).append(text)
            el_type = type(el).__name__
            if el_type == "Table" and text:
                result.assets.append(IngestedAsset(
                    asset_type="table", page_num=page_num, text=text[:3000],
                    metadata={"category": el_type},
                ))
                result.total_tables += 1
            elif el_type == "FigureCaption" and text:
                result.assets.append(IngestedAsset(
                    asset_type="caption", page_num=page_num, text=text[:1000],
                    metadata={"category": el_type},
                ))
                result.total_captions += 1
            elif el_type == "Title" and text:
                result.assets.append(IngestedAsset(
                    asset_type="title", page_num=page_num, text=text[:300],
                    metadata={"category": el_type},
                ))

        for page_num, texts in sorted(text_per_page.items()):
            page_text = "\n".join(texts)
            result.pages.append({"page_num": page_num, "text": page_text, "slide_path": ""})
            result.assets.append(IngestedAsset(
                asset_type="text", page_num=page_num, text=page_text[:5000],
            ))
        result.total_pages = len(result.pages)
        all_text = "\n".join(p.get("text", "") for p in result.pages[:3])[:3000]
        result.language = self._detect_lang(all_text)
        result.extraction_time_s = round(time.time() - start, 2)
        log.info(
            f"✅ Generic ingested : {result.total_pages} pages, "
            f"{result.total_tables} tables ({result.extraction_time_s}s)"
        )
        return result

    # ── Helpers internes ────────────────────────────────────────────────────

    def _extract_pages_text(self, pdf_path: str) -> list[str]:
        """Texte page par page via pypdf (rapide, robuste)."""
        try:
            import pypdf
            import warnings
            out: list[str] = []
            with open(pdf_path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module=r"pypdf\.generic\._base")
                    reader = pypdf.PdfReader(f, strict=False)
                    for page in reader.pages:
                        out.append((page.extract_text() or "").strip())
            return out
        except Exception as exc:
            log.warning(f"_extract_pages_text failed: {exc}")
            return []

    def _render_slides_to_png(self, pdf_path: str, out_dir: Path) -> list[Path]:
        """PDF → PNG une image par page via pdf2image."""
        try:
            import tempfile
            from pdf2image import convert_from_path
            # output_folder makes pdftoppm write PNGs to disk directly instead
            # of streaming all pages through the subprocess stdout pipe
            # (the latter can blow up with MemoryError on large/high-DPI decks).
            with tempfile.TemporaryDirectory() as tmp_dir:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.slide_dpi,
                    output_folder=tmp_dir,
                )
                paths: list[Path] = []
                for i, img in enumerate(images, start=1):
                    path = out_dir / f"slide_{i:03d}.png"
                    img.save(str(path), format="PNG", optimize=True)
                    paths.append(path)
                return paths
        except Exception as exc:
            log.warning(f"_render_slides_to_png failed: {exc}")
            return []

    def _render_pptx_slides_to_png(self, pptx_path: str, out_dir: Path) -> list[Path]:
        """PPTX → PNG via LibreOffice headless conversion (PPTX → PDF → PNG).

        Requires LibreOffice ou soffice installe et accessible dans PATH.
        Returns empty list si LibreOffice non disponible.
        """
        import subprocess
        import shutil
        import tempfile

        bin_path = shutil.which(self.libreoffice_bin) or shutil.which("soffice")
        if not bin_path:
            log.info(f"LibreOffice non trouve ({self.libreoffice_bin}/soffice) → skip PNG render")
            return []

        # Step 1 : convert PPTX → PDF in temp dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                result_proc = subprocess.run(
                    [bin_path, "--headless", "--convert-to", "pdf",
                     "--outdir", tmp_dir, pptx_path],
                    capture_output=True, timeout=120, check=True,
                )
                pdf_name = Path(pptx_path).stem + ".pdf"
                pdf_path = Path(tmp_dir) / pdf_name
                if not pdf_path.exists():
                    log.warning(f"LibreOffice didn't produce PDF: stdout={result_proc.stdout[:200]!r}")
                    return []
                # Step 2 : PDF → PNG via pdf2image (existant)
                return self._render_slides_to_png(str(pdf_path), out_dir)
            except subprocess.TimeoutExpired:
                log.warning("LibreOffice conversion timeout (>120s)")
                return []
            except subprocess.CalledProcessError as exc:
                log.warning(f"LibreOffice conversion failed: {exc.stderr[:200]!r}")
                return []
            except Exception as exc:
                log.warning(f"_render_pptx_slides_to_png error: {exc}")
                return []

    def _extract_and_ocr_images(
        self, pdf_path: str, out_dir: Path,
    ) -> tuple[int, list[IngestedAsset]]:
        """Extract images embarquees du PDF + OCR via Tesseract.

        Prefere PyMuPDF (fitz) — donne les bbox spatiales (Fix #5). Fallback
        pypdf si pymupdf indisponible. Le bbox est stocke dans
        ``IngestedAsset.bbox`` pour retrieval spatial ("show me the figure
        for concept X").
        """
        # Chemin prefere : PyMuPDF (donne bbox)
        try:
            import fitz  # type: ignore  # pymupdf
            return self._extract_and_ocr_images_fitz(pdf_path, out_dir)
        except ImportError:
            log.info("pymupdf not installed → fallback pypdf (no bbox metadata)")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"pymupdf extraction failed ({exc}) → fallback pypdf")

        # Fallback : pypdf (pas de bbox)
        return self._extract_and_ocr_images_pypdf(pdf_path, out_dir)

    def _extract_and_ocr_images_fitz(
        self, pdf_path: str, out_dir: Path,
    ) -> tuple[int, list[IngestedAsset]]:
        """Extract via PyMuPDF — inclut bbox (x0, y0, x1, y1) par image."""
        import fitz  # type: ignore
        assets: list[IngestedAsset] = []
        n_total = 0
        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                page_num = page_idx + 1
                page_w = float(page.rect.width)
                page_h = float(page.rect.height)
                images = page.get_images(full=True)
                for img_idx, img_info in enumerate(images):
                    n_total += 1
                    xref = img_info[0]
                    # Bbox : page.get_image_rects(xref) — peut renvoyer plusieurs
                    # rectangles si la meme image est placee 2x sur la page
                    try:
                        rects = page.get_image_rects(xref) or []
                    except Exception:
                        rects = []
                    rect = rects[0] if rects else None

                    # Extract image bytes
                    try:
                        base = doc.extract_image(xref)
                        ext = "." + (base.get("ext") or "png").lstrip(".")
                        img_bytes = base["image"]
                    except Exception as exc:
                        log.debug(f"extract_image xref={xref} failed: {exc}")
                        continue

                    out_name = f"page{page_num:03d}_img{img_idx:02d}{ext}"
                    out_path = out_dir / out_name
                    try:
                        with open(out_path, "wb") as fh:
                            fh.write(img_bytes)
                    except Exception as exc:
                        log.debug(f"failed to write image: {exc}")
                        continue

                    ocr_text = self._ocr_image(str(out_path))
                    vision_desc = self._describe_image_with_vision_llm(str(out_path))
                    combined = ocr_text
                    if vision_desc:
                        combined = (combined + "\n" if combined else "") + f"[Vision] {vision_desc}"

                    bbox: dict[str, float] | None = None
                    if rect is not None:
                        x0, y0, x1, y1 = (
                            float(rect.x0), float(rect.y0),
                            float(rect.x1), float(rect.y1),
                        )
                        bbox = {
                            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                            "page_width": page_w, "page_height": page_h,
                            "x0_norm": x0 / page_w if page_w else 0.0,
                            "y0_norm": y0 / page_h if page_h else 0.0,
                            "x1_norm": x1 / page_w if page_w else 0.0,
                            "y1_norm": y1 / page_h if page_h else 0.0,
                        }

                    assets.append(IngestedAsset(
                        asset_type="image",
                        page_num=page_num,
                        text=combined[:2000],
                        image_path=str(out_path),
                        image_index_in_page=img_idx,
                        bbox=bbox,
                        metadata={
                            "name": f"xref_{xref}",
                            "size_bytes": len(img_bytes),
                            "ocr_chars": len(ocr_text),
                            "vision_description": vision_desc[:500],
                            "extractor": "pymupdf",
                        },
                    ))
            return n_total, assets
        finally:
            doc.close()

    def _extract_and_ocr_images_pypdf(
        self, pdf_path: str, out_dir: Path,
    ) -> tuple[int, list[IngestedAsset]]:
        """Fallback extraction via pypdf — pas de bbox."""
        assets: list[IngestedAsset] = []
        n_total = 0
        try:
            import pypdf
            import warnings
            with open(pdf_path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    reader = pypdf.PdfReader(f, strict=False)
                    for page_idx, page in enumerate(reader.pages, start=1):
                        try:
                            images = list(page.images)
                        except Exception:
                            images = []
                        for img_idx, img_obj in enumerate(images):
                            n_total += 1
                            ext = self._guess_image_ext(img_obj.name)
                            out_name = f"page{page_idx:03d}_img{img_idx:02d}{ext}"
                            out_path = out_dir / out_name
                            try:
                                with open(out_path, "wb") as fh:
                                    fh.write(img_obj.data)
                            except Exception as exc:
                                log.debug(f"failed to write image: {exc}")
                                continue
                            ocr_text = self._ocr_image(str(out_path))
                            vision_desc = self._describe_image_with_vision_llm(str(out_path))
                            combined = ocr_text
                            if vision_desc:
                                combined = (combined + "\n" if combined else "") + f"[Vision] {vision_desc}"
                            assets.append(IngestedAsset(
                                asset_type="image",
                                page_num=page_idx,
                                text=combined[:2000],
                                image_path=str(out_path),
                                image_index_in_page=img_idx,
                                bbox=None,
                                metadata={
                                    "name": getattr(img_obj, "name", ""),
                                    "size_bytes": len(img_obj.data) if hasattr(img_obj, "data") else 0,
                                    "ocr_chars": len(ocr_text),
                                    "vision_description": vision_desc[:500],
                                    "extractor": "pypdf",
                                },
                            ))
            return n_total, assets
        except Exception as exc:
            log.warning(f"_extract_and_ocr_images_pypdf failed: {exc}")
            return n_total, assets

    @staticmethod
    def _looks_like_math_garbage(text: str) -> bool:
        """Heuristic: did Tesseract fail on a math image?

        Tesseract on a formula like ``Σ xᵢ / n`` typically produces
        garbled output ("£ x; / n", ":2 a;", "ic - ie") because it
        tries to read math glyphs as Latin letters. We flag short
        outputs where alphanumeric chars are at most half the length,
        leaving the rest as punctuation/symbols — a clear sign that
        Tesseract gave up on a math image.

        Pure structural — no vocabulary. False positives just trigger
        an extra (cheap) pix2tex call that returns "" if the image
        wasn't a formula.
        """
        s = (text or "").strip()
        if not s:
            return False
        if len(s) < 25:
            alpha = sum(1 for c in s if c.isalpha())
            digits = sum(1 for c in s if c.isdigit())
            if (alpha + digits) / max(len(s), 1) <= 0.5:
                return True
        return False

    def _try_math_ocr(self, image_path: str) -> str:
        """Try pix2tex (LaTeX-OCR) on a formula-shaped image.

        Returns a string of the form ``[FORMULA: \\frac{x}{y}]`` on
        success, empty string on failure. The wrapper marker is what
        downstream RAG / TTS code uses to recognise math content
        (so it gets verbalised correctly during narration).

        Opt-in via ``ENABLE_MATH_OCR=true`` because pix2tex pulls
        ~500 MB of model weights on first run and is CPU-slow
        (1-3s per image). On a fresh machine without the env flag
        we just skip and let Tesseract own the OCR.
        """
        import os as _os
        if _os.getenv("ENABLE_MATH_OCR", "false").lower() != "true":
            return ""
        try:
            from pix2tex.cli import LatexOCR  # heavy import — guarded
        except ImportError:
            log.debug("pix2tex not installed — math OCR disabled")
            return ""
        # Lazy-load and cache the model on the instance.
        if getattr(self, "_pix2tex", None) is None:
            try:
                self._pix2tex = LatexOCR()
                log.info("✅ pix2tex (LaTeX-OCR) loaded for math images")
            except Exception as exc:                                        # noqa: BLE001
                log.warning(f"pix2tex load failed: {exc}")
                self._pix2tex = False
                return ""
        if self._pix2tex is False:
            return ""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                latex = self._pix2tex(img.convert("RGB"))
            latex = (latex or "").strip()
            if latex and any(c in latex for c in ("\\", "_", "^", "{", "}")):
                return f"[FORMULA: {latex}]"
            return ""
        except Exception as exc:                                            # noqa: BLE001
            log.debug(f"pix2tex inference failed: {exc}")
            return ""

    def _ocr_image(self, image_path: str) -> str:
        """OCR optimise via pytesseract :
          - cache par hash MD5 (meme image → skip Tesseract)
          - filtre size (< min_image_size_bytes)
          - filtre dimensions (< min_image_dim_px)
          - early-exit si image solide-color
          - fallback math OCR (pix2tex) si Tesseract retourne garbage
        """
        import hashlib
        import os as _os
        try:
            # Size filter (rapide)
            try:
                file_size = _os.path.getsize(image_path)
                if file_size < self.min_image_size_bytes:
                    return ""
            except OSError:
                return ""

            # Hash pour cache
            with open(image_path, "rb") as fh:
                img_bytes = fh.read()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in self._ocr_cache:
                return self._ocr_cache[img_hash]

            # Dimensions + entropy filter
            from PIL import Image
            with Image.open(image_path) as img:
                w, h = img.size
                if w < self.min_image_dim_px or h < self.min_image_dim_px:
                    self._ocr_cache[img_hash] = ""
                    return ""
                # Entropy quick check : si image nearly mono-color, skip
                try:
                    extrema = img.convert("L").getextrema()
                    if extrema and (extrema[1] - extrema[0]) < 30:
                        self._ocr_cache[img_hash] = ""
                        return ""
                except Exception:
                    pass

                import pytesseract
                txt = (pytesseract.image_to_string(img, lang=self.ocr_languages) or "").strip()
                # Math OCR fallback: if Tesseract output looks like
                # gibberish AND the image looks like a formula
                # (small/wide), try pix2tex. The result is wrapped as
                # [FORMULA: \\frac{...}] so downstream RAG/TTS can
                # detect and verbalise it correctly.
                if self._looks_like_math_garbage(txt):
                    is_formula_shape = (h <= 200 and w / max(h, 1) >= 1.2)
                    if is_formula_shape:
                        latex = self._try_math_ocr(image_path)
                        if latex:
                            log.info(
                                "🧮 math OCR rescued garbage Tesseract output: %s",
                                latex[:80],
                            )
                            txt = latex if not txt else f"{txt}\n{latex}"
                self._ocr_cache[img_hash] = txt
                return txt
        except Exception as exc:
            log.debug(f"_ocr_image failed for {image_path}: {exc}")
            return ""

    def _describe_image_with_vision_llm(self, image_path: str, lang: str = "en") -> str:
        """Delegate to ``services.vision_describe.describe_slide_image``.

        The previous implementation here was a 50-line OpenAI-only inline
        version with an in-RAM cache and no fallback. The runtime path
        (``handlers/ws.py``) already used ``services.vision_describe``,
        which provides :

          - a multi-provider chain (OpenAI gpt-4o-mini → Ollama LLaVA),
          - a disk cache keyed on image MD5 (survives restarts and
            re-ingestions, bilingual cache keys),
          - single-flight coalescing for parallel callers,
          - process-local stats for observability.

        Reusing it here means re-ingestion no longer pays the OpenAI
        bill twice and the ingester gets the Ollama fallback for free
        when the OpenAI quota is exhausted mid-run.

        The service is async; the ingester runs in a sync context, so
        we drive it via ``asyncio.run`` (creating a fresh loop each
        call). When the ingester itself runs inside an event loop
        (rare — most ingestion paths are scripts), we fall through to
        a no-op.
        """
        if not self.enable_vision_llm:
            return ""
        try:
            import os as _os
            if _os.path.getsize(image_path) < self.min_image_size_bytes:
                return ""
        except Exception as exc:                                          # noqa: BLE001
            log.debug(f"vision describe: stat failed for {image_path}: {exc}")
            return ""

        try:
            import asyncio
            from services.vision_describe import describe_slide_image
        except Exception as exc:                                          # noqa: BLE001
            log.debug(f"vision describe import failed: {exc}")
            return ""

        try:
            # Run the async service synchronously. ``asyncio.run`` raises
            # if a loop is already running in this thread; in that case
            # the ingester is being driven from an async context and the
            # caller should refactor — for now we degrade to no-op.
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is not None:
                log.debug("vision describe: skipped (already inside an event loop)")
                return ""
            description = asyncio.run(describe_slide_image(image_path, lang=lang))
            if description:
                log.info(f"🔭 Vision LLM described image ({len(description)} chars, lang={lang})")
            return description or ""
        except Exception as exc:                                          # noqa: BLE001
            log.debug(f"_describe_image_with_vision_llm failed: {exc}")
            return ""

    def _run_unstructured(self, pdf_path: str) -> list[Any]:
        """Run unstructured.partition.auto for Tables + FigureCaptions.

        Strategy resolution
        -------------------
        We try ``hi_res`` first because it's the only strategy that
        actually extracts tables (it runs a layout detection model
        + table-transformer to recover row/column structure). If
        ``hi_res`` fails — typically because the heavy optional deps
        aren't installed (``detectron2``, ``paddleocr``, etc.) — we
        fall back to ``auto`` which gives text only (current
        behaviour).

        For pure-Python table detection without the heavy deps, we
        also try pdfplumber as a complementary extractor and merge
        its tables into the elements list. Pdfplumber catches
        vector-based tables that ``hi_res`` sometimes misses on
        slides with thin borders.

        Both extractors are best-effort: failure on either path is
        logged at WARN level and the pipeline continues with what's
        available.
        """
        from unstructured.partition.auto import partition
        elements: list[Any] = []

        # Tier 1: unstructured hi_res (with table inference). Falls
        # back to auto if optional layout deps are missing.
        try:
            elements = partition(
                filename=pdf_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng", "fra"],
            ) or []
        except Exception as exc:                                            # noqa: BLE001
            log.info(
                "unstructured hi_res strategy unavailable (%s) → falling back to auto",
                str(exc)[:120],
            )
            try:
                elements = partition(filename=pdf_path) or []
            except Exception as exc2:                                       # noqa: BLE001
                log.warning(f"unstructured auto failed too: {exc2}")
                elements = []

        # Tier 2: pdfplumber table extraction (pure-Python, fast).
        # Adds Table-like elements that downstream code can pick up.
        try:
            extra_tables = self._extract_tables_pdfplumber(pdf_path)
            if extra_tables:
                log.info("pdfplumber extracted %d tables", len(extra_tables))
                elements.extend(extra_tables)
        except Exception as exc:                                            # noqa: BLE001
            log.debug(f"pdfplumber table extraction skipped: {exc}")

        return elements

    def _extract_tables_pdfplumber(self, pdf_path: str) -> list[Any]:
        """Extract tables from a PDF using pdfplumber and wrap each one
        in a lightweight Element-shaped object compatible with the
        unstructured downstream consumers.

        Each table is rendered as Markdown so embedding/retrieval can
        index it as text. Header row (first non-empty row) becomes
        the markdown header. Empty rows/cells are skipped.
        """
        try:
            import pdfplumber
        except ImportError:
            return []

        class _TableElement:
            """Duck-typed substitute for unstructured.Table — has .text,
            .category, and .metadata.page_number that downstream code
            reads via ``getattr``."""
            def __init__(self, text: str, page_number: int):
                self.text = text
                self.category = "Table"
                class _Meta:
                    pass
                self.metadata = _Meta()
                self.metadata.page_number = page_number

            def __str__(self) -> str:
                return self.text

        def _table_to_markdown(rows: list[list[Any]]) -> str:
            cleaned = [
                [(c if c is not None else "").strip().replace("\n", " ") for c in row]
                for row in rows
                if any((c or "").strip() for c in row)
            ]
            if not cleaned:
                return ""
            header = cleaned[0]
            body = cleaned[1:]
            ncols = max(len(r) for r in cleaned)
            header += [""] * (ncols - len(header))
            md_lines = ["| " + " | ".join(header) + " |"]
            md_lines.append("| " + " | ".join(["---"] * ncols) + " |")
            for r in body:
                r = r + [""] * (ncols - len(r))
                md_lines.append("| " + " | ".join(r) + " |")
            return "\n".join(md_lines)

        out: list[Any] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    continue
                for tbl in tables:
                    md = _table_to_markdown(tbl)
                    if md:
                        out.append(_TableElement(md, page_number=i))
        return out

    @staticmethod
    def _safe_page_num(element) -> int:
        """Extract page_number depuis un Element unstructured (1-based)."""
        try:
            metadata = getattr(element, "metadata", None)
            if metadata is None:
                return 0
            page = getattr(metadata, "page_number", 0)
            return int(page) if page else 0
        except Exception:
            return 0

    @staticmethod
    def _guess_image_ext(name: str) -> str:
        if not name:
            return ".png"
        lower = name.lower()
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"):
            if lower.endswith(ext):
                return ext
        return ".png"

    @staticmethod
    def _detect_lang(text: str) -> str:
        try:
            from langdetect import detect
            return detect(text)[:5] if text else "en"
        except Exception:
            return "en"

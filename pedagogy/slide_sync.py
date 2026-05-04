"""
╔══════════════════════════════════════════════════════════════════════╗
║        SMART TEACHER — SlideSynchronizer                           ║
║                                                                      ║
║  Génère des slides structurées depuis les sections du cours.        ║
║  Chaque slide a un TYPE visuel distinct rendu dans index.html.      ║
║                                                                      ║
║  Types de slides :                                                   ║
║    chapter_intro  — titre du chapitre + plan des sections           ║
║    section        — titre + bullet points extraits du texte         ║
║    concept        — terme + définition + exemple en évidence        ║
║    image          — image extraite du PDF (GPT-4o Vision)           ║
║    summary        — récap en fin de section longue                  ║
║                                                                      ║
║  Principe fondamental :                                              ║
║    slide.content = TEXTE ORIGINAL, jamais modifié, utilisé par TTS  ║
║    slide.bullets  = points extraits du texte pour l'affichage       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("SmartTeacher.SlideSync")


# ══════════════════════════════════════════════════════════════════════
#  DATACLASS SLIDE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Slide:
    index:      int
    slide_type: str              # "chapter_intro"|"section"|"concept"|"image"|"summary"
    title:      str
    content:    str              # texte ORIGINAL → lu par le TTS, jamais modifié
    bullets:    list[str]        # points visuels extraits → affichage uniquement
    keywords:   list[str]        # mots-clés surlignés pendant la lecture
    image_url:  Optional[str] = None   # URL image extraite par RAG (GPT-4o Vision)
    duration_s: float = 30.0
    # Champs spécifiques type "concept"
    concept_term: Optional[str] = None
    concept_def:  Optional[str] = None
    concept_ex:   Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
#  SLIDE SYNCHRONIZER
# ══════════════════════════════════════════════════════════════════════

class SlideSynchronizer:
    """
    Génère les slides visuelles depuis une section de cours et les
    convertit en événements WebSocket envoyés au frontend.

    Utilisation dans main.py (present_section) :
        slides = slide_sync.generate_slides_from_section(section)
        # Envoyer slide 0 immédiatement (affichage pendant le TTS)
        await ws.send_json(slide_sync.slide_to_ws_event(slides[0], chapter, pct))
        # TTS lit content_txt (texte original)
        audio = await voice.generate_audio_async(section["content"])
        # Envoyer les slides suivantes (concepts, images, résumé)
        for slide in slides[1:]:
            await ws.send_json(slide_sync.slide_to_ws_event(slide, chapter, pct))
    """

    # ── API principale ─────────────────────────────────────────────────────────

    def generate_slides_from_section(self, section: dict) -> list[Slide]:
        """
        Génère toutes les slides d'une section de cours.

        Args:
            section: dict avec les clés :
                - title    (str)           titre de la section
                - content  (str)           texte original complet
                - concepts (list[dict])    [{term, definition, example}]
                - image_url (str|None)     image extraite par le RAG (optionnel)
                - duration_s (float)       durée estimée (optionnel)

        Returns:
            Liste de Slide ordonnées :
              [0]   slide principale (section)
              [1..] slides concepts (une par concept)
              [-2]  slide image si disponible
              [-1]  slide résumé si section > 150 mots
        """
        slides: list[Slide] = []
        title     = section.get("title", "")
        content   = section.get("content", "")
        concepts  = section.get("concepts", [])
        image_url = section.get("image_url")

        # ── Slide 0 : Section principale ─────────────────────────────────────
        word_count = len(content.split())
        slides.append(Slide(
            index      = 0,
            slide_type = "section",
            title      = title,
            content    = content,                        # texte original → TTS
            bullets    = self._extract_bullets(content), # points visuels
            keywords   = self._extract_keywords(content),
            image_url  = image_url,
            duration_s = max(10.0, word_count / 2.5),
        ))

        # ── Slides concepts : une par concept clé ─────────────────────────────
        for i, concept in enumerate(concepts):
            term = concept.get("term", "").strip()
            defn = concept.get("definition", "").strip()
            ex   = concept.get("example", "").strip()
            if not term:
                continue

            # Texte lu à voix haute = texte original du concept (pas reformulé)
            tts_text = term
            if defn:
                tts_text += f" : {defn}"
            if ex:
                tts_text += f". Par exemple : {ex}"

            slides.append(Slide(
                index        = i + 1,
                slide_type   = "concept",
                title        = term,
                content      = tts_text,   # lu par TTS
                bullets      = [defn] + ([f"Ex : {ex}"] if ex else []),
                keywords     = [term],
                duration_s   = 12.0,
                concept_term = term,
                concept_def  = defn,
                concept_ex   = ex or None,
            ))

        # ── Slide image si disponible (extraite par GPT-4o Vision via RAG) ───
        if image_url:
            slides.append(Slide(
                index      = len(concepts) + 1,
                slide_type = "image",
                title      = title,
                content    = "",          # pas de TTS pour la slide image
                bullets    = [],
                keywords   = [],
                image_url  = image_url,
                duration_s = 6.0,
            ))

        # ── Slide résumé si section longue (> 150 mots) ───────────────────────
        if word_count > 150:
            summary_bullets = self._extract_bullets(content, max_points=5)
            if summary_bullets:
                slides.append(Slide(
                    index      = len(slides),
                    slide_type = "summary",
                    title      = f"Résumé — {title}",
                    content    = "",        # pas lu à voix haute séparément
                    bullets    = summary_bullets,
                    keywords   = self._extract_keywords(content),
                    duration_s = 6.0,
                ))

        return slides

    def generate_chapter_intro_slide(
        self,
        chapter_title:  str,
        section_titles: list[str],
        chapter_index:  int = 0,
    ) -> Slide:
        """
        Crée la slide d'introduction d'un chapitre avec son plan.
        Envoyée avant la première section du chapitre.
        """
        return Slide(
            index      = 0,
            slide_type = "chapter_intro",
            title      = f"Chapitre {chapter_index + 1} — {chapter_title}",
            content    = f"Au programme : {', '.join(section_titles[:6])}",
            bullets    = section_titles[:8],   # max 8 sections affichées
            keywords   = [chapter_title],
            duration_s = 5.0,
        )

    def slide_to_ws_event(
        self,
        slide:        Slide,
        chapter_title: str = "",
        progress_pct:  int = 0,
    ) -> dict:
        """
        Convertit une slide en dict WebSocket prêt à envoyer.
        Le frontend (index.html updateSlide) utilise slide_type pour
        choisir le rendu visuel approprié.
        """
        event: dict = {
            "type":          "slide_update",
            "slide_type":    slide.slide_type,
            "slide_index":   slide.index,
            "slide_title":   slide.title,
            "slide_content": slide.content,
            "bullets":       slide.bullets,
            "keywords":      slide.keywords,
            "image_url":     slide.image_url,
            "chapter":       chapter_title,
            "progress_pct":  progress_pct,
            "duration_s":    slide.duration_s,
        }
        # Données supplémentaires pour le rendu "concept"
        if slide.slide_type == "concept":
            event["concept_term"] = slide.concept_term
            event["concept_def"]  = slide.concept_def
            event["concept_ex"]   = slide.concept_ex
        return event

    # ── Utilitaires privés ─────────────────────────────────────────────────────

    def _extract_bullets(self, text: str, max_points: int = 6) -> list[str]:
        """
        Extrait des points visuels depuis le texte original.
        Ordre de priorité :
          1. Listes existantes (•, -, *, 1. 2.)
          2. Phrases définitoires courtes (contenant "est", "sont", ":")
          3. Premières phrases du texte
        Le texte original n'est JAMAIS modifié.
        """
        if not text or not text.strip():
            return []

        # 1. Listes existantes
        list_items = re.findall(
            r"^[ \t]*(?:[•\-\*]|\d+[.)]) +(.+)$",
            text, re.MULTILINE
        )
        if list_items:
            return [p.strip() for p in list_items[:max_points] if len(p.strip()) > 5]

        # 2. Phrases définitoires
        sentences = re.split(r"(?<=[.!?])\s+", text)
        points: list[str] = []
        for s in sentences:
            s = s.strip()
            if 15 < len(s) < 160:
                low = s.lower()
                if any(kw in low for kw in [
                    "est ", "sont ", "signifie", " : ", "désigne",
                    "correspond", "permet", "consiste", "définit",
                    "is ", "are ", "means", "refers to",
                    "هو ", "هي ", "تعني", "يُعرَّف",
                ]):
                    points.append(s)
                    if len(points) >= max_points:
                        break

        # 3. Fallback : premières phrases
        if not points:
            for s in sentences[:max_points]:
                s = s.strip()
                if 15 < len(s) < 160:
                    points.append(s)

        return points[:max_points]

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extrait les mots-clés : termes entre guillemets + mots techniques en majuscule.
        Limité à 5 mots-clés uniques.
        """
        if not text:
            return []
        quoted   = re.findall(r'[«»"""]([^«»"""]{2,30})[«»"""]', text)
        capitals = re.findall(
            r'(?<![.!?\n])\b[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜ][a-záàâäéèêëîïôùûü]{3,}\b',
            text,
        )
        return list(dict.fromkeys(quoted + capitals))[:5]

    def extract_slides_from_pdf(self, pdf_path: str) -> list[dict]:
        """
        Extrait les pages d'un PDF comme slides brutes.
        Chaque page = un dict {page, content, total}.
        Utilisé pour prévisualiser un PDF avant ingestion.
        """
        try:
            import pypdf
            slides = []
            with open(pdf_path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module=r"pypdf\.generic\._base")
                    reader = pypdf.PdfReader(f, strict=False)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            slides.append({
                                "page":    i + 1,
                                "content": text.strip(),    # contenu complet, non tronqué
                                "total":   len(reader.pages),
                            })
            return slides
        except Exception as exc:
            log.warning(f"PDF extraction failed: {exc}")
            return []
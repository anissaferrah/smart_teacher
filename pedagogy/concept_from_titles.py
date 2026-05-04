"""Concept extraction by slide-title grouping.

# Approche

Le cours est structuré en sections nommées. Chaque slide porte un
``section_title`` qui EST le concept enseigné — c'est déjà dans les
metadata après ingestion. Donc :

  1. Group by ``section_title`` (gratuit, déterministe)
  2. LLM enrichit chaque titre (canonical, description, bloom_level)
  3. Construit ``Concept`` pour le KG

# Avantages

  - Pas de bruit (les titres sont écrits par le prof, pas extraits du texte)
  - Couverture 100% des chunks (chaque chunk a un titre)
  - 1 appel LLM par concept (~9 calls pour un cours typique)
  - Reflète la pédagogie réelle du cours

# Limites

  - Suppose un cours bien structuré (PDF avec ``section_title`` extrait
    pendant l'ingestion). Sur transcript brut sans titres → ne marche pas.
  - Si le prof réutilise le même titre pour 2 concepts différents,
    on les fusionne (limite acceptable : même contexte pédagogique).
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Optional

from pedagogy.concept_types import Concept, _slugify

log = logging.getLogger("SmartTeacher.ConceptFromTitles")


# ── Tunables ────────────────────────────────────────────────────────────

# Titres trop génériques à ignorer (ne sont pas des concepts pédagogiques)
_NOISE_TITLES = {
    "", "...", "slide", "intro", "introduction", "outline",
    "summary", "résumé", "resume", "agenda", "plan",
    "references", "bibliographie", "bibliography",
    "thanks", "merci", "questions", "q&a", "qna",
}

# Min de chunks par titre. À 1, on garde TOUS les titres rencontrés —
# même ceux apparus une seule fois — comme concepts candidats. Le filtre
# de bruit `_NOISE_TITLES` reste actif pour écarter ("intro", "summary"…).
MIN_CHUNKS_PER_TITLE = 1

# Limite haute du contenu envoyé au LLM, en caractères. Mistral 7B a ~8K
# tokens de contexte (~24K chars) ; on laisse de la marge pour le prompt
# + la réponse. À 20K chars on couvre des sections de ~80 slides.
# Si une section dépasse, on tronque pour éviter de planter le LLM.
EXCERPT_TRUNCATE = 20000

# Pas de cap sur le nombre de sous-concepts par titre : on laisse le LLM
# en lister autant qu'il en trouve dans le contenu. Le substring match
# filtre naturellement les hallucinations (sous-concept jamais cité dans
# le texte → fallback au parent, score réduit). Si une section dense
# comme "Ensemble learning" en mentionne 15, on les garde tous.

# Score multiplicateur appliqué aux sous-concepts (parent garde priorité).
# Un sous-concept avec 8 chunks aura score = 8 × 0.8 = 6.4 ; le parent
# avec 73 chunks reste largement au-dessus dans le tri.
SUB_CONCEPT_SCORE_FACTOR = 0.8


class ConceptFromTitles:
    """Extracteur de concepts à partir des titres de section.

    Retourne ``list[Concept]`` (compatible ``ensure_concepts_loaded`` + KG).

    Usage :
        extractor = ConceptFromTitles(rag.embeddings)
        concepts = extractor.extract(documents=rag.all_docs, course_id="...")
    """

    def __init__(self, embedding_model: Any | None = None) -> None:
        # ``embedding_model`` accepté pour compatibilité de signature mais
        # pas utilisé : cette méthode ne dépend pas des embeddings.
        self.embedding_model = embedding_model
        log.info("✅ ConceptFromTitles initialized (title-based extraction + LLM enrich)")

    # ── Public ──────────────────────────────────────────────────────────

    def extract(
        self,
        documents: list,
        course_id: Optional[str] = None,
        max_concepts: int = 50,
        lang: str = "fr",
    ) -> list[Concept]:
        """Extrait les concepts en groupant par ``section_title``.

        Pipeline :
          1. Filtre par course_id
          2. Group by section_title (case-insensitive, stripped)
          3. Filtre les titres-bruit + ceux trop rares
          4. Pour chaque titre, LLM génère canonical/description/bloom
          5. Convertit en list[Concept]
        """
        if course_id:
            filtered = [
                d for d in documents
                if (getattr(d, "metadata", None) or {}).get("course") == course_id
            ]
        else:
            filtered = list(documents)

        if not filtered:
            log.warning(f"No documents for course_id={course_id}")
            return []

        log.info(
            f"⏳ Title-based concept extraction "
            f"({len(filtered)} chunks, course={course_id}, lang={lang})…"
        )

        # 1) Group by normalized title
        groups = self._group_by_title(filtered)
        log.info(f"📂 {len(groups)} unique section titles found")

        # 2) Filtre noise + min size
        valid_titles = {
            title: docs
            for title, docs in groups.items()
            if self._is_valid_title(title) and len(docs) >= MIN_CHUNKS_PER_TITLE
        }
        log.info(
            f"🧹 {len(valid_titles)} titles kept after filtering "
            f"(dropped {len(groups) - len(valid_titles)} noise/rare)"
        )

        if not valid_titles:
            return []

        # 3) Fusionne les titres quasi-identiques (typos/variants)
        merged = self._merge_similar_titles(valid_titles)
        log.info(f"🔀 {len(merged)} concepts after merging similar titles")

        # 4) LLM enrich par concept (renvoie parent + sous-concepts)
        concepts: list[Concept] = []
        seen_slugs: set[str] = set()      # dedup global (un sous-concept peut
                                          # apparaître dans plusieurs sections)
        n_parents = 0
        n_subs = 0
        for title, docs in merged.items():
            built = self._build_concepts(title, docs, lang)
            for i, c in enumerate(built):
                if c.label in seen_slugs:
                    continue
                seen_slugs.add(c.label)
                concepts.append(c)
                if i == 0:
                    n_parents += 1
                else:
                    n_subs += 1

        # 5) Tri par taille (importance) + cap
        concepts.sort(key=lambda c: -len(c.chunk_ids))
        concepts = concepts[:max_concepts]

        log.info(
            f"✅ {len(concepts)} concepts extracted via title method "
            f"({n_parents} parents + {n_subs} sub-concepts, cap={max_concepts})"
        )
        return concepts

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _group_by_title(documents: list) -> dict[str, list]:
        """Group docs by ``section_title``. Falls back to ``chapter_title`` if
        ``section_title`` is missing.
        """
        groups: dict[str, list] = defaultdict(list)
        for d in documents:
            meta = getattr(d, "metadata", None) or {}
            title = (meta.get("section_title") or "").strip()
            if not title:
                title = (meta.get("chapter_title") or "").strip()
            if not title:
                continue
            # Normalise lightly (collapse whitespace) but preserve case for
            # display. The merge step below handles case-insensitive matching.
            normalised = re.sub(r"\s+", " ", title).strip()
            groups[normalised].append(d)
        return dict(groups)

    @staticmethod
    def _is_valid_title(title: str) -> bool:
        """True if the title looks like a real concept (not noise)."""
        if not title:
            return False
        low = title.lower().strip()
        if low in _NOISE_TITLES:
            return False
        # Pure numbers ("1", "2.3"), single chars
        if len(low) < 3:
            return False
        if low.replace(".", "").replace(" ", "").isdigit():
            return False
        return True

    @staticmethod
    def _merge_similar_titles(groups: dict[str, list]) -> dict[str, list]:
        """Merge titles that differ only by case/typos.

        Strategy: build a canonical form (lowercased, stripped of common
        suffixes/separators) and merge entries sharing it. The longest
        original form wins as display title (more informative).
        """
        canonical: dict[str, list[tuple[str, list]]] = defaultdict(list)

        def _canon(t: str) -> str:
            c = t.lower()
            # Strip common separators / qualifiers
            c = re.sub(r"\s*[&,/\-]\s*", " ", c)
            c = re.sub(r"\s+", " ", c).strip()
            return c

        for title, docs in groups.items():
            canonical[_canon(title)].append((title, docs))

        merged: dict[str, list] = {}
        for canon_key, entries in canonical.items():
            # Pick the longest variant as the display title
            entries.sort(key=lambda e: -len(e[0]))
            display = entries[0][0]
            all_docs: list = []
            for _t, ds in entries:
                all_docs.extend(ds)
            merged[display] = all_docs
        return merged

    def _build_concepts(
        self,
        title: str,
        docs: list,
        lang: str,
    ) -> list[Concept]:
        """Build the parent ``Concept`` for a title group **plus any
        sub-concepts** the LLM detects inside the body text.

        Returns a list :
          - [0] : parent concept (always present, even if LLM fails)
          - [1..] : sub-concepts (e.g. for "Ensemble learning" parent →
            [Bagging, Random Forest, AdaBoost, Stacking, Voting])

        Sub-concepts inherit their ``chunk_ids`` by **substring matching**
        their canonical name against ``page_content`` of each doc in the
        group. A sub-concept the LLM hallucinated (no chunks contain it)
        falls back to the parent's chunks rather than being dropped.
        """
        chunk_ids: set[str] = set()
        chapter_idxs: set[int] = set()
        excerpts: list[str] = []
        # (chunk_id, lowercased_text) for fast substring matching when we
        # later hunt for each sub-concept across this section's chunks.
        per_chunk_text: list[tuple[str, str]] = []

        for d in docs:
            meta = getattr(d, "metadata", None) or {}
            cid = meta.get("idea_id") or meta.get("content_hash") or ""
            if cid:
                chunk_ids.add(str(cid))
            ch = meta.get("chapter_idx")
            if ch is not None:
                try:
                    chapter_idxs.add(int(ch))
                except Exception:
                    pass
            text = (getattr(d, "page_content", "") or "").strip()
            if text:
                excerpts.append(text)
                if cid:
                    per_chunk_text.append((str(cid), text.lower()))

        joined = "\n\n".join(excerpts)[:EXCERPT_TRUNCATE]
        canonical, description, bloom, sub_specs = self._llm_enrich(
            title, joined, lang,
        )

        # ── Parent concept ──────────────────────────────────────────────
        display = canonical or title
        parent = Concept(
            label=_slugify(display),
            name=display,
            keywords=[title],
            chunk_ids=chunk_ids,
            chapter_idxs=chapter_idxs,
            score=float(len(chunk_ids)),
            description=description or "",
            bloom_level=bloom or "",
        )
        out: list[Concept] = [parent]

        # ── Sub-concepts ────────────────────────────────────────────────
        # For each sub-concept the LLM identified, find which chunks of
        # this section actually mention it. Substring match keeps the
        # graph layer accurate at retrieval time.
        seen_sub_slugs: set[str] = {parent.label}
        for spec in sub_specs:
            sub_name = (spec.get("name") or "").strip()
            sub_desc = (spec.get("description") or spec.get("desc") or "").strip()
            if not sub_name or len(sub_name) < 2:
                continue
            sub_slug = _slugify(sub_name)
            if sub_slug in seen_sub_slugs:
                continue   # dedup intra-section

            needle = sub_name.lower()
            sub_chunk_ids: set[str] = {
                cid for cid, lower_text in per_chunk_text
                if needle in lower_text
            }
            # LLM hallucinated → DROP this sub-concept entirely.
            # Earlier behaviour fell back to the parent's chunks, which
            # let LLM-invented names ("RI = Research Interface") into
            # the KG attached to real text. That's worse than missing a
            # concept : a hallucinated node with real chunks pollutes
            # downstream agents (path_recommender, retriever KG-aug).
            # Operator-visible bug : student asked about "Définitions et
            # terminologie", responder hallucinated "Research Interface"
            # because a sub-concept invented by Ollama got grafted onto
            # the section's real chunks.
            if not sub_chunk_ids:
                log.debug(
                    "concept_from_titles: dropping hallucinated sub-concept "
                    "'%s' under '%s' (no substring match in section text)",
                    sub_name, title,
                )
                continue

            seen_sub_slugs.add(sub_slug)
            out.append(Concept(
                label=sub_slug,
                name=sub_name,
                keywords=[sub_name, title],   # keep parent linkage
                chunk_ids=sub_chunk_ids,
                chapter_idxs=chapter_idxs,
                score=float(len(sub_chunk_ids)) * SUB_CONCEPT_SCORE_FACTOR,
                description=sub_desc[:300],
                bloom_level=bloom or "",
            ))

        return out

    @staticmethod
    def _llm_enrich(
        title: str,
        excerpts: str,
        lang: str,
    ) -> tuple[str, str, str, list[dict]]:
        """1 LLM call to canonicalise the title, describe it, AND extract
        sub-concepts from the body text.

        Returns ``(canonical_name, description, bloom_level, sub_concepts)``.
        ``sub_concepts`` is a list of ``{"name": str, "description": str}``
        dicts (possibly empty). All defaults to empty on failure.

        When ``Config.KG_DISABLE_LLM_ENRICH`` is set, this returns empty
        values immediately — useful when only Ollama (which hallucinates
        canonical names like "RI = Research Interface") is reachable. The
        caller falls back to using the raw section_title as the concept
        name, which is deterministic and PDF-grounded.
        """
        try:
            from core.config import Config
            if Config.KG_DISABLE_LLM_ENRICH:
                log.debug(
                    "concept_from_titles: skipping LLM enrich for '%s' "
                    "(KG_DISABLE_LLM_ENRICH=true) — using title as canonical",
                    title,
                )
                return "", "", "", []
        except Exception:
            pass
        from ai.llm_router import get_default_router

        if lang == "fr":
            prompt = (
                f"Tu es un assistant pédagogique. Voici une section d'un cours, "
                f"avec son titre et le contenu intégral.\n\n"
                f"TITRE : {title}\n\n"
                f"CONTENU :\n{excerpts}\n\n"
                f"Donne :\n"
                f"  - canonical : le nom canonique du concept principal de cette section "
                f"(forme courte standard, ex: 'k-NN', 'Random Forest', 'Naive Bayes', 'Decision Tree')\n"
                f"  - description : 1 phrase qui résume ce que cette section enseigne (max 25 mots)\n"
                f"  - bloom : niveau Bloom dominant (remember/understand/apply/analyze)\n"
                f"  - sub_concepts : liste EXHAUSTIVE des SOUS-CONCEPTS nommément mentionnés "
                f"dans le contenu (ex: pour 'Ensemble learning' on s'attend à voir 'Bagging', "
                f"'Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', "
                f"'CatBoost', 'Stacking', 'Blending', 'Voting' ; pour 'Decision Trees' : "
                f"'Entropy', 'Information Gain', 'ID3', 'Gain Ratio'). Liste TOUS ceux qui "
                f"apparaissent réellement dans le contenu, sans limite de nombre. "
                f"Ne répète pas le titre parent.\n\n"
                f"Réponds UNIQUEMENT en JSON STRICT (pas de markdown) :\n"
                f'{{"canonical": "<nom court>", "description": "<1 phrase>", '
                f'"bloom": "remember|understand|apply|analyze", '
                f'"sub_concepts": [{{"name": "<nom>", "description": "<1 phrase>"}}, ...]}}'
            )
        else:
            prompt = (
                f"You are a pedagogical assistant. Below is a course section "
                f"with its title and full content.\n\n"
                f"TITLE: {title}\n\n"
                f"CONTENT:\n{excerpts}\n\n"
                f"Provide:\n"
                f"  - canonical: the canonical name of the main concept of this section "
                f"(standard short form, e.g. 'k-NN', 'Random Forest', 'Naive Bayes', 'Decision Tree')\n"
                f"  - description: 1 sentence summarising what this section teaches (max 25 words)\n"
                f"  - bloom: dominant Bloom level (remember/understand/apply/analyze)\n"
                f"  - sub_concepts: EXHAUSTIVE list of SUB-CONCEPTS explicitly mentioned in "
                f"the content (e.g. for 'Ensemble learning' expect 'Bagging', 'Random Forest', "
                f"'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', "
                f"'Stacking', 'Blending', 'Voting'; for 'Decision Trees': 'Entropy', "
                f"'Information Gain', 'ID3', 'Gain Ratio'). List ALL that ACTUALLY appear "
                f"in the content above, no count limit. Do not repeat the parent title.\n\n"
                f"Reply STRICT JSON ONLY (no markdown):\n"
                f'{{"canonical": "<short name>", "description": "<1 sentence>", '
                f'"bloom": "remember|understand|apply|analyze", '
                f'"sub_concepts": [{{"name": "<name>", "description": "<1 sentence>"}}, ...]}}'
            )

        router = get_default_router()
        # max_tokens generous so the LLM can list every sub-concept it finds
        # without being truncated mid-list. 1500 tokens covers ~30 sub-concepts
        # with descriptions, which is more than any pedagogical section needs.
        response = router.invoke(prompt, prefer="openai", temperature=0.0, max_tokens=1500)
        if not response:
            return "", "", "", []

        import json as _json

        raw = response.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return "", "", "", []
        try:
            data = _json.loads(raw[start:end + 1])
        except _json.JSONDecodeError:
            return "", "", "", []

        canonical = (data.get("canonical") or "").strip()[:120]
        description = (data.get("description") or "").strip()[:300]
        bloom = (data.get("bloom") or "").strip().lower()
        if bloom not in {"remember", "understand", "apply", "analyze", "evaluate", "create"}:
            bloom = ""

        subs_raw = data.get("sub_concepts") or []
        sub_concepts: list[dict] = []
        if isinstance(subs_raw, list):
            for item in subs_raw:
                if isinstance(item, dict):
                    sub_concepts.append(item)
                elif isinstance(item, str):
                    # Some LLMs emit a flat list of strings instead of dicts
                    sub_concepts.append({"name": item, "description": ""})

        return canonical, description, bloom, sub_concepts

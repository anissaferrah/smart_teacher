"""Smart Teacher — Multi-Modal RAG with Qdrant"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from core.config import Config
from rag.embedding_cache import embedding_cache
from rag.metadata import RAGDocumentMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SmartTeacher.RAG")


def _normalize_text_for_diversity(text: str) -> str:
    text = re.sub(r"[^\w\sÀ-ÿ]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_openai_embedding_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith("text-embedding-")


def _embedding_dim_for_model(model_name: str) -> int:
    normalized = model_name.strip().lower()
    if normalized.startswith("text-embedding-3-large"):
        return 3072
    if normalized.startswith("text-embedding-3-small"):
        return 1536
    if "bge-m3" in normalized:
        return 1024
    if "all-minilm-l6-v2" in normalized:
        return 384
    return 1536 if _is_openai_embedding_model(normalized) else 1024


COLLECTION_NAME  = "smart_teacher_multimodal"
EMBEDDING_DIM_OPENAI     = 1536      # OpenAI text-embedding-3-small
EMBEDDING_DIM_LOCAL      = 1024      # BAAI/bge-m3
EMBEDDING_DIM_LEGACY     = 384       # sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_OPENAI   = "text-embedding-3-small"
EMBEDDING_MODEL_LOCAL    = "BAAI/bge-m3"  # Modèle local par défaut
EMBEDDING_MODEL_LEGACY   = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback léger
EMBEDDING_MODEL  = Config.RAG_EMBEDDING_MODEL or EMBEDDING_MODEL_LOCAL
EMBEDDING_DIM    = _embedding_dim_for_model(EMBEDDING_MODEL)
LLM_SUMMARY      = "gpt-4o-mini"
LLM_ANSWER       = "gpt-4o-mini"


def _make_chat_llm(model: str, temperature: float, max_tokens: int):
    """Build a ChatOpenAI instance, repointed at Groq's OpenAI-compatible
    endpoint when ``Config.DISABLE_OPENAI=true`` and a Groq API key is set.

    Centralises the routing so every RAG call site in this module benefits
    from Groq fallback without duplicating the if/else logic. Falls back
    silently to plain OpenAI when Groq isn't configured.
    """
    if Config.DISABLE_OPENAI and getattr(Config, "GROQ_API_KEY", None):
        return ChatOpenAI(
            model=getattr(Config, "GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=Config.GROQ_API_KEY,
            base_url=getattr(Config, "GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=0,
        )
    return ChatOpenAI(
        model=model, temperature=temperature,
        max_tokens=max_tokens, max_retries=0,
    )

class MultiModalRAG:
    def __init__(self, db_dir: str = "data/rag_cache", force_local_embeddings: bool = False):
        """
        Init RAG with optional forced local embeddings for ingestion.
        
        Args:
            db_dir: Database directory
            force_local_embeddings: If True, skip OpenAI and use local HuggingFace embeddings directly.
                                   Used during course ingestion to avoid quota issues.
        """
        self.db_dir      = Path(db_dir)
        self.docs_cache  = self.db_dir / "docs_cache.json"
        self.summary_cache_path = self.db_dir / "summary_cache.json"
        self.idea_cache_path    = self.db_dir / "idea_cache.json"

        self.vectorstore:     QdrantVectorStore | None = None
        self.client:          QdrantClient | None      = None
        self.qdrant_dir = self.db_dir / "qdrant"
        self.collection_name = COLLECTION_NAME
        self.bm25_retriever:  BM25Retriever | None     = None
        self.vector_retriever = None
        self.all_docs:        list[Document]           = []
        self.summary_cache:   dict[str, str]           = {}
        # idea_cache : md5(text) -> list[{"idea": str, "label": str}]
        # evite re-LLM lors des re-ingestions
        self.idea_cache:      dict[str, list]          = {}
        self.is_ready = False
        self.embedding_source = "none"  # "openai" ou "huggingface"
        self.embedding_model_name = "none"
        self.preferred_embedding_model = Config.RAG_EMBEDDING_MODEL or EMBEDDING_MODEL_LOCAL
        self.current_embedding_dim = _embedding_dim_for_model(self.preferred_embedding_model)
        # LLM router : single source of truth pour OpenAI/Ollama + disable state
        # (avant : 2 helpers divergents `_invoke_llm_text` ici et `_call_llm` dans
        # concept_extractor avec priorites inversees, doublon non documente).
        from ai.llm_router import LLMRouter
        self._llm = LLMRouter(openai_model=LLM_SUMMARY)

        # Reranker cross-encoder (lazy) — None=pas tente, False=tente mais echoue, instance=ok
        self._reranker = None

        log.info("Initializing Smart Teacher Multi-Modal RAG (Qdrant) v2…")

        # ── Embeddings : modèle configuré -> fallback local ───────────────────
        self.embeddings = None
        self._embeddings_ok = False
        
        if force_local_embeddings or not _is_openai_embedding_model(self.preferred_embedding_model):
            local_model = (
                self.preferred_embedding_model
                if not _is_openai_embedding_model(self.preferred_embedding_model)
                else EMBEDDING_MODEL_LOCAL
            )
            log.info(f"🔧 MODE LOCAL: Using HuggingFace embeddings ({local_model})…")
            try:
                self._set_embeddings(
                    "huggingface",
                    local_model,
                    self._build_local_embeddings(local_model),
                )
                log.info(
                    f"✅ Local embeddings ready ({local_model}) — "
                    f"dim={self.current_embedding_dim}"
                )
            except Exception as hf_exc:
                if local_model != EMBEDDING_MODEL_LEGACY:
                    log.info(
                        f"ℹ️ Local embeddings failed ({hf_exc.__class__.__name__}). "
                        f"Fallback to {EMBEDDING_MODEL_LEGACY}…"
                    )
                    try:
                        self._set_embeddings(
                            "huggingface",
                            EMBEDDING_MODEL_LEGACY,
                            self._build_local_embeddings(EMBEDDING_MODEL_LEGACY),
                        )
                        log.info(
                            f"✅ Fallback embeddings ready ({EMBEDDING_MODEL_LEGACY}) — "
                            f"dim={self.current_embedding_dim}"
                        )
                    except Exception as legacy_exc:
                        log.error(
                            f"❌ Local embeddings failed: {legacy_exc}. "
                            f"RAG disabled — system continues with BM25 search only."
                        )
                else:
                    log.error(
                        f"❌ Local embeddings failed: {hf_exc}. "
                        f"RAG disabled — system continues with BM25 search only."
                    )
        else:
            try:
                self._set_embeddings(
                    "openai",
                    self.preferred_embedding_model,
                    self._build_openai_embeddings(self.preferred_embedding_model),
                )
                log.info(f"✅ Embeddings OpenAI ready ({self.preferred_embedding_model})")
            except Exception as exc:
                log.info(
                    f"ℹ️ Modèle distant indisponible ({exc.__class__.__name__}). "
                    f"Fallback to local BAAI/bge-m3…"
                )
                try:
                    self._set_embeddings(
                        "huggingface",
                        EMBEDDING_MODEL_LOCAL,
                        self._build_local_embeddings(EMBEDDING_MODEL_LOCAL),
                    )
                    log.info(
                        f"✅ Fallback embeddings ready ({EMBEDDING_MODEL_LOCAL}) — "
                        f"dim={self.current_embedding_dim} (quota OpenAI probable)"
                    )
                except Exception as hf_exc:
                    log.error(
                        f"❌ Both OpenAI and HuggingFace embeddings failed: {hf_exc}. "
                        f"RAG disabled — system continues with BM25 search only."
                    )

        self._load_summary_cache()
        self._load_idea_cache()

        # ── Qdrant Docker Container (redis + postgres backend) ──────────────────────
        try:
            self.client = QdrantClient(
                host=Config.QDRANT_HOST,
                port=Config.QDRANT_PORT,
                timeout=5.0
            )
            log.info(f"✅ Qdrant Docker ready → {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
            self.collection_name = self._collection_name_for_current_backend()
            if self._embeddings_ok and self.client.collection_exists(self.collection_name):
                self._load_existing_db()
            elif not self._embeddings_ok:
                self._activate_local_retrieval_fallback("no embeddings available")
            else:
                self._activate_local_retrieval_fallback(f"collection '{self.collection_name}' not found")
        except Exception as exc:
            self.client = None
            log.info(f"ℹ️ Qdrant local storage unavailable ({exc}) — local BM25 fallback active.")
            self._activate_local_retrieval_fallback(str(exc))

    # ══════════════════════════════════════════════════════════════════════════
    #  STATUS & DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def get_status(self) -> dict[str, str | bool | int]:
        """
        Retourne le statut actuel du RAG pour monitoring/diagnostics.
        Utile pour vérifier quel backend d'embedding est utilisé.
        """
        return {
            "rag_ready": self.is_ready,
            "embeddings_ok": self._embeddings_ok,
            "embedding_source": self.embedding_source,  # "openai", "huggingface", ou "none"
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.current_embedding_dim,
            "vectorstore_available": self.vectorstore is not None,
            "bm25_available": self.bm25_retriever is not None,
            "qdrant_connected": self.client is not None,
            "docs_loaded": len(self.all_docs),
        }

    # ── OpenAI disable state : delegate to LLMRouter (single source of truth) ──
    @property
    def _openai_disabled_reason(self) -> str | None:
        return self._llm.openai_disabled_reason

    @staticmethod
    def _should_disable_openai(exc: Exception) -> bool:
        from ai.llm_router import LLMRouter
        return LLMRouter._is_permanent_openai_error(exc)

    def _disable_openai(self, reason: str) -> None:
        self._llm.disable_openai(reason)

    def _embedding_cache_namespace(self) -> str:
        return f"{self.embedding_source}:{self.embedding_model_name}"

    def _collection_name_for_current_backend(self) -> str:
        namespace = self._embedding_cache_namespace()
        safe_namespace = re.sub(r"[^a-zA-Z0-9]+", "_", namespace.lower()).strip("_") or "default"
        namespace_hash = hashlib.md5(namespace.encode()).hexdigest()[:8]
        return f"{COLLECTION_NAME}__{safe_namespace}__{namespace_hash}"

    def _build_openai_embeddings(self, model_name: str) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=model_name,
            max_retries=0,
        )

    def _build_local_embeddings(self, model_name: str = EMBEDDING_MODEL_LOCAL) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _set_embeddings(self, source: str, model_name: str, embeddings_obj: Any) -> None:
        self.embeddings = embeddings_obj
        self._embeddings_ok = True
        self.embedding_source = source
        self.embedding_model_name = model_name
        self.current_embedding_dim = _embedding_dim_for_model(model_name)
        self.collection_name = self._collection_name_for_current_backend()

    def _switch_to_local_embeddings(self, reason: str) -> bool:
        if (
            self.embedding_source == "huggingface"
            and self.embedding_model_name == EMBEDDING_MODEL_LOCAL
            and self.embeddings is not None
        ):
            return True

        try:
            self._set_embeddings(
                "huggingface",
                EMBEDDING_MODEL_LOCAL,
                self._build_local_embeddings(EMBEDDING_MODEL_LOCAL),
            )
            log.info(
                f"ℹ️ Modèle distant indisponible ({reason}) — switching to local HuggingFace embeddings ({EMBEDDING_MODEL_LOCAL})."
            )
            return True
        except Exception as hf_exc:
            self.embeddings = None
            self._embeddings_ok = False
            log.error(
                f"❌ HuggingFace fallback failed after OpenAI error ({reason}): {hf_exc}"
            )
            return False

    @staticmethod
    def _should_fallback_to_local_embeddings(exc: Exception) -> bool:
        message = f"{exc.__class__.__module__}:{exc.__class__.__name__}:{exc}".lower()
        return any(
            token in message
            for token in (
                "openai",
                "quota",
                "rate limit",
                "ratelimit",
                "429",
                "insufficient_quota",
                "authentication",
                "api error",
            )
        )

    def _activate_local_retrieval_fallback(self, reason: str) -> None:
        """Active une recherche locale BM25 quand Qdrant n'est pas disponible."""
        if self.docs_cache.exists():
            self._load_docs_cache()

        if self.all_docs:
            self._build_hybrid_retriever()
            self.is_ready = True
            log.info(
                f"ℹ️ Mode local activé ({reason}) — BM25 uniquement avec {len(self.all_docs)} documents"
            )
        else:
            self.is_ready = False
            log.info(f"ℹ️ Mode local activé ({reason}) — aucun cache local disponible")

    def _documents_from_course_data(
        self,
        course_data: dict,
        domain: str = "general",
        course: str = "generic",
        course_id: str | None = None,
    ) -> list[Document]:
        """Build Documents from pre-structured course_data (CourseBuilder output).

        UNIFIE le pipeline avec _summarise_chunks :
          - meme schema de metadata (section_idx, idea_id, idea_label, idea_index_in_chunk)
          - idea-level chunking via _chunk_by_ideas (LLM OpenAI -> Ollama Mistral fallback)
          - cache idea_cache.json partage cross-paths
        """
        documents: list[Document] = []
        source_file = course_data.get("file_path", "")
        slides = course_data.get("slides", []) or []
        language = course_data.get("language", "")
        total_ideas = 0
        total_sections = 0

        # Phase 1 : collecter toutes les sections traitables (1 passe sequentielle, pas de LLM)
        pending: list[dict] = []
        for chapter_index, chapter in enumerate(course_data.get("chapters", []), start=1):
            chapter_idx = int(chapter.get("order") or chapter.get("chapter_idx") or chapter_index)
            chapter_title = chapter.get("title") or course_data.get("title") or f"Chapter {chapter_idx}"

            chapter_sections = chapter.get("sections", []) or []
            # Page fusion : merge consecutive short sections AVANT idea-chunking
            # (evite 1 appel LLM pour 2 phrases — qualite pedagogique + cout)
            chapter_sections = self._fuse_short_sections(chapter_sections)

            for section_index, section in enumerate(chapter_sections, start=1):
                content = (section.get("content") or "").strip()
                if len(content) < 10:
                    continue

                section_idx = int(section.get("section_idx") or section.get("order") or section_index)
                section_title = (section.get("title") or "").strip()
                section_lang = section.get("language") or language

                page_index = int(section.get("page_index") or section.get("order") or section_index)
                slide_index = max(0, page_index - 1)
                image_url = (section.get("image_url") or "").strip()
                if not image_url and 0 <= slide_index < len(slides):
                    image_url = slides[slide_index]
                total_sections += 1

                pending.append({
                    "content": content,
                    "chapter_idx": chapter_idx,
                    "chapter_title": chapter_title,
                    "section_idx": section_idx,
                    "section_title": section_title,
                    "section_lang": section_lang,
                    "page_index": page_index,
                    "image_url": image_url,
                    "section_index": section_index,
                })

        # Phase 2 : appels LLM idea-chunking en parallele (Fix #3 — speed-up principal)
        ideas_batch = self._chunk_by_ideas_batch(
            [(p["content"], p["chapter_title"], p["section_title"]) for p in pending]
        )

        # Phase 3 : construire les Documents depuis les resultats parallelises
        for p, ideas in zip(pending, ideas_batch):
            chapter_idx = p["chapter_idx"]
            chapter_title = p["chapter_title"]
            section_idx = p["section_idx"]
            section_title = p["section_title"]
            section_lang = p["section_lang"]
            page_index = p["page_index"]
            image_url = p["image_url"]
            section_index = p["section_index"]

            # First pass : compute global idea_id for every idea so we can
            # resolve LOCAL relations (depends_on / illustrates) into
            # GLOBAL idea_ids that survive cross-chunk lookups.
            local_to_global: dict[str, str] = {}
            idea_ids_for_section: list[str] = []
            for j, idea_dict in enumerate(ideas):
                idea_text = idea_dict["idea"]
                content_hash = hashlib.md5(idea_text.encode()).hexdigest()[:8]
                idea_id = hashlib.md5(
                    f"{course_id or course}|{chapter_idx}|{section_idx}|{section_index}|{j}|{content_hash}".encode()
                ).hexdigest()[:12]
                idea_ids_for_section.append(idea_id)
                local_id = idea_dict.get("local_id") or f"idea_{j}"
                local_to_global[local_id] = idea_id

            for j, idea_dict in enumerate(ideas):
                idea_text = idea_dict["idea"]
                idea_label = idea_dict["label"]
                idea_id = idea_ids_for_section[j]

                # Resolve relations local_id → global idea_id
                depends_on_ids = [
                    local_to_global[d] for d in idea_dict.get("depends_on", [])
                    if d in local_to_global
                ]
                illustrates_local = idea_dict.get("illustrates")
                illustrates_id = local_to_global.get(illustrates_local) if illustrates_local else None

                content_hash = hashlib.md5(idea_text.encode()).hexdigest()[:8]

                # Type hint pour catcher les fautes de frappe sur les cles —
                # voir rag.metadata.RAGDocumentMetadata
                meta: RAGDocumentMetadata = {
                    "source_file":         source_file,
                    "chunk_idx":           len(documents),
                    "idea_index_in_chunk": j,
                    "domain":              domain,
                    "course":              course_id or course,
                    "language":            section_lang,
                    "original_text":       idea_text[:500],
                    "content_hash":        content_hash,
                    "chapter_idx":         chapter_idx,
                    "chapter_title":       chapter_title,
                    "section_idx":         section_idx,
                    "section_title":       section_title,
                    "idea_id":             idea_id,
                    "idea_label":          idea_label,
                    "slide_idx":           page_index,
                    "image_url":           image_url,
                    # ── Knowledge graph edges (intra-section) ──
                    "depends_on_ids":      depends_on_ids,
                    "illustrates_id":      illustrates_id,
                }
                documents.append(Document(page_content=idea_text, metadata=meta))
                total_ideas += 1

        # Persist idea cache cross-restart (partage avec _summarise_chunks)
        if self.idea_cache:
            self._save_idea_cache()

        log.info(
            f"📝 Documents produits (structured): {len(documents)} "
            f"(ideas: {total_ideas} sur {total_sections} sections → "
            f"avg {total_ideas / max(1, total_sections):.1f} idées/section)"
        )
        return documents

    def run_ingestion_pipeline_from_course_data(
        self,
        course_data: dict,
        domain: str = "general",
        course: str = "generic",
        course_id: str | None = None,
        incremental: bool = True,
    ) -> bool:
        if not self._embeddings_ok or self.embeddings is None:
            if not self._switch_to_local_embeddings("embeddings unavailable before structured ingestion"):
                log.info(
                    "⚠️  run_ingestion_pipeline_from_course_data ignoré : embeddings indisponibles."
                )
                return False

        t0 = time.time()
        documents = self._documents_from_course_data(
            course_data,
            domain=domain,
            course=course,
            course_id=course_id,
        )

        if not documents:
            log.error("Structured ingestion produced no documents.")
            return False

        ok = self._store_documents(documents, incremental=incremental)
        if ok:
            self.is_ready = True
            elapsed = time.time() - t0
            log.info(f"✅ Structured ingestion terminée en {elapsed:.1f}s ({len(documents)} docs)")
        return ok

    # ══════════════════════════════════════════════════════════════════════════
    #  INGESTION GÉNÉRIQUE (Tous domaines/cours)
    # ══════════════════════════════════════════════════════════════════════════

    def index_from_elements(
        self,
        elements: list[Any],
        file_paths: list[str],
        domain: str = "general",
        course: str = "generic",
        course_id: str | None = None,
        incremental: bool = True,
    ) -> bool:
        """Index pre-extracted unstructured Elements (skip the partition step).

        Used by the unified ingestion path (IntelligentIngester → here) so we
        don't re-parse the same PDF twice. ``run_ingestion_pipeline_for_files``
        is the legacy path that does its own partitioning.

        Args:
            elements   : pre-extracted unstructured Elements (Tables, Captions,
                         NarrativeText, Title…). Typically from
                         ``IntelligentIngester.ingest_file().elements``.
            file_paths : original source paths — used for TOC extraction and
                         metadata only, NOT re-parsed.
        """
        if not self._embeddings_ok or self.embeddings is None:
            if not self._switch_to_local_embeddings("embeddings unavailable before ingestion"):
                log.info(
                    "⚠️  index_from_elements ignored: embeddings unavailable. "
                    "Course is still saved in DB / local."
                )
                return False

        if not elements:
            log.error("index_from_elements: no elements provided.")
            return False

        t0 = time.time()
        log.info("=" * 60)
        log.info(
            f"🚀 Indexation (pre-extracted) — {len(elements)} element(s) "
            f"from {len(file_paths)} file(s) | incremental={incremental}"
        )
        log.info("=" * 60)

        # TOC for chapter/section enrichment in metadata (cheap; reads PDFs only).
        file_tocs = self._extract_tocs(file_paths)

        chunks = self._create_chunks_by_title(elements)
        if not chunks:
            log.error("Chunking produced no chunks.")
            return False

        documents = self._summarise_chunks(
            chunks, domain=domain, course=course, course_id=course_id, file_tocs=file_tocs,
        )
        if not documents:
            log.error("Summarization produced no documents.")
            return False

        ok = self._store_documents(documents, incremental=incremental)
        if ok:
            self.is_ready = True
            elapsed = time.time() - t0
            log.info(f"✅ Indexation terminée en {elapsed:.1f}s ({len(documents)} docs)")
        return ok


    # NOTE: run_ingestion_pipeline_for_files was removed (legacy path).
    # All ingestion now goes through index_from_elements() above, fed by
    # IntelligentIngester. See routes/rest.py:_run_ingestion_background.

    # ══════════════════════════════════════════════════════════════════════════
    #  RETRIEVAL HYBRIDE AVEC ISOLATION PAR CHAPITRE
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _doc_source_key(doc: Document) -> tuple:
        metadata = doc.metadata or {}
        return (
            metadata.get("source_file", ""),
            metadata.get("chapter_idx"),
            metadata.get("slide_idx"),
        )

    @staticmethod
    def _doc_signature(doc: Document) -> str:
        return _normalize_text_for_diversity(doc.page_content)

    # Near-duplicate cutoffs.
    # 0.9 SequenceMatcher ratio is the conventional "essentially identical"
    # threshold (e.g. used by difflib's HtmlDiff, Mercurial's similarity
    # detection). 0.88 token-overlap is the matching threshold used in
    # Lin (2004, ROUGE) for text reuse / paraphrase. Both are well-attested
    # in NLP literature, so they're documented operational constants
    # rather than arbitrary tuning knobs.
    _NEAR_DUP_SEQ_RATIO  = 0.90
    _NEAR_DUP_TOKEN_JACC = 0.88

    @classmethod
    def _is_near_duplicate(cls, left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True

        ratio = SequenceMatcher(None, left, right).ratio()
        if ratio >= cls._NEAR_DUP_SEQ_RATIO:
            return True

        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return False

        overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
        return overlap >= cls._NEAR_DUP_TOKEN_JACC

    def _dedupe_scored_docs(
        self,
        scored_docs: list[tuple[Document, float, str]],
        max_results: int,
    ) -> list[tuple[Document, float, str]]:
        unique_docs: list[tuple[Document, float, str]] = []
        seen_signatures: list[str] = []

        for doc, confidence, source_info in scored_docs:
            signature = self._doc_signature(doc)

            if any(self._is_near_duplicate(signature, seen) for seen in seen_signatures[-6:]):
                continue

            unique_docs.append((doc, confidence, source_info))
            if signature:
                seen_signatures.append(signature)

            if len(unique_docs) >= max_results:
                break

        return unique_docs or scored_docs[:max_results]

    @staticmethod
    def _build_chat_messages(
        system_prompt: str,
        history: list[dict],
        user_content: str,
    ) -> list[Any]:
        messages: list[Any] = [SystemMessage(content=system_prompt)]

        for msg in history[-6:]:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=user_content))
        return messages

    def _dedupe_answer_text(self, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return clean_text

        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        kept_sentences: list[str] = []
        seen_signatures: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            signature = _normalize_text_for_diversity(sentence)
            if not signature:
                continue
            if any(self._is_near_duplicate(signature, seen) for seen in seen_signatures[-4:]):
                continue

            kept_sentences.append(sentence)
            seen_signatures.append(signature)

        deduped = " ".join(kept_sentences).strip()
        return deduped or clean_text

    def retrieve_chunks(
        self,
        query: str,
        k: int = 5,
        current_chapter_idx: int | None = None,
        strict_chapter: bool = False,
        min_chunk_length: int = 50,  # Filter out tiny chunks
        course_id: str | None = None,  # ✅ Filter results by course_id
    ) -> list[tuple[Document, float, str]]:
        """
        Recherche hybride BM25 + Vectorielle + RRF avec scores de confiance.

        AMÉLIORATIONS :
        - Score de confiance pour chaque chunk
        - Filtrage des petits chunks inutiles
        - Source (chapter + file) incluse
        - Isolation par chapitre optionnelle
        - Isolation par cours optionnelle (évite contamination cross-course)

        Args:
            query:               Question de l'étudiant
            k:                   Nombre de chunks à retourner
            current_chapter_idx: Index du chapitre en cours (1-7 pour DM)
            strict_chapter:      Forcer l'isolation au chapitre courant
            min_chunk_length:    Longueur minimum du chunk (défaut 50 chars)
            course_id:           Cours actuel (optionnel, scoped retrieval)

        Returns:
            List of (Document, confidence_score, source_info)
        """
        if not self.is_ready:
            log.warning("RAG not ready — returning empty results")
            return []

        log.info(f"🔍 Retrieval | ch={current_chapter_idx} | strict={strict_chapter} | course={course_id} | q='{query[:60]}'")

        # ── Recherche vectorielle (avec filtre Qdrant si strict) ──────────────
        vector_docs = self._vector_search(query, k * 3, current_chapter_idx if strict_chapter else None, course_id=course_id)

        # ── Recherche BM25 ─────────────────────────────────────────────────────
        bm25_docs = self._bm25_search(query, k * 3, course_id=course_id)

        # ── Adaptive BM25 weight from query IDF (helps under-represented
        # concepts like "regression" with 1 doc out of N) ────────────────
        alpha = self._bm25_alpha(query)
        if alpha > 1.05:
            log.info(f"  ⚖️ BM25 boost α={alpha:.2f} (query has rare terms)")

        # ── RRF fusion (chapter focus via the Qdrant filter, not a soft boost) ──
        fused = self._rrf_fuse(vector_docs, bm25_docs, bm25_weight=alpha)

        # ── Filter (taille min) puis rerank cross-encoder OU heuristique fallback ──
        candidates = [d for d in fused if len(d.page_content) >= min_chunk_length]
        candidates = candidates[: Config.RAG_RERANKER_TOP_N]

        reranked = self._rerank_chunks(query, candidates)

        results_with_scores = []
        if reranked is not None:
            confidence_source = "cross-encoder"
            for doc, score in reranked:
                source_info = self._format_source_info(doc)
                results_with_scores.append((doc, score, source_info))
        else:
            # Reranker unavailable — fall back to RRF order. The position
            # in the fused list IS the relative score (lower index = more
            # relevant); we expose 1 - rank/len as a normalised score so
            # downstream uses still get a number in [0, 1] without
            # synthesising fake confidence.
            confidence_source = "rrf-position"
            n = max(1, len(candidates))
            for rank, doc in enumerate(candidates):
                score = 1.0 - (rank / n)
                source_info = self._format_source_info(doc)
                results_with_scores.append((doc, score, source_info))

        # Return top k with diversity to avoid repeated explanations
        top_results = self._dedupe_scored_docs(results_with_scores, k)
        n = max(1, len(top_results))
        avg = sum(s[1] for s in top_results) / n
        # Score cosine Qdrant pose dans metadata par _vector_search (None si BM25-only)
        vector_scores = [doc.metadata.get("_vector_score") for doc, _, _ in top_results]
        vector_scores = [s for s in vector_scores if s is not None]
        if vector_scores:
            avg_vec = sum(vector_scores) / len(vector_scores)
            log.info(
                f"✅ {len(top_results)} chunks retenus "
                f"(rerank: {avg:.2f} [{confidence_source}], cosine: {avg_vec:.2f} [bge-m3])"
            )
        else:
            log.info(
                f"✅ {len(top_results)} chunks retenus "
                f"(rerank: {avg:.2f} [{confidence_source}], cosine: n/a [BM25 only])"
            )

        # Detail log : per-chunk scoring breakdown so the operator
        # understands WHY each chunk was retained. Shows :
        #   - rank position
        #   - rerank score (the one downstream uses)
        #   - vector cosine score (if vector search was used)
        #   - source / chapter / section
        for i, (doc, score, src) in enumerate(top_results):
            meta = getattr(doc, "metadata", {}) or {}
            vs = meta.get("_vector_score")
            log.info(
                "🔍   rag[%d] rerank=%.3f cosine=%s ch=%s sec=%s src=%s | %r",
                i, score,
                f"{vs:.3f}" if vs is not None else "n/a",
                meta.get("chapter_idx", meta.get("chapter")),
                meta.get("section_idx"),
                str(meta.get("source") or "?")[-40:],
                (getattr(doc, "page_content", "") or "")[:120],
            )
        return top_results

    def _vector_search(
        self, query: str, k: int,
        chapter_filter: int | None = None,
        course_id: str | None = None,
    ) -> list[Document]:
        """Recherche vectorielle Qdrant avec filtres optionnels sur chapter_idx et course_id.
        
        Utilise embedding_cache pour éviter recalcul des embeddings.
        """
        if not self.vectorstore:
            return []
        try:
            cache_namespace = self._embedding_cache_namespace()
            # 🔄 Vérifier cache avant de générer embedding
            query_embedding = embedding_cache.get(query, namespace=cache_namespace)
            if query_embedding is None:
                # Générer embedding et sauvegarder en cache
                query_embedding = self.embeddings.embed_query(query)
                embedding_cache.set(query, query_embedding, namespace=cache_namespace)
                log.debug(f"📍 Query embedding calculé et cachéé: {query[:50]}...")
            else:
                log.debug(f"📍 Query embedding récupéré du cache: {query[:50]}...")
            
            # ✅ Build Qdrant filter with both chapter_idx and course_id
            filter_conditions = []
            
            if chapter_filter is not None:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.chapter_idx",
                        match=MatchValue(value=chapter_filter)
                    )
                )
            
            if course_id is not None and course_id.strip():
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.course",
                        match=MatchValue(value=course_id)
                    )
                )
            
            qdrant_filter = None
            if filter_conditions:
                qdrant_filter = Filter(must=filter_conditions) if len(filter_conditions) > 1 else Filter(must=[filter_conditions[0]])

            # similarity_search_with_score : on capture le score cosine Qdrant
            # pour le logger separement du score reranker (cf retrieve_chunks)
            import time as _qt
            _t0 = _qt.time()
            log.info(
                "🟣 QDRANT search START | collection=%s k=%d | course=%s chapter=%s | "
                "filters=%d | query='%s%s'",
                getattr(self, "collection_name", "?"), k,
                course_id or "*", str(chapter_filter) if chapter_filter is not None else "*",
                len(filter_conditions),
                query[:80], "..." if len(query) > 80 else "",
            )
            if qdrant_filter:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=qdrant_filter
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)

            docs: list[Document] = []
            for doc, score in results:
                # Avec distance=COSINE et embeddings normalises, langchain_qdrant
                # retourne la cosine similarity (~ [0, 1] ; 1 = match parfait).
                doc.metadata["_vector_score"] = float(score)
                docs.append(doc)
            elapsed = (_qt.time() - _t0) * 1000
            top_score = float(results[0][1]) if results else 0.0
            mean_score = (sum(float(s) for _, s in results) / len(results)) if results else 0.0
            log.info(
                "🟣 QDRANT search DONE | hits=%d/%d | top_score=%.3f mean=%.3f | took=%.0fms",
                len(results), k, top_score, mean_score, elapsed,
            )
            return docs
        except Exception as exc:
            log.warning(f"🟣 QDRANT search ERR | {exc}")
            return []

    def _bm25_search(self, query: str, k: int, course_id: str | None = None) -> list[Document]:
        if not self.bm25_retriever:
            return []
        try:
            self.bm25_retriever.k = k
            docs = self.bm25_retriever.invoke(query)
            
            # ✅ Filter by course_id if specified
            if course_id is not None and course_id.strip():
                docs = [
                    doc for doc in docs
                    if doc.metadata.get("course") == course_id
                ]
            
            return docs
        except Exception as exc:
            log.warning(f"BM25 search error: {exc}")
            return []

    def _bm25_alpha(self, query: str) -> float:
        """Adaptive BM25 weight for the RRF fusion, derived from query
        term rarity (IDF).

        # Why adaptive

        Equiweighted RRF (alpha=1.0, Cormack 2009) is optimal on average,
        but **under-represented concepts** (e.g. ``regression`` appearing
        in 1 doc out of 384) lose to high-frequency semantic neighbors
        when dense embeddings dominate the fusion. BM25 catches the exact
        match — but only if its rank carries enough weight in the fusion.

        # Logic

        - High average IDF of query tokens → query contains rare terms
          → BM25 exact match should win → boost alpha towards 2.0
        - Low average IDF → common terms → dense embeddings handle it
          better → keep alpha at 1.0 (Cormack default)

        # Mapping

        ``alpha = 1.0 + clamp(avg_query_idf / max_corpus_idf, 0, 1)``

        Bounds:
          - alpha=1.0 : default, equivalent to Cormack 2009 RRF
          - alpha=2.0 : maximum boost, when all query terms are at the
            corpus's most rare quantile

        # Design choice (kept minimal)

        We tested adding stopword filtering and top-K rarest selection
        but they hurt aggregate metrics on our 12-query eval (R@5 +F1
        +nDCG dropped). The pure ``mean over all tokens`` proves more
        robust : tokens absent from the BM25 vocab return 0.0 and
        contribute nothing to the average — the IDF lookup IS the
        natural filter. Common words (``the``, ``what``, ``comment``)
        contribute their actual low IDF, which is the honest signal.

        Caveat : on a small corpus (N=384 docs), IDF estimates are
        noisier than at web scale. Acceptable here; the effect would
        tighten with more docs.

        # References

          - Cormack et al. 2009, *Reciprocal Rank Fusion*, SIGIR.
          - Robertson & Sparck Jones 1976, *Relevance weighting of search
            terms*, JASIS — IDF as informational rarity.
          - Bruch et al. 2023, *An Analysis of Fusion Functions for Hybrid
            Retrieval*, ACM TOIS — weighted RRF can outperform equiweight,
            optimum is corpus-dependent → motivates per-query adaptation.

        Returns 1.0 if the BM25 retriever isn't built yet (graceful).
        """
        if not self.bm25_retriever or not getattr(self.bm25_retriever, "vectorizer", None):
            return 1.0
        idf_map = getattr(self.bm25_retriever.vectorizer, "idf", None)
        if not idf_map:
            return 1.0

        # Tokenise query. No length filter, no stopword filter — the BM25
        # vocabulary IS the natural filter : tokens absent from the vocab
        # return IDF 0.0 (they don't contribute to the average), tokens
        # present carry their actual rarity. Common words have low IDF
        # (they appear in many docs), rare technical terms have high IDF.
        tokens = [t.lower() for t in re.findall(r"\w+", query)]
        if not tokens:
            return 1.0

        token_idfs = [idf_map.get(t, 0.0) for t in tokens]
        avg_idf = sum(token_idfs) / len(token_idfs)

        # Normalise against the corpus max IDF (rarest term in the corpus).
        # This auto-calibrates alpha to the corpus's IDF distribution.
        max_idf = max(idf_map.values()) if idf_map else 1.0
        if max_idf <= 0:
            return 1.0
        normalised = max(0.0, min(1.0, avg_idf / max_idf))
        return 1.0 + normalised

    def _rrf_fuse(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
    ) -> list[Document]:
        """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher 2009).

        Combines two ranked lists by summing reciprocal ranks 1/(k + rank).
        ``rrf_k = 60`` is the constant from the original paper —
        empirically validated as outperforming Condorcet and CombMNZ in
        their retrieval experiments. Not a hand-picked tuning knob.

        ``bm25_weight`` (default 1.0 = Cormack equiweight) lets callers
        bias the fusion towards exact-match retrieval when the query
        contains rare terms. Use ``_bm25_alpha(query)`` to compute it
        adaptively from the query's term rarity (IDF).

        The previous version added a ``chapter_boost`` (+0.35) and an
        off-chapter penalty (×0.7) to bias toward the student's current
        chapter. Those weights were arbitrary. Chapter focus is now
        handled exclusively by the optional Qdrant filter in
        ``_vector_search`` (strict mode) — an honest hard filter rather
        than a magic-number soft preference.

        Reference: Cormack, G. V., Clarke, C. L. A., & Buettcher, S.
        (2009). *Reciprocal Rank Fusion outperforms Condorcet and
        individual Rank Learning Methods.* Proc. SIGIR '09.
        """
        scores: dict[str, float] = {}
        docs_map: dict[str, Document] = {}

        def _doc_id(doc: Document) -> str:
            return doc.metadata.get("content_hash", "") or \
                   hashlib.md5(doc.page_content[:100].encode()).hexdigest()[:12]

        for rank, doc in enumerate(vector_docs):
            did = _doc_id(doc)
            scores[did] = scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs_map[did] = doc

        for rank, doc in enumerate(bm25_docs):
            did = _doc_id(doc)
            scores[did] = scores.get(did, 0.0) + bm25_weight / (rrf_k + rank + 1)
            docs_map[did] = doc

        sorted_ids = sorted(scores, key=lambda x: -scores[x])
        return [docs_map[did] for did in sorted_ids if did in docs_map]

    def _get_reranker(self):
        """Charge le cross-encoder reranker (lazy). Retourne None si desactive ou si echec."""
        if not Config.RAG_USE_RERANKER:
            return None
        if self._reranker is False:  # tentative anterieure echouee
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                log.info(f"⏳ Loading reranker {Config.RAG_RERANKER_MODEL}…")
                self._reranker = CrossEncoder(
                    Config.RAG_RERANKER_MODEL,
                    max_length=2048,
                    device="cpu",
                )
                log.info(f"✅ Reranker {Config.RAG_RERANKER_MODEL} ready")
            except Exception as exc:
                log.warning(f"⚠️ Reranker load failed ({exc}) — falling back to heuristic confidence")
                self._reranker = False
                return None
        return self._reranker

    def warmup(self) -> None:
        """Eagerly preload heavy lazy components so the FIRST user query
        doesn't pay their cold-start cost.

        Warms two things:
          1. Cross-encoder reranker (~20-43s on CPU, ~5-10s on GPU).
             Without warmup the first Q&A turn shows a "⏳ Loading
             reranker…" log line that adds 30+ seconds.
          2. IdeaGraph from the RAG corpus (~1-3s for 2000 docs).
             Without warmup the first retriever call ingests 800+
             nodes inline. Pre-built means instant on first query.

        Called once from main.py at startup. Failures are non-fatal —
        on miss components stay in lazy mode and load on first query
        (current behaviour preserved).
        """
        try:
            self._get_reranker()
        except Exception as exc:                                            # noqa: BLE001
            log.debug(f"rag warmup: reranker skipped: {exc}")
        try:
            from pedagogy.knowledge_graph.builder import get_or_build
            get_or_build(self)
            log.info("✅ IdeaGraph preloaded")
        except Exception as exc:                                            # noqa: BLE001
            log.debug(f"rag warmup: knowledge graph skipped: {exc}")

    def _rerank_chunks(
        self, query: str, docs: list[Document]
    ) -> list[tuple[Document, float]] | None:
        """Rerank docs avec cross-encoder. Retourne [(doc, prob_in_0_1), ...] trie desc.
        Si reranker indisponible, retourne None pour signaler au caller d'utiliser le fallback heuristique.
        """
        reranker = self._get_reranker()
        if reranker is None or not docs:
            return None
        try:
            import numpy as np
            pairs = [[query, doc.page_content[:4000]] for doc in docs]
            raw_scores = reranker.predict(pairs)
            # bge-reranker emet des logits ; sigmoid pour [0, 1]
            probs = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float32)))
            scored = list(zip(docs, [float(p) for p in probs]))
            scored.sort(key=lambda x: -x[1])
            return scored
        except Exception as exc:
            log.warning(f"⚠️ Rerank failed ({exc}) — falling back to heuristic confidence")
            return None

    @staticmethod
    def _query_overlap(doc: Document, query: str) -> float:
        """Token overlap of ``query`` words with ``doc.page_content``, in [0, 1].

        This is the **Jaccard-like overlap on the query side** — the
        fraction of the query's distinct words that appear in the chunk.
        It's not a trained relevance score; it's a transparent textual
        feature that callers can use however they want (display, filter,
        etc.). The previous ``_compute_chunk_confidence`` mixed this
        overlap with arbitrary additive bonuses (length > 500 → +0.15,
        has chapter_idx → +0.05, baseline 0.5) — those weights had no
        justification, so they were removed. When the cross-encoder
        reranker is unavailable, the caller falls back to RRF order
        (already a defensible relative ranking) instead of synthesizing
        a fake confidence score.
        """
        if not query:
            return 0.0
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        chunk_words = set(doc.page_content.lower().split())
        return len(query_words & chunk_words) / len(query_words)

    def _format_source_info(self, doc: Document) -> str:
        """Formate les infos source pour affichage."""
        metadata = doc.metadata or {}
        parts = []
        
        if chapter_idx := metadata.get("chapter_idx"):
            if chapter_title := metadata.get("chapter_title"):
                parts.append(f"Ch{chapter_idx}: {chapter_title}")
            else:
                parts.append(f"Ch{chapter_idx}")
        
        if slide_idx := metadata.get("slide_idx"):
            parts.append(f"p{slide_idx}")
        
        if source_file := metadata.get("source_file"):
            parts.append(f"({Path(source_file).stem})")
        
        return " | ".join(parts) if parts else "Unknown source"

    # ══════════════════════════════════════════════════════════════════════════
    #  DEBUG ENDPOINT SUPPORT
    # ══════════════════════════════════════════════════════════════════════════

    def debug_retrieve(
        self,
        query: str,
        k: int = 10,
        current_chapter_idx: int | None = None,
    ) -> dict:
        """
        DEBUG: Retourne chunks avec ALL details (scores, sources, confiance).
        Utile pour l'endpoint /debug/rag_test
        """
        chunks_with_scores = self.retrieve_chunks(
            query, k=k, current_chapter_idx=current_chapter_idx
        )
        
        return {
            "query": query,
            "total_chunks": len(chunks_with_scores),
            "chunks": [
                {
                    "content": doc.page_content[:300],
                    "confidence": round(confidence, 3),
                    "source": source_info,
                    "metadata": {
                        "chapter": doc.metadata.get("chapter_idx"),
                        "chapter_title": doc.metadata.get("chapter_title"),
                        "language": doc.metadata.get("language"),
                        "content_length": len(doc.page_content),
                    }
                }
                for doc, confidence, source_info in chunks_with_scores
            ]
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  GÉNÉRATION DE RÉPONSE PÉDAGOGIQUE
    # ══════════════════════════════════════════════════════════════════════════

    def generate_final_answer(
        self,
        retrieved_chunks: list[tuple] | list[Document],  # Accept both formats
        question: str | None = None,
        query: str | None = None,
        history: list[dict] | None = None,
        language: str = "fr",
        student_level: str = "université",
        current_chapter_title: str = "",
        current_section_title: str = "",
    ) -> tuple[str, float]:  # Returns (answer, confidence)
        """
        Génère une réponse pédagogique avec score de confiance.
        
        Returns:
            Tuple of (answer_text, confidence_score)
            - confidence_score = moyenne des scores des chunks utilisés
        """
        question = question if question is not None else (query or "")
        history = history or []

        if not retrieved_chunks:
            return (self._no_answer_message(language), 0.0)

        # Gérer two formats: new (doc, conf, source) ou old (doc only)
        docs = []
        chunk_confidences = []
        
        for item in retrieved_chunks:
            if isinstance(item, tuple) and len(item) >= 2:
                docs.append(item[0])  # doc
                chunk_confidences.append(item[1])  # confidence
            else:
                docs.append(item)
                chunk_confidences.append(0.5)

        deduped_pairs = self._dedupe_scored_docs(
            list(zip(docs, chunk_confidences, [""] * len(docs))),
            max_results=5,
        )
        docs = [doc for doc, _, _ in deduped_pairs]
        chunk_confidences = [confidence for _, confidence, _ in deduped_pairs]
        
        # Construire le contexte RAG
        context_parts = []
        for i, doc in enumerate(docs[:5]):
            ch_title  = doc.metadata.get("chapter_title", "")
            slide_idx = doc.metadata.get("slide_idx")
            slide_ref = f" (slide {slide_idx})" if slide_idx else ""
            prefix    = f"[Ch: {ch_title}{slide_ref}]\n" if ch_title else ""
            context_parts.append(f"{prefix}{doc.page_content}")

        context_str = "\n\n---\n\n".join(context_parts)

        # Construire le prompt système du cours
        system_prompt = self._build_course_system_prompt(
            language=language,
            student_level=student_level,
            current_chapter_title=current_chapter_title,
            current_section_title=current_section_title,
        )

        user_content = f"EXTRAITS DU COURS :\n{context_str}\n\n---\n\nQUESTION : {question}"

        try:
            if self._openai_disabled_reason:
                raise RuntimeError(f"OpenAI disabled: {self._openai_disabled_reason}")

            if not self._openai_disabled_reason:
                llm = _make_chat_llm(LLM_ANSWER, temperature=0.4, max_tokens=500)
                response = llm.invoke(self._build_chat_messages(system_prompt, history, user_content))
                answer = self._clean_for_speech(response.content.strip())
                answer = self._dedupe_answer_text(answer)

                # Moyenne des confiances des chunks utilisés
                avg_confidence = sum(chunk_confidences) / max(len(chunk_confidences), 1) if chunk_confidences else 0.5

                log.info(f"✅ Réponse générée : {len(answer)} chars | confidence={avg_confidence:.2f}")
                return (answer, avg_confidence)
        except Exception as exc:
            is_disabled_exc = str(exc).lower().startswith("openai disabled:")
            if is_disabled_exc:
                log.info(f"ℹ️ OpenAI désactivé pour ce RAG ({self._openai_disabled_reason}) → Ollama prioritaire")
            elif self._should_disable_openai(exc):
                self._disable_openai(str(exc))
                log.error(f"❌ OpenAI error: {exc} → Trying Ollama fallback...")
            elif not is_disabled_exc:
                log.error(f"❌ OpenAI error: {exc} → Trying Ollama fallback...")
            
            # Try Ollama fallback
            try:
                from ai.local_llm import LocalLLMFallback
                import requests
                
                fallback_llm = LocalLLMFallback(model="mistral")
                if fallback_llm.available:
                    log.info(f"🖥️ Ollama fallback ({fallback_llm.base_url}) with Mistral...")
                    
                    payload = {
                        "model": "mistral",
                        "prompt": f"{system_prompt}\n\n{user_content}",
                        "temperature": 0.4,
                        "num_predict": 500,
                        "stream": False,
                    }
                    response = requests.post(
                        f"{fallback_llm.base_url}/api/generate",
                        json=payload,
                        timeout=None,  # Pas de timeout pour laisser Ollama répondre à son rythme
                    )
                    
                    if response.status_code == 200:
                        ollama_text = response.json().get("response", "").strip()
                        if ollama_text:
                            answer = self._clean_for_speech(ollama_text)
                            answer = self._dedupe_answer_text(answer)
                            avg_confidence = sum(chunk_confidences) / max(len(chunk_confidences), 1) if chunk_confidences else 0.5
                            log.info(f"✅ Ollama OK: {len(answer)} chars")
                            return (answer, avg_confidence)
            except requests.exceptions.Timeout:
                log.error("❌ Ollama request failed or was interrupted")
            except Exception as fallback_err:
                log.error(f"❌ Ollama fallback failed: {fallback_err}")
            
            # All failed - return error
            return (self._error_message(language), 0.0)

    async def generate_final_answer_stream(
        self,
        retrieved_chunks: list[Document],
        question: str | None = None,
        query: str | None = None,
        history: list[dict] | None = None,
        language: str = "fr",
        student_level: str = "université",
        current_chapter_title: str = "",
        current_section_title: str = "",
    ):
        """
        🚀 STREAMING VERSION: Génère une réponse par chunks (phrases complètes).
        
        Yields:
            Tuples of (sentence_text, full_response_so_far)
            Permet streaming LLM → TTS en temps réel
        """
        question = question if question is not None else (query or "")
        history = history or []

        if not retrieved_chunks:
            yield (self._no_answer_message(language), "")
            return

        if isinstance(retrieved_chunks[0], tuple):
            docs_only = [item[0] for item in retrieved_chunks if item]
        else:
            docs_only = list(retrieved_chunks)

        deduped_docs = self._dedupe_scored_docs(
            [(doc, 0.5, "") for doc in docs_only],
            max_results=5,
        )
        docs_only = [doc for doc, _, _ in deduped_docs]

        # Construire le contexte RAG
        context_parts = []
        for i, doc in enumerate(docs_only[:5]):
            ch_title  = doc.metadata.get("chapter_title", "")
            slide_idx = doc.metadata.get("slide_idx")
            slide_ref = f" (slide {slide_idx})" if slide_idx else ""
            prefix    = f"[Ch: {ch_title}{slide_ref}]\n" if ch_title else ""
            context_parts.append(f"{prefix}{doc.page_content}")

        context_str = "\n\n---\n\n".join(context_parts)

        # Construire le prompt système du cours
        system_prompt = self._build_course_system_prompt(
            language=language,
            student_level=student_level,
            current_chapter_title=current_chapter_title,
            current_section_title=current_section_title,
        )

        user_content = f"EXTRAITS DU COURS :\n{context_str}\n\n---\n\nQUESTION : {question}"

        try:
            if self._openai_disabled_reason:
                raise RuntimeError(f"OpenAI disabled: {self._openai_disabled_reason}")

            if not self._openai_disabled_reason:
                llm = _make_chat_llm(LLM_ANSWER, temperature=0.4, max_tokens=500)

                # Stream tokens from LLM (synchronous iterator)
                full_response = ""
                display_response = ""
                display_sentences: list[str] = []
                seen_signatures: list[str] = []
                buffer = ""
                sentence_endings = (".", "!", "?", ":\n", ";\n")

                for chunk in llm.stream(self._build_chat_messages(system_prompt, history, user_content)):
                    token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += token
                    buffer += token

                    # Check for sentence endings
                    if any(buffer.endswith(ending) for ending in sentence_endings):
                        sentence = buffer.strip()
                        if len(sentence) > 3:  # Minimum meaningful sentence length
                            cleaned = self._clean_for_speech(sentence)
                            signature = _normalize_text_for_diversity(cleaned)
                            if signature and any(self._is_near_duplicate(signature, seen) for seen in seen_signatures[-4:]):
                                log.info(f"⏭️ Repetition skipped: {cleaned[:60]}…")
                            else:
                                if signature:
                                    seen_signatures.append(signature)
                                display_sentences.append(cleaned)
                                display_response = " ".join(display_sentences).strip()
                                log.info(f"📤 Streaming chunk: {cleaned[:60]}…")
                                yield (cleaned, display_response)
                        buffer = ""

                # Yield remaining buffer
                if buffer.strip():
                    cleaned = self._clean_for_speech(buffer.strip())
                    signature = _normalize_text_for_diversity(cleaned)
                    if len(cleaned) > 3 and not (signature and any(self._is_near_duplicate(signature, seen) for seen in seen_signatures[-4:])):
                        if signature:
                            seen_signatures.append(signature)
                        display_sentences.append(cleaned)
                        display_response = " ".join(display_sentences).strip()
                        log.info(f"📤 Final chunk: {cleaned[:60]}…")
                        yield (cleaned, display_response)

                if not display_response:
                    display_response = self._dedupe_answer_text(self._clean_for_speech(full_response.strip()))

                log.info(f"✅ Stream complété : {len(display_response or full_response)} chars total")

        except Exception as exc:
            is_disabled_exc = str(exc).lower().startswith("openai disabled:")
            if is_disabled_exc:
                log.info(f"ℹ️ OpenAI désactivé pour ce RAG ({self._openai_disabled_reason}) → Ollama prioritaire")
            elif self._should_disable_openai(exc):
                self._disable_openai(str(exc))
                log.error(f"❌ OpenAI LLM stream error: {exc}")
            elif not is_disabled_exc:
                log.error(f"❌ OpenAI LLM stream error: {exc}")
            log.info("🖥️ Activating Ollama fallback for streaming...")
            
            # ══════════════════════════════════════════════════════════
            # FALLBACK: Ollama + Mistral (synchrone, mais garantit une réponse)
            # ══════════════════════════════════════════════════════════
            try:
                from ai.local_llm import LocalLLMFallback
                fallback_llm = LocalLLMFallback(model="mistral")
                
                if fallback_llm.available:
                    log.info(f"🖥️ Utilizing Ollama ({fallback_llm.base_url}) with Mistral model...")
                    
                    # Build fallback prompt
                    fallback_prompt = f"{system_prompt}\n\n{user_content}"
                    
                    # Call Ollama with streaming (much faster response)
                    import requests
                    import json
                    try:
                        payload = {
                            "model": "mistral",
                            "prompt": fallback_prompt,
                            "temperature": 0.4,
                            "num_predict": 500,
                            "stream": True,  # ✅ STREAMING for fast incremental response
                        }
                        response = requests.post(
                            f"{fallback_llm.base_url}/api/generate",
                            json=payload,
                            timeout=None,  # Pas de timeout pour laisser Ollama répondre à son rythme
                            stream=True,  # ✅ Stream chunks from requests
                        )
                        
                        if response.status_code == 200:
                            full_ollama_response = ""
                            display_sentences_ollama = []
                            seen_signatures_ollama = []
                            buffer_ollama = ""
                            
                            # Stream NDJSON response from Ollama
                            for line in response.iter_lines():
                                if not line:
                                    continue
                                try:
                                    chunk_data = json.loads(line)
                                    chunk_text = chunk_data.get("response", "")
                                    if chunk_text:
                                        full_ollama_response += chunk_text
                                        buffer_ollama += chunk_text
                                        
                                        # Check for sentence endings to yield progressively
                                        sentence_endings = (".", "!", "?", ":\n", ";\n")
                                        if any(buffer_ollama.endswith(ending) for ending in sentence_endings):
                                            sentence = buffer_ollama.strip()
                                            if len(sentence) > 3:
                                                cleaned = self._clean_for_speech(sentence)
                                                signature = _normalize_text_for_diversity(cleaned)
                                                if signature and any(self._is_near_duplicate(signature, seen) for seen in seen_signatures_ollama[-4:]):
                                                    log.info("⏭️ Ollama: Repetition skipped")
                                                else:
                                                    if signature:
                                                        seen_signatures_ollama.append(signature)
                                                    display_sentences_ollama.append(cleaned)
                                                    display_response_ollama = " ".join(display_sentences_ollama).strip()
                                                    log.info(f"📤 Ollama chunk: {cleaned[:60]}…")
                                                    yield (cleaned, display_response_ollama)
                                            buffer_ollama = ""
                                except json.JSONDecodeError:
                                    continue
                            
                            # Yield remaining buffer from Ollama
                            if buffer_ollama.strip():
                                cleaned = self._clean_for_speech(buffer_ollama.strip())
                                signature = _normalize_text_for_diversity(cleaned)
                                if len(cleaned) > 3 and not (signature and any(self._is_near_duplicate(signature, seen) for seen in seen_signatures_ollama[-4:])):
                                    if signature:
                                        seen_signatures_ollama.append(signature)
                                    display_sentences_ollama.append(cleaned)
                                    display_response_ollama = " ".join(display_sentences_ollama).strip()
                                    log.info(f"📤 Ollama final chunk: {cleaned[:60]}…")
                                    yield (cleaned, display_response_ollama)
                            
                            log.info(f"✅ Ollama fallback OK: {len(full_ollama_response)} chars")
                            return
                    except requests.exceptions.Timeout:
                        log.error("❌ Ollama request failed or was interrupted")
                    except Exception as ollama_err:
                        log.error(f"❌ Ollama call failed: {ollama_err}")
            except Exception as fallback_err:
                log.error(f"❌ Ollama fallback activation failed: {fallback_err}")
            
            # If all else fails, yield error message
            error_msg = self._error_message(language)
            log.error("❌ Both OpenAI and Ollama failed - returning error message")
            yield (error_msg, "")

    def generate_quiz(
        self,
        retrieved_chunks: list[tuple] | list[Document],
        question: str | None = None,
        query: str | None = None,
        history: list[dict] | None = None,
        language: str = "fr",
        student_level: str = "université",
        current_chapter_title: str = "",
        current_section_title: str = "",
        question_count: int = 3,
    ) -> tuple[dict, float]:
        """Generate a short multiple-choice quiz grounded in retrieved course chunks."""
        topic = (question if question is not None else (query or "")).strip()
        history = history or []
        desired_question_count = max(1, min(int(question_count or 3), 5))

        def _fallback_quiz_payload() -> dict:
            base_topic = topic or current_section_title or current_chapter_title or "ce cours"
            anchor = current_section_title or current_chapter_title or base_topic
            return {
                "title": "Quiz rapide",
                "topic": base_topic,
                "difficulty": student_level,
                "language": language[:2].lower(),
                "chapter_title": current_chapter_title or "",
                "section_title": current_section_title or "",
                "questions": [
                    {
                        "question": f"Quel est le point principal de {anchor} ?",
                        "options": [
                            f"L'idee principale de {anchor}",
                            "Un detail secondaire du cours",
                            "Un element hors sujet",
                            "Une erreur de formulation",
                        ],
                        "correct_index": 0,
                        "explanation": f"La bonne reponse reprend le theme central de {anchor}.",
                    }
                ],
            }

        def _parse_quiz_payload(raw_text: str) -> dict | None:
            if not raw_text:
                return None

            text = raw_text.strip()
            if not text:
                return None

            candidates = [text]
            if text.startswith("```"):
                fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
                if fenced:
                    candidates.insert(0, fenced)

            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidates.insert(0, text[start : end + 1])

            for candidate in candidates:
                try:
                    payload = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    return payload
            return None

        def _normalize_quiz_payload(payload: dict | None) -> dict | None:
            if not isinstance(payload, dict):
                return None

            questions_raw = payload.get("questions")
            if not isinstance(questions_raw, list):
                return None

            normalized_questions: list[dict] = []
            for item in questions_raw[:desired_question_count]:
                if not isinstance(item, dict):
                    continue

                question_text = str(item.get("question") or item.get("prompt") or "").strip()
                options_raw = item.get("options") or item.get("choices") or []
                if isinstance(options_raw, str):
                    options_raw = [line.strip() for line in options_raw.splitlines() if line.strip()]
                if not isinstance(options_raw, list):
                    options_raw = []
                options = [str(option).strip() for option in options_raw if str(option).strip()]
                if len(options) < 2:
                    continue

                try:
                    correct_index = int(item.get("correct_index", item.get("answer_index", 0)))
                except Exception:
                    correct_index = 0
                correct_index = max(0, min(correct_index, len(options) - 1))

                explanation = str(item.get("explanation") or item.get("feedback") or "").strip()
                if not question_text:
                    continue

                normalized_questions.append(
                    {
                        "question": question_text,
                        "options": options[:4],
                        "correct_index": correct_index,
                        "explanation": explanation,
                    }
                )

            if not normalized_questions:
                return None

            return {
                "title": str(payload.get("title") or payload.get("quiz_title") or "Quiz rapide").strip() or "Quiz rapide",
                "topic": str(payload.get("topic") or topic or current_section_title or current_chapter_title or "").strip(),
                "difficulty": str(payload.get("difficulty") or student_level).strip(),
                "language": str(payload.get("language") or language[:2].lower()).strip(),
                "chapter_title": str(payload.get("chapter_title") or current_chapter_title or "").strip(),
                "section_title": str(payload.get("section_title") or current_section_title or "").strip(),
                "questions": normalized_questions,
            }

        if not retrieved_chunks:
            return (_fallback_quiz_payload(), 0.0)

        if isinstance(retrieved_chunks[0], tuple):
            docs = [item[0] for item in retrieved_chunks if item]
            chunk_confidences = [float(item[1]) if len(item) > 1 and isinstance(item[1], (int, float)) else 0.5 for item in retrieved_chunks if item]
        else:
            docs = list(retrieved_chunks)
            chunk_confidences = [0.5 for _ in docs]

        deduped_docs = self._dedupe_scored_docs(
            [(doc, confidence, "") for doc, confidence in zip(docs, chunk_confidences)],
            max_results=5,
        )
        docs = [doc for doc, _, _ in deduped_docs]
        chunk_confidences = [confidence for _, confidence, _ in deduped_docs]

        context_parts = []
        for i, doc in enumerate(docs[:5]):
            ch_title = doc.metadata.get("chapter_title", "")
            slide_idx = doc.metadata.get("slide_idx")
            slide_ref = f" (slide {slide_idx})" if slide_idx else ""
            prefix = f"[Ch: {ch_title}{slide_ref}]\n" if ch_title else ""
            context_parts.append(f"{prefix}{doc.page_content}")

        context_str = "\n\n---\n\n".join(context_parts)
        avg_confidence = sum(chunk_confidences) / max(len(chunk_confidences), 1) if chunk_confidences else 0.5

        if language[:2].lower() == "en":
            system_prompt = (
                f"You are Smart Teacher. Create a short multiple-choice quiz grounded only in the provided course extracts. "
                f"Return ONLY valid JSON, no markdown, no extra text.\n\n"
                f"Required schema:\n"
                f"{{\n"
                f"  \"title\": \"Quick quiz\",\n"
                f"  \"topic\": \"...\",\n"
                f"  \"difficulty\": \"{student_level}\",\n"
                f"  \"language\": \"en\",\n"
                f"  \"chapter_title\": \"...\",\n"
                f"  \"section_title\": \"...\",\n"
                f"  \"questions\": [\n"
                f"    {{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"correct_index\": 0, \"explanation\": \"...\"}}\n"
                f"  ]\n"
                f"}}\n\n"
                f"Rules: use 2 to {desired_question_count} questions, exactly 4 options per question, one correct answer only, and keep the questions concise."
            )
        else:
            system_prompt = (
                f"Tu es Smart Teacher. Cree un mini quiz a choix multiples base uniquement sur les extraits du cours fournis. "
                f"Reponds UNIQUEMENT en JSON valide, sans markdown ni texte en plus.\n\n"
                f"Schema attendu:\n"
                f"{{\n"
                f"  \"title\": \"Quiz rapide\",\n"
                f"  \"topic\": \"...\",\n"
                f"  \"difficulty\": \"{student_level}\",\n"
                f"  \"language\": \"fr\",\n"
                f"  \"chapter_title\": \"...\",\n"
                f"  \"section_title\": \"...\",\n"
                f"  \"questions\": [\n"
                f"    {{\"question\": \"...\", \"options\": [\"...\", \"...\", \"...\", \"...\"], \"correct_index\": 0, \"explanation\": \"...\"}}\n"
                f"  ]\n"
                f"}}\n\n"
                f"Regles: propose 2 a {desired_question_count} questions, exactement 4 options par question, une seule bonne reponse, et des distracteurs plausibles."
            )

        user_content = (
            f"LANGUE: {language}\n"
            f"NIVEAU: {student_level}\n"
            f"CHAPITRE: {current_chapter_title or 'N/A'}\n"
            f"SECTION: {current_section_title or 'N/A'}\n"
            f"THEME: {topic or current_section_title or current_chapter_title or 'le cours'}\n"
            f"NOMBRE_DE_QUESTIONS: {desired_question_count}\n\n"
            f"EXTRAITS DU COURS:\n{context_str}\n\n"
            f"Rends uniquement le JSON demande par le schema. Chaque question doit rester courte et couvrir un point verifiable dans les extraits."
        )

        normalized_payload: dict | None = None

        try:
            if self._openai_disabled_reason:
                raise RuntimeError(f"OpenAI disabled: {self._openai_disabled_reason}")

            if not self._openai_disabled_reason:
                llm = _make_chat_llm(LLM_ANSWER, temperature=0.35, max_tokens=800)
                response = llm.invoke(self._build_chat_messages(system_prompt, history, user_content))
                normalized_payload = _normalize_quiz_payload(_parse_quiz_payload(response.content))
        except Exception as exc:
            is_disabled_exc = str(exc).lower().startswith("openai disabled:")
            if is_disabled_exc:
                log.info(f"ℹ️ OpenAI désactivé pour ce RAG ({self._openai_disabled_reason}) → Ollama quiz fallback prioritaire")
            elif self._should_disable_openai(exc):
                self._disable_openai(str(exc))
                log.error(f"❌ OpenAI quiz error: {exc} → Trying Ollama fallback...")
            elif not is_disabled_exc:
                log.error(f"❌ OpenAI quiz error: {exc} → Trying Ollama fallback...")

            try:
                from ai.local_llm import LocalLLMFallback
                import requests

                fallback_llm = LocalLLMFallback(model="mistral")
                if fallback_llm.available:
                    log.info(f"🖥️ Ollama quiz fallback ({fallback_llm.base_url}) with Mistral...")

                    payload = {
                        "model": "mistral",
                        "prompt": f"{system_prompt}\n\n{user_content}",
                        "temperature": 0.35,
                        "num_predict": 800,
                        "stream": False,
                    }
                    response = requests.post(
                        f"{fallback_llm.base_url}/api/generate",
                        json=payload,
                        timeout=None,
                    )

                    if response.status_code == 200:
                        ollama_text = response.json().get("response", "").strip()
                        if ollama_text:
                            normalized_payload = _normalize_quiz_payload(_parse_quiz_payload(ollama_text))
            except requests.exceptions.Timeout:
                log.error("❌ Ollama quiz request failed or was interrupted")
            except Exception as fallback_err:
                log.error(f"❌ Ollama quiz fallback failed: {fallback_err}")

        if not normalized_payload:
            normalized_payload = _fallback_quiz_payload()

        normalized_payload["confidence"] = round(avg_confidence, 3)
        normalized_payload["question_count"] = len(normalized_payload.get("questions", []))
        return (normalized_payload, avg_confidence)

    # ══════════════════════════════════════════════════════════════════════════
    #  PIPELINE D'INGESTION INTERNE
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_pdf_toc(self, pdf_path: str) -> list[tuple[int, int, str]]:
        """Extrait la table des matieres d'un PDF.

        Retourne une liste triee [(page_1based, depth, title), ...] ou
        depth=0 designe un chapitre, depth=1 une section, etc.
        Liste vide si pas de TOC ou erreur.
        """
        if not pdf_path.lower().endswith(".pdf"):
            return []
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            outline = getattr(reader, "outline", None)
            if not outline:
                return []

            entries: list[tuple[int, int, str]] = []

            def walk(items, depth: int = 0) -> None:
                for item in items:
                    if isinstance(item, list):
                        walk(item, depth + 1)
                    else:
                        try:
                            page_num = reader.get_destination_page_number(item) + 1
                            title = (getattr(item, "title", "") or "").strip()
                            if title:
                                entries.append((page_num, depth, title))
                        except Exception:
                            continue

            walk(outline)
            entries.sort(key=lambda x: (x[0], x[1]))
            return entries
        except Exception as exc:
            log.warning(f"PDF TOC extraction failed for {Path(pdf_path).name}: {exc}")
            return []

    def _extract_tocs(self, file_paths: list[str]) -> dict[str, list[tuple[int, int, str]]]:
        """Pour chaque PDF dans file_paths, extrait la TOC. Cle = nom de fichier (basename)."""
        tocs: dict[str, list[tuple[int, int, str]]] = {}
        for fp in file_paths:
            try:
                resolved = Path(fp).expanduser().resolve()
                toc = self._extract_pdf_toc(str(resolved))
                if toc:
                    tocs[resolved.name] = toc
                    log.info(f"  📚 TOC extracted from {resolved.name}: {len(toc)} entries")
            except Exception as exc:
                log.warning(f"_extract_tocs error on {fp}: {exc}")
        return tocs

    def _toc_lookup(
        self,
        toc: list[tuple[int, int, str]],
        page: int | None,
    ) -> tuple[int, str, int, str]:
        """Trouve (chapter_idx, chapter_title, section_idx, section_title) pour une page.

        Retourne (0, '', 0, '') si pas de TOC ou page absente.
        Les indices sont 1-based (premier chapitre = 1, premiere section = 1).
        """
        if not toc or page is None:
            return (0, "", 0, "")
        chapter_idx = 0
        chapter_title = ""
        section_idx = 0
        section_title = ""
        chapter_counter = 0
        section_counter = 0
        for entry_page, depth, title in toc:
            if entry_page > page:
                break
            if depth == 0:
                chapter_counter += 1
                chapter_idx = chapter_counter
                chapter_title = title
                section_counter = 0
                section_idx = 0
                section_title = ""
            elif depth == 1:
                section_counter += 1
                section_idx = section_counter
                section_title = title
        return (chapter_idx, chapter_title, section_idx, section_title)

    # NOTE: _partition_files removed — IntelligentIngester now handles all PDF
    # parsing upstream. Elements arrive pre-extracted via index_from_elements().

    # NOTE: ``_wordify_math`` and ``_MATH_SYMBOL_MAP`` were removed.
    # The previous version mapped 39 unicode math symbols (∫, ∑, ², α …)
    # to English words ("integral", "squared", "alpha") at INGEST time,
    # before chunking. Three structural problems made it unfit for a
    # retrieval system :
    #
    #   1. Wrong layer. Wordification is a TTS-output concern (so the
    #      voice doesn't say "x squared symbol"), not an index concern.
    #      Applying it at index time destroyed retrieval precision: a
    #      student searching for "x²" found nothing because the index
    #      held "x squared".
    #   2. English-only. "alpha", "squared" — a French course lost its
    #      vocabulary register, and the choice of language was hidden in
    #      a hardcoded dict, not driven by ``language`` metadata.
    #   3. Irreversible. Once ingested, the original symbols were gone.
    #      Switching strategy required full re-ingestion.
    #
    # Math symbols now flow through the index UNTOUCHED, preserving
    # retrieval precision. The TTS layer ``audio/math_speech.py``
    # handles symbol → spoken word conversion at output time, with
    # bilingual support (FR + EN) and a sympy LaTeX parser when
    # available. See ``audio.math_speech.to_speech``.

    @staticmethod
    def _table_to_markdown(table_element) -> str:
        """Convertit un unstructured.Table en Markdown table (preserve la structure)."""
        try:
            html = getattr(getattr(table_element, "metadata", None), "text_as_html", None)
            if not html:
                # fallback : texte brut
                return str(table_element).strip()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            if not rows:
                return str(table_element).strip()
            md_lines: list[str] = []
            for i, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                cell_texts = [c.get_text(strip=True) or " " for c in cells]
                md_lines.append("| " + " | ".join(cell_texts) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
            return "\n".join(md_lines)
        except Exception:
            return str(table_element).strip()

    def _create_chunks_by_title(self, elements: list) -> list:
        try:
            # Pre-process : table → markdown + figure captions on each element.
            # Math symbols are NOT wordified here — they're preserved in the
            # index for retrieval precision and converted at TTS time only
            # (see ``audio.math_speech``).
            from unstructured.documents.elements import Table
            try:
                # FigureCaption peut s'appeler differemment selon la version
                from unstructured.documents.elements import FigureCaption  # type: ignore
            except Exception:
                FigureCaption = None  # graceful fallback si la classe n'existe pas

            n_tables_md = 0
            n_captions = 0
            for el in elements:
                # Tables : injecter Markdown dans le text de l'element
                if isinstance(el, Table):
                    md = self._table_to_markdown(el)
                    if md and md != str(el).strip():
                        try:
                            el.text = md
                            n_tables_md += 1
                        except Exception:
                            pass

                # FigureCaption : prefixer pour que le LLM sache que c'est une legende
                #   d'image (donc parle d'un schema absent du texte). Les keyphrases
                #   extraites de cette caption restent associees au concept de la figure.
                if FigureCaption is not None and isinstance(el, FigureCaption):
                    caption_text = (getattr(el, "text", "") or "").strip()
                    if caption_text and not caption_text.lower().startswith(("figure", "fig.", "schema")):
                        try:
                            el.text = f"[Légende figure] {caption_text}"
                            n_captions += 1
                        except Exception:
                            pass
                    elif caption_text:
                        n_captions += 1

            chunks = chunk_by_title(elements, max_characters=1500, new_after_n_chars=1200)
            log.info(
                f"✂️  Chunks créés : {len(chunks)} "
                f"(tables→markdown: {n_tables_md}, figure captions: {n_captions})"
            )
            return chunks
        except Exception as exc:
            log.error(f"❌ Chunking error: {exc}")
            return []

    # ── LLM invocation : delegue au LLMRouter (OpenAI prioritaire) ──────────
    def _invoke_llm_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str | None:
        """Thin wrapper sur LLMRouter — OpenAI prioritaire, Ollama fallback."""
        return self._llm.invoke(
            prompt,
            prefer="openai",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Idea-level chunking (LLM-driven) ────────────────────────────────────
    _IDEA_LABELS = {"definition", "theorem", "procedure", "example", "warning", "note"}

    # Prompt version — bump invalidates idea_cache (new prompt = new structure).
    _IDEA_PROMPT_VERSION = "v2_graph"

    def _chunk_by_ideas_batch(
        self,
        items: list[tuple[str, str, str]],
    ) -> list[list[dict]]:
        """Parallel idea-chunking pour N (text, chapter_title, section_title).

        Why: chaque ``_chunk_by_ideas`` fait 1 appel LLM (3-10s OpenAI, plus sur
        Ollama CPU). Sequentiel sur 20 sections = 60-200s. I/O-bound → threads.
        """
        if not items:
            return []
        max_workers = max(1, min(Config.RAG_LLM_PARALLELISM, len(items)))
        if max_workers == 1:
            return [self._chunk_by_ideas(*it) for it in items]

        from concurrent.futures import ThreadPoolExecutor
        results: list[list[dict] | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(self._chunk_by_ideas, *it): idx
                for idx, it in enumerate(items)
            }
            for fut in future_to_idx:
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"_chunk_by_ideas batch[{idx}] failed: {exc}")
                    results[idx] = [{
                        "idea": items[idx][0],
                        "label": "fragment",
                        "local_id": "frag_1",
                        "depends_on": [],
                        "illustrates": None,
                    }]
        return [r if r is not None else [] for r in results]

    def _chunk_by_ideas(
        self,
        text: str,
        chapter_title: str = "",
        section_title: str = "",
    ) -> list[dict]:
        """Decoupe un title-chunk en idees pedagogiques atomiques via LLM.

        Retourne [{"local_id": str, "idea": str, "label": str,
                   "depends_on": list[str], "illustrates": str|None}, ...].

        ``local_id``    : identifiant lisible local au chunk (ex: "def_1", "thm_2").
                          Utilise pour les relations DANS LE MEME chunk.
        ``depends_on``  : liste de local_id pre-requis (ex: un theorem depends_on
                          ses lemmes / definitions).
        ``illustrates`` : local_id du concept illustre (cas d'un example).

        Fallback : 1 seul "idea" qui est le texte original si:
          - texte trop court (<200 chars)
          - LLM disabled
          - parse JSON echoue
          - feature flag off
        """
        # Fallback simple si feature off ou texte trop court
        if not Config.RAG_USE_IDEA_CHUNKING or len(text) < 200:
            return [{"idea": text, "label": "fragment", "local_id": "frag_1",
                     "depends_on": [], "illustrates": None}]

        # Cache check — clé inclut la version du prompt pour invalidation propre
        key = hashlib.md5(f"{self._IDEA_PROMPT_VERSION}|{text}".encode()).hexdigest()
        if key in self.idea_cache:
            cached = self.idea_cache[key]
            if isinstance(cached, list) and cached:
                return cached

        ctx = ""
        if chapter_title:
            ctx = f"\nChapitre : {chapter_title}"
            if section_title:
                ctx += f"\nSection : {section_title}"

        max_ideas = Config.RAG_IDEAS_PER_CHUNK_MAX
        prompt = (
            "Tu es un expert pédagogique. Découpe ce texte en idées atomiques + leurs RELATIONS.\n\n"
            "Une idée atomique = UN concept enseignable indépendamment :\n"
            "  - definition / theorem / procedure / example / warning / note\n\n"
            "Règles strictes :\n"
            f"  - Maximum {max_ideas} idées (minimum 1)\n"
            "  - Chaque idée est autonome (compréhensible hors contexte)\n"
            "  - Préserve la terminologie technique exacte du texte\n"
            "  - Pour chaque idée, donne un `local_id` court ASCII (ex: \"def_grad\", \"thm_pyth\")\n"
            "  - `depends_on` : liste des local_id pré-requis pour comprendre cette idée\n"
            "      (ex: un théorème depends_on ses définitions / lemmes)\n"
            "  - `illustrates` : si l'idée est un exemple, donne le local_id du concept illustré\n"
            "      (sinon null)\n"
            "  - Les relations sont LOCALES au texte fourni (pas d'inventions externes)\n"
            "  - JSON STRICT en sortie, sans markdown ni préambule\n\n"
            "Format JSON attendu :\n"
            '[{"local_id": "def_X", "idea": "<contenu>", '
            '"label": "definition|theorem|procedure|example|warning|note", '
            '"depends_on": ["local_id_a", ...], "illustrates": "local_id_b"|null}, ...]'
            f"{ctx}\n\n"
            f"TEXTE À DÉCOUPER :\n{text[:1800]}"
        )

        try:
            # Invoke via helper : OpenAI prioritaire, Ollama Mistral fallback
            raw = self._invoke_llm_text(prompt, temperature=0.0, max_tokens=1200)
            if not raw:
                raise ValueError("All LLM backends unavailable (OpenAI + Ollama)")
            # Robustness contre markdown fences (Ollama tend a en mettre)
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].lstrip()
                raw = raw.rstrip("`").strip()
            # Robustness : Ollama peut prefixer du texte avant le JSON
            json_start = raw.find("[")
            json_end = raw.rfind("]")
            if json_start != -1 and json_end > json_start:
                raw = raw[json_start:json_end + 1]
            ideas = json.loads(raw)
            if not isinstance(ideas, list) or not ideas:
                raise ValueError("LLM did not return a non-empty list")

            cleaned: list[dict] = []
            seen_local_ids: set[str] = set()
            for i, item in enumerate(ideas[:max_ideas]):
                if not isinstance(item, dict):
                    continue
                idea_text = (item.get("idea") or "").strip()
                if len(idea_text) < Config.RAG_IDEA_MIN_LENGTH:
                    continue
                label = (item.get("label") or "note").strip().lower()
                if label not in self._IDEA_LABELS:
                    label = "note"

                # local_id : LLM-provided, with fallback + collision avoidance
                raw_lid = (item.get("local_id") or "").strip()
                local_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_lid)[:32] or f"idea_{i}"
                if local_id in seen_local_ids:
                    local_id = f"{local_id}_{i}"
                seen_local_ids.add(local_id)

                # depends_on : list of local_ids — sanitize, keep only known later
                deps_raw = item.get("depends_on") or []
                deps = []
                if isinstance(deps_raw, list):
                    for d in deps_raw[:8]:    # cap at 8 dependencies per idea
                        if isinstance(d, str):
                            d_clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", d.strip())[:32]
                            if d_clean and d_clean != local_id:
                                deps.append(d_clean)

                # illustrates : single local_id (or None)
                illus_raw = item.get("illustrates")
                illus = None
                if isinstance(illus_raw, str) and illus_raw.strip():
                    illus_clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", illus_raw.strip())[:32]
                    if illus_clean and illus_clean != local_id:
                        illus = illus_clean

                cleaned.append({
                    "local_id":    local_id,
                    "idea":        idea_text,
                    "label":       label,
                    "depends_on":  deps,
                    "illustrates": illus,
                })

            if not cleaned:
                raise ValueError("All ideas filtered out")

            # Second pass: drop relations to unknown local_ids (LLM hallucinations)
            valid_ids = {item["local_id"] for item in cleaned}
            for item in cleaned:
                item["depends_on"] = [d for d in item["depends_on"] if d in valid_ids]
                if item["illustrates"] not in valid_ids:
                    item["illustrates"] = None

            # Persiste en cache
            self.idea_cache[key] = cleaned
            return cleaned
        except Exception as exc:
            log.warning(f"⚠️ Idea chunking failed ({exc}) — fallback chunk entier")
            return [{"idea": text, "label": "fragment", "local_id": "frag_1",
                     "depends_on": [], "illustrates": None}]


    def _summarise_chunks(
        self,
        chunks: list,
        domain: str = "general",
        course: str = "generic",
        course_id: str | None = None,
        file_tocs: dict[str, list[tuple[int, int, str]]] | None = None,
    ) -> list[Document]:
        """Process chunks into Documents enrichies.

        Pipeline par title-chunk :
          1. TOC lookup → chapter_idx, section_idx
          2. Idea-level split via LLM (configurable, fallback chunk entier)
          3. Pour chaque idée : summary + embedding-ready Document avec metadata complete
        """
        file_tocs = file_tocs or {}
        documents: list[Document] = []
        total_ideas = 0

        # Phase 1 : pre-traiter chaque chunk (TOC lookup, source, page) — pas de LLM
        prepared: list[dict] = []
        for i, chunk in enumerate(chunks):
            text = str(chunk).strip()
            if len(text) < 30:
                continue
            source = self._extract_source_file(chunk)
            page = self._extract_page_number(chunk)

            toc_entries = file_tocs.get(source) or file_tocs.get(Path(source).name)
            if toc_entries:
                chapter_idx, chapter_title, section_idx, section_title = self._toc_lookup(toc_entries, page)
            else:
                chapter_idx, chapter_title, section_idx, section_title = (0, "", 0, "")

            prepared.append({
                "i": i, "text": text, "source": source, "page": page,
                "chapter_idx": chapter_idx, "chapter_title": chapter_title,
                "section_idx": section_idx, "section_title": section_title,
            })

        # Phase 2 : idea-chunking en parallele (Fix #3 — speed-up principal)
        ideas_batch = self._chunk_by_ideas_batch(
            [(p["text"], p["chapter_title"], p["section_title"]) for p in prepared]
        )

        # Phase 3 : pre-collecter tous les idea_texts pour summarisation parallele
        idea_texts: list[str] = []
        for ideas in ideas_batch:
            idea_texts.extend(idea["idea"] for idea in ideas)
        # Map (idx) -> chapter_title pour le contexte de summarisation
        # (le summary depend du chapter_title, donc on aligne par idx)
        idea_chapter_titles: list[str] = []
        for p, ideas in zip(prepared, ideas_batch):
            idea_chapter_titles.extend([p["chapter_title"]] * len(ideas))
        summaries_flat = self._summaries_batch(idea_texts, idea_chapter_titles)

        # Phase 4 : construire les Documents
        summary_cursor = 0
        for p, ideas in zip(prepared, ideas_batch):
            i = p["i"]
            chapter_idx = p["chapter_idx"]
            chapter_title = p["chapter_title"]
            section_idx = p["section_idx"]
            section_title = p["section_title"]
            source = p["source"]
            page = p["page"]

            local_to_global: dict[str, str] = {}
            idea_ids_for_chunk: list[str] = []
            for j, idea_dict in enumerate(ideas):
                idea_text = idea_dict["idea"]
                content_hash = hashlib.md5(idea_text.encode()).hexdigest()[:8]
                idea_id = hashlib.md5(
                    f"{course_id or course}|{chapter_idx}|{section_idx}|{i}|{j}|{content_hash}".encode()
                ).hexdigest()[:12]
                idea_ids_for_chunk.append(idea_id)
                local_id = idea_dict.get("local_id") or f"idea_{j}"
                local_to_global[local_id] = idea_id

            for j, idea_dict in enumerate(ideas):
                idea_text = idea_dict["idea"]
                idea_label = idea_dict["label"]
                idea_id = idea_ids_for_chunk[j]

                depends_on_ids = [
                    local_to_global[d] for d in idea_dict.get("depends_on", [])
                    if d in local_to_global
                ]
                illustrates_local = idea_dict.get("illustrates")
                illustrates_id = local_to_global.get(illustrates_local) if illustrates_local else None

                summary = summaries_flat[summary_cursor] if summary_cursor < len(summaries_flat) else idea_text
                summary_cursor += 1
                lang = self._detect_language(idea_text)
                content_hash = hashlib.md5(idea_text.encode()).hexdigest()[:8]

                doc = Document(
                    page_content=summary or idea_text,
                    metadata={
                        "source_file":         source,
                        "chunk_idx":           i,                # parent title-chunk
                        "idea_index_in_chunk": j,                # ordre dans le title-chunk
                        "domain":              domain,
                        "course":              course_id or course,
                        "language":            lang,
                        "original_text":       idea_text[:500],
                        "content_hash":        content_hash,
                        "chapter_idx":         chapter_idx,
                        "chapter_title":       chapter_title,
                        "section_idx":         section_idx,
                        "section_title":       section_title,
                        "idea_id":             idea_id,
                        "idea_label":          idea_label,       # definition|theorem|procedure|example|warning|note
                        "slide_idx":           page,
                        "image_url":           "",               # vide en flow /ingest (pas de slide PNG extraite)
                        # ── Knowledge graph edges (intra-chunk for now) ──
                        "depends_on_ids":      depends_on_ids,   # list of idea_ids
                        "illustrates_id":      illustrates_id,   # idea_id|None
                    }
                )
                documents.append(doc)
                total_ideas += 1

        # Persist caches (summary + idea) cross-restart
        if self.summary_cache:
            self._save_summary_cache()
        if self.idea_cache:
            self._save_idea_cache()

        log.info(
            f"📝 Documents produits : {len(documents)} "
            f"(ideas: {total_ideas} sur {len(chunks)} title-chunks → "
            f"avg {total_ideas / max(1, len(chunks)):.1f} idées/chunk)"
        )
        return documents

    def _get_or_create_summary(self, text: str, chapter_context: str) -> str:
        """Résumé IA avec cache. Adapté pour le contenu DM."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.summary_cache:
            return self.summary_cache[key]

        if self._openai_disabled_reason:
            self.summary_cache[key] = text
            return text

        if len(text) < 120:
            self.summary_cache[key] = text
            return text

        try:
            llm = _make_chat_llm(LLM_SUMMARY, temperature=0.0, max_tokens=300)
            ctx = f" (contexte : {chapter_context})" if chapter_context else ""
            prompt = (
                f"Tu es un assistant pédagogique expert dans ce cours{ctx}.\n"
                f"Résume ce contenu en 2-3 phrases claires et précises.\n"
                f"Conserve TOUS les termes techniques importants.\n"
                f"Ne simplifie pas la terminologie technique.\n\n"
                f"CONTENU :\n{text[:1200]}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            summary  = response.content.strip()
            self.summary_cache[key] = summary
            return summary
        except Exception as exc:
            log.warning(f"Summary error: {exc}")
            return text

    def _summaries_batch(
        self,
        texts: list[str],
        chapter_contexts: list[str],
    ) -> list[str]:
        """Parallel summarisation. Cache hits restent instant; uniquement les
        misses paient le cout LLM, en parallele.

        Why: en sequentiel, N idees = N requetes OpenAI. Sur un PDF de 100 idees
        a 1-3s par appel, c'est 100-300s. En parallele a P=6 → 17-50s.
        """
        n = len(texts)
        if n == 0:
            return []
        results: list[str] = [""] * n
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self.summary_cache:
                results[i] = self.summary_cache[key]
            elif self._openai_disabled_reason or len(text) < 120:
                self.summary_cache[key] = text
                results[i] = text
            else:
                miss_indices.append(i)

        if not miss_indices:
            return results

        max_workers = max(1, min(Config.RAG_LLM_PARALLELISM, len(miss_indices)))
        if max_workers == 1:
            for i in miss_indices:
                results[i] = self._get_or_create_summary(texts[i], chapter_contexts[i])
            return results

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self._get_or_create_summary, texts[i], chapter_contexts[i]): i
                for i in miss_indices
            }
            for fut in futures:
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"summary batch[{i}] failed: {exc}")
                    results[i] = texts[i]
        return results

    def _fuse_short_sections(self, sections: list[dict]) -> list[dict]:
        """Fix #2 — Fusion des sections trop courtes en sections logiques.

        Why: extraction PDF produit souvent des sections de 1-2 phrases (titres
        orphelins, notes de bas de page, sauts de page). Idea-chunking sur 2
        phrases = 1 appel LLM gaspille pour 1 idee fragmentaire. La qualite
        pedagogique en souffre aussi (concepts coupes en deux).

        Strategie: glisser sur les sections, si len(content) < FUSION_MIN_CHARS
        et le buffer accumule + section <= FUSION_MAX_CHARS, fusionner. Sinon
        flush et continuer. Preserve l'ordre, le titre du premier element du
        groupe, et concatene les pages.
        """
        if not sections:
            return sections
        min_chars = max(0, Config.RAG_FUSION_MIN_CHARS)
        max_chars = max(min_chars + 200, Config.RAG_FUSION_MAX_CHARS)
        if min_chars == 0:
            return sections

        fused: list[dict] = []
        buf: dict | None = None
        merged_pages: list[int] = []

        def _flush():
            nonlocal buf, merged_pages
            if buf is not None:
                if merged_pages:
                    buf["fused_pages"] = sorted(set(merged_pages))
                fused.append(buf)
            buf = None
            merged_pages = []

        for sec in sections:
            content = (sec.get("content") or "").strip()
            page = sec.get("page_index") or sec.get("order")

            if buf is None:
                buf = dict(sec)
                buf["content"] = content
                if page is not None:
                    merged_pages = [int(page)]
                else:
                    merged_pages = []
                # Si la 1ere section est deja assez longue, on flush directement
                if len(content) >= min_chars:
                    _flush()
                continue

            # buf existe → fusion possible si combinaison reste sous max_chars
            combined_len = len(buf["content"]) + len(content) + 2
            if combined_len <= max_chars:
                # Fusionne
                buf["content"] = (buf["content"] + "\n\n" + content).strip()
                if page is not None:
                    merged_pages.append(int(page))
                # Si maintenant assez long, flush
                if len(buf["content"]) >= min_chars:
                    _flush()
            else:
                # Trop gros pour fusionner → flush buf, demarre nouveau buffer
                _flush()
                buf = dict(sec)
                buf["content"] = content
                if page is not None:
                    merged_pages = [int(page)]
                if len(content) >= min_chars:
                    _flush()

        _flush()

        if len(fused) != len(sections):
            log.info(
                f"📑 Page fusion: {len(sections)} sections brutes → {len(fused)} "
                f"sections fusionnees (min={min_chars}, max={max_chars} chars)"
            )
        return fused

    def _store_documents_once(
        self,
        documents: list[Document],
        full_documents: list[Document] | None = None,
    ) -> bool:
        documents_to_keep = full_documents if full_documents is not None else documents

        try:
            if not self.client:
                raise RuntimeError("Qdrant client unavailable")

            import time as _qt
            collection_exists = self.client.collection_exists(self.collection_name)
            log.info(
                "🟣 QDRANT collection_check | name=%s exists=%s | dim=%s distance=COSINE",
                self.collection_name, collection_exists, self.current_embedding_dim,
            )
            if not collection_exists:
                _t0 = _qt.time()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.current_embedding_dim, distance=Distance.COSINE),
                )
                log.info(
                    "🟣 QDRANT CREATE collection | name=%s dim=%s | took=%.0fms",
                    self.collection_name, self.current_embedding_dim,
                    (_qt.time() - _t0) * 1000,
                )

            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            docs_to_store = documents if collection_exists else documents_to_keep
            _t1 = _qt.time()
            log.info(
                "🟣 QDRANT UPSERT START | collection=%s | docs=%d | sample_meta=%s",
                self.collection_name, len(docs_to_store),
                str(docs_to_store[0].metadata)[:150] if docs_to_store else "{}",
            )
            self.vectorstore.add_documents(docs_to_store)
            elapsed_s = _qt.time() - _t1
            docs_per_s = (len(docs_to_store) / elapsed_s) if elapsed_s > 0 else 0.0
            log.info(
                "🟣 QDRANT UPSERT DONE | collection=%s | docs=%d | took=%.2fs (%.0f docs/s)",
                self.collection_name, len(docs_to_store), elapsed_s, docs_per_s,
            )
            self.all_docs = documents_to_keep
            self._build_hybrid_retriever()
            self._save_docs_cache()
            self.is_ready = True
            try:
                _count = self.client.count(self.collection_name, exact=False).count
                log.info(
                    "🟣 QDRANT collection_total | name=%s | total_vectors=%d (after upsert)",
                    self.collection_name, _count,
                )
            except Exception:
                pass
            return True
        except Exception as exc:
            self.vectorstore = None
            self.all_docs = documents_to_keep
            self._build_hybrid_retriever()
            self._save_docs_cache()
            self.is_ready = bool(self.all_docs)
            log.info(
                f"ℹ️ Qdrant local indisponible ({exc}) — documents conservés localement, BM25 actif."
            )
            return bool(self.all_docs)

    def _store_documents(self, documents: list[Document], incremental: bool) -> bool:
        existing_docs = list(self.all_docs)
        full_documents = [*existing_docs, *documents]

        try:
            return self._store_documents_once(
                documents,
                full_documents=full_documents,
            )
        except Exception as exc:
            if self.embedding_source != "huggingface" and self._should_fallback_to_local_embeddings(exc):
                log.warning(
                    "⚠️ Local primary embeddings unavailable during ingestion; retrying with HuggingFaceEmbeddings."
                )
                if self._switch_to_local_embeddings(str(exc)):
                    try:
                        return self._store_documents_once(
                            full_documents,
                            full_documents=full_documents,
                        )
                    except Exception as retry_exc:
                        log.error(f"❌ Local fallback store error: {retry_exc}")
                        return False

            log.error(f"❌ Store error: {exc}")
            return False

    def _build_hybrid_retriever(self) -> None:
        if not self.all_docs:
            return
        try:
            self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
            log.info(f"✅ BM25 retriever construit ({len(self.all_docs)} docs)")
        except Exception as exc:
            log.warning(f"BM25 build error: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PROMPTS PÉDAGOGIQUES
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_course_system_prompt(
        domain: str = "general",
        course: str = "generic",
        language: str = "fr",
        student_level: str = "licence",
        current_chapter_title: str = "",
        current_section_title: str = "",
    ) -> str:
        """
        Prompt système générique par domaine/cours.
        Contextualisé avec le chapitre et la section en cours.

        LANGUE : Strictement respectée. Instructions très claires.
        """
        domain_descriptions = {
            "general": "cours",
            "computer_science": "informatique",
            "mathematics": "mathématiques",
            "science": "sciences",
            "languages": "langues",
            "humanities": "humanités",
            "business": "gestion",
            "engineering": "ingénierie",
        }
        
        domain_desc = domain_descriptions.get(course, domain_descriptions.get(domain, domain.replace("_", " ").capitalize()))
        
        # Contexte chapitre/section
        ch_ctx = ""
        if current_chapter_title:
            if language[:2].lower() == "en":
                ch_ctx = f"\nWe are currently in chapter: '{current_chapter_title}'."
            else:
                ch_ctx = f"\nNous sommes actuellement dans le chapitre : '{current_chapter_title}'."
        if current_section_title:
            if language[:2].lower() == "en":
                ch_ctx += f" Section: '{current_section_title}'."
            else:
                ch_ctx += f" Section : '{current_section_title}'."

        base = {
            "fr": (
                f"Tu es Smart Teacher, un professeur expert en {domain_desc} "
                f"qui enseigne à des étudiants de niveau {student_level}.{ch_ctx}\n\n"
                f"RÈGLES ABSOLUES — tu PARLES, tu n'écris PAS :\n"
                f"- JAMAIS de markdown : pas de **, pas de #, pas de tirets -, pas de listes.\n"
                f"- JAMAIS de LaTeX : écris les formules en clair.\n"
                f"- Réponds comme tu PARLERAIS en cours : phrases naturelles, transitions fluides.\n"
                f"- Commence par : 'Bonne question !', 'Exactement !', 'Alors, pour ce concept...' etc.\n"
                f"- Utilise la terminologie technique appropriée.\n"
                f"- Si disponible, base-toi sur les extraits du cours fournis.\n"
                f"- Si plusieurs extraits disent la même chose, fusionne-les en une seule explication.\n"
                f"- Ne répète pas la même idée avec des mots proches.\n"
                f"- Un exemple concret et pertinent.\n"
                f"- 4 à 6 phrases naturelles. Termine par une question si concept difficile.\n"
                f"- [!!!CRITIQUE!!!] Réponds UNIQUEMENT en français. Aucune autre langue acceptée."
            ),
            "en": (
                f"You are Smart Teacher, an expert professor in {domain_desc} "
                f"teaching {student_level} level students.{ch_ctx}\n\n"
                f"ABSOLUTE RULES — you are SPEAKING, not writing:\n"
                f"- NEVER use markdown: no **, #, bullet points, numbered lists.\n"
                f"- NEVER use LaTeX: write formulas in plain words.\n"
                f"- Reply as if TALKING in class: natural sentences, smooth transitions.\n"
                f"- Start with: 'Great question!', 'Exactly!', 'So, for this concept...' etc.\n"
                f"- Use technical terminology accurately.\n"
                f"- Use course extracts provided when available.\n"
                f"- If several extracts say the same thing, merge them into one explanation.\n"
                f"- Do not repeat the same idea with different wording.\n"
                f"- Include a concrete relevant example.\n"
                f"- 4 to 6 natural sentences. End with a comprehension question if complex.\n"
                f"- [!!!CRITICAL!!!] Reply ONLY in English. No other language accepted."
            ),
        }
        return base.get(language[:2].lower(), base["fr"])

    # ══════════════════════════════════════════════════════════════════════════
    #  UTILITAIRES
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte."""
        if not text:
            return "en"
        fr_count  = len(re.findall(r'\b(le|la|les|de|du|des|un|une|et|est|qui|que|dans|pour|avec|sur|par)\b', text.lower()))
        en_count  = len(re.findall(r'\b(the|of|and|is|are|in|for|with|on|by|this|that|which|from)\b', text.lower()))
        if fr_count > en_count:
            return "fr"
        return "en"

    def _extract_source_file(self, chunk) -> str:
        meta = getattr(chunk, "metadata", None)
        if meta is None:
            return "unknown"
        return (
            getattr(meta, "filename", None)
            or getattr(meta, "file_path", None)
            or "unknown"
        )

    def _extract_page_number(self, chunk) -> int | None:
        meta = getattr(chunk, "metadata", None)
        page = getattr(meta, "page_number", None) if meta else None
        return int(page) if page is not None else None

    def _load_existing_db(self) -> None:
        if not self._embeddings_ok or self.embeddings is None:
            log.warning("⚠️  _load_existing_db ignoré : embeddings non disponibles (quota OpenAI ?)")
            return

        if self.docs_cache.exists():
            self._load_docs_cache()

        try:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            self.is_ready = True
            log.info(f"✅ Collection '{self.collection_name}' chargée.")
            if not self.all_docs:
                log.info("🔄 Pas de cache — chargement depuis Qdrant…")
                retriever  = self.vectorstore.as_retriever(search_kwargs={"k": 1000})
                self.all_docs = retriever.invoke(" ")
            self._build_hybrid_retriever()
        except Exception as exc:
            log.info(f"ℹ️ Load DB error ({exc}) — local fallback active")
            self._activate_local_retrieval_fallback(str(exc))

    def _save_docs_cache(self) -> None:
        try:
            data = [{"page_content": d.page_content, "metadata": d.metadata}
                    for d in self.all_docs]
            with open(self.docs_cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log.info(f"💾 Cache sauvegardé ({len(data)} docs → {self.docs_cache})")
        except Exception as exc:
            log.warning(f"Cache save error: {exc}")

    def _load_docs_cache(self) -> None:
        try:
            with open(self.docs_cache, encoding="utf-8") as f:
                data = json.load(f)
            self.all_docs = [Document(page_content=d["page_content"], metadata=d["metadata"])
                             for d in data]
            log.info(f"✅ Cache chargé ({len(self.all_docs)} docs)")
        except Exception as exc:
            log.warning(f"Cache load error: {exc}")

    def _save_summary_cache(self) -> None:
        try:
            with open(self.summary_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.summary_cache, f, ensure_ascii=False)
        except Exception as exc:
            log.warning(f"Summary cache save error: {exc}")

    def _load_summary_cache(self) -> None:
        if self.summary_cache_path.exists():
            try:
                with open(self.summary_cache_path, encoding="utf-8") as f:
                    self.summary_cache = json.load(f)
                log.info(f"✅ Summary cache chargé ({len(self.summary_cache)} entrées)")
            except Exception:
                self.summary_cache = {}

    def _save_idea_cache(self) -> None:
        try:
            with open(self.idea_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.idea_cache, f, ensure_ascii=False)
        except Exception as exc:
            log.warning(f"Idea cache save error: {exc}")

    def _load_idea_cache(self) -> None:
        if self.idea_cache_path.exists():
            try:
                with open(self.idea_cache_path, encoding="utf-8") as f:
                    self.idea_cache = json.load(f)
                log.info(f"✅ Idea cache chargé ({len(self.idea_cache)} entrées)")
            except Exception:
                self.idea_cache = {}

    @staticmethod
    def _clean_for_speech(text: str, language: str = "fr") -> str:
        """Render LLM output as TTS-ready spoken text.

        Math notation is VERBALIZED (``$x^2$`` → "x squared" / "x au
        carré") via ``audio.math_speech.to_speech``, not deleted. The
        previous version stripped LaTeX blocks entirely, which silently
        dropped the math content from the spoken answer. After
        verbalization, markdown structure (headings, bullets, code) is
        flattened — TTS doesn't render structure.
        """
        import re
        try:
            from audio.math_speech import to_speech as _math_to_speech
            text = _math_to_speech(text, lang=(language or "fr")[:2])
        except Exception:
            # Failsafe: drop LaTeX blocks rather than echo raw symbols.
            text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
            text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
            text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
            text = re.sub(r'\$[^$\n]+\$', '', text)
            text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
            text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)
        text = re.sub(r'^\s*[-•–—]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+[.)]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\\n|\\t|\\r', ' ', text)
        text = text.replace('\\', '')
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    @staticmethod
    def _no_answer_message(language: str) -> str:
        return {
            "fr": "Je n'ai pas trouvé d'information pertinente dans le cours. Pouvez-vous reformuler votre question ?",
            "en": "I couldn't find relevant information in the course material. Could you rephrase your question?",
        }.get(language, "Je n'ai pas trouvé de réponse dans le cours.")

    @staticmethod
    def _error_message(language: str) -> str:
        return {
            "fr": "Une erreur s'est produite. Veuillez réessayer.",
            "en": "An error occurred. Please try again.",
        }.get(language, "Une erreur s'est produite.")

    def get_stats(self) -> dict:
        by_ch: dict[int, int] = {}
        subjects: dict[str, int] = {}
        languages: dict[str, int] = {}
        for doc in self.all_docs:
            ch = doc.metadata.get("chapter_idx", 0)
            by_ch[ch] = by_ch.get(ch, 0) + 1
            
            subj = doc.metadata.get("subject", "unknown")
            subjects[subj] = subjects.get(subj, 0) + 1
            
            lang = doc.metadata.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        
        # Ajouter stats cache
        cache_stats = embedding_cache.stats()
        
        return {
            "is_ready":        self.is_ready,
            "embeddings_ok":   self._embeddings_ok,
            "total_docs":      len(self.all_docs),
            "collection":      self.collection_name,
            "by_chapter":      by_ch,
            "subjects":        subjects,
            "languages":       languages,
            "cache_entries":   len(self.summary_cache),
            "bm25_ready":      self.bm25_retriever is not None,
            "embedding_cache": cache_stats,  # 🔄 Nouveau!
        }

    def delete_collection(self) -> None:
        if self.client and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            log.info(f"🗑️  Collection '{self.collection_name}' supprimée.")
        self.vectorstore = None
        self.bm25_retriever = None
        self.all_docs = []
        self.is_ready = False

    def reset(self) -> None:
        """
        Réinitialise complètement la base vectorielle Qdrant et le cache BM25.
        Utilisé avant une réingestion complète.
        """
        log.warning("🔄 Réinitialisation de la base RAG…")
        self.delete_collection()
        self.summary_cache.clear()
        if self.docs_cache.exists():
            self.docs_cache.unlink()
        if self.summary_cache_path.exists():
            self.summary_cache_path.unlink()
        log.info("✅ Base RAG réinitialisée (collection, BM25, caches)")
# Smart Teacher — Architecture (RAG + Agentic RAG)

> Reference document for the **RAG + Agentic RAG** subsystem.
> Every diagram below is renderable as-is in any Markdown viewer that
> supports Mermaid (GitHub, GitLab, VS Code with the Markdown
> Preview Mermaid extension, MkDocs, Obsidian…).
>
> File-and-line cross-references point at concrete code locations so a
> reviewer can trust each box on the diagrams against the source.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Q&A graph (Self-RAG)](#2-qa-graph-self-rag)
3. [Teaching graph (with reflection fallback)](#3-teaching-graph-with-reflection-fallback)
4. [RAG retrieval pipeline](#4-rag-retrieval-pipeline)
5. [RAG ingestion pipeline](#5-rag-ingestion-pipeline)
6. [TutorState lifecycle](#6-tutorstate-lifecycle)
7. [Self-RAG decision tree](#7-self-rag-decision-tree)
8. [Citation grounding flow](#8-citation-grounding-flow)
9. [Resilience layers (breaker + OTel + fallback)](#9-resilience-layers-breaker--otel--fallback)
10. [Math handling : index vs output layer](#10-math-handling--index-vs-output-layer)
11. [Personalization bandit (Phase 1)](#11-personalization-bandit-phase-1)
12. [References](#references)

---

## 1. System overview

High-level flow from a student utterance to a spoken answer.

```mermaid
flowchart LR
    Student(["🎓 Student"]) -- "speaks / types" --> Capture
    Capture["Audio capture"] --> STT["Whisper STT"]
    STT -- "transcription + prosody" --> Orchestrator["Global Orchestrator"]

    Orchestrator -. "event_type=present_section" .-> Teaching["Teaching Graph"]
    Orchestrator -. "event_type=student_speech" .-> QA["Q&A Graph"]

    Teaching --> StateOut["TutorState"]
    QA --> StateOut

    StateOut -- "answer + citations" --> Speech["_clean_for_speech<br/>+ math_speech.to_speech"]
    Speech --> TTS["Edge TTS / Ollama TTS"]
    TTS --> Audio(["🔊 Audio out"])

    StateOut -. "citations, confidence, fallback flag" .-> UI["Frontend UI"]

    classDef rag fill:#e1f5ff,stroke:#01579b,color:#01579b
    classDef agentic fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef io fill:#f3e5f5,stroke:#4a148c,color:#4a148c
    class Teaching,QA agentic
    class STT,TTS io
```

**Key sources:**
- `main.py` — bootstrap
- `handlers/audio_pipeline.py` — STT → orchestrator → response wiring
- `agentic/orchestrator.py` — Global Orchestrator dispatching to subgraphs
- `agentic/state.py` — Shared TutorState

---

## 2. Q&A graph (Self-RAG)

`agentic/qa/graph.py:build_qa_graph` — the LangGraph for student questions.
Implements **Self-RAG** (Asai et al. 2023, arXiv:2310.11511) via the
`retrieval_decision` node.

```mermaid
flowchart TD
    START(["START"]) --> Intent["<b>IntentAgent</b><br/>SIGHT classifier + LLM<br/>outputs VoiceIntent<br/>+ needs_retrieval flag<br/>+ feedback_polarity"]

    Intent -- "intent.type" --> IntentRouter{"intent_router"}

    IntentRouter -- "navigation / feedback / confusion_signal" --> Responder
    IntentRouter -- "question" --> RetrievalDecision["retrieval_decision<br/>pass-through node"]

    RetrievalDecision -- "intent.needs_retrieval" --> RetrievalRouter{"retrieval_router<br/>Self-RAG"}
    RetrievalRouter -- "skip — meta / arithmetic / chitchat" --> Responder
    RetrievalRouter -- "retrieve" --> Rewriter

    Rewriter["<b>QueryRewriter</b><br/>step-back prompting<br/>resolves pronouns + ellipses<br/>outputs rewritten_query<br/>+ anchored_concept"]

    Rewriter --> Retriever["<b>Retriever</b><br/>BM25 + Dense + RRF<br/>+ cross-encoder rerank<br/>+ KG augmentation<br/>outputs retrieved_chunks"]

    Retriever --> Responder["<b>Responder</b><br/>JSON output<br/>answer + supporting_chunks<br/>derives citations + confidence"]

    Responder --> END(["END"])

    classDef llm fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef router fill:#fce4ec,stroke:#880e4f,color:#880e4f
    classDef rag fill:#e1f5ff,stroke:#01579b,color:#01579b
    class Intent,Rewriter,Responder llm
    class IntentRouter,RetrievalRouter router
    class Retriever rag
```

**Key sources:**
- `agentic/qa/graph.py` — graph builder
- `agentic/qa/intent.py` — SIGHT + LLM intent classifier
- `agentic/qa/rewriter.py` — step-back rewriter (Zheng et al. 2023, arXiv:2310.06117)
- `agentic/qa/retriever.py` — RAG + KG augmentation
- `agentic/qa/responder.py` — grounded JSON output

**Self-RAG semantics:**

| Path | Trigger | Latency cost |
|---|---|---|
| `intent_router → short` | navigation / feedback / confusion | 1 LLM call (Responder) |
| `retrieval_router → skip` | needs_retrieval=False (meta question) | 1 LLM call (Responder) |
| `retrieval_router → retrieve` | needs_retrieval=True (factual question) | 3 LLM calls (Rewriter + Responder, retriever is sub-LLM) |

---

## 3. Teaching graph (with reflection fallback)

`agentic/teaching/graph.py:build_teaching_graph` — the LangGraph for
proactive course narration. Implements **self-reflection fallback**
(Asai et al. 2023): when the reviewer rejects the narration after the
retry budget, control transfers to a `FallbackNarrator` that builds a
grounded-by-construction reply rather than silently shipping the
hallucinated draft.

```mermaid
flowchart TD
    START(["START"]) --> Planner

    Planner["<b>Planner</b><br/>LLM<br/>decomposes slide → 3-5 ideas"]
    Planner --> Context["<b>Context</b><br/>RAG retrieval per idea"]
    Context --> Adaptation["<b>Adaptation</b><br/>profile lookup<br/>adjusts depth"]
    Adaptation --> Narrator

    Narrator["<b>Narrator</b><br/>LLM<br/>generates spoken narration"]
    Narrator --> Review

    Review["<b>Reviewer</b><br/>LLM binary verdict<br/>grounded in slide + RAG ?"]
    Review --> ReviewRouter{"review_router"}

    ReviewRouter -- "grounded" --> END(["END"])
    ReviewRouter -- "ungrounded<br/>retries less than MAX_RETRIES" --> Narrator
    ReviewRouter -- "ungrounded<br/>retries ≥ MAX_RETRIES" --> Fallback

    Fallback["<b>FallbackNarrator</b><br/>LLM with slide-only context<br/>3-tier cascade:<br/>llm_paraphrase → verbatim → no_source"]
    Fallback --> END

    classDef llm fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef rag fill:#e1f5ff,stroke:#01579b,color:#01579b
    classDef router fill:#fce4ec,stroke:#880e4f,color:#880e4f
    classDef reflect fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    class Planner,Narrator,Review llm
    class Context rag
    class ReviewRouter router
    class Fallback reflect
    class Adaptation router
```

**Fallback cascade** in `agentic/teaching/fallback.py:FallbackNarratorAgent` :

```mermaid
flowchart TD
    Trigger["Reviewer rejects after MAX_RETRIES"] --> SlideCheck{"Slide<br/>content ?"}
    SlideCheck -- "empty" --> NoSource["<b>Tier 3 : no_source</b><br/>Je n'ai pas assez d'éléments<br/>honest I don't know"]
    SlideCheck -- "has content" --> LLMTry["<b>Tier 1 : LLM paraphrase</b><br/>strict prompt, slide-only context<br/>JSON output answer"]

    LLMTry -- "exception<br/>or invalid JSON" --> Verbatim["<b>Tier 2 : verbatim</b><br/>Voici ce que dit la slide :<br/>+ first 1500 chars verbatim"]
    LLMTry -- "valid answer" --> Done["Set state.answer<br/>+ Action.payload.fallback=True<br/>+ Action.payload.kind=llm_paraphrase<br/>+ Review.grounded=True<br/>by construction"]

    Verbatim --> Done
    NoSource --> Done

    classDef tier1 fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef tier2 fill:#fff8e1,stroke:#f57f17,color:#f57f17
    classDef tier3 fill:#ffebee,stroke:#b71c1c,color:#b71c1c
    class LLMTry tier1
    class Verbatim tier2
    class NoSource tier3
```

---

## 4. RAG retrieval pipeline

`rag/multimodal_rag.py:MultiModalRAG.retrieve_chunks` — the hybrid
retrieval engine. RRF fusion (Cormack, Clarke & Buettcher 2009) +
cross-encoder rerank + ROUGE-style near-duplicate dedup (Lin 2004).

```mermaid
flowchart TD
    Query["query"] --> Expand{"anchored_concept<br/>already in query ?"}
    Expand -- "no" --> AppendConcept["expanded_query =<br/>query + anchored_concept"]
    Expand -- "yes" --> SkipExpand["expanded_query = query"]
    AppendConcept --> Parallel["Parallel retrieval"]
    SkipExpand --> Parallel

    Parallel --> BM25["<b>BM25</b><br/>sparse lexical<br/>top-K×3"]
    Parallel --> Vector["<b>Dense vectors</b><br/>Qdrant<br/>cosine similarity<br/>top-K×3"]

    BM25 --> RRF["<b>RRF Fusion</b><br/>Cormack et al. 2009<br/>k=60<br/>score = Σ 1/k+rank_i"]
    Vector --> RRF

    RRF --> SizeFilter["Filter chunks<br/>length ≥ min_chunk_length"]
    SizeFilter --> Rerank["<b>Cross-encoder rerank</b><br/>bge-reranker<br/>sigmoid → prob in 0..1"]

    Rerank -- success --> RerankOK["score = sigmoid logit"]
    Rerank -- fail/disabled --> RerankFallback["score = 1 minus rank over n<br/>RRF position fallback"]

    RerankOK --> Dedup
    RerankFallback --> Dedup

    Dedup["<b>Near-duplicate dedup</b><br/>SequenceMatcher ratio ≥ 0.90<br/>OR Jaccard tokens ≥ 0.88<br/>Lin 2004 ROUGE"]
    Dedup --> KGAug["<b>KG augmentation</b><br/>fetch direct prerequisites<br/>of top-1 chunk idea_id<br/>cap: 3 prereqs"]
    KGAug --> Out["retrieved_chunks<br/>= direct + kg_augmented"]

    classDef sparse fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef dense fill:#e1f5ff,stroke:#01579b,color:#01579b
    classDef fusion fill:#fce4ec,stroke:#880e4f,color:#880e4f
    classDef kg fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    class BM25 sparse
    class Vector dense
    class RRF,Rerank,Dedup fusion
    class KGAug kg
```

**Citations cited in code comments:**
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR'09.
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL.
- Doignon, J.-P., & Falmagne, J.-C. (1985). *Spaces for the assessment of knowledge.* IJMMS — for KG augmentation.

---

## 5. RAG ingestion pipeline

`pedagogy/intelligent_ingester.py:IntelligentIngester` — drives the
parsing/chunking/indexing flow.

```mermaid
flowchart TD
    PDF["📄 PDF / PPTX / DOCX"] --> Partition["<b>unstructured.partition.auto</b><br/>extracts elements:<br/>Text, Title, Table, FigureCaption, Image"]

    Partition --> Tables["Tables → Markdown<br/>preserve structure"]
    Partition --> Captions["FigureCaption →<br/>prefix Légende figure ..."]
    Partition --> Images["Images"]
    Partition --> TextEls["Text elements"]

    Images --> Vision["<b>vision_describe.describe_slide_image</b><br/>OpenAI gpt-4o-mini → Ollama LLaVA<br/>disk cache + single-flight"]
    Vision --> VisualText["Visual description<br/>2-4 sentences"]

    Tables --> Concat
    Captions --> Concat
    VisualText --> Concat
    TextEls --> Concat["Concat preserving<br/>math symbols verbatim<br/>NO wordification at index time"]

    Concat --> Chunker["<b>chunk_by_title</b><br/>max 1500 chars<br/>new after 1200 chars"]
    Chunker --> ChunkMeta["Chunks with metadata:<br/>idea_id, idea_label,<br/>chapter_idx, section_idx,<br/>course_id, depends_on_ids"]

    ChunkMeta --> Embed["<b>Embeddings</b><br/>BAAI/bge-m3 local<br/>or text-embedding-3-small<br/>+ embedding_cache"]
    Embed --> Qdrant[("Qdrant<br/>vector store<br/>collection: smart_teacher_multimodal")]
    ChunkMeta --> BM25Build["BM25Retriever build<br/>over all_docs"]

    ChunkMeta --> KGBuild["<b>IdeaGraph build</b><br/>nodes from idea_id metadata<br/>edges from depends_on_ids"]

    classDef parser fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef vision fill:#e1f5ff,stroke:#01579b,color:#01579b
    classDef indexer fill:#fce4ec,stroke:#880e4f,color:#880e4f
    classDef store fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    class Partition,Chunker parser
    class Vision vision
    class Embed,BM25Build,KGBuild indexer
    class Qdrant store
```

**Why math symbols are preserved at index time** (vs the previous
`_wordify_math` approach): a student searching for `"x²"` finds chunks
containing `"x²"`. Wordifying the symbols at index time destroyed
retrieval precision; verbalization is now done at TTS output time only
(see [section 10](#10-math-handling--index-vs-output-layer)).

---

## 6. TutorState lifecycle

`agentic/state.py:TutorState` — the TypedDict shared between the Q&A
and Teaching graphs. The diagram below shows which agent **writes**
which field, and which agent **reads** it.

```mermaid
flowchart LR
    EvtPayload["event_payload"]
    Lang["language"]
    SessionId["session_id"]
    StudentId["student_id"]
    LastSlide["last_slide_content"]
    History["history"]

    Intent_W["<b>IntentAgent</b><br/>writes:"]
    EvtPayload --> Intent_W
    Lang --> Intent_W
    Intent_W --> IntentField["intent : VoiceIntent<br/>+ needs_retrieval<br/>+ feedback_polarity"]

    Rewriter_W["<b>QueryRewriter</b><br/>reads intent + history + slide<br/>writes:"]
    IntentField --> Rewriter_W
    History --> Rewriter_W
    LastSlide --> Rewriter_W
    Rewriter_W --> Rewritten["rewritten_query"]
    Rewriter_W --> Anchored["anchored_concept"]

    Retriever_W["<b>Retriever</b><br/>reads rewritten + anchored<br/>writes:"]
    Rewritten --> Retriever_W
    Anchored --> Retriever_W
    Retriever_W --> RetrievedField["retrieved_chunks<br/>each tagged via_kg if augmented"]

    Responder_W["<b>Responder</b><br/>reads chunks + slide + history<br/>writes:"]
    RetrievedField --> Responder_W
    LastSlide --> Responder_W
    History --> Responder_W
    Responder_W --> AnswerField["answer"]
    Responder_W --> CitationsField["citations"]
    Responder_W --> ConfField["confidence"]
    Responder_W --> ActionsField["actions"]

    classDef agent fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef field fill:#e1f5ff,stroke:#01579b,color:#01579b
    classDef input fill:#f3e5f5,stroke:#4a148c,color:#4a148c
    class Intent_W,Rewriter_W,Retriever_W,Responder_W agent
    class IntentField,Rewritten,Anchored,RetrievedField,AnswerField,CitationsField,ConfField,ActionsField field
    class EvtPayload,Lang,SessionId,StudentId,LastSlide,History input
```

---

## 7. Self-RAG decision tree

How a single utterance is routed through the Q&A graph based on
intent type and the LLM-decided `needs_retrieval` flag.

```mermaid
flowchart TD
    Start["Student utterance"] --> Intent

    Intent["IntentAgent<br/>SIGHT first, LLM fallback"]
    Intent --> Confused{"SIGHT detects<br/>confusion ?"}
    Confused -- "yes" --> ConfusionPath["type=confusion_signal<br/>needs_retrieval=False"]
    Confused -- "no" --> LLMClassify["LLM classifies<br/>intent type + Self-RAG flag"]

    LLMClassify --> TypeCheck{"intent.type ?"}
    TypeCheck -- "navigation" --> NavPath["Static reply<br/>D'accord, je reviens sur ce point"]
    TypeCheck -- "feedback" --> FeedbackCheck{"feedback_polarity ?"}
    FeedbackCheck -- "positive" --> ContinuePath["Static reply<br/>Parfait, je continue"]
    FeedbackCheck -- "negative" --> RepeatPath["Static reply<br/>D'accord, je vais réexpliquer"]
    TypeCheck -- "question" --> SelfRAG{"needs_retrieval ?"}

    SelfRAG -- "false<br/>meta / arithmetic / chitchat" --> SkipRetrieval["Responder direct<br/>uses history + slide only"]
    SelfRAG -- "true<br/>factual question" --> FullPath["Full path:<br/>Rewriter → Retriever → Responder"]

    ConfusionPath --> ReformulationPath["Responder<br/>reformulation prompt<br/>compose_reformulation_prompt"]

    NavPath --> Out["state.answer"]
    ContinuePath --> Out
    RepeatPath --> Out
    ReformulationPath --> Out
    SkipRetrieval --> Out
    FullPath --> Out

    classDef sight fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef static fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    classDef rag fill:#e1f5ff,stroke:#01579b,color:#01579b
    class Intent,LLMClassify sight
    class NavPath,ContinuePath,RepeatPath static
    class FullPath,SkipRetrieval rag
```

---

## 8. Citation grounding flow

How `state.citations` is populated by the Responder. Implements
hallucination detection: if the LLM didn't cite any chunk, the answer
is flagged as ungrounded (confidence=0.0).

```mermaid
sequenceDiagram
    participant R as Retriever
    participant Resp as Responder
    participant LLM as Brain.ask
    participant Parse as _parse_qa_response
    participant State as TutorState

    R->>Resp: retrieved_chunks (5 items)
    Resp->>Resp: _format_chunks_with_ids<br/>injects [id:concept_knn] tags<br/>builds id_to_chunk mapping
    Resp->>LLM: prompt with chunks_block<br/>+ JSON output instruction
    LLM-->>Resp: '{"answer": "...", "supporting_chunks": ["concept_knn", "concept_distance"]}'

    Resp->>Parse: raw_text + id_to_chunk
    Parse->>Parse: extract JSON<br/>filter hallucinated IDs<br/>dedup
    Parse-->>Resp: (answer, citations)

    alt citations non-empty
        Resp->>State: citations = [{chunk_id, idea_label, source, score}]<br/>confidence = avg(citation.score)
    else citations empty (ungrounded)
        Resp->>State: citations = []<br/>confidence = 0.0<br/>Action.payload.grounded=False
    end

    Note over State: Frontend can read confidence + citations<br/>to render trust badges
```

---

## 9. Resilience layers (breaker + OTel + fallback)

`agentic/resilience/wrap.py:build_resilient_node` — wraps every graph
node uniformly. Three concentric layers : circuit breaker → OpenTelemetry
span → exception-to-fallback.

```mermaid
flowchart TD
    Call["Graph node call:<br/>resilient_planner state"] --> Breaker{"Breaker.allow ?"}

    Breaker -- "open<br/>too many recent failures" --> ShortCircuit["fallback_for node<br/>deterministic state update"]
    Breaker -- "closed" --> Span["<b>OTel span</b><br/>agentic.&lt;node_name&gt;"]

    Span --> Invoke["await node state"]
    Invoke -- "success" --> RecordOK["breaker.record_success<br/>attach_node_attrs span<br/>emit_event ok"]
    RecordOK --> Result["return state update dict"]

    Invoke -- "exception" --> RecordFail["breaker.record_failure<br/>span.set_attribute exception<br/>emit_event error"]
    RecordFail --> ExcFallback["fallback_for node<br/>same fallback as breaker-open path"]

    ShortCircuit --> Result
    ExcFallback --> Result

    classDef breaker fill:#ffebee,stroke:#b71c1c,color:#b71c1c
    classDef otel fill:#e8eaf6,stroke:#1a237e,color:#1a237e
    classDef fallback fill:#fff8e1,stroke:#f57f17,color:#f57f17
    class Breaker breaker
    class Span,RecordOK,RecordFail otel
    class ShortCircuit,ExcFallback fallback
```

**OTel attributes attached per node** (in `agentic/observability.py:attach_node_attrs`):

| Node | Attributes |
|---|---|
| `intent` | type, confidence, needs_retrieval, feedback_polarity, source |
| `rewriter` | length, anchored_concept |
| `retriever` | chunks_count, kg_augmented_count |
| `responder` | answer_length, citations_count, grounded, confidence, fallback flag |
| `review` | grounded, score |
| `fallback` | kind (llm_paraphrase / verbatim / no_source) |
| `planner` | ideas_count |
| `narrator` | answer_length, retries |
| Generic | output_keys |

---

## 10. Math handling : index vs output layer

Architectural shift from the previous `_wordify_math` design (math
words injected at INDEX time) to the current `audio/math_speech.py`
design (verbalization at OUTPUT time only).

```mermaid
flowchart LR
    subgraph Old ["❌ Previous - lossy"]
        OldIngest["PDF ingestion"] --> OldWordify["_wordify_math<br/>x² → squared<br/>α → alpha<br/>39 hardcoded symbols<br/>English-only"]
        OldWordify --> OldIndex["Index :<br/>x squared<br/>+ alpha"]
        OldIndex --> OldQuery{"Student searches x²"}
        OldQuery -- "no match" --> OldEmpty["❌ 0 results<br/>retrieval broken"]
    end

    subgraph New ["✅ Current - lossless"]
        NewIngest["PDF ingestion"] --> NewIndex["Index :<br/>x²<br/>+ α<br/>SYMBOLS PRESERVED"]
        NewIndex --> NewQuery{"Student searches x²"}
        NewQuery -- "match" --> NewHit["✅ Chunks containing x²"]

        NewHit --> NewLLM["LLM generates<br/>response with math"]
        NewLLM --> NewSpeech["<b>math_speech.to_speech</b><br/>at OUTPUT time<br/>3-tier:<br/>1. sympy LaTeX parser<br/>2. regex commands<br/>3. unicode lexicon<br/>BILINGUAL FR + EN"]
        NewSpeech --> NewTTS["TTS:<br/>x au carré or x squared<br/>by language"]
    end

    classDef bad fill:#ffebee,stroke:#b71c1c,color:#b71c1c
    classDef good fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    class OldIngest,OldWordify,OldIndex,OldQuery,OldEmpty bad
    class NewIngest,NewIndex,NewQuery,NewHit,NewLLM,NewSpeech,NewTTS good
```

---

## 11. Personalization bandit (Phase 1)

`pedagogy/personalization/bandit/` — contextual Thompson sampling
bandit that selects a pedagogical strategy + speech rate per turn,
based on the student's profile bucket.

```mermaid
flowchart TD
    Q["Student question"] --> Resp["Responder"]
    Resp --> Mastery["Fetch current mastery<br/>MasteryRepo.get_scores_bulk"]
    Mastery --> EndPrev["BanditController.end_turn<br/>resolve previous decision<br/>compute_reward → update bandit"]
    EndPrev --> Profile["Fetch profile<br/>learning_style + avg_response_time"]
    Profile --> Bucket["context_from_profile<br/>style x pace x mastery_level<br/>discretized bucket"]
    Bucket --> Sample["bandit.select<br/>Thompson sample<br/>argmax over 18 arms"]
    Sample --> Inject["Inject strategy_prompt<br/>into personalization_prefix"]
    Inject --> LLM["LLM generates answer"]
    LLM --> Stash["record_pending<br/>Redis bandit:pending:session_id<br/>TTL 1h"]
    Stash --> ActPayload["actions[0].payload :<br/>bandit_strategy<br/>bandit_speech_rate<br/>bandit_context"]
    ActPayload --> TTS["TTS layer reads bandit_speech_rate<br/>tts_adapter.bandit_rate_multiplier"]

    classDef bandit fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    classDef store fill:#fff3e0,stroke:#e65100,color:#e65100
    classDef llm fill:#e1f5ff,stroke:#01579b,color:#01579b
    class Sample,EndPrev,Bucket bandit
    class Stash store
    class LLM,Inject llm
```

**Action space** : 18 arms = 6 strategies × 3 speech rates.

| Strategy | Description |
|---|---|
| `analogy` | Map concept onto a familiar domain |
| `example` | Worked-out concrete case |
| `decomposition` | Split into sub-concepts (uses IdeaGraph prereqs) |
| `socratic` | Guiding questions before the answer |
| `recap` | Summarize previous concepts before new one |
| `simpler_words` | Plain-language rephrasing |

**Context buckets** : 5 styles × 3 paces × 3 mastery levels = 45 buckets.

**Phases (M2 thesis roadmap)** :

```mermaid
flowchart LR
    P1["<b>Phase 1</b><br/>Online bandit<br/>+ live logging"] --> P2["<b>Phase 2</b><br/>Synthetic simulator<br/>bootstrap_bandit"]
    P2 --> P3["<b>Phase 3</b><br/>Offline training<br/>IPS evaluation"]
    P3 --> P4["<b>Phase 4</b><br/>DRL in production<br/>future work"]

    classDef done fill:#e8f5e9,stroke:#1b5e20,color:#1b5e20
    classDef partial fill:#fff8e1,stroke:#f57f17,color:#f57f17
    classDef future fill:#fce4ec,stroke:#880e4f,color:#880e4f
    class P1,P2 done
    class P3 partial
    class P4 future
```

**Reward function** :

```
reward = 0.50 × (1 - confusion_signal)        # Bloom 1968
       + 0.40 × clip(mastery_delta, 0, 0.30)  # Sweller 1985 cognitive load
       + 0.10 × engagement_signal              # tie-breaker
```

**Key files** :
- `bandit/strategies.py` — action space taxonomy
- `bandit/thompson.py` — Beta posterior + Thompson sampling
- `bandit/reward.py` — reward function
- `bandit/repo.py` — Redis persistence + pending decisions (TTL 1h)
- `bandit/controller.py` — start_turn / end_turn lifecycle
- `bandit/simulator.py` — Phase 2 synthetic episodes
- `bandit/offline.py` — Phase 3 offline training + IPS evaluation

**Observability** : `GET /admin/bandit/stats` returns per-bucket arm
distributions, posterior means, and global coverage.

---

## References

Academic citations made by the codebase, each tied to a specific design decision.

| Reference | Used in | Section |
|---|---|---|
| Asai et al. 2023, *Self-RAG* arXiv:2310.11511 | Self-RAG router + reflection fallback | [§2](#2-qa-graph-self-rag), [§3](#3-teaching-graph-with-reflection-fallback), [§7](#7-self-rag-decision-tree) |
| Zheng et al. 2023, *Step-Back Prompting* arXiv:2310.06117 | Rewriter step-back reasoning | [§2](#2-qa-graph-self-rag), [§6](#6-tutorstate-lifecycle) |
| Cormack, Clarke & Buettcher 2009, *RRF outperforms Condorcet…* SIGIR'09 | RRF fusion in retrieval | [§4](#4-rag-retrieval-pipeline) |
| Lin 2004, *ROUGE: A Package…* ACL | Near-duplicate dedup | [§4](#4-rag-retrieval-pipeline) |
| Doignon & Falmagne 1985, *Spaces for the assessment of knowledge* | IdeaGraph + KG-augmented retrieval | [§4](#4-rag-retrieval-pipeline), [§5](#5-rag-ingestion-pipeline) |
| Manning, Raghavan, Schütze 2008, *Introduction to IR* | Recall@K, P@K, F1, MAP | `rag/evaluation/metrics.py` |
| Voorhees 1999, *TREC-8 QA Track Report* | Mean Reciprocal Rank | `rag/evaluation/metrics.py` |
| Järvelin & Kekäläinen 2002, *Cumulated gain-based evaluation…* | nDCG@K | `rag/evaluation/metrics.py` |
| W3C MathML / SSML | math_speech verbalization rules | [§10](#10-math-handling--index-vs-output-layer) |
| Soiffer 2005, *Reading Mathematics* | Linguistics of spoken math | `audio/math_speech.py` |
| GPT-4V (OpenAI 2023) + LLaVA (Liu et al. NeurIPS 2023) | Vision describe providers | [§5](#5-rag-ingestion-pipeline) |
| Nygard 2007, *Release It!* | Circuit breaker pattern | [§9](#9-resilience-layers-breaker--otel--fallback) |
| W3C OpenTelemetry semantic conventions | OTel attribute namespacing | [§9](#9-resilience-layers-breaker--otel--fallback) |
| Russo et al. 2018, *A Tutorial on Thompson Sampling* | Contextual bandit selection | [§11](#11-personalization-bandit-phase-1) |
| Agrawal & Goyal 2013, *Further Optimal Regret Bounds for TS* | Beta-Bernoulli conjugate updates | [§11](#11-personalization-bandit-phase-1) |
| Li et al. 2010, *A contextual-bandit approach to personalized news* | Disjoint contextual bandit pattern | [§11](#11-personalization-bandit-phase-1) |
| Bloom 1968, *Learning for Mastery* | Reward weighting (confusion dominance) | [§11](#11-personalization-bandit-phase-1) |
| Sweller 1985, *Cognitive Load Theory* | Reward weighting (mastery delta) | [§11](#11-personalization-bandit-phase-1) |
| Horvitz & Thompson 1952, *Sampling Without Replacement* | IPS off-policy evaluation | `bandit/offline.py` |
| Dudík, Langford & Li 2011, *Doubly Robust Policy Evaluation* | Phase 3 offline RL methodology | `bandit/offline.py` |
| Swaminathan & Joachims 2015, *Counterfactual Risk Minimization* | Clipped IPS variance reduction | `bandit/offline.py` |

---

## Tooling notes

**To render these diagrams**:

- **GitHub** : automatic, just open this file in the repo.
- **VS Code** : install *Markdown Preview Mermaid Support* extension.
- **Local** : `npm install -g @mermaid-js/mermaid-cli` then
  `mmdc -i ARCHITECTURE.md -o ARCHITECTURE.pdf` to export to PDF for the thesis appendix.
- **Obsidian / MkDocs / GitLab** : built-in Mermaid support.

**To regenerate** : the diagrams are hand-written Mermaid; no auto-extraction is performed. When you change the graph topology, update the relevant section in this file.

"""
╔══════════════════════════════════════════════════════════════════════╗
║        SMART TEACHER — Dashboard Professeur                        ║
║                                                                      ║
║  Routes FastAPI pour le tableau de bord du professeur :             ║
║    GET  /dashboard              — page HTML du dashboard            ║
║    GET  /dashboard/stats        — statistiques globales             ║
║    GET  /dashboard/sessions     — sessions actives                  ║
║    GET  /dashboard/analytics    — métriques d'apprentissage         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import logging
import time

log = logging.getLogger("SmartTeacher.Dashboard")
router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ── Stats en mémoire (complétées par PostgreSQL si dispo) ─────────────
_SESSIONS_CACHE: list[dict] = []
_CHECKPOINTS_CACHE: list[dict] = []
_TRACE_CACHE: list[dict] = []

def record_session_event(event: dict):
    """Appelé depuis main.py à chaque interaction."""
    _SESSIONS_CACHE.append({**event, "ts": time.time()})
    if len(_SESSIONS_CACHE) > 1000:
        _SESSIONS_CACHE.pop(0)


def record_checkpoint_event(event: dict):
    """Enregistre un point de pause/reprise pour le dashboard."""
    _CHECKPOINTS_CACHE.append({**event, "ts": time.time()})
    if len(_CHECKPOINTS_CACHE) > 1000:
        _CHECKPOINTS_CACHE.pop(0)


def record_trace_event(event: dict):
  """Enregistre une étape backend courte pour la vue opérationnelle."""
  _TRACE_CACHE.append({**event, "ts": time.time()})
  if len(_TRACE_CACHE) > 1000:
    _TRACE_CACHE.pop(0)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_pointer_short(event: dict) -> str:
    chapter_index = _safe_int(event.get("chapter_index"))
    section_index = _safe_int(event.get("section_index"))
    char_position = event.get("char_position")

    parts = []
    if chapter_index is not None:
        parts.append(f"C{chapter_index + 1}")
    if section_index is not None:
        parts.append(f"S{section_index + 1}")
    if char_position is not None:
        parts.append(f"@{char_position}")

    return " / ".join(parts) if parts else "—"


def _format_pointer_detail(event: dict) -> str:
    point_text = str(event.get("point_text") or event.get("pointer_text") or "").strip()
    if point_text:
        return point_text

    location_label = str(event.get("location_label") or "").strip()
    cursor_label = str(event.get("cursor_label") or "").strip()
    if location_label and cursor_label:
        return f"{location_label}, position {cursor_label}"
    if location_label:
        return location_label
    if cursor_label:
        return cursor_label
    return _format_pointer_short(event)


@router.get("/stats")
async def get_stats():
    """Statistiques globales pour le dashboard."""
    total    = len(_SESSIONS_CACHE)
    kpi_ok   = sum(1 for e in _SESSIONS_CACHE if e.get("meets_kpi"))
    avg_time = (sum(e.get("total_time", 0) for e in _SESSIONS_CACHE) / total) if total else 0

    # Par langue
    langs: dict[str, int] = {}
    for e in _SESSIONS_CACHE:
        l = e.get("language", "unknown")
        langs[l] = langs.get(l, 0) + 1

    # Par matière
    subjects: dict[str, int] = {}
    for e in _SESSIONS_CACHE:
        s = e.get("subject") or "unknown"
        subjects[s] = subjects.get(s, 0) + 1

    # Tentatives PostgreSQL
    try:
        from database.init_db import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as db:
            r = await db.execute(text("SELECT COUNT(*) FROM interactions"))
            total_db = r.scalar() or 0
            r2 = await db.execute(text("SELECT COUNT(*) FROM learning_sessions"))
            sessions_db = r2.scalar() or 0
    except Exception:
        total_db = total
        sessions_db = len(set(e.get("session_id","") for e in _SESSIONS_CACHE))

    return {
        "total_interactions": total_db,
        "total_sessions":     sessions_db,
        "kpi_rate_pct":       round(kpi_ok / total * 100, 1) if total else 0,
        "avg_response_time":  round(avg_time, 2),
        "by_language":        langs,
        "by_subject":         subjects,
        "last_24h":           sum(1 for e in _SESSIONS_CACHE if time.time() - e.get("ts",0) < 86400),
    }


@router.get("/sessions")
async def get_active_sessions():
    """Sessions récentes (dernières 24h)."""
    cutoff = time.time() - 86400
    recent_questions = [e for e in _SESSIONS_CACHE if e.get("ts", 0) > cutoff]
    recent_checkpoints = [e for e in _CHECKPOINTS_CACHE if e.get("ts", 0) > cutoff]

    by_session: dict[str, list] = {}
    for e in recent_questions:
        sid = str(e.get("session_id", "unknown"))
        by_session.setdefault(sid, []).append(e)

    checkpoints_by_session: dict[str, list] = {}
    for e in recent_checkpoints:
        sid = str(e.get("session_id", "unknown"))
        checkpoints_by_session.setdefault(sid, []).append(e)

    sessions = []
    for sid in sorted(set(by_session) | set(checkpoints_by_session)):
        question_events = by_session.get(sid, [])
        checkpoint_events = checkpoints_by_session.get(sid, [])
        latest_question = question_events[-1] if question_events else None
        latest_checkpoint = checkpoint_events[-1] if checkpoint_events else None
        combined_events = question_events + checkpoint_events
        latest_event = max(combined_events, key=lambda x: x.get("ts", 0)) if combined_events else None
        pointer_source = latest_event or latest_checkpoint or latest_question or {}

        checkpoint_type = (latest_checkpoint or {}).get("checkpoint_type")
        is_paused = checkpoint_type == "pause"
        sessions.append({
            "session_id": sid[:8],
            "language": (pointer_source or {}).get("language", "?"),
            "turns": len(question_events),
            "avg_time": round(sum(e.get("total_time", 0) for e in question_events) / len(question_events), 2) if question_events else 0,
            "kpi_rate": round(sum(1 for e in question_events if e.get("meets_kpi")) / len(question_events) * 100, 1) if question_events else 0,
            "pointer": _format_pointer_short(pointer_source),
            "pointer_detail": _format_pointer_detail(pointer_source),
          "checkpoint_state": "pause" if is_paused else "active",
          "checkpoint_label": "Pause" if is_paused else "Actif",
          "paused": is_paused,
            "last_activity": (latest_event or pointer_source).get("ts", 0),
        })

    return {"sessions": sorted(sessions, key=lambda x: -x["last_activity"])}


@router.get("/analytics")
async def get_analytics():
    """Métriques d'apprentissage détaillées."""
    if not _SESSIONS_CACHE:
        return {"message": "Pas encore de données"}

    times = [e["total_time"] for e in _SESSIONS_CACHE if e.get("total_time")]
    stt_t = [e["stt_time"]   for e in _SESSIONS_CACHE if e.get("stt_time")]
    llm_t = [e["llm_time"]   for e in _SESSIONS_CACHE if e.get("llm_time")]
    tts_t = [e["tts_time"]   for e in _SESSIONS_CACHE if e.get("tts_time")]

    def stats(lst):
        if not lst: return {}
        return {"min": round(min(lst),2), "max": round(max(lst),2),
                "avg": round(sum(lst)/len(lst),2)}

    return {
        "total_time": stats(times),
        "stt_time":   stats(stt_t),
        "llm_time":   stats(llm_t),
        "tts_time":   stats(tts_t),
        "kpi_threshold_s": 5.0,
        "kpi_rate_pct": round(sum(1 for e in _SESSIONS_CACHE if e.get("meets_kpi")) / len(_SESSIONS_CACHE) * 100, 1),
    }

@router.get("/checkpoints")
async def get_recent_checkpoints():
    """Retourne les derniers points de pause et de reprise."""
    cutoff = time.time() - 86400
    recent = [e for e in _CHECKPOINTS_CACHE if e.get("ts", 0) > cutoff]

    checkpoints = []
    for e in reversed(recent):
        checkpoint_type = str(e.get("checkpoint_type") or "checkpoint")
        checkpoints.append({
            "session_id": str(e.get("session_id", "unknown"))[:8],
            "checkpoint_type": checkpoint_type,
            "checkpoint_label": "Pause" if checkpoint_type == "pause" else "Reprise" if checkpoint_type == "resume" else checkpoint_type,
            "pointer": _format_pointer_short(e),
            "pointer_detail": _format_pointer_detail(e),
            "reason": e.get("reason") or "",
            "language": e.get("language") or "?",
            "subject": e.get("subject") or "unknown",
            "slide_title": e.get("slide_title") or "",
            "ts": e.get("ts", 0),
        })
        if len(checkpoints) >= 20:
            break

    return {"checkpoints": checkpoints}


@router.get("/trace")
async def get_recent_trace():
  """Retourne les dernières étapes backend visibles brièvement."""
  cutoff = time.time() - 120
  recent = [e for e in _TRACE_CACHE if e.get("ts", 0) > cutoff]

  traces = []
  for e in reversed(recent):
    ts = e.get("ts", 0)
    traces.append({
      "session_id": str(e.get("session_id", "unknown"))[:8],
      "turn_id": e.get("turn_id"),
      "state": e.get("state") or "",
      "state_name": e.get("state_name") or e.get("state") or "",
      "substep": e.get("substep") or "",
      "display_message": e.get("display_message") or "",
      "details": e.get("details") or {},
      "metrics": e.get("metrics") or {},
      "emoji": e.get("emoji") or "",
      "age_sec": round(max(time.time() - ts, 0), 1),
      "ts": ts,
    })
    if len(traces) >= 30:
      break

  return {"traces": traces}


@router.get("/questions")
async def get_recent_questions():
    """Retourne les questions récentes avec leurs métriques par tour."""
    cutoff = time.time() - 86400
    recent = [e for e in _SESSIONS_CACHE if e.get("ts", 0) > cutoff]

    questions = []
    for e in reversed(recent):
        question_text = (e.get("question") or e.get("input_text") or e.get("question_text") or "").strip()
        if not question_text:
            continue
        questions.append({
            "session_id": str(e.get("session_id", "unknown"))[:8],
            "turn_id": e.get("turn_id"),
            "language": e.get("language", "?"),
            "subject": e.get("subject") or "unknown",
            "question": question_text,
            "stt_text": (e.get("stt_text") or question_text).strip(),
            "answer": (e.get("answer") or e.get("output_text") or "").strip(),
            "slide_title": e.get("slide_title") or "",
            "chapter_title": e.get("chapter_title") or "",
            "section_title": e.get("section_title") or "",
            "tts_engine": e.get("tts_engine") or "",
            "tts_voice": e.get("tts_voice") or "",
          "chapter_index": e.get("chapter_index"),
          "section_index": e.get("section_index"),
          "char_position": e.get("char_position"),
          "pointer": _format_pointer_short(e),
          "pointer_detail": _format_pointer_detail(e),
            "llm_time": round(float(e.get("llm_time") or 0.0), 2),
          "stt_time": round(float(e.get("stt_time") or 0.0), 2),
            "tts_time": round(float(e.get("tts_time") or 0.0), 2),
            "total_time": round(float(e.get("total_time") or 0.0), 2),
            "kpi_ok": bool(e.get("meets_kpi")),
          "confusion": bool(e.get("confusion")),
          "confusion_reason": e.get("confusion_reason") or "",
          "source": e.get("source") or "",
            "ts": e.get("ts", 0),
        })
        if len(questions) >= 20:
            break

    return {"questions": questions}


@router.post("/slide_ideas")
async def slide_ideas(payload: dict):
    """Generates up to 5 short presentation ideas per slide.

    Payload example:
      {"slides": [{"title": "Titre", "content": "..."}, ...], "language": "fr"}

    Returns: {"results":[{"title":"...","ideas":["...",...]}, ...]}
    """
    slides = payload.get("slides") or payload.get("data") or []
    lang = (payload.get("language") or "fr").lower()[:2]

    if not slides or not isinstance(slides, list):
        return {"error": "Provide a list of slides under key 'slides'"}

    try:
        from ai.llm import Brain
    except Exception:
        return {"error": "LLM module unavailable"}

    brain = Brain()

    import asyncio

    async def _gen_for_slide(slide):
        title = (slide.get("title") if isinstance(slide, dict) else None) or ""
        content = (slide.get("content") if isinstance(slide, dict) else str(slide)) or ""
        if not content.strip():
            return {"title": title, "ideas": []}

        # call blocking LLM.present in thread
        def sync_call():
            text, _ = brain.present(
                section_content=content,
                language=lang,
                student_level="université",
                chapter_title=title,
            )
            return text

        text = await asyncio.to_thread(sync_call)
        # split into sentences and keep up to 5 short ideas
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        ideas = []
        for s in sentences:
            if len(ideas) >= 5:
                break
            # shorten long sentences to a concise idea (max 140 chars)
            idea = s if len(s) <= 140 else s[:137].rsplit(' ', 1)[0] + '…'
            ideas.append(idea)

        return {"title": title, "ideas": ideas}

    tasks = [_gen_for_slide(s) for s in slides]
    generated = await asyncio.gather(*tasks)

    return {"results": generated}


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Dashboard HTML complet pour le professeur."""
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Teacher — Dashboard Professeur</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d0f14;--panel:#151820;--border:#1e2330;
  --accent:#6c63ff;--accent2:#00d4aa;--accent3:#ff6b6b;
  --text:#e2e8f0;--muted:#64748b;--success:#00d4aa;--warn:#f59e0b;
}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;padding:24px}
h1{font-size:1.6em;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.subtitle{color:var(--muted);font-size:.85em;margin-bottom:28px}

/* Grid */
.grid{display:grid;gap:16px}
.grid-4{grid-template-columns:repeat(4,1fr)}
.grid-2{grid-template-columns:1fr 1fr}
@media(max-width:900px){.grid-4{grid-template-columns:1fr 1fr}.grid-2{grid-template-columns:1fr}}

/* Cards */
.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:20px}
.card-title{font-size:.72em;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px}

/* KPI */
.kpi-value{font-size:2.4em;font-weight:700;font-family:'DM Mono',monospace;line-height:1}
.kpi-sub{font-size:.78em;color:var(--muted);margin-top:6px}
.green{color:var(--success)}.red{color:var(--accent3)}.yellow{color:var(--warn)}.purple{color:var(--accent)}

/* Sessions table */
table{width:100%;border-collapse:collapse;font-size:.83em}
th{color:var(--muted);font-weight:600;text-align:left;padding:8px 12px;border-bottom:1px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid var(--border)}
tr:hover td{background:rgba(108,99,255,.06)}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.75em;font-weight:600}
.badge-green{background:rgba(0,212,170,.12);color:var(--success)}
.badge-red{background:rgba(255,107,107,.12);color:var(--accent3)}
.badge-yellow{background:rgba(245,158,11,.12);color:var(--warn)}
.badge-purple{background:rgba(108,99,255,.12);color:var(--accent)}

/* Chart bars */
.bar-group{display:flex;flex-direction:column;gap:8px}
.bar-row{display:flex;align-items:center;gap:10px;font-size:.8em}
.bar-label{width:80px;color:var(--muted);text-align:right;flex-shrink:0}
.bar-track{flex:1;background:var(--border);border-radius:4px;height:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width 1s ease}
.bar-val{width:50px;color:var(--text);font-family:'DM Mono',monospace;font-size:.85em}

/* Services + trace */
.service-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.service-card{padding:16px;border:1px solid var(--border);border-radius:14px;background:rgba(255,255,255,.02);display:flex;flex-direction:column;gap:10px;min-height:180px}
.service-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.service-title{font-weight:700;font-size:.9em}
.service-state{font-size:.72em;font-weight:700;padding:2px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.06em}
.service-state.ok{background:rgba(0,212,170,.12);color:var(--success)}
.service-state.warn{background:rgba(245,158,11,.12);color:var(--warn)}
.service-state.bad{background:rgba(255,107,107,.12);color:var(--accent3)}
.service-meta{font-size:.75em;color:var(--muted);line-height:1.5}
.service-detail{padding-top:8px;border-top:1px solid rgba(255,255,255,.05);display:flex;flex-direction:column;gap:3px}
.service-detail-label{font-size:.68em;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.service-detail-value{font-size:.8em;color:var(--text);line-height:1.45}
.trace-list{display:flex;flex-direction:column;gap:10px;max-height:280px;overflow-y:auto}
.trace-item{display:flex;flex-direction:column;gap:6px;padding:12px;border:1px solid var(--border);border-radius:14px;background:rgba(255,255,255,.02)}
.trace-top{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
.trace-title{font-size:.88em;font-weight:700}
.trace-sub{font-size:.78em;color:var(--muted);line-height:1.5;white-space:pre-wrap}
.trace-badges{display:flex;flex-wrap:wrap;gap:6px}
.trace-age{font-size:.72em;color:var(--muted);font-family:'DM Mono',monospace}

/* Timeline */
.timeline{display:flex;flex-direction:column;gap:8px;max-height:280px;overflow-y:auto}
.tl-item{display:flex;gap:10px;align-items:flex-start;font-size:.8em}
.tl-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);flex-shrink:0;margin-top:4px}
.tl-dot.ok{background:var(--success)}.tl-dot.bad{background:var(--accent3)}
.tl-text{color:var(--muted)}.tl-time{color:var(--text);font-family:'DM Mono',monospace}

/* Refresh */
.refresh-btn{padding:6px 14px;background:var(--accent);color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.8em;font-weight:600;transition:opacity .2s}
.refresh-btn:hover{opacity:.8}
.last-update{font-size:.75em;color:var(--muted);margin-left:10px}

/* Loading */
.loading{display:inline-block;width:12px;height:12px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div style="max-width:1200px;margin:0 auto">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px">
    <div>
      <h1>🎓 Dashboard Professeur</h1>
      <div class="subtitle">Smart Teacher — Monitoring des sessions en temps réel</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <button class="refresh-btn" onclick="loadAll()">↻ Actualiser</button>
      <span class="last-update" id="lastUpdate">—</span>
    </div>
  </div>

  <!-- KPIs -->
  <div class="grid grid-4" style="margin-bottom:16px">
    <div class="card">
      <div class="card-title">Interactions totales</div>
      <div class="kpi-value purple" id="kpi-total"><span class="loading"></span></div>
      <div class="kpi-sub" id="kpi-24h">—</div>
    </div>
    <div class="card">
      <div class="card-title">Sessions</div>
      <div class="kpi-value green" id="kpi-sessions"><span class="loading"></span></div>
      <div class="kpi-sub">sessions enregistrées</div>
    </div>
    <div class="card">
      <div class="card-title">Taux KPI (&lt; 5s)</div>
      <div class="kpi-value" id="kpi-rate"><span class="loading"></span></div>
      <div class="kpi-sub">objectif : &gt; 80%</div>
    </div>
    <div class="card">
      <div class="card-title">Temps moyen réponse</div>
      <div class="kpi-value yellow" id="kpi-avg"><span class="loading"></span></div>
      <div class="kpi-sub">STT + LLM + TTS</div>
    </div>
  </div>

  <!-- AI Techniques Overview -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
      <span>🤖 Techniques IA Actives</span>
      <span style="font-size:.85em;color:var(--accent)">v2.0 Agentic</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px">
      <div style="padding:12px;border:1px solid rgba(108,99,255,.2);border-radius:8px;background:rgba(108,99,255,.05)">
        <div style="font-size:.9em;font-weight:700;color:var(--accent);margin-bottom:4px">Agentic RAG Pipeline</div>
        <div style="font-size:.75em;color:var(--muted);line-height:1.4">Orchestration LangGraph-like avec auto-correction et review</div>
      </div>
      <div style="padding:12px;border:1px solid rgba(0,212,170,.2);border-radius:8px;background:rgba(0,212,170,.05)">
        <div style="font-size:.9em;font-weight:700;color:var(--accent2);margin-bottom:4px">Reviewer LLM</div>
        <div style="font-size:.75em;color:var(--muted);line-height:1.4">Évaluation automatique des réponses générées</div>
      </div>
      <div style="padding:12px;border:1px solid rgba(245,158,11,.2);border-radius:8px;background:rgba(245,158,11,.05)">
        <div style="font-size:.9em;font-weight:700;color:var(--warn);margin-bottom:4px">FSRS Spaced Repetition</div>
        <div style="font-size:.75em;color:var(--muted);line-height:1.4">Algorithme moderne remplaçant SM-2</div>
      </div>
      <div style="padding:12px;border:1px solid rgba(255,107,107,.2);border-radius:8px;background:rgba(255,107,107,.05)">
        <div style="font-size:.9em;font-weight:700;color:var(--accent3);margin-bottom:4px">Confusion Detection (BERT)</div>
        <div style="font-size:.75em;color:var(--muted);line-height:1.4">Détection avancée avec feedback pédagogique</div>
      </div>
      <div style="padding:12px;border:1px solid rgba(108,99,255,.2);border-radius:8px;background:rgba(108,99,255,.05)">
        <div style="font-size:.9em;font-weight:700;color:var(--accent);margin-bottom:4px">Agent Router</div>
        <div style="font-size:.75em;color:var(--muted);line-height:1.4">Routage intelligent: quiz/code/RAG</div>
      </div>
      <div style="padding:12px;border:1px solid rgba(0,212,170,.2);border-radius:8px;background:rgba(0,212,170,.05)">
        <div style="font-size:.9em;font-weight:700;color:var(--accent2);margin-bottom:4px">Code Executor (Sandbox)</div>
        <div style="font-size:.75em;color:var(--muted);line-height:1.4">Exécution sécurisée Python pour programmation</div>
      </div>
    </div>
  </div>

  <div class="grid grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="card-title" style="display:flex;align-items:center;justify-content:space-between;gap:12px">
        <span>🩺 Services en direct</span>
        <span id="services-count" style="font-size:.85em;color:var(--accent)"></span>
      </div>
      <div style="color:var(--muted);font-size:.8em;line-height:1.45;margin-bottom:10px">
        Chaque carte indique le statut, le rôle réel, ce qui est récupéré et le chemin principal du code.
      </div>
      <div class="service-grid" id="serviceGrid">
        <div style="color:var(--muted);font-size:.85em">Chargement…</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title" style="display:flex;align-items:center;justify-content:space-between;gap:12px">
        <span>⚡ Trace backend récente</span>
        <span id="trace-count" style="font-size:.85em;color:var(--accent)"></span>
      </div>
      <div class="trace-list" id="traceTimeline">
        <div style="color:var(--muted);font-size:.85em">Chargement…</div>
      </div>
    </div>
  </div>

  <div class="grid grid-2" style="margin-bottom:16px">
    <!-- Répartition par composant -->
    <div class="card">
      <div class="card-title">⏱ Temps par composant</div>
      <div class="bar-group" id="timeChart">
        <div style="color:var(--muted);font-size:.85em">Chargement…</div>
      </div>
    </div>

    <!-- Répartition par langue -->
    <div class="card">
      <div class="card-title">🌍 Sessions par langue</div>
      <div class="bar-group" id="langChart">
        <div style="color:var(--muted);font-size:.85em">Chargement…</div>
      </div>
    </div>
  </div>

  <!-- Sessions récentes -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
      <span>📋 Sessions récentes (24h)</span>
      <span id="sessions-count" style="font-size:.85em;color:var(--accent)"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Session ID</th>
          <th>Langue</th>
          <th>Tours</th>
          <th>Temps moyen</th>
          <th>KPI</th>
          <th>Pointeur</th>
          <th>État</th>
          <th>Dernière activité</th>
        </tr>
      </thead>
      <tbody id="sessionsTable">
        <tr><td colspan="8" style="color:var(--muted);text-align:center;padding:20px">Chargement…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Repères pause / reprise -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
      <span>📍 Pointeur, pause et reprise</span>
      <span id="checkpoints-count" style="font-size:.85em;color:var(--accent)"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Session</th>
          <th>Type</th>
          <th>Pointeur</th>
          <th>Détail</th>
          <th>Raison</th>
          <th>Horodatage</th>
        </tr>
      </thead>
      <tbody id="checkpointsTable">
        <tr><td colspan="6" style="color:var(--muted);text-align:center;padding:20px">Chargement…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Questions récentes -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
      <span>❓ Questions récentes (24h)</span>
      <span id="questions-count" style="font-size:.85em;color:var(--accent)"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Session</th>
          <th>Tour</th>
          <th>Question</th>
          <th>Sujet</th>
          <th>Pointeur</th>
          <th>Langue</th>
          <th>STT</th>
          <th>LLM</th>
          <th>TTS</th>
          <th>Temps total</th>
          <th>KPI</th>
          <th>Confusion</th>
        </tr>
      </thead>
      <tbody id="questionsTable">
        <tr><td colspan="12" style="color:var(--muted);text-align:center;padding:20px">Chargement…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Accès rapide -->
  <div class="card">
    <div class="card-title">🔗 Accès rapide</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <a href="/static/index.html" style="padding:8px 16px;background:var(--accent);color:#fff;border-radius:8px;text-decoration:none;font-size:.85em;font-weight:600">🎓 Interface Étudiant</a>
      <a href="/docs" style="padding:8px 16px;background:var(--border);color:var(--text);border-radius:8px;text-decoration:none;font-size:.85em;font-weight:600">📚 API Docs</a>
      <a href="/rag/stats" style="padding:8px 16px;background:var(--border);color:var(--text);border-radius:8px;text-decoration:none;font-size:.85em;font-weight:600">🔍 RAG Stats</a>
      <a href="/health" style="padding:8px 16px;background:var(--border);color:var(--text);border-radius:8px;text-decoration:none;font-size:.85em;font-weight:600">💚 Health Check</a>
    </div>
  </div>
</div>

<script>
async function loadAll() {
  await Promise.all([loadStats(), loadServices(), loadSessions(), loadAnalytics(), loadCheckpoints(), loadTrace(), loadQuestions()]);
  document.getElementById('lastUpdate').textContent = 'Mis à jour : ' + new Date().toLocaleTimeString();
}

async function loadStats() {
  try {
    const d = await fetch('/dashboard/stats').then(r => r.json());
    document.getElementById('kpi-total').textContent    = d.total_interactions?.toLocaleString() || '0';
    document.getElementById('kpi-sessions').textContent = d.total_sessions?.toLocaleString() || '0';
    document.getElementById('kpi-24h').textContent      = `${d.last_24h || 0} dernières 24h`;

    const rate = d.kpi_rate_pct || 0;
    const rateEl = document.getElementById('kpi-rate');
    rateEl.textContent = rate + '%';
    rateEl.className   = 'kpi-value ' + (rate >= 80 ? 'green' : rate >= 60 ? 'yellow' : 'red');

    // Graphe langues
    const langs = d.by_language || {};
    const total = Object.values(langs).reduce((a,b)=>a+b, 0) || 1;
    const colors = {fr:'#6c63ff',en:'#f59e0b'};
    document.getElementById('langChart').innerHTML = Object.entries(langs)
      .sort((a,b) => b[1]-a[1])
      .map(([lang, cnt]) => `
        <div class="bar-row">
          <div class="bar-label">${lang.toUpperCase()}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${cnt/total*100}%;background:${colors[lang]||'#888'}"></div></div>
          <div class="bar-val">${cnt}</div>
        </div>`).join('') || '<div style="color:var(--muted);font-size:.85em">Aucune donnée</div>';
  } catch(e) { console.error(e); }
}

async function loadAnalytics() {
  try {
    const d = await fetch('/dashboard/analytics').then(r => r.json());
    if (d.message) return;

    document.getElementById('kpi-avg').textContent = (d.total_time?.avg || 0) + 's';

    const components = [
      {label:'STT',  val: d.stt_time?.avg || 0,  color:'#6c63ff', max:2},
      {label:'LLM',  val: d.llm_time?.avg || 0,  color:'#00d4aa', max:3},
      {label:'TTS',  val: d.tts_time?.avg || 0,  color:'#f59e0b', max:2},
      {label:'TOTAL',val: d.total_time?.avg || 0, color:'#ff6b6b', max:5},
    ];
    document.getElementById('timeChart').innerHTML = components.map(c => `
      <div class="bar-row">
        <div class="bar-label">${c.label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.min(c.val/c.max*100,100)}%;background:${c.color}"></div></div>
        <div class="bar-val">${c.val}s</div>
      </div>`).join('');
  } catch(e) { console.error(e); }
}

async function loadServices() {
  try {
    const d = await fetch('/dashboard/services').then(r => r.json());
    const services = d.services || {};

    const cards = [
      {
        key: 'rag',
        title: 'Qdrant / RAG',
        icon: '🔍',
        ok: !!services.rag?.healthy || !!services.rag?.bm25_ready,
        stateLabel: services.rag?.healthy ? 'Actif' : (services.rag?.bm25_ready ? 'BM25' : 'HS'),
        stateClass: services.rag?.healthy ? 'ok' : (services.rag?.bm25_ready ? 'warn' : 'bad'),
        meta: [
          services.rag?.embedding_source ? `Embeddings ${services.rag.embedding_source}` : '',
          services.rag?.embedding_model || '',
          services.rag?.qdrant_connected ? 'Qdrant connecté' : 'Qdrant indisponible',
          services.rag?.vectorstore_available ? 'Vectorstore OK' : 'Vectorstore KO',
          services.rag?.collection ? `Collection ${services.rag.collection}` : '',
          services.rag?.docs_loaded != null ? `${services.rag.docs_loaded} docs` : '',
        ].filter(Boolean),
        details: [
          { label: 'Rôle', value: services.rag?.role || 'Orchestre la recherche hybride et la génération de réponses' },
          { label: 'Récupère', value: services.rag?.retrieves || "Chunks vectoriels Qdrant, scores BM25 et cache d'embeddings" },
          { label: 'Code', value: services.rag?.used_in || 'modules/multimodal_rag.py' },
        ],
      },
      {
        key: 'redis',
        title: 'Redis',
        icon: '🧠',
        ok: !!services.redis?.connected,
        stateLabel: services.redis?.connected ? 'Actif' : 'HS',
        stateClass: services.redis?.connected ? 'ok' : 'bad',
        meta: [services.redis?.endpoint || '', services.redis?.latency_ms != null ? `${services.redis.latency_ms} ms` : ''].filter(Boolean),
        details: [
          { label: 'Rôle', value: services.redis?.role || 'Cache et état temps réel' },
          { label: 'Récupère', value: services.redis?.retrieves || 'Sessions WebSocket, état temporaire et latence de traitement' },
          { label: 'Code', value: services.redis?.used_in || 'handlers/session_manager.py, modules/llm.py, main.py get_redis' },
        ],
      },
      {
        key: 'postgres',
        title: 'PostgreSQL',
        icon: '🗄️',
        ok: !!services.postgres?.connected,
        stateLabel: services.postgres?.connected ? 'Actif' : 'HS',
        stateClass: services.postgres?.connected ? 'ok' : 'bad',
        meta: [services.postgres?.endpoint || '', services.postgres?.latency_ms != null ? `${services.postgres.latency_ms} ms` : ''].filter(Boolean),
        details: [
          { label: 'Rôle', value: services.postgres?.role || 'Persistance transactionnelle' },
          { label: 'Récupère', value: services.postgres?.retrieves || "Sessions, interactions, profils étudiants et événements d'apprentissage" },
          { label: 'Code', value: services.postgres?.used_in || 'database/models.py, database/init_db.py, main.py /course/build, /session, /dashboard/stats' },
        ],
      },
      {
        key: 'ollama',
        title: 'Ollama',
        icon: '🖥️',
        ok: !!services.ollama?.available,
        stateLabel: services.ollama?.available ? 'Actif' : 'HS',
        stateClass: services.ollama?.available ? 'ok' : 'bad',
        meta: [services.ollama?.model ? `Model ${services.ollama.model}` : '', services.ollama?.endpoint || ''].filter(Boolean),
        details: [
          { label: 'Rôle', value: services.ollama?.role || 'LLM local de secours' },
          { label: 'Récupère', value: services.ollama?.retrieves || 'Modèles disponibles via /api/tags et réponses locales via Ollama' },
          { label: 'Code', value: services.ollama?.used_in || 'modules/llm.py, main.py /ask, quiz fallback' },
        ],
      },
      {
        key: 'elasticsearch',
        title: 'Elasticsearch',
        icon: '🧭',
        ok: !!services.elasticsearch?.available,
        stateLabel: services.elasticsearch?.available ? 'Actif' : 'Mémoire',
        stateClass: services.elasticsearch?.available ? 'ok' : 'warn',
        meta: [services.elasticsearch?.backend ? `Backend ${services.elasticsearch.backend}` : '', services.elasticsearch?.host || '', services.elasticsearch?.index ? `Index ${services.elasticsearch.index}` : ''].filter(Boolean),
        details: [
          { label: 'Rôle', value: services.elasticsearch?.role || 'Recherche full-text historique' },
          { label: 'Récupère', value: services.elasticsearch?.retrieves || 'Questions, réponses et index texte des transcriptions' },
          { label: 'Code', value: services.elasticsearch?.used_in || 'modules/transcript_search.py, main.py /dashboard/services' },
        ],
      },
      {
        key: 'minio',
        title: 'MinIO',
        icon: '🪣',
        ok: true,
        stateLabel: services.minio?.provider === 'minio' ? 'MinIO' : 'Local',
        stateClass: services.minio?.provider === 'minio' ? 'ok' : 'warn',
        meta: [
          services.minio?.provider ? `Provider ${services.minio.provider}` : '',
          services.minio?.endpoint && services.minio.provider === 'minio' ? services.minio.endpoint : '',
          services.minio?.bucket ? `Bucket ${services.minio.bucket}` : '',
          services.minio?.local_root ? `Local ${services.minio.local_root}` : '',
        ].filter(Boolean),
        details: [
          { label: 'Rôle', value: services.minio?.role || 'Stockage objet des médias' },
          { label: 'Récupère', value: services.minio?.retrieves || 'PDF, slides, audio et objets listés via /media-list' },
          { label: 'Code', value: services.minio?.used_in || 'modules/media_storage.py, main.py /media/{path}, /media-list, modules/course_builder.py' },
        ],
      },
    ];

    const healthyCount = cards.filter(card => card.ok).length;
    document.getElementById('services-count').textContent = `${healthyCount}/${cards.length} actifs`;

    document.getElementById('serviceGrid').innerHTML = cards.map(card => `
      <div class="service-card">
        <div class="service-head">
          <div class="service-title">${card.icon} ${card.title}</div>
          <span class="service-state ${card.stateClass}">${esc(card.stateLabel)}</span>
        </div>
        <div class="service-meta">${card.meta.length ? card.meta.map(esc).join('<br>') : 'Aucune donnée'}</div>
        ${card.details.map(detail => detail.value ? `
          <div class="service-detail">
            <div class="service-detail-label">${esc(detail.label)}</div>
            <div class="service-detail-value">${esc(detail.value)}</div>
          </div>
        ` : '').join('')}
      </div>
    `).join('');
  } catch(e) { console.error(e); }
}

async function loadSessions() {
  try {
    const d = await fetch('/dashboard/sessions').then(r => r.json());
    const sessions = d.sessions || [];
    document.getElementById('sessions-count').textContent = sessions.length + ' sessions';

    if (!sessions.length) {
      document.getElementById('sessionsTable').innerHTML =
        '<tr><td colspan="8" style="color:var(--muted);text-align:center;padding:20px">Aucune session récente</td></tr>';
      return;
    }

    document.getElementById('sessionsTable').innerHTML = sessions.map(s => {
      const kpiClass = s.kpi_rate >= 80 ? 'badge-green' : s.kpi_rate >= 60 ? 'badge-yellow' : 'badge-red';
      const stateLabel = s.checkpoint_label || (s.paused ? 'Pause' : 'Actif');
      const stateClass = s.paused ? 'badge-yellow' : 'badge-purple';
      const ago = Math.round((Date.now()/1000 - s.last_activity) / 60);
      const agoStr = ago < 60 ? `il y a ${ago}min` : `il y a ${Math.round(ago/60)}h`;
      return `<tr>
        <td style="font-family:'DM Mono',monospace;color:var(--accent)">${esc(s.session_id)}…</td>
        <td><span class="badge badge-green">${(s.language||'?').toUpperCase()}</span></td>
        <td>${s.turns}</td>
        <td style="font-family:'DM Mono',monospace">${s.avg_time}s</td>
        <td><span class="badge ${kpiClass}">${s.kpi_rate}%</span></td>
        <td style="font-family:'DM Mono',monospace;color:var(--text)" title="${esc(s.pointer_detail || s.pointer || '—')}">${esc(s.pointer || '—')}</td>
        <td><span class="badge ${stateClass}">${esc(stateLabel)}</span></td>
        <td style="color:var(--muted)">${agoStr}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

async function loadCheckpoints() {
  try {
    const d = await fetch('/dashboard/checkpoints').then(r => r.json());
    const checkpoints = d.checkpoints || [];
    document.getElementById('checkpoints-count').textContent = checkpoints.length + ' repères';

    if (!checkpoints.length) {
      document.getElementById('checkpointsTable').innerHTML =
        '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:20px">Aucun repère récent</td></tr>';
      return;
    }

    document.getElementById('checkpointsTable').innerHTML = checkpoints.map(c => {
      const kindClass = c.checkpoint_type === 'pause' ? 'badge-yellow' : 'badge-green';
      const kindLabel = c.checkpoint_label || (c.checkpoint_type === 'pause' ? 'Pause' : 'Reprise');
      const detail = c.pointer_detail || c.pointer || '—';
      const detailShort = detail.length > 90 ? detail.slice(0, 90) + '…' : detail;
      const stamp = new Date((c.ts || 0) * 1000).toLocaleString();
      return `<tr>
        <td style="font-family:'DM Mono',monospace;color:var(--accent)">${esc(c.session_id)}…</td>
        <td><span class="badge ${kindClass}">${esc(kindLabel)}</span></td>
        <td style="font-family:'DM Mono',monospace" title="${esc(detail)}">${esc(c.pointer || '—')}</td>
        <td title="${esc(detail)}">${esc(detailShort)}</td>
        <td style="color:var(--muted)" title="${esc(c.slide_title || '')}">${esc(c.reason || '—')}</td>
        <td style="font-family:'DM Mono',monospace">${stamp}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

async function loadTrace() {
  try {
    const d = await fetch('/dashboard/trace').then(r => r.json());
    const traces = d.traces || [];
    document.getElementById('trace-count').textContent = `${traces.length} étapes`;

    if (!traces.length) {
      document.getElementById('traceTimeline').innerHTML =
        '<div style="color:var(--muted);font-size:.85em">Aucune étape récente</div>';
      return;
    }

    document.getElementById('traceTimeline').innerHTML = traces.map(t => {
      const state = (t.state || '').toUpperCase();
      const stateClass = state === 'RESPONDING' ? 'badge-green' : state === 'PROCESSING' ? 'badge-yellow' : state === 'PRESENTING' ? 'badge-purple' : 'badge-red';
      const rawSummary = (t.display_message || t.substep || t.state_name || '').replace(/\\n+/g, ' • ');
      const summary = rawSummary.length > 180 ? rawSummary.slice(0, 180) + '…' : rawSummary;
      const details = t.details || {};
      const detailParts = [];
      if (details.course_title) detailParts.push(`Course: ${details.course_title}`);
      if (details.chapter_title) detailParts.push(`Chapter: ${details.chapter_title}`);
      if (details.section_title) detailParts.push(`Section: ${details.section_title}`);
      if (details.slide_title) detailParts.push(`Slide: ${details.slide_title}`);
      if (details.question_text) detailParts.push(`Question: ${details.question_text}`);
      if (details.transcription) detailParts.push(`STT: ${details.transcription}`);
      if (details.tts_engine || details.tts_voice) detailParts.push(`TTS: ${details.tts_engine || 'unknown'}${details.tts_voice ? ` / ${details.tts_voice}` : ''}`);
      if (details.answer_preview) detailParts.push(`Answer: ${details.answer_preview}`);
      const detailLine = detailParts.join(' • ');
      const age = typeof t.age_sec === 'number' ? `${t.age_sec.toFixed(1)}s` : '—';
      const metrics = t.metrics || {};
      const badges = [];
      if (t.turn_id != null) badges.push(`<span class="badge badge-purple">Tour ${esc(t.turn_id)}</span>`);
      if (metrics.stt_time != null) badges.push(`<span class="badge badge-yellow">STT ${Number(metrics.stt_time).toFixed(2)}s</span>`);
      if (metrics.retrieval_time != null) badges.push(`<span class="badge badge-yellow">RAG ${Number(metrics.retrieval_time).toFixed(2)}s</span>`);
      if (metrics.llm_time != null) badges.push(`<span class="badge badge-green">LLM ${Number(metrics.llm_time).toFixed(2)}s</span>`);
      if (metrics.tts_time != null) badges.push(`<span class="badge badge-green">TTS ${Number(metrics.tts_time).toFixed(2)}s</span>`);
      if (metrics.total_time != null) badges.push(`<span class="badge badge-red">Total ${Number(metrics.total_time).toFixed(2)}s</span>`);
      if (metrics.chunks != null) badges.push(`<span class="badge badge-yellow">Docs ${metrics.chunks}</span>`);
      return `
        <div class="trace-item">
          <div class="trace-top">
            <div>
              <div class="trace-title">${esc(t.emoji || '•')} ${esc(t.state_name || t.state || 'Trace')}</div>
                <div class="trace-sub">${esc(summary || '—')}${detailLine ? `\\n${esc(detailLine)}` : ''}</div>
            </div>
            <div class="trace-age">${age}</div>
          </div>
          <div class="trace-badges">
            <span class="badge ${stateClass}">${esc(state || 'STEP')}</span>
            ${badges.join('')}
          </div>
        </div>`;
    }).join('');
  } catch(e) { console.error(e); }
}

function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function loadQuestions() {
  try {
    const d = await fetch('/dashboard/questions').then(r => r.json());
    const questions = d.questions || [];
    document.getElementById('questions-count').textContent = questions.length + ' questions';

    if (!questions.length) {
      document.getElementById('questionsTable').innerHTML =
        '<tr><td colspan="12" style="color:var(--muted);text-align:center;padding:20px">Aucune question récente</td></tr>';
      return;
    }

    document.getElementById('questionsTable').innerHTML = questions.map(q => {
      const kpiClass = q.kpi_ok ? 'badge-green' : 'badge-red';
      const confusionClass = q.confusion ? 'badge-yellow' : 'badge-green';
      const question = (q.question || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      const answer = (q.answer || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      const slideTitle = (q.slide_title || q.section_title || q.chapter_title || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      const pointer = q.pointer || '—';
      const pointerDetail = q.pointer_detail || pointer;
      const turnLabel = q.turn_id != null ? q.turn_id : '—';
      const sttTime = Number(q.stt_time || 0).toFixed(2);
      const llmTime = Number(q.llm_time || 0).toFixed(2);
      const ttsTime = Number(q.tts_time || 0).toFixed(2);
      const totalTime = Number(q.total_time || 0).toFixed(2);
      const ttsVoice = (q.tts_voice || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      const ttsEngine = (q.tts_engine || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
      return `<tr>
        <td style="font-family:'DM Mono',monospace;color:var(--accent)">${esc(q.session_id)}…</td>
        <td style="font-family:'DM Mono',monospace">${esc(turnLabel)}</td>
        <td title="${question}">${question.length > 82 ? question.slice(0, 82) + '…' : question}${slideTitle ? `<div style="margin-top:6px;color:var(--muted);font-size:.78em;line-height:1.45" title="${slideTitle}">🖼️ ${slideTitle}</div>` : ''}${answer ? `<div style="margin-top:6px;color:var(--muted);font-size:.78em;line-height:1.45" title="${answer}">${answer.length > 110 ? answer.slice(0, 110) + '…' : answer}</div>` : ''}</td>
        <td><span class="badge badge-green">${esc(q.subject || 'unknown')}</span></td>
        <td style="font-family:'DM Mono',monospace" title="${esc(pointerDetail)}">${esc(pointer)}</td>
        <td><span class="badge badge-yellow">${(q.language || '?').toUpperCase()}</span></td>
        <td style="font-family:'DM Mono',monospace">${sttTime}s</td>
        <td style="font-family:'DM Mono',monospace">${llmTime}s</td>
        <td style="font-family:'DM Mono',monospace">${ttsTime}s${ttsVoice ? `<div style="margin-top:4px;color:var(--muted);font-size:.76em;line-height:1.35" title="${ttsEngine || ttsVoice}">${ttsEngine ? `${ttsEngine} / ` : ''}${ttsVoice}</div>` : ''}</td>
        <td style="font-family:'DM Mono',monospace">${totalTime}s</td>
        <td><span class="badge ${kpiClass}">${q.kpi_ok ? 'OK' : 'SLOW'}</span></td>
        <td><span class="badge ${confusionClass}">${q.confusion ? 'Oui' : 'Non'}</span></td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>"""

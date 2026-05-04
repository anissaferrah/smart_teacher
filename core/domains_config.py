"""
╔══════════════════════════════════════════════════════════════════════╗
║     SMART TEACHER — Configuration Domaines & Cours v5              ║
║                                                                      ║
║  Structure 100% DYNAMIQUE : Découverte depuis les dossiers         ║
║  - Domaines & Cours : découverts depuis courses/{domain}/          ║
║  - Chapitre : découverts dynamiquement depuis les fichiers         ║
║  - Métadonnées : générées automatiquement depuis les noms          ║
║                                                                      ║
║  Aucun hardcodage des cours ! Tout est automatique.                ║
║                                                                      ║
║  Exemple :                                                           ║
║    courses/informatique/                                            ║
║    ├── calcul/                                                     ║
║    │   ├── Chapter 1.pdf                                           ║
║    │   └── Chapter 2.pdf                                           ║
║    └── linguistique/                                               ║
║        ├── Chapter 1.pdf                                           ║
║        └── Chapter 2.pdf                                           ║
║                                                                      ║
║  Pour AJOUTER UN COURS : Créez simplement le dossier !            ║
║    mkdir -p courses/informatique/"MonNouveauCours"                 ║
║    cp mon_pdf.pdf "courses/informatique/MonNouveauCours/"          ║
║  C'est tout ! Le cours sera automatiquement détecté.               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from typing import Dict
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION DOMAINES & COURS (ACTUELS)
# ═══════════════════════════════════════════════════════════════════════
#  - INFORMATIQUE : 4 cours principaux
#  - Autres domaines : à ajouter progressivement
# ═══════════════════════════════════════════════════════════════════════

DOMAINS: Dict[str, list] = {
    "informatique": [],  # Découvert dynamiquement depuis courses/informatique/
}

# ═══════════════════════════════════════════════════════════════════════
#  MÉTADONNÉES OPTIONNELLES (SPÉCIALISÉES)
# ═══════════════════════════════════════════════════════════════════════
#  Utilisé seulement si vous voulez personnaliser les métadonnées
#  Sinon, les métadonnées sont auto-générées depuis le nom du cours

COURSE_METADATA: Dict[str, Dict[str, dict]] = {}
# Exemple si vous voulez personnaliser un cours:
# {
#     "informatique": {
#         "mon_cours": {
#             "title": "Mon cours",
#             "description": "Contenu du cours",
#             "level": "licence",
#             "language": "fr",
#         },
#     },
# }

# ═══════════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════

def get_domains() -> list[str]:
    """Retourne la liste des domaines disponibles."""
    from pathlib import Path

    domains = set(DOMAINS.keys())
    courses_dir = Path("courses")
    if courses_dir.exists():
        for domain_folder in courses_dir.iterdir():
            if domain_folder.is_dir():
                domains.add(domain_folder.name)

    return sorted(domains)


def get_courses(domain: str) -> list[str]:
    """
    Retourne les cours d'un domaine.
    Les cours sont découverts DYNAMIQUEMENT depuis les dossiers dans courses/{domain}/
    """
    from pathlib import Path
    
    # Lire dynamiquement les dossiers dans courses/{domain}/
    domain_dir = Path("courses") / domain
    
    if not domain_dir.exists():
        return []
    
    # Récupérer tous les sous-dossiers (qui représentent les cours)
    courses = []
    for course_folder in sorted(domain_dir.iterdir()):
        if course_folder.is_dir():
            courses.append(course_folder.name)
    
    return courses


def get_courses_list(domain: str) -> list[str]:
    """Alias pour get_courses()."""
    return get_courses(domain)


def get_course_metadata(domain: str, course: str) -> dict:
    """
    Retourne les métadonnées d'un cours (titre, description, etc.).
    
    Les métadonnées sont découvertes DYNAMIQUEMENT:
    1. D'abord, cherche dans COURSE_METADATA (pour les cours spécialisés)
    2. Sinon, génère automatiquement à partir du nom du dossier
    """
    # Si dans COURSE_METADATA, retourner ça
    if domain in COURSE_METADATA and course in COURSE_METADATA[domain]:
        return COURSE_METADATA[domain][course]
    
    # Sinon, générer automatiquement depuis le nom du dossier
    # "mon_cours" → "Mon Cours"
    # "Traitement Automatique du Langage Naturel" → "Traitement Automatique du Langage Naturel"
    title = course.replace("_", " ").title() if "_" in course else course
    
    return {
        "title": title,
        "description": f"Cours : {title}",
        "level": "licence",
        "language": "fr",
    }


def get_course_title(domain: str, course: str) -> str:
    """Retourne le titre d'un cours."""
    return get_course_metadata(domain, course).get("title", course)


# ═══════════════════════════════════════════════════════════════════════
#  FONCTIONS DE DÉCOUVERTE DE CHAPITRES (DYNAMIQUE)
# ═══════════════════════════════════════════════════════════════════════

from pathlib import Path
import re

def discover_chapters(domain: str, course: str) -> Dict[int, str]:
    """
    Découvre automatiquement les chapitres depuis le dossier cours.
    
    Structure attendue :
        courses/{domain}/{course}/
        ├── Chapter 1.pdf
        ├── Chapter 2.pdf
        ├── Chapter 3.pdf
        └── ...
    
    Retourne un dict {1: "Nom du chapitre", 2: "...", ...}
    Les noms peuvent être :
    1. Extraits du contenu PDF (futur)
    2. Extraits du titre du fichier (actuellement)
    3. Auto-générés si absent
    
    Returns:
        Dict[int, str] : {chapter_number: chapter_title}
    """
    course_path = Path("courses") / domain / course
    
    if not course_path.exists():
        return {}

    def _extract_chapter_num(name: str) -> int | None:
        patterns = [
            r"[Cc]hapter\s+(\d+)",
            r"[Cc]hapitre\s+(\d+)",
            r"[Cc]h(\d+)",
            r"(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                for group in match.groups():
                    if group:
                        return int(group)
        return None
    
    # Patterns de fichiers acceptés
    chapters = {}

    # Chapitres au niveau racine
    for pdf_file in sorted(course_path.glob("*.pdf")):
        chapter_num = _extract_chapter_num(pdf_file.stem) or _extract_chapter_num(pdf_file.name)
        if chapter_num is None:
            continue
        chapters[chapter_num] = f"Chapter {chapter_num}"

    # Chapitres dans des sous-dossiers chapter_*/
    for subdir in sorted(course_path.iterdir()):
        if not subdir.is_dir():
            continue
        chapter_num = _extract_chapter_num(subdir.name)
        pdfs = sorted(list(subdir.glob("*.pdf")))
        if chapter_num is None and pdfs:
            chapter_num = _extract_chapter_num(pdfs[0].stem) or _extract_chapter_num(pdfs[0].name)
        if chapter_num is None:
            continue
        chapters[chapter_num] = f"Chapter {chapter_num}"
    
    return chapters


def get_chapters(domain: str, course: str) -> Dict[int, str]:
    """
    Retourne les chapitres d'un cours.
    Chapitres découverts dynamiquement depuis les fichiers.
    
    Returns:
        Dict[int, str] : {1: "Chapter 1", 2: "Chapter 2", ...}
    """
    courses_list = get_courses(domain)
    if course not in courses_list:
        raise ValueError(
            f"Cours '{course}' introuvable dans '{domain}'. "
            f"Disponibles : {courses_list}"
        )
    
    return discover_chapters(domain, course)


def get_chapter_title(domain: str, course: str, chapter_idx: int) -> str:
    """Retourne le titre d'un chapitre spécifique."""
    chapters = get_chapters(domain, course)
    if chapter_idx not in chapters:
        raise ValueError(
            f"Chapitre {chapter_idx} introuvable dans {course}. "
            f"Disponibles : {list(chapters.keys())}"
        )
    return chapters[chapter_idx]


# ═══════════════════════════════════════════════════════════════════════
#  VALEURS PAR DÉFAUT (Génériques)
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_DOMAIN = "general"  # Domaine générique par défaut
DEFAULT_COURSE = "generic"  # Cours générique par défaut

print(f"✅ Config Domaines & Cours chargée : {len(DOMAINS)} domaine(s)")
print(f"   Domaines actuels : {list(DOMAINS.keys())}")
print("   📝 Chapitres découverts DYNAMIQUEMENT depuis courses/{domain}/{course}/")
print("   💡 Ajouter un nouveau domaine : modifier ce fichier + créer le dossier courses/{new_domain}/")
print(f"   Defaults: domain='{DEFAULT_DOMAIN}', course='{DEFAULT_COURSE}' (Génériques)")

# ═══════════════════════════════════════════════════════════════════════
#  DÉTECTION AUTOMATIQUE DU DOMAINE & COURS
# ═══════════════════════════════════════════════════════════════════════
#  
#  La détection fonctionne 100% dynamiquement:
#  1. Cherche dans les dossiers existants de courses/
#  2. Compare le nom du fichier et le contenu avec les noms des cours
#  3. Aucun mot-clé hardcodé, tout est automatique!
#

def auto_detect_course(file_path: str) -> tuple[str, str]:
    """
    Détecte le domaine et course depuis un fichier PDF.
    
    Stratégie (par ordre de priorité):
    1. Cherche si le fichier est DÉJÀ dans courses/{domain}/{course}/
    2. Compare le nom du fichier avec les noms des cours existants
    3. Compare le contenu PDF avec les noms des cours
    4. Fallback: retourne le domaine/cours par défaut
    
    Args:
        file_path: Chemin vers le fichier PDF
    
    Returns:
        tuple[str, str] : (domain, course) - découvert AUTOMATIQUEMENT
    """
    from pathlib import Path
    
    file_path_obj = Path(file_path)
    filename_lower = file_path_obj.stem.lower()
    
    # STRATÉGIE 1: Vérifier si le fichier est déjà dans courses/
    courses_dir = Path("courses")
    if courses_dir.exists():
        # Chercher dans tous les domaines/cours
        for domain_folder in courses_dir.iterdir():
            if not domain_folder.is_dir():
                continue
            
            domain_name = domain_folder.name
            
            for course_folder in domain_folder.iterdir():
                if not course_folder.is_dir():
                    continue
                
                course_name = course_folder.name
                
                # Vérifier si le fichier existe déjà dans ce dossier
                for existing_file in course_folder.glob(f"*{file_path_obj.suffix}"):
                    if existing_file.name.lower() == file_path_obj.name.lower():
                        return (domain_name, course_name)
    
    # STRATÉGIE 2: Comparer avec les noms des cours existants
    if courses_dir.exists():
        for domain_folder in courses_dir.iterdir():
            if not domain_folder.is_dir():
                continue
            
            domain_name = domain_folder.name
            
            for course_folder in domain_folder.iterdir():
                if not course_folder.is_dir():
                    continue
                
                course_name = course_folder.name
                course_name_lower = course_name.lower()
                
                # Comparer avec le NOM DU FICHIER
                if course_name_lower in filename_lower or filename_lower in course_name_lower:
                    return (domain_name, course_name)
    
    # STRATÉGIE 3: Comparer avec le CONTENU DU PDF
    if courses_dir.exists():
        try:
            from unstructured.partition.auto import partition
            elements = partition(file_path)
            text = "\n".join([el.text for el in elements]).lower()
            
            for domain_folder in courses_dir.iterdir():
                if not domain_folder.is_dir():
                    continue
                
                domain_name = domain_folder.name
                
                for course_folder in domain_folder.iterdir():
                    if not course_folder.is_dir():
                        continue
                    
                    course_name = course_folder.name
                    course_name_lower = course_name.lower()
                    
                    # Si le contenu mentionne le cours, c'est probablement ce cours
                    if course_name_lower in text or course_name_lower.replace(" ", "") in text:
                        return (domain_name, course_name)
        except:
            pass
    
    # FALLBACK : retourner le domaine/cours par défaut
    return (DEFAULT_DOMAIN, DEFAULT_COURSE)


def classify_course_via_llm(file_path: str, max_pages: int = 2) -> dict | None:
    """Auto-classification du cours via LLM (Ollama Mistral prioritaire).

    Lit les premieres pages du PDF et demande au LLM de classer ce cours
    dans un domaine/cours existant ou de creer un nouveau slug.

    Returns:
        {"domain": str, "course": str, "title": str, "reason": str} ou None
    """
    import json
    import logging
    import re
    from pathlib import Path

    log = logging.getLogger("SmartTeacher.AutoClassify")

    # ── 1. Extract first pages text (pypdf) ───────────────────────────────
    try:
        import pypdf
        text_parts = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f, strict=False)
            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    break
                txt = (page.extract_text() or "").strip()
                if txt:
                    text_parts.append(txt)
        text_sample = "\n\n".join(text_parts)
    except Exception as exc:
        log.warning(f"PDF extract failed: {exc}")
        return None

    if not text_sample or len(text_sample.strip()) < 100:
        log.info(f"⏭️ LLM classify skip: extrait trop court ({len(text_sample)} chars)")
        return None

    # ── 2. List existing courses structure ────────────────────────────────
    existing: dict[str, list[str]] = {}
    courses_dir = Path("courses")
    if courses_dir.exists():
        for domain_folder in courses_dir.iterdir():
            if not domain_folder.is_dir():
                continue
            courses = [c.name for c in domain_folder.iterdir() if c.is_dir()]
            if courses:
                existing[domain_folder.name] = courses
    existing_str = "\n".join(
        f"  - {dom}/{c}" for dom, lst in existing.items() for c in lst
    ) or "  (aucun cours existant)"

    # ── 3. Build prompt ───────────────────────────────────────────────────
    prompt = (
        "Tu es un assistant pédagogique. Classifie ce cours dans un domaine/cours.\n\n"
        f"COURS EXISTANTS :\n{existing_str}\n\n"
        "Règles :\n"
        "  - Réutilise un domaine/cours existant si le contenu correspond\n"
        "  - Sinon crée un slug court en snake_case (ex: machine_learning, math, physique)\n"
        "  - Évite 'general' et 'generic' si possible\n\n"
        "Réponds UNIQUEMENT en JSON STRICT (sans markdown, sans préambule) :\n"
        '{"domain": "<snake_case>", "course": "<snake_case>", '
        '"title": "<2-6 mots lisibles>", "reason": "<justification courte>"}\n\n'
        f"CONTENU DU COURS (extrait des {max_pages} 1ères pages) :\n{text_sample[:2000]}"
    )

    # ── 4. Call LLM (Ollama prioritaire car OpenAI souvent en quota) ─────
    response_text: str | None = None

    # 4a. Try OpenAI first (may be rate-limited) — unless the global
    #     DISABLE_OPENAI kill-switch is set. The local import keeps the
    #     domains_config module importable even when langchain_openai
    #     isn't installed, which is the lightweight test path.
    # When OpenAI is off but Groq is configured, we use Groq's OpenAI-
    # compatible endpoint as a high-quality replacement for OpenAI in
    # this classification call (~1-2s instead of 60-300s on Ollama CPU).
    try:
        from core.config import Config as _Config
        _openai_off = bool(getattr(_Config, "DISABLE_OPENAI", False))
        _groq_key = getattr(_Config, "GROQ_API_KEY", None)
        _groq_model = getattr(_Config, "GROQ_MODEL", "llama-3.3-70b-versatile")
        _groq_base = getattr(_Config, "GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    except Exception:
        _openai_off = False
        _groq_key = None
        _groq_model = "llama-3.3-70b-versatile"
        _groq_base = "https://api.groq.com/openai/v1"

    if _openai_off and _groq_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            log.info("🟢 LLM classify : Groq activé (model=%s)", _groq_model)
            llm = ChatOpenAI(
                model=_groq_model,
                api_key=_groq_key,
                base_url=_groq_base,
                temperature=0.0, max_tokens=300, max_retries=0,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = (response.content or "").strip()
        except Exception as exc:
            log.warning(f"🟢 Groq classify failed: {exc} → fallback Ollama")
    elif _openai_off:
        log.info("LLM classify : DISABLE_OPENAI=true (et GROQ_API_KEY manquant) → skip OpenAI/Groq, Ollama only")
    else:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=300, max_retries=0)
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = (response.content or "").strip()
        except Exception as exc:
            log.debug(f"OpenAI classify failed or rate-limited: {exc}")

    # 4b. Fallback to Ollama if OpenAI not available or returned nothing
    ollama_error: str | None = None
    if not response_text:
        try:
            import requests
            # Ping with a small timeout — this is just an availability
            # check, not the classification call. If Ollama isn't even
            # reachable we want to know fast and fall through.
            ping = requests.get("http://localhost:11434/api/tags", timeout=3)
            if ping.status_code == 200:
                # Ollama expects model parameters inside ``options`` (the
                # previous payload had them at top level, so they were
                # silently ignored — the model used its defaults).
                _opts = {"temperature": 0.0, "num_predict": 300}
                try:
                    from core.config import Config as _Cfg
                    _n_threads = int(getattr(_Cfg, "OLLAMA_NUM_THREADS", 0) or 0)
                    if _n_threads > 0:
                        _opts["num_thread"] = _n_threads
                except Exception:
                    pass
                payload = {
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": _opts,
                }
                # Classification call — NO timeout. Mistral on CPU can be
                # slow on first call (model load) or on contention; an
                # arbitrary cap was forcing valid classifications to fail
                # and the system to fall back to heuristics. Better to
                # wait than to mis-classify the course.
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=None,
                )
                if response.status_code == 200:
                    response_text = (response.json().get("response", "") or "").strip()
                else:
                    ollama_error = f"HTTP {response.status_code}"
            else:
                ollama_error = f"ping HTTP {ping.status_code}"
        except requests.exceptions.ConnectionError:
            ollama_error = "connection refused"
        except Exception as exc:
            ollama_error = f"{type(exc).__name__}: {str(exc)[:80]}"

    if not response_text:
        # Visible at INFO so the operator sees WHY classification was skipped.
        log.info(f"⏭️ LLM classify skip — Ollama: {ollama_error or 'no response'}")
        return None

    # ── 5. Parse JSON robuste (markdown fences + prefixes) ────────────────
    raw = response_text
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
        raw = raw.rstrip("`").strip()
    json_start = raw.find("{")
    json_end = raw.rfind("}")
    if json_start == -1 or json_end <= json_start:
        log.warning(f"LLM classify: pas de JSON dans la reponse: {raw[:200]}")
        return None
    try:
        result = json.loads(raw[json_start:json_end + 1])
    except json.JSONDecodeError as exc:
        log.warning(f"LLM classify: JSON parse error: {exc}")
        return None

    # ── 6. Sanitize (slug propre) ─────────────────────────────────────────
    domain_raw = (result.get("domain") or "").strip().lower().replace(" ", "_")
    course_raw = (result.get("course") or "").strip().lower().replace(" ", "_")
    domain = re.sub(r"[^a-z0-9_]", "", domain_raw) or DEFAULT_DOMAIN
    course = re.sub(r"[^a-z0-9_]", "", course_raw) or DEFAULT_COURSE
    title = (result.get("title") or "").strip() or course.replace("_", " ").title()
    reason = (result.get("reason") or "").strip()

    log.info(
        f"🤖 LLM classify → domain='{domain}', course='{course}', "
        f"title='{title}' (reason: {reason[:80]})"
    )
    return {"domain": domain, "course": course, "title": title, "reason": reason}


#
#  ✅ POUR AJOUTER UN COURS DYNAMIQUEMENT:
#
#  C'est très simple maintenant! Aucune modification de code n'est nécessaire.
#  
#  MÉTHODE 1: Ajouter un nouveau cours dans un domaine existant
#  ─────────────────────────────────────────────────────────────
#    mkdir -p "courses/informatique/Mon Nouveau Cours"
#    cp mon_pdf.pdf "courses/informatique/Mon Nouveau Cours/"
#
#    ✅ Le cours sera AUTOMATIQUEMENT découvert!
#    ✅ Les métadonnées seront AUTO-GÉNÉRÉES!
#    ✅ Les chapitres seront AUTOMATIQUEMENT listés!
#
#
#  MÉTHODE 2: Ajouter un nouveau domaine
#  ──────────────────────────────────────
#    mkdir -p courses/data_science/machine_learning
#    mkdir -p courses/data_science/statistics
#    cp mon_pdf.pdf courses/data_science/machine_learning/
#    
#  Puis modifier seulement la ligne DOMAINS:
#    DOMAINS = {
#        "informatique": [],
#        "data_science": [],  # ← Ajouter cette ligne
#    }
#
#    ✅ Tout le reste est automatique!
#
#
#  🎯 RÉSUMÉ:
#  ──────────
#  - Domaines & Cours: découverts depuis courses/
#  - Métadonnées: générées automatiquement
#  - Chapitres: détectés depuis les fichiers PDF
#  - AUCUNE configuration supplémentaire nécessaire!
#
# ═══════════════════════════════════════════════════════════════════════



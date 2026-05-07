# Smart Teacher

Assistant pédagogique vocal interactif basé sur **STT + RAG + LLM + TTS**.

Smart Teacher charge un cours depuis les fichiers du dossier `courses/`, extrait sa structure en chapitres et sections, puis le présente à voix haute avec interaction en temps réel par micro ou texte.

Le flux principal est volontairement direct : import des fichiers de cours -> construction de la structure -> présentation -> questions/réponses -> reprise depuis le point d’arrêt quand c’est possible.

---

## Vue d’ensemble

### Ce que fait l’application

1. L’utilisateur sélectionne un cours importé ou en cours de préparation.
2. L’application affiche la structure du cours et la slide courante.
3. Le professeur IA lit le contenu à voix haute.
4. L’étudiant peut poser une question, interrompre la présentation, puis reprendre.
5. Le système combine transcription, recherche documentaire, génération de réponse et synthèse vocale.

### Chaîne technique

```text
Micro / texte
   -> VAD Silero
   -> STT Whisper
   -> RAG Qdrant + BM25
   -> LLM OpenAI ou Ollama
   -> TTS Edge-TTS
   -> Réponse audio + chat
```

### Machine d’état

```text
IDLE -> PRESENTING -> LISTENING -> PROCESSING -> RESPONDING
```

---

## Fonctionnalités principales

- Présentation de cours par chapitres et sections.
- Interaction vocale en temps réel via WebSocket.
- Réponses textuelles et audio.
- Transcription affichée dès la détection STT.
- Interruption immédiate quand l’étudiant pose une question.
- Reprise du cours depuis le point sauvegardé quand c’est possible.
- Métriques de performance pour STT, LLM, TTS et temps total.
- Fallback local via Ollama si OpenAI est indisponible ou en quota insuffisant.

---

## Prérequis

- Python 3.13+
- Docker et Docker Compose
- Une clé `OPENAI_API_KEY` dans un fichier `.env`
- Optionnel : `ELEVENLABS_API_KEY` si vous utilisez ElevenLabs pour le TTS

### Services Docker

#### Minimum

- PostgreSQL 15
- Redis 7
- Qdrant

#### Profile complet (`--profile full`)

- MinIO
- Elasticsearch
- ClickHouse
- Ollama

---

## Installation rapide

### 1. Créer et activer l’environnement Python

```powershell
envProject\Scripts\Activate
```

### 2. Installer les dépendances

Le dépôt fournit maintenant un fichier `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
```

Vous pouvez aussi utiliser l’environnement Python déjà préparé (`envProject`) si vous voulez éviter de recréer un environnement local.

### 3. Préparer le fichier `.env`

```bash
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=
STT_BACKEND=faster-whisper
TTS_PROVIDER=edge
GPT_MODEL=gpt-4o-mini
RAG_EMBEDDING_MODEL=BAAI/bge-m3
```

Pour activer le backend STT optionnel basé sur WhisperLiveKit, réglez `STT_BACKEND=whisperlivekit` dans le même fichier `.env`.

### 4. Démarrer l’infrastructure

```bash
docker-compose up -d
```

Si vous voulez les services optionnels :

```bash
docker-compose --profile full up -d
```


On a lancé cette commande pour **démarrer toute l’infrastructure backend de ton projet Smart Teacher automatiquement**, avec **Docker Compose**.
Au lieu d’installer chaque service séparément sur Windows, Docker les exécute dans des conteneurs isolés.

### 🔍 Décomposition :

### 1️⃣ `docker-compose`

Outil qui lit un fichier `docker-compose.yml` et démarre plusieurs services ensemble.

Dans ton projet, il contient probablement :

* PostgreSQL
* Redis
* Qdrant
* Elasticsearch
* Ollama
* MinIO
* ClickHouse

---

### 2️⃣ `--profile full`

Cela veut dire :

👉 lancer le **profil complet** du projet.

Souvent il existe :

* `minimal` → services essentiels seulement
* `dev` → développement
* `full` → tous les composants avancés

Donc ici tu veux la **version complète Smart Teacher**.

---

### 3️⃣ `up`

Créer + démarrer les conteneurs.

---

### 4️⃣ `-d`

Mode détaché :

👉 Les services tournent en arrière-plan.

Tu récupères directement la main dans le terminal.

Sans `-d`, les logs restent affichés.

---

# 📦 Pourquoi Smart Teacher a besoin de tout ça ?

Le dashboard "Services" affiche les briques réellement utilisées par le backend, avec leur rôle, ce qu'elles stockent ou récupèrent, et les chemins de code qui les appellent.

| Service | Ce qu'il stocke / récupère | Code principal |
|---------|----------------------------|----------------|
| PostgreSQL | sessions, interactions, profils étudiants, logs, événements d'apprentissage | [database/models.py](database/models.py), [database/init_db.py](database/init_db.py), [main.py](main.py), [modules/dashboard.py](modules/dashboard.py) |
| Redis | sessions WebSocket, état temporaire, cache de travail, latence de traitement | [handlers/session_manager.py](handlers/session_manager.py), [modules/llm.py](modules/llm.py), [main.py](main.py) |
| Qdrant | embeddings et chunks vectorisés du cours pour la recherche RAG | [modules/multimodal_rag.py](modules/multimodal_rag.py), [main.py](main.py) |
| Elasticsearch | historique des transcriptions et recherche full-text | [modules/transcript_search.py](modules/transcript_search.py), [main.py](main.py) |
| Ollama | modèles locaux et réponses LLM de secours | [modules/llm.py](modules/llm.py), [main.py](main.py) |
| MinIO | PDF, slides, audio, objets média et URLs temporaires | [modules/media_storage.py](modules/media_storage.py), [main.py](main.py), [modules/course_builder.py](modules/course_builder.py) |
| ClickHouse | agrégations analytiques et métriques de session | [modules/analytics.py](modules/analytics.py), exports dashboard |

Notes pratiques :

- `Qdrant` alimente la carte `Qdrant / RAG` du dashboard, avec l'état du vectorstore et le nombre de documents indexés.
- `MinIO` est optionnel. Si `MINIO_ENDPOINT` n'est pas défini, Smart Teacher utilise le stockage local dans `media/` et le dashboard affiche `Local`.
- `Elasticsearch` peut tomber en fallback mémoire si le service n'est pas disponible.

---

# 🎯 Pourquoi lancer tout ça ?

Parce que Smart Teacher semble vouloir faire :

✅ Chat IA
✅ RAG sur documents
✅ Upload PDF/audio
✅ Analyse étudiants
✅ Temps réel websocket
✅ Historique sessions
✅ LLM local
✅ Dashboard analytique

Donc architecture sérieuse.

---

# 🧠 Pourquoi Docker est idéal ici ?

Sans Docker tu devrais installer :

* PostgreSQL
* Redis
* Qdrant
* Java pour ES
* MinIO
* Ollama
* ClickHouse

Avec conflits versions + ports + config.

Docker règle ça.

---

### 5. Lancer l’application

```bash
python main.py
```


### 6. Ouvrir l’interface

```text
http://localhost:8000/static/index.html
```

---


Les principales variables sont définies dans `config.py`.

| Variable | Rôle |
|----------|------|
| `OPENAI_API_KEY` | Clé OpenAI utilisée pour le LLM, et pour les embeddings si vous basculez vers un modèle OpenAI |
| `ELEVENLABS_API_KEY` | Clé optionnelle si `TTS_PROVIDER=elevenlabs` |
| `POSTGRES_HOST` / `POSTGRES_PORT` | Connexion PostgreSQL |
| `REDIS_HOST` / `REDIS_PORT` | Connexion Redis |
| `QDRANT_HOST` / `QDRANT_PORT` | Connexion Qdrant |
| `GPT_MODEL` | Modèle OpenAI utilisé par défaut |
| `RAG_EMBEDDING_MODEL` | Modèle d'embedding RAG, défaut local `BAAI/bge-m3` |
| `WHISPER_MODEL_SIZE` | Taille du modèle STT |
| `STT_BACKEND` | Backend STT (`faster-whisper` par défaut ou `whisperlivekit`) |
| `TTS_PROVIDER` | Moteur TTS (`edge` ou `elevenlabs`) |
| `COURSES_DIR` | Dossier racine des cours |
| `SERVER_PORT` | Port HTTP du serveur |

---

## API exposée

### WebSocket

| Route | Rôle |
|-------|------|
| `WS /ws/{session_id}` | Pipeline vocal temps réel |

### REST

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | État général du serveur |
| `GET` | `/health` | Vérification de santé |
| `POST` | `/session` | Création / initialisation d’une session |
| `GET` | `/session/{session_id}` | Lecture d’une session |
| `POST` | `/session/clear` | Réinitialisation d’une session |
| `POST` | `/ask` | Question texte -> réponse audio |
| `POST` | `/course/build` | Construction d’un cours depuis des fichiers |
| `POST` | `/ingest` | Ingestion manuelle / indexation |
| `GET` | `/course/list` | Liste des cours disponibles |
| `GET` | `/course/{course_id}/structure` | Structure complète d’un cours |
| `GET` | `/rag/stats` | Statistiques RAG |
| `GET` | `/cache/stats` | Statistiques cache |
| `GET` | `/ingestion/status` | Avancement de l’ingestion |
| `GET` | `/debug/rag_test` | Test RAG avec requête personnalisée |

---

## Structure du projet

```text
smart_teacher/
├── main.py              # Serveur FastAPI principal
├── config.py            # Configuration centralisée
├── analyze_metrics.py   # Analyse des métriques
├── handlers/            # Pipeline audio et gestion de session
├── modules/             # STT, LLM, TTS, RAG, analytics, etc.
├── database/            # Modèles, CRUD, init DB, SQL
├── courses/             # Cours importés
├── static/              # Interface web
├── media/               # Audio, PDFs, slides
├── data/                # Caches et données locales
├── logs/                # Métriques et logs CSV
├── analytics/           # Exports d’événements
├── docker-compose.yml   # Infrastructure locale
├── Dockerfile           # Image applicative
└── init_ollama.sh       # Initialisation Ollama locale
```

---

## Commandes utiles

### Démarrage / arrêt

```bash
docker-compose down
docker-compose down -v
```

### Base de données

```bash
python -m database.init_db
```

### Analyse des métriques

```bash
python analyze_metrics.py
python analyze_metrics.py --stt
```

### Ingestion / indexation

Le dépôt actuel n’inclut pas de script `ingest.py`.

Le flux d’import passe par l’API :

- `POST /course/build` pour construire un cours à partir de fichiers importés.
- `POST /ingest` pour une ingestion / indexation manuelle si besoin.

Si vous automatisez l’import, appelez directement ces endpoints depuis un script ou un client HTTP.

---

## Notes d’exploitation

- Le système privilégie l’upload direct et la présentation du cours à partir des fichiers importés.
- Si OpenAI retourne `insufficient_quota`, l’application peut basculer sur Ollama quand il est disponible.
- Les métriques sont écrites dans `logs/` et `analytics/`.
- L’interface principale est `static/index.html`.
- Les services optionnels du profile complet servent au stockage média, à la recherche et à l’analytics.

---

## Vision produit

Smart Teacher est pensé comme un professeur IA capable de suivre un cours importé sans réécrire la matière à partir de zéro.

Le flux cible reste direct : import du cours -> structuration -> présentation vocale -> questions -> reformulation -> quiz -> synthèse.

Les capacités visées par le projet sont les suivantes :

- Présentation automatique du cours avec synchronisation des slides.
- Détection des signes de confusion et adaptation du rythme.
- Reformulation pédagogique quand l’étudiant bloque.
- Quiz courts et révision espacée pour consolider l’apprentissage.
- Résumé final du chapitre et analytique de session.

Une partie de cette logique existe déjà dans le code, notamment la détection de confusion, les signaux prosodiques, la reformulation adaptative, la journalisation des événements et les caches de résumé.

**Architecture agentique** : Smart Teacher intègre désormais un pipeline RAG agentique inspiré de projets open source comme GiovanniPasq/agentic-rag-for-dummies, avec un reviewer LLM pour l'évaluation automatique des réponses, un routeur d'agents pour aiguiller les requêtes (quiz, code, RAG), et une répétition espacée FSRS.

---

## Références GitHub

Les ressources suivantes servent d’inspiration architecturale et fonctionnelle pour la suite du projet :

- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) pour le STT temps réel et les flux live.
- [RAG-AI-Voice-assistant](https://github.com/Adii2202/RAG-AI-Voice-assistant) pour le dialogue vocal RAG.
- [AI-Teaching-Assistant](https://github.com/goldmanau/AI-Teaching-Assistant) pour l’orchestration tutorielle.
- Adaptive-RAG pour la récupération adaptative et les stratégies de réponse.
- [fsrs4anki](https://github.com/open-spaced-repetition/fsrs4anki) pour la répétition espacée.
- LECTOR pour les idées de détection de confusion et d’état étudiant.
- [asinghcsu/AgenticRAG-Survey](https://github.com/asinghcsu/AgenticRAG-Survey) pour la taxonomie académique des systèmes RAG agentiques (CRAG, Adaptive RAG, Self-RAG).

Ces références ne remplacent pas l’implémentation locale, mais elles donnent une cible claire pour faire évoluer Smart Teacher vers un tuteur plus adaptatif.

---

## Feuille de route recommandée

1. ✅ Stabiliser le flux cours -> présentation -> questions -> reprise.
2. ✅ Renforcer la détection de confusion et la reformulation contextuelle (avec feedback négatif et BERT).
3. ✅ Ajouter un vrai moteur de quiz post-leçon (amélioré avec prompts pédagogiques).
4. ✅ Exporter les résumés et les métriques de progression.
5. ✅ Introduire la répétition espacée sur les notions mal comprises (FSRS).
6. ✅ Implémenter un pipeline RAG agentique (LangGraph-like) avec auto-correction.
7. ✅ Ajouter un reviewer LLM pour l'évaluation automatique des réponses.
8. ✅ Intégrer un routeur d'agents (quiz, code, RAG) dans la pipeline audio.
9. ✅ Ajouter un agent code executor en sandbox pour les questions de programmation.
10. 🔄 Créer un mode offline complet (whisper.cpp + llama.cpp + Piper) pour déploiement autonome.

---

## Dépannage rapide

- Vérifier que PostgreSQL, Redis et Qdrant sont bien démarrés.
- Vérifier la présence de `OPENAI_API_KEY` dans `.env`.
- Si le serveur refuse de démarrer, libérer le port 8000 puis relancer `python main.py`.
- Si le fallback local est attendu, vérifier que Ollama est actif.

### Libérer le port 8000

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Sous PowerShell, vous pouvez aussi tuer directement le ou les processus qui écoutent sur 8000 :

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
```
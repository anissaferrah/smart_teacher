# Personnalisation pédagogique adaptative — Modèle théorique et implémentation

> Document de référence pour le composant de personnalisation de Smart Teacher.
> Niveau : M2 / thèse de spécialisation IA pédagogique.

## 1. Cadre théorique

### 1.1 Modèles de styles d'apprentissage retenus

Smart Teacher s'appuie sur le modèle **VARK** de Fleming & Mills (1992)¹ — quatre
dimensions perceptives :

| Code | Style | Préférence dominante |
|------|-------|---------------------|
| **V**isual | Visuel | Diagrammes, schémas, cartes mentales |
| **A**uditory | Auditif | Discours, dialogue, répétition vocale |
| **R**eading | Lecture/écriture | Texte écrit, prise de notes |
| **K**inesthetic | Kinesthésique | Pratique, manipulation, exemples concrets |

Choix justifié :
- Validation psychométrique (test-retest r ≈ 0.79 sur 4 dimensions, Fleming 1995)
- Couvre la majorité des préférences d'apprentissage des étudiants en sciences
- Compatible avec l'instrumentation de Smart Teacher (audio, texte, slides, quiz)

Modèles complémentaires considérés mais non implémentés (à ce stade) :
- **Felder-Silverman**² : 4 axes bipolaires (actif/réflexif, sensoriel/intuitif,
  visuel/verbal, séquentiel/global). Plus riche, plus complexe à instrumenter.
- **Kolb's experiential learning**³ : 4 modes (concret/abstrait × actif/réflexif).
  Centré sur le cycle d'apprentissage plus que sur la modalité.

### 1.2 Cadre cognitif

- **Zone proximale de développement** (Vygotsky, 1978)⁴ : enseigner légèrement
  au-dessus de la maîtrise actuelle, avec scaffolding.
- **Cognitive Load Theory** (Sweller, 1988)⁵ : adapter la charge cognitive
  selon l'expertise (novice → simplification, expert → terminologie technique).
- **Mastery learning** (Bloom, 1968)⁶ : seuil de maîtrise (≥ 0.85) avant
  passage au concept suivant — implémenté dans `pedagogy/skill_tree.py`.

## 2. Modèle probabiliste

### 2.1 Inférence Bayésienne (Dirichlet-Multinomial)

Soit la distribution a priori de la dimension d'apprentissage de l'étudiant `s` :

```
P(θ_s) = Dirichlet(α₀)
```

avec `α₀ = (α_V, α_A, α_K, α_R) = (2, 2, 2, 2)` (prior uniforme symétrique
faiblement informatif — équivalent à 8 observations virtuelles).

Chaque signal observé `x ∈ {audio_question, text_question, practice_attempt,
interrupt_during_slide, vark_*}` ajoute des pseudo-counts pondérés :

```
α_post[i] = α_prior[i] + Σ_{x ∈ Obs} w_x[i]
```

où `w_x` est le vecteur de poids du signal `x` (cf. `SIGNAL_WEIGHTS` dans
`services/learning_style_bayes.py`). La distribution a posteriori reste
Dirichlet (conjugaison) :

```
P(θ_s | Obs) = Dirichlet(α_post)
```

**Espérance a posteriori** (le score communiqué à l'étudiant) :

```
E[θ_s,i | Obs] = α_post[i] / Σⱼ α_post[j]
```

**Intervalle de crédibilité 95%** (Highest Density Interval) sur chaque
dimension marginale, approximé par la marginale Beta :

```
θ_s,i | Obs ~ Beta(α_post[i], Σⱼ α_post[j] - α_post[i])
```

### 2.2 Pourquoi pas un simple comptage normalisé ?

Le comptage naïf (heuristique v1, conservé dans `services/learning_style.py`)
souffre de :

| Problème | Heuristique | Bayes |
|----------|-------------|-------|
| Cold start (nouvel étudiant) | Score = 0/0 indéterminé | Prior uniforme |
| Self-report incompatible avec comportement | Aucun cadre | Cross-validation cosine |
| Communication d'incertitude | Score ponctuel | HDI 95% |
| Mise à jour incrémentale | Recompute O(N) | O(1) |
| Robustesse aux outliers | Sensible | Lissage par prior |

### 2.3 Cross-validation behavior vs self-report

À chaque mise à jour, on compare la posterior comportementale aux pseudo-counts
issus du questionnaire VARK :

```
sim(behavioral, self_report) = cos(θ_behavioral, θ_self_report)
```

- Cosine ≥ 0.85 → bonne concordance, modèle validé pour cet étudiant
- Cosine < 0.5 → divergence — peut indiquer self-report biaisé (désirabilité
  sociale) ou comportement non-typique. Endpoint `/student/me/learning-style/cross-validate`.

## 3. Pipeline d'adaptation

### 3.1 Boucle complète

```
Comportement       Détection            Adaptation              Sortie
─────────          ─────────            ──────────              ──────
audio_question  →  Bayes update     →  posterior.dominant() →  TTS rate (-30..+30%)
practice_attempt → save_posterior   →  style_to_prompt_hint →  Inject in LangGraph state
                                     →  Narrator system_prompt → Voice TTS adaptée
```

### 3.2 Adaptations actives

| Dimension | Style visuel | Style auditif | Style lecture | Style kinesthésique |
|-----------|--------------|---------------|---------------|---------------------|
| Prompt LLM | Analogies visuelles, schémas mentaux | Ton conversationnel, rythme | Structure écrite, définitions | Exemples pratiques |
| TTS rate | Selon `pace` profil | Standard | Lecture plus lente | Standard |
| Confusion (×3) | + analogie visuelle | + reformulation orale | + définition écrite | + exemple numérique |

### 3.3 Adaptations envisagées (non implémentées)

- **Sélection RAG par modalité** : favoriser chunks contenant figures pour visuel, sections de définition pour reading
- **Sequencing** : alterner les exemples (kinesthésique-friendly) et la théorie (reading-friendly)
- **Multimodal slides** : déclencher la génération d'image (Vision LLM) si style = visuel

## 4. Validation et évaluation

### 4.1 KPIs cibles

| KPI | Cible | Mesure |
|-----|-------|--------|
| Concordance behavioral/VARK | cosine ≥ 0.7 | `cross_validate()` |
| Confidence après 30 interactions | ≥ 0.6 | `posterior.confidence()` |
| Engagement (durée session) | +15% vs groupe contrôle | A/B test (à mettre en place) |
| Learning gain (post - pre) | +10% vs contrôle | Quiz pre/post + corrélation avec mastery |

### 4.2 Méthodologie de validation expérimentale (à venir)

1. **A/B test** : 50% étudiants en mode adaptatif (style hint injecté), 50% en
   mode contrôle (prompt générique). Stratification par niveau initial.
2. **Métriques** : learning gain (Hake's normalized gain), engagement (durée
   moyenne par session), satisfaction (échelle Likert 5 items).
3. **Test statistique** : t-test apparié sur learning gain. Significativité
   p < 0.05, effet minimum d = 0.3 (small-to-medium).

## 5. Architecture technique

### 5.1 Modules

```
services/
├── learning_style.py            # Heuristique v1 (legacy, conservé)
├── learning_style_bayes.py      # Modèle Bayes Dirichlet (production)
└── vark_questionnaire.py        # Instrument psychométrique VARK 8 items

handlers/
├── auth.py                      # JWT + bcrypt + rate limit + audit log
└── ws.py                        # WebSocket — extrait student_id du JWT

routes/
├── auth.py                      # /auth/{register,login,me,logout}
├── student.py                   # /student/me/* + /vark/* + /learning-style/*
└── admin.py                     # /admin/students/* (requires teacher/admin role)

agentic/teaching/
└── narrator.py                  # Lit state["learning_style_hint"] → STYLE_BLOCK
```

### 5.2 Persistance

- `students` table : credentials + role (account_level)
- `student_profiles.preferences` (JSONB) : `{bayes_style: {alpha, n_observations}, vark_self_report: {V, A, K, R}}`
- `student_profiles.learning_style` : champ dénormalisé pour requêtes rapides

### 5.3 Endpoints publics (résumé)

```
POST /auth/register             — bcrypt hash, password strength check
POST /auth/login                — rate-limited (5 tentatives / 5 min)
POST /auth/logout
GET  /auth/me                   — JWT-protected

GET  /vark/questionnaire        — 8 items (FR/EN)
POST /student/me/vark/submit    — seed Bayes prior + persist self-report
GET  /student/me/learning-style/bayes        — posterior + HDI + confidence
POST /student/me/learning-style/bayes/rebuild — force recompute from history
GET  /student/me/learning-style/cross-validate — behavior vs self-report
```

## 6. Sécurité et conformité

- Mots de passe : bcrypt rounds=12 (résistance brute force)
- JWT : HS256, expiration 24h, secret key via env var `JWT_SECRET_KEY`
- Cookies : httponly + samesite=lax (résiste XSS / CSRF basique)
- Rate limiting : Redis sliding window (5 fails / 5 min)
- Audit log : `logs/sec_audit.csv` (timestamp, event, actor, ip, outcome)
- Password strength : min 8 chars, alpha + digit, exclusion top-N common passwords

À ajouter pour conformité RGPD :
- Fields `consent_given`, `consent_date` dans `students`
- Endpoint `GET /student/me/data-export` (JSON dump complet)
- Endpoint `DELETE /student/me/account` (soft delete + 30j de rétention)
- Politique de confidentialité accessible

## 7. Limites connues

1. **Cold start** : avant 5+ interactions, le score est dominé par le prior.
   Mitigation : VARK obligatoire au premier login.
2. **Stabilité du style** : les préférences évoluent — le modèle ne capture pas
   les drifts long-terme. Mitigation : window glissante 90 jours.
3. **Cohérence inter-domaines** : un étudiant peut être visuel en math, auditif
   en histoire. Modèle actuel agnostique du domaine. Mitigation future :
   posterior par (student × course) au lieu de (student) global.
4. **Self-report désirabilité sociale** : VARK auto-déclaratif est sensible à
   la réponse "que je devrais donner". Cross-validation atténue mais n'élimine
   pas. Mitigation : pondération du self-report < observations comportementales.

## 8. Références

¹ Fleming, N. D., & Mills, C. (1992). *Not another inventory, rather a catalyst
   for reflection.* To Improve the Academy, 11(1), 137–155.

² Felder, R. M., & Silverman, L. K. (1988). *Learning and teaching styles in
   engineering education.* Engineering Education, 78(7), 674–681.

³ Kolb, D. A. (1984). *Experiential learning: Experience as the source of
   learning and development.* Prentice-Hall.

⁴ Vygotsky, L. S. (1978). *Mind in society: The development of higher
   psychological processes.* Harvard University Press.

⁵ Sweller, J. (1988). *Cognitive load during problem solving: Effects on
   learning.* Cognitive Science, 12(2), 257–285.

⁶ Bloom, B. S. (1968). *Learning for mastery.* Evaluation Comment, 1(2).

⁷ Corbett, A. T., & Anderson, J. R. (1995). *Knowledge tracing: Modeling the
   acquisition of procedural knowledge.* User Modeling and User-Adapted
   Interaction, 4(4), 253–278.

⁸ Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall.
   Chapter 3 — Single-parameter models. Chapter 5 — Hierarchical models.

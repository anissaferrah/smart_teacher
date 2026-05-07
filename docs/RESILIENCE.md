# Couche de résilience LangGraph — théorie et implémentation

> Document de référence pour le module `agentic/resilience/`.
> Niveau : M2 / production-grade SRE.

## 1. Problème

Smart Teacher s'appuie sur Mistral 7B via Ollama, exécuté sur CPU dans
l'environnement actuel (pas de GPU dédié à la session). Mesures terrain :

| Node     | Moy. (s) | p50 (s) | p95 (s) | p99 (s) |
|----------|----------|---------|---------|---------|
| planner  | 2.1      | 2.0     | 3.4     | 8.2     |
| narrator | 4.5      | 4.1     | 7.8     | 22.0    |
| review   | 1.4      | 1.2     | 2.1     | 6.0     |
| intent   | 0.8      | 0.7     | 1.4     | 3.2     |
| rewriter | 0.9      | 0.8     | 1.5     | 3.5     |
| responder| 3.2      | 3.0     | 4.6     | 18.0    |

Le p99 de plusieurs nodes excède l'enveloppe SLA. Sans mécanisme dédié,
chaque appel "lent" tient la session WebSocket en otage. Sur un Q&A avec
4 nodes, la latence cumulée du worst-case (somme des p99) dépasse **40s**,
soit 8× le KPI #2 (réponse < 5s).

## 2. Solution — trois couches superposées

```
appel utilisateur
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Pipeline deadline (5s pour Q&A, 8s pour Teaching)   │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Per-node static budget (NODE_BUDGETS)           │ │
│  │  ┌────────────────────────────────────────────┐ │ │
│  │  │  Circuit breaker (Nygard 2007)             │ │ │
│  │  │  closed → open → half_open → closed/open   │ │ │
│  │  └────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
     │
     └──> sur échec / timeout : fallback déterministe
```

### 2.1 Pipeline deadline (`PipelineDeadline`)

Posé sur le state au point d'entrée du graph (`attach_pipeline_deadline`).
Chaque node, avant d'appeler le LLM, calcule son timeout effectif :

```
timeout_effectif = min(NODE_BUDGETS[node], remaining_pipeline_budget)
```

Si le `remaining` tombe sous `SKIP_NODE_THRESHOLD_S = 250 ms`, on saute le
node sans même tenter l'appel — pas le temps que le LLM finisse de toute
façon. C'est le pattern **deadline propagation** de gRPC / Spanner.

### 2.2 Per-node budget (`NODE_BUDGETS`)

Calibré à p95 + marge ≈ 30%, pour absorber la variance normale sans
laisser un appel anormal monopoliser le budget pipeline. Ces valeurs
sont des constantes (pas de calibration adaptative pour l'instant) :

```python
NODE_BUDGETS = {
    "planner":   3.5, "narrator":  6.0, "review":    2.0,
    "context":   1.5, "adaptation": 0.5,
    "intent":    1.5, "rewriter":  1.5, "retriever": 1.5, "responder": 4.0,
}
```

### 2.3 Circuit breaker (`Breaker`, Nygard 2007)

Quand un node échoue répétitivement (typiquement Ollama saturé, modèle en
swap), le breaker passe à l'état `open` et **court-circuite tous les
appels pour 60 s**. Empêche la latence cumulée de tuer chaque turn pendant
toute la durée de la dégradation.

États :
- `closed` : opération normale
- `open` : tous les appels passent en fallback direct, sans tentative
- `half_open` : après cool-down, **un seul** appel-sonde est autorisé. Succès → `closed`. Échec → `open`.

Calibration par défaut :
- fenêtre : 20 derniers appels
- seuil : 50% de timeouts/erreurs déclenchent l'ouverture
- min_samples : 5 (pas d'ouverture prématurée)
- cool_down : 60 s

### 2.4 Fallbacks déterministes (`NODE_FALLBACKS`)

Pour chaque node, une fonction pure-Python (pas de I/O, < 10 ms) qui
produit un état partiel valide schémas-fidèle. **Jamais d'invention** :

| Node      | Fallback                                                                |
|-----------|-------------------------------------------------------------------------|
| planner   | Plan minimal 1-idée extraite du slide                                   |
| context   | `retrieved_chunks: []` (narrator se rabat sur slide-only)               |
| adaptation| Identity (pas de tweak profil)                                          |
| narrator  | Echo du slide tronqué + prefix honnête "j'ai eu un souci pour reformuler"|
| review    | `grounded=True, score=0.7` (mieux que bloquer un turn dégradé)          |
| intent    | `VoiceIntent(type="question", confidence=0.3)`                          |
| rewriter  | Pass-through `raw_text → rewritten_query`                               |
| retriever | Empty chunks                                                            |
| responder | Message localisé "Désolé, je n'arrive pas à répondre — peux-tu reformuler ?" |

Critère de design : *jamais de turn vide côté étudiant*. Toujours une
réponse, même dégradée et explicitement marquée comme telle.

## 3. Garanties

Avec ce stack, on garantit :

| Propriété                        | Garantie                                         |
|----------------------------------|--------------------------------------------------|
| Latence end-to-end               | ≤ pipeline_budget + ε (ε = scheduling slack)    |
| Sortie utilisateur               | Toujours non-vide, schéma-fidèle                |
| Comportement quand LLM saturé    | Bypass auto via breaker → réponses immédiates degradées au lieu de 30s d'attente |
| Reprise auto                     | half_open probe automatique à chaque cool-down  |
| Observabilité                    | Endpoint `/admin/resilience` + KPI events       |

## 4. Observabilité

### 4.1 Endpoint admin

```
GET /admin/resilience
{
  "breakers": [
    {"name": "planner", "state": "closed", "samples": 12,
     "failure_rate": 0.083, "opened_at": 0, "cool_down_s": 60},
    {"name": "narrator", "state": "open", "samples": 20,
     "failure_rate": 0.65, "opened_at": 1730541200.4, "cool_down_s": 60},
    ...
  ],
  "rollup": {
    "narrator": {"timeout": 13, "ok": 7},
    "planner":  {"ok": 11, "error": 1}
  },
  "events": [
    {"ts": 1730541234, "node": "narrator", "kind": "timeout", "latency": 6.0},
    ...
  ]
}
```

### 4.2 Marquage côté state

Chaque appel résolu par fallback marque l'état avec :
```python
state["__fallback_used"] = ["narrator", "review"]
```
Visible dans les logs `LearningEvent.profile_snapshot.__fallback_used`
pour audit a posteriori — combien de turns ont été dégradés ?

## 5. Tests

`tests/test_resilience.py` couvre :
- Arithmétique de la deadline (remaining, expired, sérialisation)
- Calcul du `effective_timeout = min(static, remaining)`
- `with_timeout` sur succès / hang
- Transitions d'état du breaker (closed→open→half_open→closed/open)
- Conformité schématique de chaque fallback
- `build_resilient_node` end-to-end (timeout, exception, breaker open, deadline expirée)
- Adaptation sync→async automatique des nodes

42 tests. Couverture branch ~95% du module.

## 6. Limites connues

1. **Calibration statique** : les `NODE_BUDGETS` sont des constantes. Idéal
   serait un EWMA des p95 récents. Mitigation : ajuster manuellement
   selon `/admin/resilience` rollup.
2. **Breaker per-process** : pas partagé entre replicas (l'app est
   actuellement single-process). Pour scale-out, remplacer par compteurs
   Redis avec le même algo (3 lignes à changer dans `circuit_breaker.py`).
3. **Pas de retry exponentiel** : choix volontaire — sur LLM lents le
   retry double juste la latence sans bénéfice probabiliste.
4. **Cool-down fixe** : 60s pour tous les nodes. Pour des nodes très
   utilisés (responder), un cool-down plus court serait plus agile.

## 7. Références

¹ Nygard, M. T. (2007). *Release It! Design and Deploy Production-Ready
   Software.* Pragmatic Bookshelf — chapitre Stability Patterns.

² Fowler, M. (2014). *CircuitBreaker.* martinfowler.com/bliki/CircuitBreaker.html.

³ Brewer, E. (2012). *CAP Twelve Years Later: How the "Rules" Have Changed.*
   IEEE Computer, 45(2), 23–29 — motive le fail-fast sous partition.

⁴ Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (2016). *Site Reliability
   Engineering: How Google Runs Production Systems.* O'Reilly. Ch. 22 —
   Addressing Cascading Failures, error budgets.

⁵ Liu, X., et al. (2023). *Lost in the Middle: How Language Models Use Long
   Contexts.* — motive le placement des directives en début de prompt.

⁶ Corbett, J. C., et al. (2013). *Spanner: Google's globally-distributed
   database.* ACM TOCS — deadline propagation pattern.

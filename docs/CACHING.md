# Cache 2-niveaux du learning-style — théorie et implémentation

> Document de référence pour le module `services/learning_style_cache.py`.
> Niveau : M2 / production-grade.

## 1. Problème

À chaque appel de `services/presentation.py:_resolve_style()` (= 1 fois par
slide présentée), on exécute :

  1. `load_posterior(student_id)` — 1 SELECT Postgres avec déserialisation JSON
  2. Si `confidence < 0.30` : `compute_learning_style(student_id, days=30)` —
     5 SELECT COUNT(*) sur `Interaction`, `LearningEvent`, `PracticeAttempt`
  3. `style_to_prompt_hint(dominant)` + `style_to_params(dominant)` —
     deux dict lookups (négligeable)

Pour un cours de 50 slides, c'est **50× la même séquence DB** alors que la
posterior change tous les ~10 signaux comportementaux, soit ~10× toute une
session. Le ratio est ridicule. Mesures locales avec Postgres en réseau
local : ~80–150 ms par appel sur cold posterior, dominé par le SELECT
COUNT(*) du fallback heuristique.

## 2. Solution — cache 2 niveaux + single-flight + invalidation

### 2.1 Pourquoi 2 niveaux ?

Hennessy & Patterson (2017, *Computer Architecture*, ch. 2) montrent que la
combinaison **L1 rapide / faible capacité / scope étroit + L2 lent / haute
capacité / scope large** maximise le hit-ratio à coût mémoire constant. Ici :

| Niveau | Médium             | Latence hit | TTL  | Scope            | Capacité   |
|--------|--------------------|-------------|------|------------------|------------|
| L1     | Process dict       | ~50 µs      | 5min | Worker process   | 500 keys   |
| L2     | Redis              | ~500 µs     | 5min | Tous les workers | illimitée  |

Look-up : L1 → L2 → recompute. Sur L2 hit on rétro-populate L1 (warming).

### 2.2 Single-flight (coalescing)

Pattern *thundering herd* : N requêtes concurrentes pour la même clé en cold
cache → N recomputes redondants. Solution : registre `_inflight` mappant
clé → `asyncio.Future`. Le premier appelant devient leader, exécute le
slow path. Les autres awaitent le Future du leader.

Inspiration : `golang.org/x/sync/singleflight`, Bigtable read coalescing
(Chang et al., OSDI 2008), CDN cache-fill consolidation.

**Piège implementation** : ne JAMAIS `await` en tenant le `_inflight_lock`,
sinon les followers se sérialisent et le single-flight devient un mutex de
sérialisation au lieu d'un coalescing. C'est un bug d'une ligne qui annule
tout l'intérêt — couvert par `tests/test_concurrent_misses_coalesce`.

### 2.3 Invalidation event-driven

TTL pur + invalidation paresseuse offre une fraîcheur bornée par le TTL —
inacceptable ici (un signal comportemental majeur ne doit pas attendre 5 min
pour être visible côté narrator). Solution : invalidation push à chaque
mutation de la source de vérité.

| Événement                        | Action                                |
|----------------------------------|---------------------------------------|
| `save_posterior(sid, ...)`       | `invalidate(sid)` — drop L1 + L2      |
| `submit_vark_responses` (re-take)| `save_posterior` est appelé en aval, donc invalidation transitive |

Cao & Liu (2002, *IEEE TKDE*) — l'invalidation push donne une consistance
plus forte que le polling périodique pour des données rarement modifiées
mais lues souvent — exactement notre profil.

### 2.4 Observabilité

`/admin/cache/stats` retourne les compteurs process-local :

```json
{
  "counters": {
    "hits_l1":       1247,
    "hits_l2":       18,
    "misses":        12,
    "single_flight": 4,
    "invalidations": 6,
    "errors":        0
  },
  "total":     1277,
  "hit_ratio": 0.99,
  "l1_ratio":  0.977
}
```

**Cibles** après warm-up (≥ 100 requêtes) :
- `hit_ratio ≥ 0.90` (sinon le TTL est trop court ou l'invalidation trop
  agressive)
- `l1_ratio ≥ 0.85` (sinon Redis est sollicité plus que nécessaire — peut
  signaler un L1 mal dimensionné, mais 500 entries × 4 langues × ~4 KB
  ≈ 8 MB suffisent largement pour notre scale)
- `errors == 0` ou très bas (les erreurs Redis sont gracefully dégradées
  mais devraient rester rares en production)

## 3. Garanties

| Propriété               | Garantie                                          |
|-------------------------|---------------------------------------------------|
| Latence hit L1          | ~50 µs (dict lookup + ttl check)                  |
| Latence hit L2          | ~500 µs (Redis GET + JSON parse)                  |
| Latence miss            | équivaut à `_resolve_style()` original (≈80–150ms) |
| Concurrent miss (N tasks)| 1 recompute (single-flight)                      |
| Fraîcheur après update  | < 10 ms (push invalidation)                       |
| Comportement si Redis HS| L2 dégradé silencieusement, L1 + recompute restent|

## 4. Limites connues

1. **Stats process-local** — pour multi-replica il faudrait centraliser
   (Redis HINCRBY ou Prometheus). Actuel : un endpoint stats par worker.
2. **L1 eviction policy** — éviction du plus proche-de-l'expiration (cheap),
   pas LRU strict. À cette taille (500) la différence est négligeable.
3. **L2 invalidation langue-aware** — on supprime les clés `fr` et `en`
   explicitement, pas via `KEYS pattern` (lent sur grosse DB Redis). Si on
   ajoute une langue, ajouter dans `_redis_invalidate`.
4. **Pas de cache persistance L1 entre redémarrages** — c'est volontaire
   (le cold-start coût est plat ~80ms), pas un grief.

## 5. Tests

`tests/test_learning_style_cache.py` couvre :

- L1 put/get, séparation par langue, expiration TTL, eviction LRU-like
- Single-flight : N tâches concurrentes pour même clé → 1 recompute
- Pas de coalescing entre clés différentes
- Recompute échoue → return None (pas de cache pollution)
- Invalidation drop L1 + L2
- Stats counters incrémentent correctement

15 tests, ~22s d'exécution (dominé par les awaits asyncio).

## 6. Références

¹ Hennessy, J. L., & Patterson, D. A. (2017). *Computer Architecture: A
   Quantitative Approach* (6th ed.). Morgan Kaufmann. Ch. 2 — multi-level
   cache hierarchy and inclusion property.

² Chang, F., et al. (2008). *Bigtable: A Distributed Storage System for
   Structured Data.* Proc. OSDI '08 — read coalescing pattern.

³ Cao, P., & Liu, C. (2002). *Maintaining Strong Cache Consistency in the
   World Wide Web.* IEEE TKDE — invalidation push vs TTL trade-off.

⁴ Schmidt, T. M. *singleflight package documentation.* golang.org/x/sync/
   singleflight — canonical implementation of the coalescing pattern.

⁵ Karger, D., et al. (1997). *Consistent Hashing and Random Trees:
   Distributed Caching Protocols for Relieving Hot Spots on the World Wide
   Web.* Proc. STOC '97 — pertinent pour scaling le L2 multi-Redis si
   besoin futur.

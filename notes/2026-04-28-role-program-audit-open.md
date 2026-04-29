# Audit del programa de role markers (#38 + #39) -- thread abierto

Fecha: 2026-04-28 EOD
Sesión: cierre con duda sin resolver. Re-abrir en sesión limpia.

## Lo que se cerró en la sesión

- **PR #43 MERGED** (commit `661ae9b`): production wrapper `merken/role_classifier.py`, V2 SCR few-shot prototype classifier, 19 unit tests + E2E smoke con BGE-small.
- **PR #45 MERGED** (commit `7afd2b2`): vstash 0.35.0 API drift fix (`fts_only` -> `retrieval_mode`, `store_meta.embedding_model` ahora se escribe en init). Suite 401 -> 401 verde.
- **PR #46 OPEN**: `Memory(role_classifier=...)` opt-in tagging + `classify_role()` helper. 7 integration tests pasan, suite 408 verde. **No mergeado todavía** -- pendiente decisión sobre domain-restriction.

## Hallazgo verificado en #39

El eval de #39b se midió en distribución idéntica a los prototipos:

| | n | char min | mediana | max |
|---|---|---|---|---|
| Prototipos V2 | 40 | 51 | 74 | 108 |
| Eval V2 | 80 | 52 | 68 | 100 |
| LoCoMo per-session (ingesta real) | 272 | 1592 | 2808 | 5125 |
| LME oracle per-turn (ingesta real) | 10960 | 39 | 482 | 3933 |

Probe contra el clasificador V2 SCR aplicado al shape de ingesta real:
- LoCoMo per-session (n=100): 100% bajo el threshold "uncertain" (< 0.05). Distribution: 54% preference, 35% investigation, 10% SCR, 1% observation.
- LME per-turn (n=100): 93% bajo threshold. 62% preference, 38% investigation, 0% SCR, 0% observation.

**Conclusión empírica para #39**: el "STRONG_PROCEED gate met across 3 seeds" se midió **dentro de distribución**. La 0.929 macro-F1 no se trasladó a las unidades de ingesta de los benchmarks LME/LoCoMo. Domain-restriction es necesaria; alcance del classifier en su forma actual está acotado al shape sentence-level ops English que matchea los prototipos.

## Hallazgo NO concluyente en #38 (thread abierto)

#38 / #38b se midió sobre 3 escenarios (`knowledge_update_hard`, `products`, `medical`) todos con eventos sentence-level (mediana 86-97 chars). Mismo shape leakage que #39.

Probe en sesión: aplicar `cluster_metrics` de #38 (PR, EVR_1, AngDisp, NormVar) a 50 clusters random de N=4 en tres pools (200 vectores cada uno) embedidos con BGE-small:

```
metric                       short_events               lme_turn         locomo_session
pr                    2.915 [1.896-2.984]    2.905 [2.689-2.984]    2.856 [2.231-2.988]
evr_1                 0.407 [0.359-0.617]    0.408 [0.363-0.483]    0.428 [0.356-0.590]
ang_disp              1.332 [1.286-1.333]    1.332 [1.326-1.333]    1.331 [1.318-1.333]
norm_var              0.001 [0.000-0.016]    0.001 [0.000-0.005]    0.001 [0.000-0.005]
baseline_cos_mean     0.524 [0.460-0.634]    0.466 [0.392-0.647]    0.688 [0.596-0.778]
```

Las métricas residuales (PR, EVR_1, AngDisp, NormVar) salen casi idénticas en los 3 shapes. Solo `baseline_cos_mean` cambia con shape. Una lectura interpretativa propuesta en sesión: "la geometría residual a N=4/d=384 está dominada por la dimensionalidad, no por el contenido, así que el NO_GO de #38 sobrevive a cualquier shape."

**Jay lo cuestiona y NO ha aceptado esa conclusión.** Razones para dudar:
- La probe usa clusters **random**, no la construcción pos/neg de #38 (clusters topic-coherentes con knowledge updates inyectados).
- No se probó N grande (e.g., N=20). Es posible que la geometría residual gane rango dinámico con clusters más grandes.
- N=50 clusters por pool puede ser ruido estadístico bajo.
- La sospecha original (que las premisas descartadas en #38 también pudieron sufrir el mismo shape-bias que #39) sigue viva. La probe no ha refutado la sospecha de forma robusta.

## Próxima sesión: a resolver

1. Re-correr el AUC test de #38 (pos vs neg) sobre clusters construidos al shape de producción. Necesita decidir cómo construir pos/neg labels en LoCoMo / LME sin labels nativos -- opciones:
   - Per-session de la misma conv (pos topic-coherente) vs sesiones de convs distintas (neg).
   - Sub-segmentar y reusar #38 setup pero con clusters de production-shaped vectors.
2. Probar N=20 clusters (no solo N=4) para ver si la geometría residual sale del rango saturado.
3. Decidir destino de PR #46: domain-restriction explícita en código (warning runtime + docstring fuerte), o cerrar sin merge hasta tener V3 al shape correcto.
4. Decidir si #39 V1 también merece re-eval al shape correcto (la "DECISION-PREFERENCE collision" se midió en la misma distribución sentence-level).

## Archivos clave en este thread

- `notes/2026-04-28-role-classifier-organic-probe.py` -- probe inicial sobre vstash personal (descartada por premisa equivocada).
- `notes/2026-04-28-role-classifier-benchmark-probe.py` -- probe V2 SCR sobre LoCoMo per-session + LME per-turn.
- `notes/2026-04-28-geometric-shape-probe.py` -- probe `cluster_metrics` de #38 sobre los 3 shapes.
- `experiments/role_geometry/scenarios_disjoint/generated/{medical,products}.json` -- los datos de #38.
- `experiments/role_markers/{prototypes_v2,eval_v2}.json` -- los datos de #39.
- `experiments/role_geometry/directional_residual_geometry.py:cluster_metrics` -- métricas reutilizadas.

## Estado de PRs / branches al cierre

- `develop` está en `7afd2b2` (post-#45).
- `feature/role-classifier-memory-tagging` en `59a4cd4`, PR #46 OPEN.
- No hay branches abiertos de role_geometry en este programa.

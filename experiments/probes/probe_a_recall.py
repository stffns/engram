"""Probe A: measure recall@5 of brief+episodic retrieval on real queries.

Pulls real queries from the engram should_recall audit log, runs each
through `recall-briefs`, and dumps results to JSON for manual review.

The judgment (HIT / PARTIAL / MISS per query) is done by hand against
the dumped JSON -- LLM-as-judge here would introduce its own bias,
and at n=20 the manual cost is ~10 min.

Decision rule for the connector EV question:
  - recall@5 >= 80% -> connector marginal (skip nanoGPT-as-connector).
  - recall@5 50-80% -> connector might help; continue with synthesis.
  - recall@5 < 50% -> connector strongly justified.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# 20 real queries hand-picked from engram should_recall audit log.
# Mix of: state-of-project, technical decision recall, "where did we
# leave off", research lookup, status questions. Excludes hook-fixed
# strings ("recent decisions and context") and trivial acks ("si").
QUERIES = [
    "Que resultados tenemos, que es merken en este momento?",
    "En resumen que se a conseguido con Merken hasta aca?",
    "Dame un resumen de que es merken hoy en dia",
    "A quedo guardado para empezar otra sesion?",
    "Ya volvi, con que seguimos, o guardas esto e iniciamos una nueva sesion?",
    "Y hay algun experimento de la comunidad acerca de memoria agentica",
    "Trabajemos en lo de Locomo, quiero saber si finalmente la memoria sintetica es medible",
    "Peor el llm es la ultima fase, aca solo validamos ellimite que podemos lograr sin usar llm",
    "el siguiente movimiento natural es el ood_X_short solo vs ood_X_short + 0/1/2 otras interacciones",
    "Si, despues podemos volver a trabajar en v7, talvez no hayamos visto algo",
    "Check v7 training b4ma4dn1f completion",
    "Si estamos documentando todo? si lo gramos pasar 95% que habremos logrado?",
    "Con este hito que hemos logrado en que convierte a merken?",
    "No solo numeros, tecnicas, errores negativos, para no caer en los mismos errores en el futuro",
    "Si esperemos, pero mejor usemos colab para lanzar la siguiente",
    "Podemos documentar todo esto en un md son datos valiosos para nuestras futura implementaciones",
    "Si, exploremoslo y podemos docuemntar los hallazgos",
    "Explicame que creo ue me perdi, que seguiria?",
    "Si concuerdo, continuemos con tu sugerencia",
    "Si, arreglalosIgual te quedaste en modelos viejos, 2.5 cuando el ultimo es 3.1 porque?",
]


def run_recall(query: str, top_k: int = 5, brief_k: int = 3) -> dict:
    """Run merken recall-briefs and return parsed JSON."""
    result = subprocess.run(
        [
            "merken", "--json", "--project", "engram",
            "recall-briefs", query,
            "--top-k", str(top_k),
            "--brief-k", str(brief_k),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"JSONDecodeError: {e}", "stdout_head": result.stdout[:200]}


def main() -> int:
    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i:2}/{len(QUERIES)}] {q[:80]}...", flush=True)
        out = run_recall(q)
        results.append({"query": q, "result": out})

    out_path = Path(__file__).parent / "probe_a_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")
    print(f"Queries: {len(QUERIES)}")
    n_err = sum(1 for r in results if isinstance(r["result"], dict)
                and "error" in r["result"])
    print(f"Errors:  {n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

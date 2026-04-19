"""Probe B: can briefs+episodic alone support multi-hop synthesis?

Question: a Knowledge Graph layer (CONSTITUTION §6, gated) is
justified IF the LLM cannot synthesize multi-hop answers from the
existing briefs+episodic context. If the LLM CAN synthesize them,
KG is overengineering.

Method:
  1. 10 hand-crafted multi-hop queries about engram project history.
     Each requires chaining >=2 facts from different briefs / events.
  2. For each: retrieve top brief_k=3 briefs + top_k=5 episodic via
     `merken recall-briefs`, format as context, send to Gemini.
  3. Dump (query, context, answer) for manual judgment of:
     - correctness (correct / partial / wrong)
     - multi-hop usage (chained facts vs surfaced one)
     - missed-context (failed to use info that WAS in retrieved set)

Decision rule:
  - Gemini synthesizes correct multi-hop answers consistently (>=70%
    correct) -> KG is overengineering, skip.
  - Gemini misses connections often (correct < 50%) -> KG justified.
  - 50-70% -> KG might help; second-order question.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from google import genai


# Hand-crafted multi-hop queries about engram project history.
# Each requires chaining 2+ pieces of context from different briefs
# or episodic events to answer. Queries are in Spanish (Jay's
# working language) to match natural use.
QUERIES = [
    "Por que descartamos H2 contrastive loss y como se relaciona con el bug del hook PreCompact que encontramos hoy?",
    "Que cambios entre v6 y v7 del nanoGPT write filter llevaron al fix del markdown FPR de 66.7% a 0%?",
    "Como evoluciono nuestra estrategia de training data desde nanoGPT v4 hasta v7? Que aprendimos en cada salto?",
    "Por que rechazamos H8 capacity bump y que conclusion comun comparte con la falla de H2 contrastive?",
    "Cual es la diferencia entre 'training-signal limit' y 'capacity limit' que demostraron los experimentos H2 y H8?",
    "Como esta relacionado el bug de hook PreCompact con el por que Probe A inicialmente devolvio 0 briefs?",
    "Cual es la conexion entre los resultados LoCoMo H16/H17 y la decision de NO usar v7 como filtro de retrieval en conversaciones largas?",
    "Por que brief_v1 fue una breakthrough vs embedding-based consolidation, y como se prueba empiricamente esa diferencia?",
    "Como se relacionan H2 (hinge contrastive), H2b (InfoNCE), H2c (wider buckets) entre si? Cual fue la mejor variante y por que aun fallo el bar formal?",
    "Que justifica el siguiente movimiento de 'esperar a 300+ DEC labels' en lugar de seguir iterando sobre v7 ahora?",
]


def retrieve_context(query: str, top_k: int = 5, brief_k: int = 3) -> dict:
    """Run merken recall-briefs and return parsed JSON."""
    result = subprocess.run(
        ["merken", "--json", "--project", "engram", "recall-briefs",
         query, "--top-k", str(top_k), "--brief-k", str(brief_k)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}
    return json.loads(result.stdout)


def format_prompt(query: str, ctx: dict) -> str:
    briefs = ctx.get("briefs", [])
    episodic = ctx.get("episodic", [])
    parts = ["You are answering a question about an ongoing software "
             "research project called merken. Use ONLY the context "
             "provided below. If the context is insufficient for a "
             "multi-hop answer (i.e., you cannot connect two or more "
             "pieces of information), say so explicitly. Do not "
             "speculate beyond the context.\n\n"]
    if briefs:
        parts.append("## BRIEFS (compressed temporal summaries)\n")
        for i, b in enumerate(briefs, 1):
            parts.append(f"### Brief {i}\n{b}\n\n")
    if episodic:
        parts.append("## EPISODIC EVENTS (raw notes)\n")
        for i, e in enumerate(episodic, 1):
            title = e.get("title", "untitled")
            text = e.get("text", "")
            parts.append(f"### Event {i}: {title}\n{text}\n\n")
    parts.append(f"## QUESTION\n{query}\n\n## ANSWER\n")
    return "".join(parts)


def main() -> int:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    model = "gemini-2.0-flash"

    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i:2}/{len(QUERIES)}] {q[:80]}...", flush=True)
        ctx = retrieve_context(q)
        if "error" in ctx:
            results.append({"query": q, "error": ctx["error"]})
            continue
        prompt = format_prompt(q, ctx)
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
            )
            answer = resp.text or ""
        except Exception as e:
            answer = f"[ERROR] {type(e).__name__}: {e}"
        results.append({
            "query": q,
            "n_briefs": len(ctx.get("briefs", [])),
            "n_episodic": len(ctx.get("episodic", [])),
            "context_chars": sum(len(b) for b in ctx.get("briefs", []))
                             + sum(len(e.get("text", "")) for e in ctx.get("episodic", [])),
            "answer": answer,
            "context": ctx,
        })

    out_path = Path(__file__).parent / "probe_b_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

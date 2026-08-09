"""
eval/run_eval.py

Golden-set evaluation harness for the LangGraph agent. Answers the question
"how do you know this RAG/text-to-SQL system actually works?" with numbers
instead of vibes.

For every question in eval/dataset.json, this:
  1. Runs the real compiled graph (same code path as the API) end-to-end.
  2. Checks routing: did the planner pick the expected source(s)?
  3. Checks grounding: does the final answer contain the independently
     verified ground-truth facts listed in must_contain? (deterministic,
     catches wrong numbers/names cheaply)
  4. Checks faithfulness/helpfulness via an LLM-as-judge call that only
     sees the context the graph actually used, so it can catch hallucinated
     elaboration that happens to also mention the right keywords.

Two LLM calls per question (system-under-test + judge). Requires the DB
and a valid API key to be configured, same as running the app normally.

Run with:
    python -m eval.run_eval
"""

import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from app.llm import call_llm
from graph.workflow import graph

DATASET_PATH = Path("eval/dataset.json")
JUDGE_PROMPT_PATH = Path("eval/judge_prompt.txt")
RESULTS_DIR = Path("eval/results")

MAX_JUDGE_CONTEXT_CHARS = 4000


def _load_dataset() -> list:
    return json.loads(DATASET_PATH.read_text())


def _clean_json(raw: str) -> dict:
    """LLMs sometimes wrap JSON in markdown fences despite instructions — strip them."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _check_grounding(answer: str, must_contain: list, match_mode: str) -> dict:
    if not must_contain:
        return {"applicable": False, "passed": None, "hits": [], "misses": []}

    answer_lower = answer.lower()
    hits = [p for p in must_contain if p.lower() in answer_lower]
    misses = [p for p in must_contain if p.lower() not in answer_lower]

    passed = (len(hits) > 0) if match_mode == "any" else (len(misses) == 0)
    return {"applicable": True, "passed": passed, "hits": hits, "misses": misses}


def _judge(question: str, context: str, answer: str) -> dict:
    template = JUDGE_PROMPT_PATH.read_text()
    truncated_context = (context or "")[:MAX_JUDGE_CONTEXT_CHARS]
    prompt = template.format(question=question, context=truncated_context or "(no context — retrieval/SQL returned nothing)", answer=answer)

    raw = call_llm(prompt)
    try:
        parsed = _clean_json(raw)
        return {
            "faithfulness": int(parsed["faithfulness"]),
            "helpfulness": int(parsed["helpfulness"]),
            "rationale": parsed.get("rationale", ""),
            "parse_error": None,
        }
    except Exception as e:
        return {"faithfulness": None, "helpfulness": None, "rationale": None, "parse_error": f"{type(e).__name__}: {e}"}


def _empty_result_shell(item: dict, error: str) -> dict:
    """Same shape as a successful result, but with every scoring field
    left as None/not-applicable so summarize() and the report writer can
    treat it uniformly instead of needing error-specific branches."""
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_route": item.get("expected_route"),
        "actual_route": None,
        "route_correct": None,
        "answer": "",
        "sql_used": None,
        "grounding": {"applicable": False, "passed": None, "hits": [], "misses": []},
        "judge": {"faithfulness": None, "helpfulness": None, "rationale": None, "parse_error": None},
        "latency_s": None,
        "error": error,
    }


def run_item(item: dict) -> dict:
    t0 = time.perf_counter()
    try:
        # Each golden question is independent, so it gets its own thread_id —
        # the graph now requires one since it's compiled with a checkpointer
        # (for multi-turn memory in normal use). Reusing item["id"] means a
        # re-run with the same dataset reuses the same threads rather than
        # leaking a new one into the checkpointer every time.
        state = graph.invoke(
            {
                "question": item["question"],
                "k": 4,
                "sources": [],
                "execution_path": [],
            },
            config={"configurable": {"thread_id": f"eval-{item['id']}"}},
        )
    except Exception as e:
        # Most likely Gemini's daily quota (see app/llm.py) — one question
        # hitting it shouldn't cost the results already collected from
        # every question before it. Caller decides whether to keep going.
        return _empty_result_shell(item, error=f"{type(e).__name__}: {str(e)[:300]}")

    latency_s = round(time.perf_counter() - t0, 2)

    actual_route = state.get("route")
    answer = state.get("final_answer") or ""
    context = state.get("combined_context")

    expected_route = item.get("expected_route")
    route_correct = None if expected_route is None else (actual_route == expected_route)

    grounding = _check_grounding(answer, item.get("must_contain", []), item.get("match_mode", "all"))

    try:
        judge = _judge(item["question"], context, answer)
    except Exception as e:
        judge = {"faithfulness": None, "helpfulness": None, "rationale": None, "parse_error": f"{type(e).__name__}: {str(e)[:200]}"}

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_correct": route_correct,
        "answer": answer,
        "sql_used": state.get("sql_query"),
        "grounding": grounding,
        "judge": judge,
        "latency_s": latency_s,
        "error": None,
    }


def _avg(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def summarize(results: list) -> dict:
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    def category_stats(items):
        errored = [r for r in items if r.get("error")]
        completed = [r for r in items if not r.get("error")]
        route_checked = [r for r in completed if r["route_correct"] is not None]
        grounding_checked = [r for r in completed if r["grounding"]["applicable"]]
        return {
            "n": len(items),
            "errored": len(errored),
            "route_accuracy": _avg([1 if r["route_correct"] else 0 for r in route_checked]) if route_checked else None,
            "grounding_pass_rate": _avg([1 if r["grounding"]["passed"] else 0 for r in grounding_checked]) if grounding_checked else None,
            "avg_faithfulness": _avg([r["judge"]["faithfulness"] for r in completed]),
            "avg_helpfulness": _avg([r["judge"]["helpfulness"] for r in completed]),
            "avg_latency_s": _avg([r["latency_s"] for r in completed]),
        }

    return {
        "overall": category_stats(results),
        "by_category": {cat: category_stats(items) for cat, items in by_category.items()},
    }


def _write_markdown_report(results: list, summary: dict, path: Path):
    lines = ["# Evaluation Report", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | n | Route accuracy | Grounding pass rate | Avg faithfulness (/5) | Avg helpfulness (/5) | Avg latency (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, s in [("overall", summary["overall"])] + list(summary["by_category"].items()):
        lines.append(
            f"| {cat} | {s['n']} | {s['route_accuracy']} | {s['grounding_pass_rate']} | "
            f"{s['avg_faithfulness']} | {s['avg_helpfulness']} | {s['avg_latency_s']} |"
        )

    lines.append("")
    lines.append("## Per-question results")
    lines.append("")
    lines.append("| id | category | route (exp/actual) | grounding | faithfulness | helpfulness | latency (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['id']} | {r['category']} | ERROR | - | - | - | - |")
            continue
        route_str = f"{r['expected_route']} / {r['actual_route']}"
        grounding_str = "n/a" if not r["grounding"]["applicable"] else ("PASS" if r["grounding"]["passed"] else "FAIL")
        lines.append(
            f"| {r['id']} | {r['category']} | {route_str} | {grounding_str} | "
            f"{r['judge']['faithfulness']} | {r['judge']['helpfulness']} | {r['latency_s']} |"
        )

    errored = [r for r in results if r.get("error")]
    if errored:
        lines.append("")
        lines.append("## Questions that didn't complete")
        lines.append("")
        lines.append("Most likely Gemini's daily quota — re-run `python -m eval.run_eval` after it resets "
                      "to fill these in (already-completed results above are untouched by a re-run).")
        lines.append("")
        for r in errored:
            lines.append(f"- **{r['id']}** ({r['question']}): {r['error']}")

    lines.append("")
    lines.append("## Failures worth reading")
    lines.append("")
    failures = [
        r for r in results
        if not r.get("error")
        and (
            (r["route_correct"] is False)
            or (r["grounding"]["applicable"] and not r["grounding"]["passed"])
            or (r["judge"]["faithfulness"] is not None and r["judge"]["faithfulness"] < 4)
        )
    ]
    if not failures:
        lines.append("None — every completed question passed routing, grounding, and faithfulness checks.")
    else:
        for r in failures:
            lines.append(f"### {r['id']}: {r['question']}")
            lines.append(f"- Route: expected `{r['expected_route']}`, got `{r['actual_route']}`")
            if r["grounding"]["applicable"]:
                lines.append(f"- Grounding: missing {r['grounding']['misses']}" if r["grounding"]["misses"] else "- Grounding: OK")
            lines.append(f"- Judge: faithfulness={r['judge']['faithfulness']}, helpfulness={r['judge']['helpfulness']} — {r['judge']['rationale']}")
            lines.append(f"- Answer: {r['answer'][:300]}")
            lines.append("")

    path.write_text("\n".join(lines))


def _save(results: list, raw_path: Path, report_path: Path):
    """Writes both output files from whatever's been collected so far —
    called after every question, not just at the end, so a quota cutoff
    mid-run still leaves real results on disk instead of nothing."""
    summary = summarize(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    _write_markdown_report(results, summary, report_path)
    return summary


def main():
    dataset = _load_dataset()
    print(f"Running eval on {len(dataset)} questions...\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"results_{timestamp}.json"
    report_path = RESULTS_DIR / "latest_report.md"

    results = []
    for item in dataset:
        print(f"[{item['id']}] {item['question']}")
        result = run_item(item)
        results.append(result)

        if result.get("error"):
            print(f"    ERROR: {result['error']}\n")
        else:
            route_flag = "-" if result["route_correct"] is None else ("PASS" if result["route_correct"] else "FAIL")
            ground_flag = "n/a" if not result["grounding"]["applicable"] else ("PASS" if result["grounding"]["passed"] else "FAIL")
            print(
                f"    route={route_flag} grounding={ground_flag} "
                f"faithfulness={result['judge']['faithfulness']} helpfulness={result['judge']['helpfulness']} "
                f"latency={result['latency_s']}s\n"
            )

        # Save after every question, not just at the end — a quota cutoff
        # partway through should still leave real results on disk.
        _save(results, raw_path, report_path)

    summary = summarize(results)
    errored = summary["overall"]["errored"]

    print("=" * 60)
    print("OVERALL:", json.dumps(summary["overall"], indent=2))
    if errored:
        print(f"\n{errored} question(s) didn't complete (see 'Questions that didn't complete' in the report) — re-run to fill them in.")
    print(f"\nRaw results: {raw_path}")
    print(f"Report:      {report_path}")


if __name__ == "__main__":
    main()

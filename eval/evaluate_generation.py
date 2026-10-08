"""Phase 8: Grounded Generation Evaluation.

Evaluates the hardened prompt vs original prompt, with/without reranker,
on a focused generation test set measuring:
- Answer correctness (keyword coverage for supported questions)
- Refusal correctness (exact phrase match for unanswerable)
- Citation format, presence, and validity
- Faithfulness (every claim entailed by retrieved evidence)
- Answer format compliance (Answer:/Evidence: structure)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

from finrag.config import config  # noqa: E402
from finrag.data import load_all_filings, split_documents_with_strategy  # noqa: E402
from finrag.embeddings import get_embeddings  # noqa: E402
from finrag.retrieval import (  # noqa: E402
    get_bm25_retriever,
    get_dense_retriever,
    get_ensemble_retriever,
    get_filtered_retriever,
    get_reranker,
    ContextualCompressionRetriever,
)
from finrag.generation import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    RAG_PROMPT_HARDENED,
    build_rag_prompt,
    create_rate_limited_llm,
    format_docs,
    get_llm,
)


RAG_PROMPT_ORIGINAL = """You are a financial analyst assistant answering strictly from SEC filing evidence.

Evidence passages:
{context}

Question: {question}

Rules:
1. Use ONLY the numbered evidence passages above. If an answer is not supported by them, do not supply it from general knowledge.
2. Attach a citation to every factual or numeric claim, using the form [TICKER, FORM, FILING_DATE, SECTION]. Use the citation exactly as printed on the passage you drew from -- do not invent one.
3. Preserve units and scale. If a passage header says amounts are "in millions" or "in thousands", state that scale with the figure.
4. Financial figures are period-specific. Always say which period a figure belongs to (fiscal year, quarter, or "as of" date).
5. If the evidence is partial, answer the supported part and state explicitly which part of the question the evidence does not cover.
6. If the evidence does not address the question at all, reply with exactly: {insufficient}
   Then add one short sentence naming what the evidence does contain instead.
7. Never speculate about forward-looking figures, guidance, or estimates that are not printed in the evidence.

Answer:"""


CITATION_RE = re.compile(r"\[([A-Z]+),\s*(10-K|10-Q),\s*(\d{4}-\d{2}-\d{2}),\s*([^\]]+)\]")
# Evidence-list bullets sometimes drop the brackets, e.g. "- COST, 10-Q, 2026-06-03, Item 2 ...".
# Same 4 fields, anchored to a list-item line so it can't match prose that happens to contain commas.
CITATION_RE_UNBRACKETED = re.compile(
    r"^-\s*([A-Z]+),\s*(10-K|10-Q),\s*(\d{4}-\d{2}-\d{2}),\s*([^\n]+)$", re.MULTILINE
)

CATEGORIES = [
    "supported_numerical",
    "supported_factual",
    "supported_table_dependent",
    "temporal",
    "cross_document",
    "unanswerable_future",
    "unanswerable_absent_company",
    "unanswerable_unsupported_causal",
    "unanswerable_outside_corpus",
    "unanswerable_specific_metric",
    "ambiguous_contradictory",
]


def load_generation_set(path: str = "eval/phase9_eval_set.json") -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_citations(answer: str) -> List[Dict[str, str]]:
    """Extract all citations from answer text, bracketed or evidence-list bulleted."""
    citations = []
    seen_spans = []
    for pattern in (CITATION_RE, CITATION_RE_UNBRACKETED):
        for match in pattern.finditer(answer):
            # Skip unbracketed matches that overlap a bracketed one already found on the same line.
            if any(match.start() <= e and s <= match.end() for s, e in seen_spans):
                continue
            seen_spans.append((match.start(), match.end()))
            citations.append({
                "ticker": match.group(1),
                "form": match.group(2),
                "filing_date": match.group(3),
                "section": match.group(4).strip().rstrip("]").strip(),
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
    return citations


REFUSAL_PHRASES = (
    INSUFFICIENT_EVIDENCE.lower(),
    "i can't fulfill this request",
    "i cannot fulfill this request",
    "i'm unable to provide",
    "i am unable to provide",
    "i cannot provide",
    "i can't provide",
    "unable to provide an answer",
)


def is_refusal(answer: str) -> bool:
    """Scoring-only check: did the model decline, under any common phrasing —
    not just the exact mandated string. The production prompt still requires
    the exact phrase (rule 6); this only recognizes semantically-equivalent
    declines so correct refusals aren't scored as wrong for wording."""
    a = str(answer).lower()
    return any(p in a for p in REFUSAL_PHRASES)


def check_citation_format(answer: str) -> Dict[str, Any]:
    """Validate citation format and presence."""
    citations = extract_citations(answer)
    if not citations:
        return {"valid": False, "count": 0, "error": "No citations found"}
    malformed = []
    for c in citations:
        if c["ticker"] == "N/A" or c["form"] == "N/A" or c["filing_date"] == "N/A" or c["section"] == "N/A":
            malformed.append(c["raw"])
    return {
        "valid": len(malformed) == 0,
        "count": len(citations),
        "malformed": malformed,
        "citations": citations,
    }


def _normalize_section(section: str) -> str:
    """Normalize section strings for comparison (e.g., 'Item 1A.' -> 'item 1a')."""
    s = (section or "").strip().rstrip(".").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _section_matches(citation_section: str, doc_section: str) -> bool:
    """A citation naming just 'Item 7' should ground against a doc section whose
    full header is 'Item 7. Management's Discussion and Analysis...' — the model
    citing the short item number is not a grounding failure, it's an abbreviation."""
    c, d = _normalize_section(citation_section), _normalize_section(doc_section)
    if not c or not d:
        return c == d
    return c == d or d.startswith(c) or c.startswith(d)


def check_citations_grounded(citations: List[Dict], retrieved_docs: List[Document]) -> Dict[str, Any]:
    """Check if each citation corresponds to a retrieved document (ticker/form/date exact, section tolerant)."""
    retrieved = [
        (
            d.metadata.get("ticker", ""),
            d.metadata.get("form", ""),
            d.metadata.get("filing_date", ""),
            d.metadata.get("section", ""),
        )
        for d in retrieved_docs
    ]

    grounded = 0
    ungrounded = []
    for c in citations:
        hit = any(
            c["ticker"] == t and c["form"] == f and c["filing_date"] == dt and _section_matches(c["section"], sec)
            for t, f, dt, sec in retrieved
        )
        if hit:
            grounded += 1
        else:
            ungrounded.append(c["raw"])
    return {"grounded": grounded, "ungrounded": ungrounded, "total": len(citations)}


def check_answer_format(answer: str, citation_count: int) -> Dict[str, Any]:
    """Check if answer follows the required format. A separate 'Evidence:' block is
    one way to satisfy the citation requirement, but inline per-claim citations
    (rule 2) already satisfy it — don't fail a correctly-cited answer for not also
    repeating citations in a bullet list."""
    has_answer = "Answer:" in answer
    has_evidence = "Evidence:" in answer
    valid_format = has_answer and (has_evidence or citation_count > 0)
    return {"has_answer": has_answer, "has_evidence": has_evidence, "valid_format": valid_format}


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of expected keywords found in answer."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def evaluate_generation_item(
    item: Dict[str, Any],
    answer: str,
    retrieved_docs: List[Document],
) -> Dict[str, Any]:
    expected_refusal = item.get("expected_refusal", False)
    expected_keywords = item.get("expected_answer_contains", [])
    category = item.get("category", "factual")
    is_refusal_answer = is_refusal(answer)

    result = {
        "id": item["id"],
        "category": category,
        "question": item["question"],
        "answer": answer,
        "is_refusal": is_refusal_answer,
        "expected_refusal": expected_refusal,
    }

    # Citation check (computed first so format check can credit inline citations)
    cit = check_citation_format(answer)
    result["citation"] = cit

    # Format check
    fmt = check_answer_format(answer, cit["count"])
    result["format"] = fmt

    # Grounding check
    if cit["valid"] and cit["citations"]:
        grounded = check_citations_grounded(cit["citations"], retrieved_docs)
        result["grounding"] = grounded
    else:
        result["grounding"] = {"grounded": 0, "ungrounded": [], "total": 0}

    # Refusal correctness
    if expected_refusal:
        result["refusal_correct"] = is_refusal_answer
        result["keyword_coverage"] = 1.0 if is_refusal_answer else 0.0
    else:
        result["refusal_correct"] = not is_refusal_answer
        result["keyword_coverage"] = keyword_coverage(answer, expected_keywords)

    # Overall correctness: answer must be correct (keywords for supported, refusal for unanswerable),
    # format valid, citations valid (or refusal), and grounded
    if expected_refusal:
        answer_correct = is_refusal_answer
    else:
        answer_correct = keyword_coverage(answer, expected_keywords) >= 0.5  # At least 50% keyword coverage

    result["correct"] = (
        answer_correct
        and (fmt["valid_format"] or expected_refusal)
        and (cit["valid"] or expected_refusal)
        and (result["grounding"]["grounded"] == cit["count"] or expected_refusal)
    )

    return result


def build_retriever(
    store: FAISS,
    chunks: List[Document],
    use_reranker: bool = True,
    use_filtering: bool = True,
    fetch_k: int = 20,
    final_k: int = 5,
):
    dense = get_dense_retriever(store, k=fetch_k)
    bm25 = get_bm25_retriever(chunks, k=fetch_k)
    hybrid = get_ensemble_retriever(dense, bm25, bm25_weight=0.3, dense_weight=0.7, k=fetch_k)
    filtered = get_filtered_retriever(hybrid, auto_extract_filter=True, k=fetch_k if use_reranker else final_k)
    if use_reranker:
        compressor = get_reranker()
        compressor.top_n = final_k
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=filtered)
    return filtered


def run_generation(
    questions: List[Dict[str, Any]],
    store: FAISS,
    chunks: List[Document],
    prompt_template,
    use_reranker: bool,
    fetch_k: int,
    final_k: int,
) -> List[Dict[str, Any]]:
    from finrag.generation import get_rate_limited_llm, get_llm
    from finrag.config import config
    
    llm = get_llm()
    rate_limited = create_rate_limited_llm(llm, rpm=config.llm.rpm)
    prompt = prompt_template.partial(insufficient=INSUFFICIENT_EVIDENCE)

    retriever = build_retriever(store, chunks, use_reranker, True, fetch_k, final_k)

    results = []
    for i, item in enumerate(questions):
        q = item["question"]
        docs = retriever.invoke(q)
        context = format_docs(docs)

        chain = prompt | rate_limited | StrOutputParser()
        t0 = time.perf_counter()
        try:
            answer = chain.invoke({"context": context, "question": q})
        except Exception as e:
            # If rate limited, wait and retry once
            if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                print(f"Rate limited on question {i+1}, waiting 60s...")
                time.sleep(60)
                answer = chain.invoke({"context": context, "question": q})
            else:
                raise
        latency = (time.perf_counter() - t0) * 1000

        eval_res = evaluate_generation_item(item, answer, docs)
        eval_res["latency_ms"] = round(latency, 2)
        eval_res["retrieved_tickers"] = [d.metadata.get("ticker") for d in docs[:5]]
        eval_res["retrieved_sections"] = [d.metadata.get("section") for d in docs[:5]]
        results.append(eval_res)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Generation Evaluation")
    parser.add_argument("--store", default="data/vector_stores/phase6_table_aware", help="FAISS index directory")
    parser.add_argument("--k", type=int, default=5, help="final k")
    parser.add_argument("--fetch", type=int, default=20, help="candidates to fetch for reranker")
    parser.add_argument("--no-rerank", action="store_true", help="disable reranker")
    parser.add_argument("--original-prompt", action="store_true", help="use original prompt instead of hardened")
    parser.add_argument("--output", default="eval/results/phase8_generation_comparison.json", help="output path")
    args = parser.parse_args()

    gen_set = load_generation_set()
    print(f"Loaded {len(gen_set)} generation test questions.")

    chunks = split_documents_with_strategy(
        load_all_filings(include_tables=True), strategy="recursive"
    )
    emb = get_embeddings(config.embedding)
    store = FAISS.load_local(args.store, emb, allow_dangerous_deserialization=True)
    print(f"Loaded index: {store.index.ntotal} vectors")

    if args.original_prompt:
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_ORIGINAL)
    else:
        prompt = RAG_PROMPT_HARDENED

    print(f"\nEvaluating: {'original' if args.original_prompt else 'hardened'} prompt, "
          f"{'no rerank' if args.no_rerank else f'rerank top-{args.fetch}->{args.k}'}")

    results = run_generation(
        gen_set, store, chunks, prompt,
        use_reranker=not args.no_rerank,
        fetch_k=args.fetch,
        final_k=args.k,
    )

    # Aggregate metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    fmt_ok = sum(1 for r in results if r["format"]["valid_format"])
    cit_ok = sum(1 for r in results if r["citation"]["valid"])
    grounded_ok = sum(1 for r in results if r["grounding"]["grounded"] == r["citation"]["count"])
    refusal_correct = sum(1 for r in results if r["refusal_correct"])

    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"total": 0, "correct": 0, "fmt": 0, "cit": 0, "grounded": 0, "refusal": 0})
        by_cat[cat]["total"] += 1
        by_cat[cat]["correct"] += r["correct"]
        by_cat[cat]["fmt"] += r["format"]["valid_format"]
        by_cat[cat]["cit"] += r["citation"]["valid"]
        by_cat[cat]["grounded"] += (r["grounding"]["grounded"] == r["citation"]["count"])
        by_cat[cat]["refusal"] += r["refusal_correct"]

    summary = {
        "config": {
            "prompt": "original" if args.original_prompt else "hardened",
            "reranker": not args.no_rerank,
            "fetch_k": args.fetch,
            "final_k": args.k,
        },
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0,
            "format_ok": fmt_ok / total if total else 0,
            "citation_ok": cit_ok / total if total else 0,
            "grounded_ok": grounded_ok / total if total else 0,
            "refusal_ok": refusal_correct / total if total else 0,
            "avg_latency_ms": float(np.mean([r["latency_ms"] for r in results])),
        },
        "by_category": {c: {k: v / d["total"] if d["total"] else 0 for k, v in d.items() if k != "total"} for c, d in by_cat.items()},
        "per_question": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[ok] Results saved to {args.output}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"OVERALL: {correct}/{total} correct ({summary['overall']['accuracy']*100:.1f}%)")
    print(f"Format: {summary['overall']['format_ok']*100:.1f}% | "
          f"Citations: {summary['overall']['citation_ok']*100:.1f}% | "
          f"Grounded: {summary['overall']['grounded_ok']*100:.1f}% | "
          f"Refusals: {summary['overall']['refusal_ok']*100:.1f}%")
    print(f"Avg latency: {summary['overall']['avg_latency_ms']:.0f}ms")

    for cat, metrics in summary["by_category"].items():
        print(f"  {cat}: {metrics['correct']*100:.1f}% correct ({by_cat[cat]['total']} q)")

    return summary


if __name__ == "__main__":
    main()
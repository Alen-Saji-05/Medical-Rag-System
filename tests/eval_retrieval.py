"""
tests/eval_retrieval.py
Retrieval quality evaluation suite.

Focuses on the metrics that matter most for RAG accuracy:
- Context Recall: did we retrieve all the relevant information?
- Context Precision: is retrieved content relevant (not noisy)?
- Hit Rate@K: is the right answer in the top-K results?
- MRR: how high is the first relevant result ranked?

Run this after any change to chunk size, overlap, alpha, or model choice.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parent.parent))


# ─── Evaluation dataset ───────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question": "What are the main symptoms of type 2 diabetes?",
        "expected_keywords": ["thirst", "urination", "fatigue", "blurred", "glucose"],
        "specialty": "endocrinology",
    },
    {
        "question": "What is hypertension and what are its risk factors?",
        "expected_keywords": ["blood pressure", "heart", "stroke", "salt", "obesity"],
        "specialty": "cardiology",
    },
    {
        "question": "How is asthma diagnosed and treated?",
        "expected_keywords": ["inhaler", "bronchodilator", "spirometry", "airways", "inflammation"],
        "specialty": "pulmonology",
    },
    {
        "question": "What are common risk factors for heart disease?",
        "expected_keywords": ["cholesterol", "smoking", "blood pressure", "diabetes", "obesity"],
        "specialty": "cardiology",
    },
    {
        "question": "What is the recommended treatment for high blood pressure?",
        "expected_keywords": ["medication", "lifestyle", "diet", "exercise", "sodium"],
        "specialty": "cardiology",
    },
]


# ─── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    hit_rate_at_3: float    # Is any relevant chunk in top 3?
    hit_rate_at_5: float    # Is any relevant chunk in top 5?
    mrr: float              # Mean Reciprocal Rank of first relevant result
    precision_at_3: float   # Fraction of top-3 that are relevant
    avg_top_score: float    # Average reranker score of top result
    coverage: float         # Fraction of expected keywords found in top-5


def keyword_relevance(content: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in chunk content."""
    content_lower = content.lower()
    hits = sum(1 for kw in keywords if kw.lower() in content_lower)
    return hits / len(keywords) if keywords else 0.0


def evaluate_retrieval(retriever, questions: list[dict] = None) -> RetrievalMetrics:
    """
    Run retrieval evaluation over a question set.
    A chunk is considered 'relevant' if it contains ≥40% of expected keywords.
    """
    from pipeline.retriever import HybridRetriever
    questions = questions or EVAL_QUESTIONS

    hit_at_3_scores = []
    hit_at_5_scores = []
    mrr_scores = []
    precision_at_3_scores = []
    top_scores = []
    coverage_scores = []

    for q in questions:
        query = q["question"]
        keywords = q["expected_keywords"]

        results = retriever.retrieve(query, top_n=5)
        if not results:
            hit_at_3_scores.append(0)
            hit_at_5_scores.append(0)
            mrr_scores.append(0)
            precision_at_3_scores.append(0)
            top_scores.append(0)
            coverage_scores.append(0)
            continue

        # Relevance judgments
        relevances = [
            keyword_relevance(r.content, keywords) >= 0.4 for r in results
        ]

        # Hit@3 and Hit@5
        hit_at_3_scores.append(int(any(relevances[:3])))
        hit_at_5_scores.append(int(any(relevances[:5])))

        # MRR
        mrr = 0.0
        for rank, rel in enumerate(relevances, 1):
            if rel:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        # Precision@3
        precision_at_3_scores.append(sum(relevances[:3]) / 3)

        # Top reranker score
        top_scores.append(results[0].rerank_score if results else 0)

        # Keyword coverage across top-5
        all_content = " ".join(r.content for r in results[:5])
        coverage_scores.append(keyword_relevance(all_content, keywords))

    metrics = RetrievalMetrics(
        hit_rate_at_3=sum(hit_at_3_scores) / len(hit_at_3_scores),
        hit_rate_at_5=sum(hit_at_5_scores) / len(hit_at_5_scores),
        mrr=sum(mrr_scores) / len(mrr_scores),
        precision_at_3=sum(precision_at_3_scores) / len(precision_at_3_scores),
        avg_top_score=sum(top_scores) / len(top_scores),
        coverage=sum(coverage_scores) / len(coverage_scores),
    )
    return metrics


def print_report(metrics: RetrievalMetrics):
    print("\n" + "═" * 50)
    print("  RETRIEVAL EVALUATION REPORT")
    print("═" * 50)
    print(f"  Hit Rate @3:      {metrics.hit_rate_at_3:.2%}  (target: >80%)")
    print(f"  Hit Rate @5:      {metrics.hit_rate_at_5:.2%}  (target: >90%)")
    print(f"  MRR:              {metrics.mrr:.3f}    (target: >0.70)")
    print(f"  Precision @3:     {metrics.precision_at_3:.2%}  (target: >60%)")
    print(f"  Avg top score:    {metrics.avg_top_score:.3f}")
    print(f"  Keyword coverage: {metrics.coverage:.2%}  (target: >75%)")
    print("═" * 50)

    # Guidance
    if metrics.hit_rate_at_5 < 0.8:
        print("  ⚠ Low hit rate → increase RETRIEVAL_TOP_K or add more documents")
    if metrics.precision_at_3 < 0.5:
        print("  ⚠ Low precision → decrease CHUNK_SIZE or increase RERANK_TOP_N filtering")
    if metrics.mrr < 0.6:
        print("  ⚠ Low MRR → consider tuning RETRIEVAL_ALPHA or reranker threshold")
    if metrics.coverage < 0.7:
        print("  ⚠ Low coverage → increase CHUNK_OVERLAP or expand document corpus")
    print()


if __name__ == "__main__":
    from pipeline.retriever import HybridRetriever
    retriever = HybridRetriever()
    metrics = evaluate_retrieval(retriever)
    print_report(metrics)

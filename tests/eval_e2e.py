"""
tests/eval_e2e.py
End-to-end RAGAS-style evaluation suite for Phase 4.

Metrics implemented:
  1. Faithfulness          — Are all answer claims supported by retrieved context?
  2. Answer Relevancy      — Does the answer actually address the question asked?
  3. Context Recall        — Did retrieval capture all necessary information?
  4. Context Precision     — Are retrieved chunks signal or noise?
  5. Answer Correctness    — Does the answer align with expected reference answers?

Reference:
  Es et al. (2023) "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  https://arxiv.org/abs/2309.15217

Run:
  python tests/eval_e2e.py                  # full suite
  python tests/eval_e2e.py --quick          # 3 questions only
  python tests/eval_e2e.py --save           # save JSON report to logs/
"""

import re
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))


# ── Ground-truth evaluation dataset ───────────────────────────────────────────
# Each entry has: question, reference answer keywords, expected context keywords,
# and a minimal reference answer for correctness scoring.

EVAL_DATASET = [
    {
        "id": "q01",
        "question": "What are the main symptoms of type 2 diabetes?",
        "reference_answer": (
            "Common symptoms of type 2 diabetes include increased thirst, frequent urination, "
            "fatigue, blurred vision, slow-healing sores, and frequent infections."
        ),
        "answer_keywords":  ["thirst", "urination", "fatigue", "blurred", "vision"],
        "context_keywords": ["glucose", "blood sugar", "insulin", "diabetes", "symptoms"],
        "specialty": "endocrinology",
    },
    {
        "id": "q02",
        "question": "What are the risk factors for heart disease?",
        "reference_answer": (
            "Risk factors for heart disease include high blood pressure, high cholesterol, "
            "smoking, obesity, diabetes, physical inactivity, and family history."
        ),
        "answer_keywords":  ["blood pressure", "cholesterol", "smoking", "obesity", "diabetes"],
        "context_keywords": ["cardiovascular", "heart", "risk", "cholesterol", "hypertension"],
        "specialty": "cardiology",
    },
    {
        "id": "q03",
        "question": "How is high blood pressure treated?",
        "reference_answer": (
            "High blood pressure is treated with lifestyle changes such as reducing salt intake, "
            "regular exercise, and a healthy diet, as well as medications like ACE inhibitors, "
            "beta-blockers, or diuretics."
        ),
        "answer_keywords":  ["lifestyle", "medication", "diet", "exercise", "salt"],
        "context_keywords": ["hypertension", "blood pressure", "treatment", "medication", "lifestyle"],
        "specialty": "cardiology",
    },
    {
        "id": "q04",
        "question": "What causes asthma and how is it managed?",
        "reference_answer": (
            "Asthma is caused by airway inflammation triggered by allergens, exercise, or infections. "
            "It is managed with inhalers including bronchodilators for quick relief and "
            "corticosteroids for long-term control."
        ),
        "answer_keywords":  ["inflammation", "inhaler", "bronchodilator", "allergen", "airways"],
        "context_keywords": ["asthma", "airways", "inflammation", "inhaler", "trigger"],
        "specialty": "pulmonology",
    },
    {
        "id": "q05",
        "question": "What are the symptoms and warning signs of hypertension?",
        "reference_answer": (
            "Hypertension is often called the silent killer because it typically has no symptoms. "
            "When symptoms occur they may include headaches, shortness of breath, or nosebleeds, "
            "but these are not specific."
        ),
        "answer_keywords":  ["silent", "headache", "symptoms", "blood pressure", "warning"],
        "context_keywords": ["hypertension", "blood pressure", "symptoms", "silent", "headache"],
        "specialty": "cardiology",
    },
    {
        "id": "q06",
        "question": "What lifestyle changes help prevent type 2 diabetes?",
        "reference_answer": (
            "Lifestyle changes that help prevent type 2 diabetes include losing weight if overweight, "
            "eating a healthy diet low in refined carbohydrates and sugar, getting regular physical "
            "activity, and quitting smoking."
        ),
        "answer_keywords":  ["weight", "diet", "exercise", "physical activity", "prevention"],
        "context_keywords": ["diabetes", "prevention", "lifestyle", "diet", "exercise"],
        "specialty": "endocrinology",
    },
    {
        "id": "q07",
        "question": "What are the different types of heart disease?",
        "reference_answer": (
            "Types of heart disease include coronary artery disease, heart failure, arrhythmias, "
            "heart valve disease, and congenital heart defects."
        ),
        "answer_keywords":  ["coronary", "artery", "failure", "arrhythmia", "valve"],
        "context_keywords": ["heart disease", "coronary", "cardiac", "cardiovascular"],
        "specialty": "cardiology",
    },
    {
        "id": "q08",
        "question": "How is cervical cancer screened and prevented?",
        "reference_answer": (
            "Cervical cancer is screened using Pap smears and HPV tests. Prevention includes "
            "HPV vaccination and regular screening starting at age 21."
        ),
        "answer_keywords":  ["pap", "hpv", "vaccine", "screening", "cervical"],
        "context_keywords": ["cervical", "cancer", "hpv", "pap smear", "screening"],
        "specialty": "oncology",
    },
]


# ── Metric dataclasses ────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question_id: str
    question: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    answer_correctness: float
    latency_ms: int
    chunks_used: int
    safety_category: str
    has_warnings: bool
    error: Optional[str] = None

@dataclass
class EvalReport:
    timestamp: str
    total_questions: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_context_precision: float
    avg_answer_correctness: float
    avg_latency_ms: float
    pass_rate: float          # Fraction of questions meeting all thresholds
    question_results: list


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Measures how directly the answer addresses the question.

    Approach: keyword overlap between question terms and answer.
    In production, replace with an LLM-based similarity score for higher accuracy.
    Threshold: question content words that appear in the answer.
    """
    q_words = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
    a_words = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
    # Remove stop words
    stops = {"what", "that", "this", "with", "have", "from", "they",
             "your", "been", "will", "more", "about", "which", "when",
             "there", "their", "also", "some", "most", "into"}
    q_words -= stops
    if not q_words:
        return 1.0
    overlap = q_words & a_words
    return round(min(len(overlap) / len(q_words), 1.0), 3)


def score_context_recall(
    context_chunks: list, context_keywords: list[str]
) -> float:
    """
    Fraction of expected topic keywords present in the retrieved context.
    High recall = the retrieval found all the necessary information.
    """
    if not context_chunks or not context_keywords:
        return 0.0
    all_context = " ".join(c.content for c in context_chunks).lower()
    found = sum(1 for kw in context_keywords if kw.lower() in all_context)
    return round(found / len(context_keywords), 3)


def score_context_precision(
    context_chunks: list, answer_keywords: list[str]
) -> float:
    """
    Fraction of retrieved chunks that contain at least one answer keyword.
    High precision = retrieved chunks are relevant, not noisy.
    """
    if not context_chunks:
        return 0.0
    relevant = 0
    for chunk in context_chunks:
        content_lower = chunk.content.lower()
        if any(kw.lower() in content_lower for kw in answer_keywords):
            relevant += 1
    return round(relevant / len(context_chunks), 3)


def score_answer_correctness(
    answer: str, answer_keywords: list[str], reference_answer: str
) -> float:
    """
    Measures keyword coverage: fraction of expected answer concepts present.
    Lightweight proxy for semantic correctness without requiring an LLM judge.
    """
    answer_lower = answer.lower()
    found = sum(1 for kw in answer_keywords if kw.lower() in answer_lower)
    base = round(found / len(answer_keywords), 3)

    # Bonus: if answer length is reasonable (not too short)
    word_count = len(answer.split())
    length_ok = 0.1 if 30 <= word_count <= 600 else 0.0

    return round(min(base + length_ok, 1.0), 3)


# ── Per-question evaluator ─────────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.55,
    "context_recall": 0.60,
    "context_precision": 0.50,
    "answer_correctness": 0.55,
}


def evaluate_question(pipeline, item: dict) -> QuestionResult:
    """Run a single question through the full pipeline and score all metrics."""
    question = item["question"]
    start = int(time.time() * 1000)

    try:
        result = pipeline.ask(query=question)
        latency = int(time.time() * 1000) - start

        # Strip disclaimer from answer for scoring
        answer = result.answer.replace(
            "\n\n---\n⚠️ This information is for educational purposes only", ""
        ).strip()

        # Retrieve context chunks for context metrics
        context_chunks = pipeline._retriever.retrieve(question)

        return QuestionResult(
            question_id=item["id"],
            question=question,
            faithfulness=result.faithfulness,
            answer_relevancy=score_answer_relevancy(question, answer),
            context_recall=score_context_recall(context_chunks, item["context_keywords"]),
            context_precision=score_context_precision(context_chunks, item["answer_keywords"]),
            answer_correctness=score_answer_correctness(
                answer, item["answer_keywords"], item["reference_answer"]
            ),
            latency_ms=latency,
            chunks_used=result.context_chunks_used,
            safety_category=result.safety_category,
            has_warnings=result.has_warnings,
        )

    except Exception as e:
        logger.error(f"Eval error on '{question[:50]}': {e}")
        return QuestionResult(
            question_id=item["id"],
            question=question,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_recall=0.0,
            context_precision=0.0,
            answer_correctness=0.0,
            latency_ms=int(time.time() * 1000) - start,
            chunks_used=0,
            safety_category="error",
            has_warnings=True,
            error=str(e),
        )


def question_passes(r: QuestionResult) -> bool:
    return (
        r.faithfulness      >= THRESHOLDS["faithfulness"]      and
        r.answer_relevancy  >= THRESHOLDS["answer_relevancy"]  and
        r.context_recall    >= THRESHOLDS["context_recall"]    and
        r.context_precision >= THRESHOLDS["context_precision"] and
        r.answer_correctness >= THRESHOLDS["answer_correctness"]
    )


# ── Full evaluation run ────────────────────────────────────────────────────────

def run_evaluation(
    pipeline,
    dataset: list[dict] = None,
    save: bool = False,
) -> EvalReport:
    """
    Run full E2E evaluation and return an EvalReport.
    Optionally saves a JSON report to logs/eval_TIMESTAMP.json.
    """
    dataset = dataset or EVAL_DATASET
    results = []

    print(f"\nRunning E2E evaluation on {len(dataset)} questions...")
    print("─" * 60)

    for i, item in enumerate(dataset, 1):
        print(f"  [{i}/{len(dataset)}] {item['question'][:55]}...", end=" ", flush=True)
        r = evaluate_question(pipeline, item)
        results.append(r)
        status = "✓" if question_passes(r) else "✗"
        print(f"{status}  (faith={r.faithfulness:.2f} rel={r.answer_relevancy:.2f} "
              f"rec={r.context_recall:.2f} prec={r.context_precision:.2f} "
              f"corr={r.answer_correctness:.2f} {r.latency_ms}ms)")

    n = len(results)
    report = EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_questions=n,
        avg_faithfulness=    round(sum(r.faithfulness      for r in results) / n, 3),
        avg_answer_relevancy=round(sum(r.answer_relevancy  for r in results) / n, 3),
        avg_context_recall=  round(sum(r.context_recall    for r in results) / n, 3),
        avg_context_precision=round(sum(r.context_precision for r in results) / n, 3),
        avg_answer_correctness=round(sum(r.answer_correctness for r in results) / n, 3),
        avg_latency_ms=      round(sum(r.latency_ms        for r in results) / n),
        pass_rate=           round(sum(1 for r in results if question_passes(r)) / n, 3),
        question_results=[asdict(r) for r in results],
    )

    if save:
        _save_report(report)

    return report


def print_report(report: EvalReport):
    """Pretty-print the evaluation report with pass/fail per metric."""
    def fmt(val, threshold):
        tick = "✓" if val >= threshold else "✗"
        return f"{tick} {val:.2%}  (target ≥ {threshold:.0%})"

    print("\n" + "═" * 60)
    print("  PHASE 4 — END-TO-END EVALUATION REPORT")
    print("═" * 60)
    print(f"  Questions evaluated : {report.total_questions}")
    print(f"  Overall pass rate   : {report.pass_rate:.0%}")
    print(f"  Avg latency         : {report.avg_latency_ms}ms")
    print()
    print("  RAGAS-Style Metrics:")
    print(f"  Faithfulness        : {fmt(report.avg_faithfulness,     THRESHOLDS['faithfulness'])}")
    print(f"  Answer relevancy    : {fmt(report.avg_answer_relevancy,  THRESHOLDS['answer_relevancy'])}")
    print(f"  Context recall      : {fmt(report.avg_context_recall,    THRESHOLDS['context_recall'])}")
    print(f"  Context precision   : {fmt(report.avg_context_precision, THRESHOLDS['context_precision'])}")
    print(f"  Answer correctness  : {fmt(report.avg_answer_correctness,THRESHOLDS['answer_correctness'])}")
    print()

    # Per-question failures
    failed = [r for r in report.question_results if not question_passes(QuestionResult(**r))]
    if failed:
        print(f"  ✗ {len(failed)} question(s) below threshold:")
        for r in failed:
            print(f"    - [{r['question_id']}] {r['question'][:50]}...")
    else:
        print("  ✓ All questions passed all thresholds")

    print()
    _print_tuning_advice(report)
    print("═" * 60 + "\n")


def _print_tuning_advice(report: EvalReport):
    """Actionable guidance when metrics are below target."""
    advice = []
    if report.avg_faithfulness < THRESHOLDS["faithfulness"]:
        advice.append("↑ Faithfulness: tighten system prompt constraints or increase RERANK_TOP_N")
    if report.avg_answer_relevancy < THRESHOLDS["answer_relevancy"]:
        advice.append("↑ Relevancy: add query expansion or improve intent classification")
    if report.avg_context_recall < THRESHOLDS["context_recall"]:
        advice.append("↑ Context recall: increase RETRIEVAL_TOP_K or expand document corpus")
    if report.avg_context_precision < THRESHOLDS["context_precision"]:
        advice.append("↑ Context precision: decrease CHUNK_SIZE or raise RERANKER_SCORE_THRESHOLD")
    if report.avg_answer_correctness < THRESHOLDS["answer_correctness"]:
        advice.append("↑ Correctness: improve document corpus quality or refine prompt template")
    if report.avg_latency_ms > 5000:
        advice.append("↓ Latency: switch to llama-3.1-8b-instant or reduce RETRIEVAL_TOP_K")
    if advice:
        print("  Tuning advice:")
        for a in advice:
            print(f"    • {a}")
    else:
        print("  All metrics on target — system performing well.")


def _save_report(report: EvalReport):
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"eval_{ts}.json"
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\n  Report saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 E2E evaluation")
    parser.add_argument("--quick", action="store_true", help="Run 3 questions only")
    parser.add_argument("--save",  action="store_true", help="Save JSON report to logs/")
    args = parser.parse_args()

    import os
    if not os.getenv("GROQ_API_KEY"):
        print("⚠ GROQ_API_KEY not set. Set it before running the full evaluation.")
        sys.exit(1)

    from pipeline.generator import MedicalRAGPipeline
    pipeline = MedicalRAGPipeline()
    pipeline._load_components()

    dataset = EVAL_DATASET[:3] if args.quick else EVAL_DATASET
    report = run_evaluation(pipeline, dataset=dataset, save=args.save)
    print_report(report)

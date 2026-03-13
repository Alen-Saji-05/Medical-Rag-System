"""
tests/run_all.py
Master test runner — executes all Phase 1–4 test suites in sequence.

Usage:
  python tests/run_all.py                    # all suites
  python tests/run_all.py --skip-llm        # skip tests needing GROQ_API_KEY
  python tests/run_all.py --suite retrieval  # single suite only

Suites:
  1. retrieval    — Phase 1: hit rate, MRR, precision, coverage
  2. generation   — Phase 2: safety filter, prompt, post-processor
  3. e2e          — Phase 4: RAGAS-style end-to-end (requires API key)
  4. adversarial  — Phase 4: jailbreaks, boundary cases (requires API key)
"""

import os
import sys
import time
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def run_retrieval_suite() -> bool:
    section("SUITE 1 — Phase 1: Retrieval Quality")
    try:
        from pipeline.retriever import HybridRetriever
        from tests.eval_retrieval import evaluate_retrieval, print_report
        retriever = HybridRetriever()
        metrics = evaluate_retrieval(retriever)
        print_report(metrics)
        passed = (
            metrics.hit_rate_at_5 >= 0.8 and
            metrics.mrr >= 0.60 and
            metrics.precision_at_3 >= 0.50
        )
        print(f"  Suite result: {'✓ PASS' if passed else '✗ FAIL'}")
        return passed
    except Exception as e:
        print(f"  ✗ Suite failed with error: {e}")
        return False


def run_generation_suite() -> bool:
    section("SUITE 2 — Phase 2: Generation & Safety")
    try:
        from tests.test_generation import (
            test_safety_filter, test_prompt_structure, test_post_processor
        )
        results = [
            test_safety_filter(),
            test_prompt_structure(),
            test_post_processor(),
        ]
        passed = all(results)
        print(f"\n  Suite result: {'✓ PASS' if passed else '✗ FAIL'}")
        return passed
    except Exception as e:
        print(f"  ✗ Suite failed with error: {e}")
        return False


def run_e2e_suite(quick: bool = False) -> bool:
    section("SUITE 3 — Phase 4: End-to-End RAGAS Evaluation")
    if not os.getenv("GROQ_API_KEY"):
        print("  ⚠ GROQ_API_KEY not set — skipping E2E suite.")
        return True  # Don't count as failure

    try:
        from pipeline.generator import MedicalRAGPipeline
        from tests.eval_e2e import EVAL_DATASET, run_evaluation, print_report

        pipeline = MedicalRAGPipeline()
        pipeline._load_components()
        dataset = EVAL_DATASET[:3] if quick else EVAL_DATASET
        report = run_evaluation(pipeline, dataset=dataset, save=True)
        print_report(report)

        passed = (
            report.avg_faithfulness      >= 0.70 and
            report.avg_answer_relevancy  >= 0.55 and
            report.avg_context_recall    >= 0.60 and
            report.pass_rate             >= 0.70
        )
        print(f"  Suite result: {'✓ PASS' if passed else '✗ FAIL'}")
        return passed
    except Exception as e:
        print(f"  ✗ Suite failed with error: {e}")
        return False


def run_adversarial_suite() -> bool:
    section("SUITE 4 — Phase 4: Adversarial Robustness")
    if not os.getenv("GROQ_API_KEY"):
        print("  ⚠ GROQ_API_KEY not set — skipping adversarial suite.")
        return True

    try:
        from pipeline.generator import MedicalRAGPipeline
        from tests.adversarial import run_adversarial_tests, print_adversarial_report

        pipeline = MedicalRAGPipeline()
        pipeline._load_components()
        results = run_adversarial_tests(pipeline)
        print_adversarial_report(results)

        pass_rate = sum(1 for r in results if r.passed) / len(results)
        passed = pass_rate >= 0.85
        print(f"  Suite result: {'✓ PASS' if passed else '✗ FAIL'}")
        return passed
    except Exception as e:
        print(f"  ✗ Suite failed with error: {e}")
        return False


# ── Master runner ─────────────────────────────────────────────────────────────

SUITES = {
    "retrieval":   run_retrieval_suite,
    "generation":  run_generation_suite,
    "e2e":         run_e2e_suite,
    "adversarial": run_adversarial_suite,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedQuery master test runner")
    parser.add_argument("--suite", choices=list(SUITES.keys()),
                        help="Run a single suite")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip suites that require GROQ_API_KEY")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced question set for E2E")
    args = parser.parse_args()

    start = time.time()

    if args.suite:
        fn = SUITES[args.suite]
        # Pass quick flag to e2e
        result = fn(quick=args.quick) if args.suite == "e2e" else fn()
        sys.exit(0 if result else 1)

    # Run all suites
    results = {}
    skip_llm = args.skip_llm

    results["retrieval"]   = run_retrieval_suite()
    results["generation"]  = run_generation_suite()

    if not skip_llm:
        results["e2e"]         = run_e2e_suite(quick=args.quick)
        results["adversarial"] = run_adversarial_suite()
    else:
        print("\n  (Skipping LLM suites — --skip-llm flag set)")
        results["e2e"] = results["adversarial"] = None

    # Final summary
    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print("  MASTER TEST SUMMARY")
    print(f"{'═' * 60}")
    for suite, passed in results.items():
        if passed is None:
            status = "—  SKIPPED"
        elif passed:
            status = "✓  PASS"
        else:
            status = "✗  FAIL"
        print(f"  {suite:<18} {status}")

    total_run = sum(1 for v in results.values() if v is not None)
    total_pass = sum(1 for v in results.values() if v is True)
    print(f"\n  {total_pass}/{total_run} suites passed  ({elapsed:.1f}s)")
    print(f"{'═' * 60}\n")

    all_passed = all(v is not False for v in results.values())
    sys.exit(0 if all_passed else 1)

"""
cli.py — MedQuery unified command-line interface

Usage:
  python cli.py                            # interactive chat
  python cli.py --query "What is diabetes?"  # single question
  python cli.py --stream --verbose         # streaming + metadata
  python cli.py --eval retrieval           # retrieval quality eval
  python cli.py --eval e2e                 # end-to-end RAGAS eval
  python cli.py --eval adversarial         # adversarial robustness test
  python cli.py --eval all                 # all test suites
  python cli.py --monitor                  # terminal metrics dashboard
  python cli.py --monitor --html           # generate HTML report
  python cli.py --corpus-status            # corpus health check
  python cli.py --corpus-refresh           # refresh stale documents
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

HEADER = """
╔══════════════════════════════════════════════╗
║        MedQuery — Medical RAG Assistant      ║
║  Sources: NIH · CDC · WHO · Mayo Clinic      ║
╚══════════════════════════════════════════════╝
"""


# ── Response printer ──────────────────────────────────────────────────────────

def print_response(result, verbose: bool = False):
    print("\n" + "═" * 60)
    print("ANSWER")
    print("═" * 60)
    print(result.answer)

    if result.citations:
        print("\n" + "─" * 60)
        print("SOURCES")
        print("─" * 60)
        for c in result.citations:
            print(f"  [{c['number']}] {c['title']}")
            print(f"       {c['source_name']} | {c['evidence_level']} | {c['pub_date']}")
            if c.get("url") and not c["url"].startswith("local://"):
                print(f"       {c['url']}")

    if verbose:
        print("\n" + "─" * 60)
        print("METADATA")
        print("─" * 60)
        print(f"  Faithfulness score : {result.faithfulness:.2f}")
        print(f"  Chunks used        : {result.context_chunks_used}")
        print(f"  Context truncated  : {result.context_truncated}")
        print(f"  Safety category    : {result.safety_category}")
        print(f"  Latency            : {result.latency_ms}ms")
        if result.flagged_sentences:
            print(f"  ⚠ Flagged sentences ({len(result.flagged_sentences)}):")
            for s in result.flagged_sentences:
                print(f"    - {s[:100]}...")
    print()


# ── API key check ─────────────────────────────────────────────────────────────

def require_api_key():
    if not os.getenv("GROQ_API_KEY"):
        print("─" * 60)
        print("⚠  GROQ_API_KEY is not set.")
        print("   Linux/Mac: export GROQ_API_KEY='gsk_...'")
        print("   Windows:   $env:GROQ_API_KEY='gsk_...'")
        print("   Get a free key at: https://console.groq.com")
        print("─" * 60)
        sys.exit(1)


# ── Chat modes ────────────────────────────────────────────────────────────────

def run_interactive(stream=False, verbose=False):
    require_api_key()
    from pipeline.generator import MedicalRAGPipeline
    from pipeline.prompt_builder import ConversationTurn

    pipeline = MedicalRAGPipeline()
    history = []
    print(HEADER)
    print("  Type your medical question. Commands: quit | clear | verbose")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if query.lower() == "clear":
            history.clear()
            print("  [History cleared]\n")
            continue
        if query.lower() == "verbose":
            verbose = not verbose
            print(f"  [Verbose: {'on' if verbose else 'off'}]\n")
            continue

        result = pipeline.ask(query, history=history, stream=stream)
        print_response(result, verbose=verbose)
        history.append(ConversationTurn(role="user", content=query))
        history.append(ConversationTurn(role="assistant", content=result.answer))


def run_single(query, stream=False, verbose=False):
    require_api_key()
    from pipeline.generator import MedicalRAGPipeline
    pipeline = MedicalRAGPipeline()
    result = pipeline.ask(query, stream=stream)
    print_response(result, verbose=verbose)


# ── Eval modes ────────────────────────────────────────────────────────────────

def run_eval(suite: str, quick=False):
    if suite in ("retrieval",):
        from pipeline.retriever import HybridRetriever
        from tests.eval_retrieval import evaluate_retrieval, print_report
        print("\nRunning retrieval evaluation...")
        metrics = evaluate_retrieval(HybridRetriever())
        print_report(metrics)

    elif suite == "e2e":
        require_api_key()
        from pipeline.generator import MedicalRAGPipeline
        from tests.eval_e2e import EVAL_DATASET, run_evaluation, print_report
        pipeline = MedicalRAGPipeline()
        pipeline._load_components()
        dataset = EVAL_DATASET[:3] if quick else EVAL_DATASET
        report = run_evaluation(pipeline, dataset=dataset, save=True)
        print_report(report)

    elif suite == "adversarial":
        require_api_key()
        from pipeline.generator import MedicalRAGPipeline
        from tests.adversarial import run_adversarial_tests, print_adversarial_report
        pipeline = MedicalRAGPipeline()
        pipeline._load_components()
        results = run_adversarial_tests(pipeline)
        print_adversarial_report(results)

    elif suite == "all":
        import subprocess
        flags = ["--quick"] if quick else []
        subprocess.run(
            [sys.executable, "tests/run_all.py"] + flags,
            cwd=Path(__file__).parent
        )

    else:
        print(f"Unknown eval suite: {suite}. Choose: retrieval | e2e | adversarial | all")
        sys.exit(1)


# ── Monitoring ────────────────────────────────────────────────────────────────

def run_monitor(html=False, watch=None):
    from monitoring.dashboard import render_terminal, generate_html_report
    from monitoring.corpus_refresh import CorpusHealthChecker
    from pipeline.audit_log import audit_logger
    import time as _time

    checker = CorpusHealthChecker()

    if html:
        from pathlib import Path as P
        stats  = audit_logger.get_stats()
        health = checker.check()
        report = generate_html_report(stats, health)
        out = P(__file__).parent / "logs" / "dashboard.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"Dashboard saved → {out}")
    elif watch:
        print(f"Watching (refresh every {watch}s) — Ctrl+C to stop")
        while True:
            render_terminal(audit_logger.get_stats(), checker.check())
            _time.sleep(watch)
    else:
        render_terminal(audit_logger.get_stats(), checker.check())


def run_corpus(refresh=False, force=False):
    from monitoring.corpus_refresh import CorpusHealthChecker, CorpusRefreshManager
    if refresh:
        manager = CorpusRefreshManager()
        summary = manager.run_refresh(force=force)
        print(f"\nRefresh summary: {summary}")
    else:
        CorpusHealthChecker().print_status()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MedQuery CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--query",   "-q", type=str,   help="Ask a single question")
    p.add_argument("--stream",  "-s", action="store_true", help="Stream output")
    p.add_argument("--verbose", "-v", action="store_true", help="Show metadata")
    p.add_argument("--eval",    "-e", type=str,
                   metavar="SUITE",
                   help="Run eval suite: retrieval | e2e | adversarial | all")
    p.add_argument("--quick",         action="store_true", help="Short eval (3 questions)")
    p.add_argument("--monitor", "-m", action="store_true", help="Terminal metrics dashboard")
    p.add_argument("--html",          action="store_true", help="Generate HTML dashboard")
    p.add_argument("--watch",         type=int, metavar="SECS",
                   help="Auto-refresh dashboard every N seconds")
    p.add_argument("--corpus-status", action="store_true", help="Corpus health check")
    p.add_argument("--corpus-refresh",action="store_true", help="Refresh stale documents")
    p.add_argument("--force",         action="store_true", help="Force-refresh all docs")
    args = p.parse_args()

    if args.eval:
        run_eval(args.eval, quick=args.quick)
    elif args.monitor:
        run_monitor(html=args.html, watch=args.watch)
    elif args.corpus_status:
        run_corpus(refresh=False)
    elif args.corpus_refresh:
        run_corpus(refresh=True, force=args.force)
    elif args.query:
        run_single(args.query, stream=args.stream, verbose=args.verbose)
    else:
        run_interactive(stream=args.stream, verbose=args.verbose)

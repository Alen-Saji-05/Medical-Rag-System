"""
cli.py
Interactive command-line interface for testing the full RAG pipeline.

Usage:
  python cli.py                        # interactive chat mode
  python cli.py --query "What is diabetes?"  # single question mode
  python cli.py --stream               # streaming output mode
  python cli.py --eval                 # run retrieval evaluation suite
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def print_response(result, verbose: bool = False):
    """Pretty-print a RAGResponse."""
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


def check_api_key():
    if not os.getenv("GROQ_API_KEY"):
        print("─" * 60)
        print("⚠  GROQ_API_KEY is not set.")
        print("   Set it before running:")
        print("   Windows:   $env:GROQ_API_KEY='sk-...'")
        print("   Linux/Mac: export GROQ_API_KEY='sk-...'")
        print("─" * 60)
        sys.exit(1)


def run_interactive(stream: bool = False, verbose: bool = False):
    """Interactive multi-turn chat session."""
    check_api_key()
    from pipeline.generator import MedicalRAGPipeline
    from pipeline.prompt_builder import ConversationTurn

    pipeline = MedicalRAGPipeline()
    history = []

    print("\n" + "═" * 60)
    print("  Medical Knowledge Assistant")
    print("  Phase 2 — RAG + LLM Generation")
    print("─" * 60)
    print("  Type your medical question and press Enter.")
    print("  Commands: 'quit' to exit | 'clear' to reset history")
    print("           'verbose' to toggle detailed metadata")
    print("═" * 60 + "\n")

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
            print("  [Conversation history cleared]\n")
            continue
        if query.lower() == "verbose":
            verbose = not verbose
            print(f"  [Verbose mode: {'on' if verbose else 'off'}]\n")
            continue

        result = pipeline.ask(query, history=history, stream=stream)
        print_response(result, verbose=verbose)

        # Update history
        history.append(ConversationTurn(role="user", content=query))
        history.append(ConversationTurn(role="assistant", content=result.answer))


def run_single(query: str, stream: bool = False, verbose: bool = False):
    """Ask a single question and exit."""
    check_api_key()
    from pipeline.generator import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline()
    result = pipeline.ask(query, stream=stream)
    print_response(result, verbose=verbose)


def run_eval():
    """Run the retrieval evaluation suite."""
    from pipeline.retriever import HybridRetriever
    from tests.eval_retrieval import evaluate_retrieval, print_report

    print("\nRunning retrieval evaluation...")
    retriever = HybridRetriever()
    metrics = evaluate_retrieval(retriever)
    print_report(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical RAG CLI")
    parser.add_argument("--query", "-q", type=str, help="Single question mode")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show metadata")
    parser.add_argument("--eval", "-e", action="store_true", help="Run eval suite")
    args = parser.parse_args()

    if args.eval:
        run_eval()
    elif args.query:
        run_single(args.query, stream=args.stream, verbose=args.verbose)
    else:
        run_interactive(stream=args.stream, verbose=args.verbose)

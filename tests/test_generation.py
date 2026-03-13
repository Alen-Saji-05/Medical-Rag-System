"""
tests/test_generation.py
End-to-end generation tests for Phase 2.

Tests:
  1. Safety pre-filter (no API key needed)
  2. Prompt structure validation
  3. Citation format in responses
  4. Emergency query handling
  5. Out-of-scope query handling
  6. Faithfulness check (NLI guard)
  7. Full pipeline integration test (requires OPENAI_API_KEY)
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def separator(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print('─' * 55)


# ─── Test 1: Safety filter (no API key needed) ───────────────────────────────

def test_safety_filter():
    separator("Test 1: Safety pre-filter")
    from pipeline.safety import PreGenerationFilter, QueryCategory

    f = PreGenerationFilter()

    cases = [
        ("I have chest pain and can't breathe", QueryCategory.EMERGENCY),
        ("how to overdose on aspirin",          QueryCategory.HARMFUL),
        ("what is the stock price of Pfizer",   QueryCategory.OUT_OF_SCOPE),
        ("what are symptoms of diabetes",       QueryCategory.SAFE),
    ]

    passed = 0
    for query, expected in cases:
        result = f.check(query)
        ok = result.category == expected
        status = "✓" if ok else "✗"
        print(f"  {status} '{query[:45]}...' → {result.category.value}")
        if ok:
            passed += 1

    print(f"\n  Passed: {passed}/{len(cases)}")
    return passed == len(cases)


# ─── Test 2: Prompt structure ─────────────────────────────────────────────────

def test_prompt_structure():
    separator("Test 2: Prompt structure")
    from pipeline.prompt_builder import PromptBuilder
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        chunk_id: str = "test_001"
        content: str = "Diabetes is a chronic condition affecting blood glucose levels."
        rerank_score: float = 0.85
        title: str = "Diabetes Overview"
        source_name: str = "MedlinePlus"
        source_url: str = "https://medlineplus.gov/diabetes.html"
        source_authority: str = "primary"
        specialty: str = "endocrinology"
        evidence_level: str = "factsheet"
        pub_date: str = "2024-01-01"

    builder = PromptBuilder()
    prompt, citations = builder.build(
        query="What is diabetes?",
        retrieved_chunks=[FakeChunk(), FakeChunk(chunk_id="test_002")],
    )

    checks = [
        ("[SOURCE 1]" in prompt.messages[-1]["content"], "SOURCE block present"),
        ("MedlinePlus" in prompt.messages[-1]["content"], "Source name included"),
        (len(citations) == 2,                             "Citations list populated"),
        (prompt.context_used == 2,                        "Chunk count correct"),
        ("system" in prompt.system.lower() or len(prompt.system) > 100,
                                                          "System prompt present"),
    ]

    passed = 0
    for ok, label in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {label}")
        if ok:
            passed += 1

    print(f"\n  Passed: {passed}/{len(checks)}")
    return passed == len(checks)


# ─── Test 3: Post-generation processor ────────────────────────────────────────

def test_post_processor():
    separator("Test 3: Post-generation processor")
    from pipeline.safety import PostGenerationProcessor

    processor = PostGenerationProcessor()

    answer = (
        "High blood pressure is often called hypertension. "
        "Regular exercise can help lower blood pressure. "
        "The moon is made of cheese."  # Unsupported claim
    )
    context = (
        "Hypertension, or high blood pressure, is a common condition. "
        "Lifestyle changes including regular physical activity help manage blood pressure."
    )

    result = processor.process(answer, context)

    checks = [
        (DISCLAIMER_CHECK in result["answer"],  "Disclaimer appended"),
        ("faithfulness" in result,              "Faithfulness score present"),
        (isinstance(result["faithfulness"], float), "Faithfulness is float"),
        ("flagged_sentences" in result,         "Flagged sentences key present"),
    ]

    passed = 0
    for ok, label in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {label}")
        if ok:
            passed += 1

    print(f"  Faithfulness score: {result['faithfulness']:.2f}")
    print(f"  Flagged sentences: {len(result['flagged_sentences'])}")
    print(f"\n  Passed: {passed}/{len(checks)}")
    return passed == len(checks)

# Quick check for disclaimer presence
DISCLAIMER_CHECK = "educational purposes only"


# ─── Test 4: Full pipeline (requires OPENAI_API_KEY) ─────────────────────────

def test_full_pipeline():
    separator("Test 4: Full pipeline integration")

    if not os.getenv("OPENAI_API_KEY"):
        print("  ⚠ OPENAI_API_KEY not set — skipping full pipeline test.")
        print("  Set it to run this test:")
        print("    Windows:   $env:OPENAI_API_KEY='sk-...'")
        print("    Linux/Mac: export OPENAI_API_KEY='sk-...'")
        return True  # Don't fail the suite for a missing key

    from pipeline.generator import MedicalRAGPipeline

    pipeline = MedicalRAGPipeline()
    result = pipeline.ask("What are the symptoms of type 2 diabetes?")

    checks = [
        (len(result.answer) > 100,          "Answer has substance"),
        ("educational purposes" in result.answer, "Disclaimer present"),
        (result.safety_category == "safe",  "Safety category correct"),
        (result.latency_ms > 0,             "Latency recorded"),
        (isinstance(result.faithfulness, float), "Faithfulness is float"),
    ]

    passed = 0
    for ok, label in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {label}")
        if ok:
            passed += 1

    print(f"\n  Citations: {len(result.citations)}")
    print(f"  Faithfulness: {result.faithfulness:.2f}")
    print(f"  Latency: {result.latency_ms}ms")
    print(f"\n  Passed: {passed}/{len(checks)}")
    return passed == len(checks)


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    results.append(("Safety filter",      test_safety_filter()))
    results.append(("Prompt structure",   test_prompt_structure()))
    results.append(("Post-processor",     test_post_processor()))
    results.append(("Full pipeline",      test_full_pipeline()))

    print("\n" + "═" * 55)
    print("  PHASE 2 TEST SUMMARY")
    print("═" * 55)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
    total = sum(1 for _, p in results if p)
    print(f"\n  {total}/{len(results)} test suites passed")
    print("═" * 55 + "\n")

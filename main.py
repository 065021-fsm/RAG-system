"""
main.py — Orchestrate the full Agentic RAG pipeline.

Flow: Config → Ingestion → RAG Pipeline → Evaluation → Output JSON
"""

import json
import sys
import time
from pathlib import Path
from config import load_config
from ingest import run_ingestion
from run_pipeline import run_pipeline
from evaluate import evaluate_with_ragas


def main():
    """Run the complete Agentic RAG system."""
    start_time = time.time()

    print("=" * 80)
    print("  AGENTIC RAG SYSTEM")
    print("=" * 80)

    # ── Step 1: Load Configuration ────────────────────────────────────────────
    print("\n[Step 1] Loading configuration...")
    config = load_config()
    print(f"  LLM: {config.llm_name}")
    print(f"  Embedding: {config.embedding_model_name}")
    print(f"  Database: {config.db_name} @ {config.db_host}")
    print(f"  Evaluation: {config.evaluation_framework}")
    print(f"  Judge LLM: {config.judge_llm}")

    # Check if we have intermediate results to skip to evaluation
    intermediate_path = Path("results_intermediate.json")
    if intermediate_path.exists():
        print(f"\n[Step 3] Loading intermediate results from {intermediate_path}...")
        with open(intermediate_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} results.")
    else:
        # ── Step 2: Ingest Documents ──────────────────────────────────────────────
        print("\n[Step 2] Ingesting dataset into vector store...")
        run_ingestion(config)
        print("  Ingestion complete!")

        # ── Step 3: Run RAG Pipeline ──────────────────────────────────────────────
        print("\n[Step 3] Running Agentic RAG pipeline...")
        results = run_pipeline(config)
        print(f"  Pipeline complete! {len(results)} results generated.")

        # Save intermediate results
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Intermediate results saved to {intermediate_path}")

    # ── Step 4: Evaluate Results ──────────────────────────────────────────────
    print("\n[Step 4] Evaluating results with {config.evaluation_framework}...")
    results = evaluate_with_ragas(results, config)
    print("  Evaluation complete!")

    # ── Step 5: Write Output ──────────────────────────────────────────────────
    print(f"\n[Step 5] Writing results to {config.output_file}...")

    # Clean up results for output
    output_data = []
    for r in results:
        entry = {
            "Question": r["question"],
            "Generated Answer": r["generated_answer"],
            "Expected Answer": r["expected_answer"],
            "Retrieved Context": r.get("retrieved_context", []),
            "Evaluation scores": r.get("evaluation_scores", {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "contextual_precision": 0.0,
            }),
            "Evaluation framework used": r.get("evaluation_framework", config.evaluation_framework),
            "<small_llm> name": r.get("llm_model", config.llm_name),
            "<embedding_model> name": r.get("embedding_model", config.embedding_model_name),
        }
        output_data.append(entry)

    output_path = Path(config.output_file)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"  COMPLETE! Results written to: {output_path}")
    print(f"  Total entries: {len(output_data)}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"{'='*80}")

    return output_data


if __name__ == "__main__":
    main()

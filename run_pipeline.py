"""
run_pipeline.py — Load Q&A files, run RAG pipeline for each question, package results.
"""

import re
import sys
from pathlib import Path
from config import load_config, Config
from rag_agent import run_rag_query


def parse_numbered_list(filepath: str) -> list[str]:
    """Parse a numbered list .txt file (e.g., '1. question text')."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    items = []
    current_item = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Check if this line starts a new numbered item
        match = re.match(r"^\d+[\.\)]\s*", line)
        if match:
            if current_item:
                items.append(" ".join(current_item))
            # Remove the number prefix
            text = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            current_item = [text]
        else:
            # Continuation of previous item
            if current_item:
                current_item.append(line)
            else:
                current_item = [line]

    if current_item:
        items.append(" ".join(current_item))

    return items


def load_qa_pairs(config: Config) -> list[tuple[str, str]]:
    """Load and validate question-answer pairs."""
    questions = parse_numbered_list(config.questions_path)
    answers = parse_numbered_list(config.answers_path)

    print(f"[Pipeline] Loaded {len(questions)} questions from {config.questions_path}")
    print(f"[Pipeline] Loaded {len(answers)} answers from {config.answers_path}")

    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatch: {len(questions)} questions vs {len(answers)} answers. "
            "Both files must contain the same number of entries."
        )

    return list(zip(questions, answers))


def run_pipeline(config: Config = None) -> list[dict]:
    """Run the full RAG pipeline for all Q&A pairs."""
    if config is None:
        config = load_config()

    qa_pairs = load_qa_pairs(config)
    results = []

    print(f"\n[Pipeline] Processing {len(qa_pairs)} questions through Agentic RAG pipeline...")
    print(f"[Pipeline] LLM: {config.llm_name}")
    print(f"[Pipeline] Embedding: {config.embedding_model_name}")
    print("=" * 80)

    for i, (question, expected_answer) in enumerate(qa_pairs):
        print(f"\n{'='*80}")
        print(f"[Pipeline] Question {i+1}/{len(qa_pairs)}: {question[:80]}...")
        print(f"{'─'*80}")

        try:
            rag_result = run_rag_query(question, config)

            result = {
                "question_index": i + 1,
                "question": question,
                "generated_answer": rag_result["generated_answer"],
                "expected_answer": expected_answer,
                "retrieved_context": rag_result["context_snippets"],
                "is_fallback": rag_result["is_fallback"],
                "llm_model": config.llm_name,
                "embedding_model": config.embedding_model_name,
            }

            print(f"  [Result] Generated: {result['generated_answer'][:100]}...")
            print(f"  [Result] Expected: {expected_answer[:100]}...")
            print(f"  [Result] Context snippets: {len(rag_result['context_snippets'])}")

        except Exception as e:
            print(f"  [ERROR] Failed to process question {i+1}: {e}")
            result = {
                "question_index": i + 1,
                "question": question,
                "generated_answer": f"ERROR: {str(e)}",
                "expected_answer": expected_answer,
                "retrieved_context": [],
                "is_fallback": True,
                "llm_model": config.llm_name,
                "embedding_model": config.embedding_model_name,
            }

        results.append(result)

    print(f"\n{'='*80}")
    print(f"[Pipeline] Completed! Processed {len(results)} questions.")
    return results


if __name__ == "__main__":
    results = run_pipeline()
    import json
    print(json.dumps(results[:2], indent=2))

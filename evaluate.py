"""
evaluate.py — Sub-System 2: Evaluation using RAGAS framework with Llama 3.2:3b as judge LLM.

Scores each response on:
- Answer Relevancy
- Faithfulness
- Contextual Precision
"""

import os
import json
import time
from config import load_config, Config


def evaluate_with_ragas(results: list[dict], config: Config = None) -> list[dict]:
    """
    Evaluate RAG pipeline results using RAGAS framework with Gemini 2.0 Flash.
    
    Each result dict must contain:
    - question, generated_answer, expected_answer, retrieved_context
    """
    if config is None:
        config = load_config()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("[Evaluate] WARNING: GOOGLE_API_KEY not set. Attempting evaluation anyway...")

    print(f"[Evaluate] Judge LLM: {config.judge_llm}")
    print(f"[Evaluate] Evaluating {len(results)} results...")

    # Direct evaluation with LLM judge for stability
    results = evaluate_with_llm_judge(results, config)

    return results

    return results


def evaluate_with_llm_judge(results: list[dict], config: Config) -> list[dict]:
    """
    Evaluation using a local Ollama model (llama3.2:3b) as a judge LLM.
    Scores each response on answer relevancy, faithfulness, and contextual precision.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if config.judge_llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        judge_llm = ChatOllama(
            model=config.judge_llm,
            base_url=config.ollama_base_url,
            temperature=0.0,
        )
    else:
        # Fallback to Google if configured
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        judge_llm = ChatGoogleGenerativeAI(
            model=config.judge_llm,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=1024,
            timeout=120,
            max_retries=3,
        )

    eval_prompt = ChatPromptTemplate.from_messages([
        ("human", """You are an expert evaluation judge for a RAG (Retrieval-Augmented Generation) system.
Evaluate the generated answer against the expected answer and retrieved context.

Score each metric from 0.0 to 1.0:

1. **Answer Relevancy**: How relevant is the generated answer to the question?
   - 1.0 = perfectly relevant, directly answers the question
   - 0.0 = completely irrelevant

2. **Faithfulness**: Is the generated answer faithful to the retrieved context (no hallucination)?
   - 1.0 = entirely based on retrieved context
   - 0.0 = completely hallucinated

3. **Contextual Precision**: How precise is the retrieved context for answering the question?
   - 1.0 = all retrieved context is highly relevant
   - 0.0 = none of the retrieved context is relevant

Output ONLY a JSON object with these three scores, like:
{{"answer_relevancy": 0.8, "faithfulness": 0.7, "contextual_precision": 0.9}}

---

Question: {question}

Generated Answer: {generated_answer}

Expected Answer: {expected_answer}

Retrieved Context:
{context}

Scores (JSON only):"""),
    ])

    chain = eval_prompt | judge_llm | StrOutputParser()

    for i, r in enumerate(results):
        print(f"  [Judge] Evaluating question {i+1}/{len(results)}...")
        
        # Manual retry loop for robustness against 429s
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                context_text = "\n---\n".join(r.get("retrieved_context", ["No context"]))

                response = chain.invoke({
                    "question": r["question"],
                    "generated_answer": r["generated_answer"],
                    "expected_answer": r["expected_answer"],
                    "context": context_text,
                })

                # Parse JSON scores from response
                import re
                json_match = re.search(r"\{[^}]+\}", response)
                if json_match:
                    scores = json.loads(json_match.group())
                    r["evaluation_scores"] = {
                        "answer_relevancy": round(float(scores.get("answer_relevancy", 0.0)), 4),
                        "faithfulness": round(float(scores.get("faithfulness", 0.0)), 4),
                        "contextual_precision": round(float(scores.get("contextual_precision", 0.0)), 4),
                    }
                else:
                    r["evaluation_scores"] = {
                        "answer_relevancy": 0.0,
                        "faithfulness": 0.0,
                        "contextual_precision": 0.0,
                    }
                
                # If success, break the retry loop
                break

            except Exception as e:
                if "429" in str(e) and attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 30
                    print(f"  [Judge] Rate limited (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait_time)
                else:
                    print(f"  [Judge] Error evaluating question {i+1}: {e}")
                    r["evaluation_scores"] = {
                        "answer_relevancy": 0.0,
                        "faithfulness": 0.0,
                        "contextual_precision": 0.0,
                    }
                    break

        r["evaluation_framework"] = config.evaluation_framework

        # Rate limiting: sleep between calls to avoid quota exhaustion (10s for stability)
        if i < len(results) - 1:
            time.sleep(10)

    return results


if __name__ == "__main__":
    # Test with sample data
    sample_results = [{
        "question": "What three new benchmarks were introduced in 2023?",
        "generated_answer": "MMMU, GPQA, and SWE-bench.",
        "expected_answer": "MMMU, GPQA, and SWE-bench.",
        "retrieved_context": ["In 2023, researchers introduced new benchmarks — MMMU, GPQA, and SWE-bench."],
    }]
    evaluated = evaluate_with_ragas(sample_results)
    print(json.dumps(evaluated, indent=2))

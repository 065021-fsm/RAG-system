"""
rag_agent.py — LangGraph state machine implementing the Agentic RAG pipeline.

Agentic behaviors:
1. Query Rewriting — rewrite ambiguous queries before retrieval
2. Retrieval Retries — retry with revised strategy if context scores below threshold
3. Multi-hop Decomposition — break complex questions into sub-queries
4. Fallback Handling — flag responses when context is insufficient
"""

import re
from typing import TypedDict, Annotated, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from config import load_config, Config


# ── State Schema ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """State for the agentic RAG pipeline."""
    original_query: str
    current_query: str
    sub_queries: list[str]
    retrieved_docs: list[Document]
    all_retrieved_docs: list[Document]
    relevance_score: float
    retry_count: int
    max_retries: int
    generated_answer: str
    context_snippets: list[str]
    is_fallback: bool
    is_multi_hop: bool
    sub_query_index: int
    relevance_threshold: float


# ── Utility ───────────────────────────────────────────────────────────────────

def get_llm(config: Config) -> ChatOllama:
    """Get the Ollama LLM instance."""
    return ChatOllama(
        model=config.llm_name,
        base_url=config.ollama_base_url,
        temperature=0.1,
        num_ctx=4096,
    )


def get_vector_store(config: Config) -> PGVector:
    """Get the PGVector store instance."""
    embeddings = OllamaEmbeddings(
        model=config.embedding_model_name,
        base_url=config.ollama_base_url,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=config.collection_name,
        connection=config.pg_connection_string,
    )


def compute_relevance_score(docs: list[Document], query: str, embeddings) -> float:
    """Compute average relevance score based on retrieval results."""
    if not docs:
        return 0.0
    # Use the number of docs returned and their content length as a proxy
    # More sophisticated: use embedding similarity scores
    # PGVector returns docs ordered by similarity, so top docs are most relevant
    # We'll use a heuristic: if we got docs with meaningful content, score higher
    total_content_len = sum(len(d.page_content) for d in docs)
    if total_content_len < 50:
        return 0.1
    elif total_content_len < 200:
        return 0.3
    elif len(docs) >= 3:
        return 0.7
    else:
        return 0.5


# ── Node Functions ────────────────────────────────────────────────────────────

def query_rewriter(state: RAGState, config: Config) -> dict:
    """Rewrite ambiguous or underspecified queries for better retrieval."""
    llm = get_llm(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query rewriting assistant. Your task is to rewrite the user's question 
to make it more specific and better suited for document retrieval from a knowledge base about AI trends, 
benchmarks, investment, regulations, and education.

Rules:
- Keep the rewritten query concise (1-2 sentences max)
- Make implicit concepts explicit
- Add relevant keywords that might appear in the source documents
- Output ONLY the rewritten query, nothing else"""),
        ("human", "Original question: {query}\n\nRewritten query:"),
    ])

    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"query": state["original_query"]})
    rewritten = rewritten.strip()

    # If the LLM returns something too long or empty, use the original
    if not rewritten or len(rewritten) > 500:
        rewritten = state["original_query"]

    print(f"  [QueryRewriter] Original: {state['original_query'][:80]}...")
    print(f"  [QueryRewriter] Rewritten: {rewritten[:80]}...")

    return {
        "current_query": rewritten,
        "sub_queries": [],
        "retrieved_docs": [],
        "all_retrieved_docs": [],
        "relevance_score": 0.0,
        "retry_count": 0,
        "is_fallback": False,
        "is_multi_hop": False,
        "sub_query_index": 0,
    }


def retriever(state: RAGState, config: Config) -> dict:
    """Retrieve relevant documents from pgvector store."""
    vector_store = get_vector_store(config)

    query = state["current_query"]
    if state.get("sub_queries") and state.get("sub_query_index", 0) < len(state["sub_queries"]):
        query = state["sub_queries"][state["sub_query_index"]]

    print(f"  [Retriever] Searching for: {query[:80]}...")

    docs = vector_store.similarity_search(query, k=5)

    # Compute relevance score
    embeddings = OllamaEmbeddings(
        model=config.embedding_model_name,
        base_url=config.ollama_base_url,
    )
    score = compute_relevance_score(docs, query, embeddings)

    # Accumulate all retrieved docs across sub-queries
    all_docs = list(state.get("all_retrieved_docs", []))
    all_docs.extend(docs)

    # Deduplicate by content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    print(f"  [Retriever] Found {len(docs)} docs, relevance score: {score:.2f}")

    return {
        "retrieved_docs": docs,
        "all_retrieved_docs": unique_docs,
        "relevance_score": score,
    }


def retry_handler(state: RAGState, config: Config) -> dict:
    """Reformulate query using a different strategy and retry retrieval."""
    llm = get_llm(config)

    retry_count = state["retry_count"] + 1
    print(f"  [RetryHandler] Retry {retry_count}: reformulating query...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are helping reformulate a search query because the initial search didn't find 
sufficiently relevant results. Try a different angle or use different keywords.

Rules:
- Use different terminology than the original query
- Try to be more specific or broaden the scope as appropriate
- Output ONLY the reformulated query, nothing else"""),
        ("human", """Original question: {original}
Previous query that didn't work well: {current}

Reformulated query:"""),
    ])

    chain = prompt | llm | StrOutputParser()
    new_query = chain.invoke({
        "original": state["original_query"],
        "current": state["current_query"],
    })
    new_query = new_query.strip()

    if not new_query or len(new_query) > 500:
        new_query = state["original_query"]

    print(f"  [RetryHandler] New query: {new_query[:80]}...")

    return {
        "current_query": new_query,
        "retry_count": retry_count,
    }


def multi_hop_decomposer(state: RAGState, config: Config) -> dict:
    """Decompose complex questions into sub-queries."""
    llm = get_llm(config)

    print(f"  [MultiHopDecomposer] Decomposing complex question...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question decomposition assistant. Break the complex question into 2-3 
simpler sub-questions that can each be answered independently from a document store.

Rules:
- Output each sub-question on a new line, numbered (1. 2. 3.)
- Each sub-question should be self-contained
- Together they should cover all aspects of the original question
- Output ONLY the numbered sub-questions, nothing else"""),
        ("human", "Complex question: {query}\n\nSub-questions:"),
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": state["original_query"]})

    # Parse sub-queries
    sub_queries = []
    for line in result.strip().split("\n"):
        line = line.strip()
        # Remove numbering like "1.", "2.", etc.
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned) > 10:
            sub_queries.append(cleaned)

    if not sub_queries:
        sub_queries = [state["current_query"]]

    print(f"  [MultiHopDecomposer] Generated {len(sub_queries)} sub-queries")
    for i, sq in enumerate(sub_queries):
        print(f"    Sub-query {i+1}: {sq[:80]}...")

    return {
        "sub_queries": sub_queries,
        "sub_query_index": 0,
        "is_multi_hop": True,
    }


def sub_query_retriever(state: RAGState, config: Config) -> dict:
    """Retrieve documents for each sub-query in multi-hop decomposition."""
    vector_store = get_vector_store(config)

    all_docs = list(state.get("all_retrieved_docs", []))

    for i, sq in enumerate(state.get("sub_queries", [])):
        print(f"  [SubQueryRetriever] Searching sub-query {i+1}: {sq[:60]}...")
        docs = vector_store.similarity_search(sq, k=3)
        all_docs.extend(docs)

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    print(f"  [SubQueryRetriever] Total unique docs after multi-hop: {len(unique_docs)}")

    return {
        "all_retrieved_docs": unique_docs,
        "retrieved_docs": unique_docs,
        "relevance_score": 0.7 if unique_docs else 0.1,
    }


def answer_generator(state: RAGState, config: Config) -> dict:
    """Generate final answer using LLM with retrieved context."""
    llm = get_llm(config)

    docs = state.get("all_retrieved_docs", state.get("retrieved_docs", []))
    context = "\n\n".join([doc.page_content for doc in docs])
    context_snippets = [doc.page_content for doc in docs]

    print(f"  [AnswerGenerator] Generating answer with {len(docs)} context docs...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.
        
Rules:
- Answer the question using ONLY information from the context below
- Be specific and include relevant numbers, dates, and facts from the context
- If the context doesn't contain enough information to fully answer, say so clearly
- Keep your answer concise but complete"""),
        ("human", """Context:
{context}

Question: {question}

Answer:"""),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["original_query"],
    })

    print(f"  [AnswerGenerator] Answer: {answer[:100]}...")

    return {
        "generated_answer": answer.strip(),
        "context_snippets": context_snippets,
        "is_fallback": False,
    }


def fallback_handler(state: RAGState, config: Config) -> dict:
    """Handle cases where retrieved context is insufficient — abstain rather than hallucinate."""
    print(f"  [FallbackHandler] Insufficient context, providing fallback response")

    docs = state.get("all_retrieved_docs", state.get("retrieved_docs", []))
    context_snippets = [doc.page_content for doc in docs] if docs else []

    return {
        "generated_answer": (
            "INSUFFICIENT CONTEXT: The retrieved documents do not contain enough "
            "relevant information to confidently answer this question. The system "
            "is abstaining from generating a potentially inaccurate response."
        ),
        "context_snippets": context_snippets,
        "is_fallback": True,
    }


# ── Routing Functions ─────────────────────────────────────────────────────────

def check_relevance(state: RAGState) -> str:
    """Route based on relevance score and retry count."""
    score = state.get("relevance_score", 0.0)
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    threshold = state.get("relevance_threshold", 0.3)

    if score >= threshold:
        return "relevant"
    elif retries < max_retries:
        return "retry"
    else:
        return "fallback"


def check_multi_hop(state: RAGState) -> str:
    """Determine if the question needs multi-hop decomposition."""
    query = state.get("original_query", "")

    # Heuristics for multi-hop detection
    multi_hop_indicators = [
        " and ",
        "compare",
        "combined",
        "both",
        "multiple",
        "together",
        "implications",
        "trends",
        "relationship between",
        "how do",
        "what are the",
        "what does",
    ]

    query_lower = query.lower()

    # Check if query is complex (long or contains multi-hop indicators)
    is_complex = (
        len(query.split()) > 20
        or sum(1 for ind in multi_hop_indicators if ind in query_lower) >= 2
    )

    if is_complex and not state.get("is_multi_hop", False):
        return "decompose"
    else:
        return "generate"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_rag_graph(config: Config) -> StateGraph:
    """Build the LangGraph state machine for the agentic RAG pipeline."""

    # Create node wrappers that inject config
    def _query_rewriter(state):
        return query_rewriter(state, config)

    def _retriever(state):
        return retriever(state, config)

    def _retry_handler(state):
        return retry_handler(state, config)

    def _multi_hop_decomposer(state):
        return multi_hop_decomposer(state, config)

    def _sub_query_retriever(state):
        return sub_query_retriever(state, config)

    def _answer_generator(state):
        return answer_generator(state, config)

    def _fallback_handler(state):
        return fallback_handler(state, config)

    # Build the graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("query_rewriter", _query_rewriter)
    graph.add_node("retriever", _retriever)
    graph.add_node("retry_handler", _retry_handler)
    graph.add_node("multi_hop_decomposer", _multi_hop_decomposer)
    graph.add_node("sub_query_retriever", _sub_query_retriever)
    graph.add_node("answer_generator", _answer_generator)
    graph.add_node("fallback_handler", _fallback_handler)

    # Add edges
    graph.add_edge(START, "query_rewriter")
    graph.add_edge("query_rewriter", "retriever")

    # After retrieval, check relevance
    graph.add_conditional_edges(
        "retriever",
        check_relevance,
        {
            "relevant": "check_multi_hop",  # virtual node for routing
            "retry": "retry_handler",
            "fallback": "fallback_handler",
        },
    )

    # Retry loops back to retriever
    graph.add_edge("retry_handler", "retriever")

    # Add multi-hop check as a conditional from a virtual routing point
    # We need to add a routing node after relevance check
    graph.add_node("check_multi_hop", lambda state: state)  # pass-through
    graph.add_conditional_edges(
        "check_multi_hop",
        check_multi_hop,
        {
            "decompose": "multi_hop_decomposer",
            "generate": "answer_generator",
        },
    )

    # Multi-hop decomposer -> sub-query retriever -> answer generator
    graph.add_edge("multi_hop_decomposer", "sub_query_retriever")
    graph.add_edge("sub_query_retriever", "answer_generator")

    # Terminal nodes
    graph.add_edge("answer_generator", END)
    graph.add_edge("fallback_handler", END)

    return graph.compile()


def run_rag_query(query: str, config: Config = None) -> dict:
    """Run a single query through the agentic RAG pipeline."""
    if config is None:
        config = load_config()

    graph = build_rag_graph(config)

    initial_state = {
        "original_query": query,
        "current_query": query,
        "sub_queries": [],
        "retrieved_docs": [],
        "all_retrieved_docs": [],
        "relevance_score": 0.0,
        "retry_count": 0,
        "max_retries": 2,
        "generated_answer": "",
        "context_snippets": [],
        "is_fallback": False,
        "is_multi_hop": False,
        "sub_query_index": 0,
        "relevance_threshold": 0.3,
    }

    result = graph.invoke(initial_state)

    return {
        "generated_answer": result.get("generated_answer", ""),
        "context_snippets": result.get("context_snippets", []),
        "is_fallback": result.get("is_fallback", False),
    }


if __name__ == "__main__":
    config = load_config()
    result = run_rag_query("What three new benchmarks were introduced in 2023?", config)
    print(f"\nAnswer: {result['generated_answer']}")
    print(f"Context snippets: {len(result['context_snippets'])}")
    print(f"Is fallback: {result['is_fallback']}")

"""
RAG Agent for NovaTech Knowledge Base

A conversational agent that retrieves relevant information from the knowledge base
and generates accurate, grounded answers with proper source citations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

import re

from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    DEFAULT_TOP_K,
    MAX_CONVERSATION_HISTORY,
    MAX_CHUNKS_PER_SOURCE,
    SIMILARITY_THRESHOLD,
    LOG_LEVEL,
)
from retriever import (
    RetrievedChunk,
    retrieve_with_deduplication,
    format_chunks_for_context,
    get_unique_sources,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Configuration from environment
DEFAULT_MODEL = OPENAI_CHAT_MODEL


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    sources: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from the RAG agent."""

    answer: str
    source_documents: list[str]  # List of source file paths cited
    relevant_sections: list[str]  # List of section names used
    chunks_retrieved: int
    confidence: str  # "high", "medium", "low", "no_info"
    latency_ms: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "source_documents": self.source_documents,
            "relevant_sections": self.relevant_sections,
            "chunks_retrieved": self.chunks_retrieved,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


# System prompt optimized for evaluation criteria
SYSTEM_PROMPT = """You are a helpful assistant for NovaTech, an enterprise technology company. Your role is to answer questions about company policies, products, HR matters, and IT procedures using ONLY the information provided in the retrieved documents.

## Core Instructions

1. **Answer Quality**
   - Provide accurate, complete answers that directly address all parts of the question
   - ONLY use information from the provided documents - never add external knowledge
   - If the documents contain partial information, clearly state what is covered and what is not
   - Structure complex answers with numbered steps or bullet points for clarity

2. **Source Citation (MANDATORY)**
   - You will receive multiple documents, but ONLY cite the ones you actually used
   - At the END of your answer, you MUST include a citation block in this EXACT format:
   
   ---
   **source_documents:**
   - source_file_path_1
   - source_file_path_2
   
   **relevant_sections:**
   - section_name_1
   - section_name_2
   
   - List ONLY the documents that contain information you used in your answer
   - Use the full path format shown in the document source (e.g., "hr/leave/fmla_policy.md.pdf")
   - For relevant_sections, list the specific section titles or topics you referenced
   - Do NOT cite documents that were provided but not used in your answer

3. **Handling Limitations**
   - If no relevant information is found: Say "I don't have information about [topic] in the available documents."
   - If information is incomplete: Answer what you can and explicitly state what's missing
   - If the question is ambiguous: Ask for clarification before answering
   - NEVER hallucinate or make up information not in the documents

4. **Response Format**
   - Be concise but comprehensive
   - Use clear, professional language
   - For procedural questions: Use numbered steps
   - For policy questions: Quote or paraphrase relevant sections
   - For calculations: Show your work step by step

5. **Conversation Context**
   - Consider previous messages when answering follow-up questions
   - Refer back to earlier context when relevant
   - Maintain consistency with previous answers

Remember: It's better to acknowledge uncertainty than to provide incorrect information. Only cite sources you actually used."""


def build_context_prompt(
    chunks: list[RetrievedChunk],
    query: str,
    conversation_history: list[Message],
) -> list[ChatCompletionMessageParam]:
    """
    Build the full prompt with context, history, and query.

    Args:
        chunks: Retrieved document chunks.
        query: Current user query.
        conversation_history: Previous conversation messages.

    Returns:
        List of message dicts for the OpenAI API.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add conversation history (limited to recent exchanges)
    history_to_include = conversation_history[-MAX_CONVERSATION_HISTORY:]
    for msg in history_to_include:
        if msg.role == "user":
            messages.append({"role": "user", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": msg.content})

    # Build context from retrieved chunks
    if chunks:
        context = format_chunks_for_context(chunks, include_scores=False)
        context_section = f"""## Retrieved Documents

{context}

---

## User Question
{query}

Please answer the question based ONLY on the documents above. Remember to cite your sources."""
    else:
        context_section = f"""## Retrieved Documents

No relevant documents were found in the knowledge base.

---

## User Question
{query}

Please acknowledge that you don't have information to answer this question."""

    messages.append({"role": "user", "content": context_section})

    return messages


def determine_confidence(
    chunks: list[RetrievedChunk],
    answer: str,
) -> str:
    """
    Determine confidence level based on retrieval quality.

    Args:
        chunks: Retrieved chunks.
        answer: Generated answer.

    Returns:
        Confidence level: "high", "medium", "low", or "no_info"
    """
    if not chunks:
        return "no_info"

    # Check if answer indicates lack of information
    no_info_phrases = [
        "don't have information",
        "no information available",
        "not found in",
        "cannot find",
        "no relevant documents",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in no_info_phrases):
        return "no_info"

    # Confidence based on similarity scores and number of sources
    avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
    top_similarity = chunks[0].similarity_score if chunks else 0
    num_sources = len(get_unique_sources(chunks))

    if top_similarity >= 0.75 and avg_similarity >= 0.60:
        return "high"
    elif top_similarity >= 0.55 and avg_similarity >= 0.45:
        return "medium"
    else:
        return "low"


@dataclass
class ParsedCitations:
    """Parsed citations from the LLM response."""

    source_documents: list[str]
    relevant_sections: list[str]


def parse_citations(answer: str, available_sources: list[str]) -> ParsedCitations:
    """
    Parse the cited sources and sections from the LLM's response.

    Extracts source_documents and relevant_sections from the citation block.

    Expected format:
    ---
    **source_documents:**
    - source_file_path_1
    - source_file_path_2

    **relevant_sections:**
    - section_name_1
    - section_name_2

    Args:
        answer: The full answer from the LLM.
        available_sources: List of source filenames that were provided to the LLM.

    Returns:
        ParsedCitations with source_documents and relevant_sections lists.
    """
    source_documents: list[str] = []
    relevant_sections: list[str] = []

    # Extract source_documents section
    source_docs_pattern = r"\*\*source_documents:\*\*\s*((?:-[^\n]+\n?)+)"
    source_docs_match = re.search(source_docs_pattern, answer, re.IGNORECASE)

    if source_docs_match:
        docs_block = source_docs_match.group(1)
        # Extract each line starting with -
        doc_lines = re.findall(r"-\s*([^\n]+)", docs_block)
        for doc in doc_lines:
            doc = doc.strip()
            if doc:
                source_documents.append(doc)

    # Extract relevant_sections section
    sections_pattern = r"\*\*relevant_sections:\*\*\s*((?:-[^\n]+\n?)+)"
    sections_match = re.search(sections_pattern, answer, re.IGNORECASE)

    if sections_match:
        sections_block = sections_match.group(1)
        # Extract each line starting with -
        section_lines = re.findall(r"-\s*([^\n]+)", sections_block)
        for section in section_lines:
            section = section.strip()
            if section:
                relevant_sections.append(section)

    # Fallback: if structured format not found, try alternative patterns
    if not source_documents:
        # Try to find any .pdf file references in the answer
        pdf_pattern = r"(?:^|\s|-)([a-zA-Z0-9_/\-]+\.(?:md\.)?pdf)"
        pdf_matches = re.findall(pdf_pattern, answer, re.IGNORECASE)
        seen: set[str] = set()
        for pdf in pdf_matches:
            pdf = pdf.strip()
            if pdf and pdf not in seen:
                seen.add(pdf)
                source_documents.append(pdf)

    # Fallback for sections: look for "Section:" patterns
    if not relevant_sections:
        section_fallback = r"[Ss]ection[:\s]+[\"']?([^\"'\n,]+)[\"']?"
        section_matches = re.findall(section_fallback, answer)
        seen_sections: set[str] = set()
        for section in section_matches:
            section = section.strip()
            if section and section not in seen_sections:
                seen_sections.add(section)
                relevant_sections.append(section)

    return ParsedCitations(
        source_documents=source_documents,
        relevant_sections=relevant_sections,
    )


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent for answering questions.

    Maintains conversation context and provides grounded answers
    with source citations.
    """

    def __init__(
        self,
        *,
        top_k: int = DEFAULT_TOP_K,
        model: str = DEFAULT_MODEL,
        max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the RAG agent.

        Args:
            top_k: Number of chunks to retrieve per query.
            model: OpenAI model to use for generation.
            max_chunks_per_source: Max chunks from any single source (for diversity).
            similarity_threshold: Minimum similarity score for retrieved chunks.
        """
        self.top_k = top_k
        self.model = model
        self.max_chunks_per_source = max_chunks_per_source
        self.similarity_threshold = similarity_threshold
        self.conversation_history: list[Message] = []

        logger.info(
            f"RAG Agent initialized: model={model}, top_k={top_k}, "
            f"threshold={similarity_threshold}"
        )

    def clear_history(self) -> None:
        """Clear conversation history for a new session."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def answer(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> AgentResponse:
        """
        Answer a user query using RAG.

        Args:
            query: The user's question.
            top_k: Override default top_k for this query.

        Returns:
            AgentResponse with answer, sources, and metadata.
        """
        start_time = time.time()
        effective_top_k = top_k if top_k is not None else self.top_k

        # Step 1: Retrieve relevant chunks
        logger.info(f"Retrieving chunks for: {query[:80]}...")
        chunks = retrieve_with_deduplication(
            query,
            top_k=effective_top_k,
            max_chunks_per_source=self.max_chunks_per_source,
            similarity_threshold=self.similarity_threshold,
        )

        available_sources = get_unique_sources(chunks)
        logger.info(
            f"Retrieved {len(chunks)} chunks from {len(available_sources)} sources"
        )

        # Step 2: Build prompt with context and history
        messages = build_context_prompt(
            chunks=chunks,
            query=query,
            conversation_history=self.conversation_history,
        )

        # Step 3: Generate answer
        logger.info(f"Generating answer with {self.model}...")

        # Debug: Log prompt details
        total_prompt_chars = sum(
            len(c) for m in messages if (c := m.get("content")) and isinstance(c, str)
        )
        logger.info(
            f"Prompt has {len(messages)} messages, {total_prompt_chars} total chars"
        )
        if len(messages) > 1:
            user_msg = messages[-1].get("content")
            if isinstance(user_msg, str):
                logger.info(f"User message length: {len(user_msg)} chars")
                # Log first 500 chars of user message for debugging
                logger.debug(f"User message preview: {user_msg[:500]}...")

        # Reasoning models (like gpt-5.2) use internal thinking tokens that count against the limit
        # So we need much higher limits for those models
        is_reasoning_model = (
            "5" in self.model or "o1" in self.model or "o3" in self.model
        )
        token_limit = 16000 if is_reasoning_model else 1500

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,  # Low temperature for factual accuracy
            max_completion_tokens=token_limit,
        )

        # Debug: Log the full response structure
        logger.info(
            f"Response finish_reason: {response.choices[0].finish_reason if response.choices else 'N/A'}"
        )
        if response.choices:
            msg = response.choices[0].message
            logger.info(f"Message object attributes: {dir(msg)}")
            logger.info(
                f"Message content length: {len(msg.content) if msg.content else 0}"
            )
            logger.info(
                f"Message content preview: {repr(msg.content[:200]) if msg.content else 'None'}"
            )
            # Check for refusal (newer models)
            if hasattr(msg, "refusal") and msg.refusal:
                logger.warning(f"Model refused to answer: {msg.refusal}")
            # Check for reasoning content (some newer models)
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                logger.info(f"Model has reasoning_content: {len(reasoning)} chars")

        answer = response.choices[0].message.content or ""

        if not answer:
            logger.warning(
                f"Empty answer received. Finish reason: {response.choices[0].finish_reason}"
            )

        # Step 4: Parse cited sources from the answer
        citations = parse_citations(answer, available_sources)

        # Log citation info
        if citations.source_documents:
            logger.info(
                f"Cited {len(citations.source_documents)} source documents, "
                f"{len(citations.relevant_sections)} sections"
            )
            for doc in citations.source_documents:
                logger.debug(f"  - Document: {doc}")
            for section in citations.relevant_sections:
                logger.debug(f"  - Section: {section}")
        else:
            logger.warning("No sources were cited in the answer")

        # Step 5: Determine confidence
        confidence = determine_confidence(chunks, answer)

        # Step 6: Update conversation history
        self.conversation_history.append(Message(role="user", content=query))
        self.conversation_history.append(
            Message(
                role="assistant", content=answer, sources=citations.source_documents
            )
        )

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Answer generated: {len(answer)} chars, "
            f"confidence={confidence}, latency={latency_ms:.0f}ms"
        )

        return AgentResponse(
            answer=answer,
            source_documents=citations.source_documents,
            relevant_sections=citations.relevant_sections,
            chunks_retrieved=len(chunks),
            confidence=confidence,
            latency_ms=latency_ms,
        )


def run_test_questions(
    test_file: str = "../novatech-kb/test_questions.json",
    output_file: str = "test_results.json",
) -> dict:
    """
    Run the agent against test questions and save results.

    Args:
        test_file: Path to test questions JSON file.
        output_file: Path to save results.

    Returns:
        Summary statistics.
    """
    # Load test questions
    with open(test_file, "r") as f:
        data = json.load(f)

    questions = data.get("test_questions", [])
    logger.info(f"Loaded {len(questions)} test questions")

    # Initialize agent
    agent = RAGAgent(top_k=8)

    results = []
    total_latency = 0

    for q in questions:
        qid = q["id"]
        question = q["question"]

        logger.info(f"\n{'='*60}\nProcessing {qid}: {question[:60]}...")

        # Clear history for independent evaluation
        agent.clear_history()

        # Get answer
        response = agent.answer(question)
        total_latency += response.latency_ms

        result = {
            "id": qid,
            "question": question,
            "answer": response.answer,
            "source_documents": response.source_documents,
            "relevant_sections": response.relevant_sections,
            "expected_sources": q.get("source_documents", []),
            "expected_sections": q.get("relevant_sections", []),
            "chunks_retrieved": response.chunks_retrieved,
            "confidence": response.confidence,
            "latency_ms": response.latency_ms,
            "difficulty": q.get("difficulty"),
            "category": q.get("category"),
        }
        results.append(result)

        print(f"\n[{qid}] {question}")
        print(f"Answer: {response.answer[:300]}...")
        print(f"source_documents: {response.source_documents}")
        print(f"relevant_sections: {response.relevant_sections}")
        print(
            f"Confidence: {response.confidence}, Latency: {response.latency_ms:.0f}ms"
        )

    # Calculate summary statistics
    summary = {
        "total_questions": len(questions),
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / len(questions) if questions else 0,
        "confidence_distribution": {
            "high": sum(1 for r in results if r["confidence"] == "high"),
            "medium": sum(1 for r in results if r["confidence"] == "medium"),
            "low": sum(1 for r in results if r["confidence"] == "low"),
            "no_info": sum(1 for r in results if r["confidence"] == "no_info"),
        },
        "results": results,
    }

    # Save results
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Average latency: {summary['avg_latency_ms']:.0f}ms")

    return summary


def interactive_mode():
    """Run the agent in interactive conversation mode."""
    print("\n" + "=" * 60)
    print("NovaTech Knowledge Base Assistant")
    print("=" * 60)
    print("Ask questions about company policies, products, HR, and IT.")
    print("Type 'clear' to reset conversation, 'quit' to exit.\n")

    agent = RAGAgent(top_k=DEFAULT_TOP_K)

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue
            if query.lower() == "quit":
                print("Goodbye!")
                break
            if query.lower() == "clear":
                agent.clear_history()
                print("Conversation history cleared.")
                continue

            response = agent.answer(query)

            print(f"\nAssistant: {response.answer}")

            # Display cited sources and sections
            if response.source_documents:
                print(f"\n[source_documents: {response.source_documents}]")
            if response.relevant_sections:
                print(f"[relevant_sections: {response.relevant_sections}]")
            if not response.source_documents and not response.relevant_sections:
                print("\n[No sources cited]")

            print(
                f"[Confidence: {response.confidence} | Latency: {response.latency_ms:.0f}ms]"
            )

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"An error occurred: {e}")

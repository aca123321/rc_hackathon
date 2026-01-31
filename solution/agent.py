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
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Ollama client (OpenAI-compatible API)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL, api_key="ollama"
)  # Ollama doesn't need real API key

# Configuration from environment
DEFAULT_MODEL = OPENAI_CHAT_MODEL


def get_client_for_model(model: str) -> OpenAI:
    """Get the appropriate client (OpenAI or Ollama) based on model."""
    model_info = AVAILABLE_MODELS.get(model, {})
    if model_info.get("provider") == "ollama":
        return ollama_client
    return openai_client


# =============================================================================
# Prompt Cache - In-memory cache for fast repeated queries
# =============================================================================
import hashlib
from collections import OrderedDict
from threading import Lock


class PromptCache:
    """Thread-safe in-memory cache for RAG responses."""

    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.max_size = max_size
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(
        self,
        query: str,
        model: str,
        top_k: int,
        history_length: int,
        max_chunks: int,
        similarity_threshold: float,
    ) -> str:
        """Create a cache key from query parameters."""
        # Include all retrieval-affecting parameters in the key
        key_data = f"{query}|{model}|{top_k}|{history_length}|{max_chunks}|{similarity_threshold}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        model: str,
        top_k: int,
        history_length: int,
        max_chunks: int,
        similarity_threshold: float,
    ) -> Optional[dict]:
        """Get cached response if exists."""
        key = self._make_key(
            query, model, top_k, history_length, max_chunks, similarity_threshold
        )
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (LRU)
                self.cache.move_to_end(key)
                logger.info(f"Cache HIT for query: {query[:50]}... (hits: {self.hits})")
                return self.cache[key]
            self.misses += 1
            return None

    def set(
        self,
        query: str,
        model: str,
        top_k: int,
        history_length: int,
        max_chunks: int,
        similarity_threshold: float,
        response: dict,
    ):
        """Store response in cache."""
        key = self._make_key(
            query, model, top_k, history_length, max_chunks, similarity_threshold
        )
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                self.cache[key] = response
                # Evict oldest if over capacity
                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
            logger.info(
                f"Cached response for: {query[:50]}... (cache size: {len(self.cache)})"
            )

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Prompt cache cleared")

    def stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1%}",
            }


# Global prompt cache instance
prompt_cache = PromptCache(max_size=100)

# Available models for RAG queries
# Models suitable for accurate, factual RAG-informed responses
AVAILABLE_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Fast, efficient multimodal model with strong reasoning",
        "context_window": 128000,
        "recommended_for": "General RAG queries, good balance of speed and quality",
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Smaller, faster, and cheaper version of GPT-4o",
        "context_window": 128000,
        "recommended_for": "High-volume queries, cost-sensitive applications",
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "description": "High capability model with vision support",
        "context_window": 128000,
        "recommended_for": "Complex queries requiring deep understanding",
    },
    "gpt-4.1": {
        "name": "GPT-4.1",
        "description": "Enhanced GPT-4 with improved instruction following",
        "context_window": 1047576,
        "recommended_for": "Large context RAG with many retrieved chunks",
    },
    "gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "description": "Efficient version of GPT-4.1 for faster responses",
        "context_window": 1047576,
        "recommended_for": "Fast queries with large context needs",
    },
    "gpt-5.1": {
        "name": "GPT-5.1",
        "description": "Advanced reasoning model with improved accuracy",
        "context_window": 128000,
        "recommended_for": "High-accuracy RAG requiring nuanced understanding",
    },
    "gpt-5.2": {
        "name": "GPT-5.2",
        "description": "Latest reasoning model with chain-of-thought capabilities",
        "context_window": 128000,
        "recommended_for": "Complex multi-step reasoning and analysis",
    },
    "o1": {
        "name": "o1",
        "description": "Reasoning model optimized for complex problem solving",
        "context_window": 200000,
        "recommended_for": "Complex analytical queries, policy interpretation",
    },
    "o1-mini": {
        "name": "o1-mini",
        "description": "Faster, cost-effective reasoning model",
        "context_window": 128000,
        "recommended_for": "Quick reasoning tasks, moderate complexity",
    },
    "o1-pro": {
        "name": "o1-pro",
        "description": "Most capable reasoning model for complex tasks",
        "context_window": 200000,
        "recommended_for": "Most complex queries requiring deep analysis",
    },
    "o3-mini": {
        "name": "o3-mini",
        "description": "Next-gen reasoning model, compact and efficient",
        "context_window": 200000,
        "recommended_for": "Advanced reasoning with fast response times",
        "provider": "openai",
    },
    # Ollama models (local)
    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest": {
        "name": "Llama 3.2 1B (Local)",
        "description": "Fast, lightweight local model via Ollama - free to run",
        "context_window": 8192,
        "recommended_for": "Quick local queries, privacy-sensitive data, offline use",
        "provider": "ollama",
    },
}


def get_available_models() -> dict:
    """Return dict of available models with their metadata."""
    return AVAILABLE_MODELS


def is_valid_model(model: str) -> bool:
    """Check if a model name is valid."""
    return model in AVAILABLE_MODELS


# Model pricing per 1M tokens (as of Jan 2026)
# Format: {"model": {"input": price_per_1M, "output": price_per_1M}}
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.1": {"input": 5.00, "output": 15.00},
    "gpt-5.2": {"input": 8.00, "output": 24.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o1-pro": {"input": 60.00, "output": 240.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Ollama models are free (local)
    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest": {"input": 0.0, "output": 0.0},
}


@dataclass
class QueryCost:
    """Tracks token usage and cost for a query."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    breakdown: list = field(
        default_factory=list
    )  # List of (component, model, input, output, cost)

    def add_usage(
        self, component: str, model: str, input_tokens: int, output_tokens: int
    ):
        """Add token usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens

        # Calculate cost for this call
        pricing = MODEL_PRICING.get(
            model, {"input": 5.0, "output": 15.0}
        )  # Default pricing
        call_cost = (input_tokens * pricing["input"] / 1_000_000) + (
            output_tokens * pricing["output"] / 1_000_000
        )
        self.cost_usd += call_cost

        self.breakdown.append(
            {
                "component": component,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(call_cost, 6),
            }
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "breakdown": self.breakdown,
        }


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    sources: list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    """
    Rich running memory of the conversation session.

    Maintains comprehensive context that persists across turns
    to improve retrieval for follow-up questions.

    Word limits (approximate):
    - user_profile: 50 words
    - conversation_summary: 150 words
    - key_facts: 100 words
    - recent_questions: 3 questions
    - documents_referenced: 10 documents
    - current_focus: 30 words
    Total target: ~350 words max
    """

    # User profile - who is asking (role, location, department, etc.)
    user_profile: str = (
        ""  # e.g., "L4 Software Engineer in US, 3 years tenure, Engineering dept"
    )

    # Running summary of what's been discussed
    conversation_summary: str = (
        ""  # e.g., "User inquired about leave policies. First asked about Indian employee leave (20 days annual + 12 sick). Then asked about their own entitlement as US L4 employee."
    )

    # Specific facts/numbers from the conversation
    key_facts: list[str] = field(
        default_factory=list
    )  # e.g., ["Indian employees get 20 annual leave days", "US L4 employees get 15 annual + 10 public holidays"]

    # Recent questions asked (for context)
    recent_questions: list[str] = field(default_factory=list)  # Last 3 questions

    # Documents that have been cited
    documents_referenced: list[str] = field(
        default_factory=list
    )  # e.g., ["hr/leave/annual_leave_policy.md.pdf"]

    # Current topic/focus of conversation
    current_focus: str = ""  # e.g., "US employee leave entitlements for L4 level"

    # What the user might ask next
    anticipated_topics: list[str] = field(
        default_factory=list
    )  # e.g., ["sick leave policy", "public holidays US"]

    def to_context_string(self) -> str:
        """Convert memory to a rich context string for query augmentation."""
        parts = []

        if self.user_profile:
            parts.append(f"USER: {self.user_profile}")

        if self.current_focus:
            parts.append(f"CURRENT TOPIC: {self.current_focus}")

        if self.conversation_summary:
            parts.append(f"CONVERSATION: {self.conversation_summary}")

        if self.key_facts:
            facts_str = "; ".join(self.key_facts[-5:])
            parts.append(f"KEY FACTS: {facts_str}")

        if self.anticipated_topics:
            parts.append(f"RELATED TOPICS: {', '.join(self.anticipated_topics[-3:])}")

        return " | ".join(parts) if parts else ""

    def to_retrieval_context(self) -> str:
        """Get context optimized for retrieval augmentation."""
        parts = []

        if self.user_profile:
            parts.append(self.user_profile)

        if self.current_focus:
            parts.append(self.current_focus)

        if self.anticipated_topics:
            parts.extend(self.anticipated_topics[-3:])

        if self.key_facts:
            # Extract key terms from facts
            for fact in self.key_facts[-3:]:
                parts.append(fact)

        return " ".join(parts)

    def is_empty(self) -> bool:
        """Check if memory has any content."""
        return (
            not self.user_profile
            and not self.conversation_summary
            and not self.key_facts
            and not self.current_focus
        )


# Prompt for extracting rich session memory from conversation
MEMORY_EXTRACTION_PROMPT = """You are a memory manager for a RAG conversation system. Analyze this exchange and update the session memory.

## Latest Exchange
USER: {user_message}
ASSISTANT: {assistant_response}

## Current Memory State
{current_memory}

## Your Task
Update the memory with information from this exchange. Return a JSON object:

{{
    "user_profile": "...",  // Single sentence about the user. Include: role/level, location, department, tenure if known. Example: "L4 Software Engineer in US office, Engineering department"
    
    "conversation_summary": "...",  // Running summary of the conversation (2-3 sentences max, ~50 words). What has been discussed and established. UPDATE the existing summary, don't replace.
    
    "key_facts": [...],  // Specific facts, numbers, or policy details mentioned. Example: ["US L4 employees get 15 annual leave days", "Sick leave requires manager approval after 3 days"]. Keep max 5 most relevant.
    
    "current_focus": "...",  // What the user is currently trying to learn about. Be specific. Example: "US employee annual leave entitlement for L4 level"
    
    "anticipated_topics": [...],  // What the user might ask next, based on context. Example: ["US public holidays", "sick leave policy", "leave carryover rules"]. Keep 3-5 items.
    
    "documents_to_search": [...]  // Keywords/phrases that would help retrieve relevant documents for follow-up questions. Include location, role, policy types. Example: ["US leave policy", "L4 benefits", "annual leave US", "public holidays United States"]
}}

## Rules
1. MERGE with existing memory - don't lose important context
2. user_profile: Accumulate facts about the user, be concise (~20 words max)
3. conversation_summary: Update incrementally, focus on what's been established (~50 words max)
4. key_facts: Keep specific, retrievable facts with numbers/details (max 5)
5. current_focus: Should reflect what the user wants to know NOW (~15 words max)
6. anticipated_topics & documents_to_search: Be COMPREHENSIVE - include ALL related sub-topics:
   - If discussing "leave/time off": include annual leave, sick leave, public holidays, PTO, vacation, carryover
   - If discussing "benefits": include health, dental, vision, retirement, 401k, leave, insurance
   - If discussing "employment": include onboarding, termination, notice period, resignation
   - Always include the user's location (e.g., "US", "India") in search terms
6. anticipated_topics: Think about natural follow-up questions (3-5 items)
7. documents_to_search: Include terms that appear in document titles/content (5-8 items)

Return ONLY valid JSON."""


def extract_memory_update(
    user_message: str,
    assistant_response: str,
    current_memory: SessionMemory,
    sources_cited: list[str],
    model: str = "gpt-4o-mini",
    cost_tracker: Optional[QueryCost] = None,
) -> SessionMemory:
    """
    Use LLM to extract and update rich session memory from the latest exchange.

    Args:
        user_message: The user's message.
        assistant_response: The assistant's response.
        current_memory: Current session memory state.
        sources_cited: Documents cited in the response.
        model: Model to use for extraction (fast model preferred).
        cost_tracker: Optional QueryCost to track token usage.

    Returns:
        Updated SessionMemory.
    """
    try:
        current_memory_str = json.dumps(
            {
                "user_profile": current_memory.user_profile,
                "conversation_summary": current_memory.conversation_summary,
                "key_facts": current_memory.key_facts,
                "current_focus": current_memory.current_focus,
                "anticipated_topics": current_memory.anticipated_topics,
                "documents_referenced": current_memory.documents_referenced,
            },
            indent=2,
        )

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response[
                :1000
            ],  # Include more response for better context
            current_memory=current_memory_str,
        )

        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=800,
        )

        # Track cost if tracker provided
        if cost_tracker and response.usage:
            cost_tracker.add_usage(
                "memory_extraction",
                model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        result_text = response.choices[0].message.content or "{}"

        # Clean up the response - remove markdown code blocks if present
        result_text = result_text.strip()
        if result_text.startswith("```"):
            result_text = re.sub(r"^```(?:json)?\n?", "", result_text)
            result_text = re.sub(r"\n?```$", "", result_text)

        result = json.loads(result_text)

        # Merge documents referenced
        all_docs = list(
            dict.fromkeys(current_memory.documents_referenced + sources_cited)
        )[
            -10:
        ]  # Keep last 10

        # Merge recent questions
        recent_qs = current_memory.recent_questions + [user_message]
        recent_qs = recent_qs[-3:]  # Keep last 3

        # Build updated memory
        updated_memory = SessionMemory(
            user_profile=result.get("user_profile", current_memory.user_profile)
            or current_memory.user_profile,
            conversation_summary=result.get(
                "conversation_summary", current_memory.conversation_summary
            )
            or current_memory.conversation_summary,
            key_facts=(
                result.get("key_facts", current_memory.key_facts)[-5:]
                if result.get("key_facts")
                else current_memory.key_facts[-5:]
            ),
            recent_questions=recent_qs,
            documents_referenced=all_docs,
            current_focus=result.get("current_focus", current_memory.current_focus)
            or current_memory.current_focus,
            anticipated_topics=result.get("anticipated_topics", [])
            + result.get("documents_to_search", []),
        )

        # Deduplicate and limit anticipated_topics
        updated_memory.anticipated_topics = list(
            dict.fromkeys(updated_memory.anticipated_topics)
        )[-8:]

        logger.info(f"Memory updated - Focus: {updated_memory.current_focus}")
        logger.debug(f"Full memory: {updated_memory}")
        return updated_memory

    except Exception as e:
        logger.warning(f"Failed to extract memory update: {e}")
        return current_memory


QUERY_REWRITE_PROMPT = """You are a query rewriter for a RAG system. Rewrite the user's query to be self-contained and COMPREHENSIVE for document retrieval.

## Session Memory
{context}

## Original Query
{query}

## Instructions
1. Replace pronouns (me, I, my) with actual user attributes from memory
2. Replace references (it, that, this) with the actual topic
3. Include specific terms that would appear in relevant documents (location, role, policy names)
4. Make the query COMPREHENSIVE - include ALL related sub-topics
5. Output 2-3 sentences covering all aspects the user likely wants to know

## IMPORTANT: Be Comprehensive
- If user asks about "leaves" or "time off", include: annual leave, sick leave, public holidays, PTO
- If user asks about "benefits", include: health, retirement, leave, insurance
- If user asks about a policy, include related policies they'd likely want to know

## Examples
- Query: "What about me?" | Memory: "User is L4 in US, discussing leave policies" 
  → "What is the total leave entitlement for L4 employees in the United States? Include annual leave, sick leave, public holidays US, and any other paid time off for US employees."

- Query: "How many leaves do I get?" | Memory: "User is L4 US employee, discussing leave"
  → "How many total days off does an L4 employee in the US get per year? Include annual leave days, sick leave days, and US public holidays. US employee leave policy entitlement."

- Query: "What's the policy?" | Memory: "topic: remote work"
  → "What is the remote work policy? Include eligibility, requirements, equipment policy, and work from home guidelines."

Return ONLY the rewritten query."""


def rewrite_query_with_context(
    query: str,
    memory: SessionMemory,
    model: str = "gpt-4o-mini",
    cost_tracker: Optional[QueryCost] = None,
) -> str:
    """
    Use LLM to rewrite an ambiguous query into a clear, self-contained query.
    """
    try:
        # Build rich context from memory
        context_parts = []

        if memory.user_profile:
            context_parts.append(f"User: {memory.user_profile}")

        if memory.current_focus:
            context_parts.append(f"Current topic: {memory.current_focus}")

        if memory.conversation_summary:
            context_parts.append(f"Conversation so far: {memory.conversation_summary}")

        if memory.key_facts:
            context_parts.append(
                f"Key facts established: {'; '.join(memory.key_facts[-3:])}"
            )

        if memory.anticipated_topics:
            context_parts.append(
                f"Related topics: {', '.join(memory.anticipated_topics[-5:])}"
            )

        if memory.recent_questions:
            context_parts.append(
                f"Recent questions: {' | '.join(memory.recent_questions[-2:])}"
            )

        context_str = "\n".join(context_parts) if context_parts else "No prior context"

        prompt = QUERY_REWRITE_PROMPT.format(
            context=context_str,
            query=query,
        )

        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=200,
        )

        # Track cost if tracker provided
        if cost_tracker and response.usage:
            cost_tracker.add_usage(
                "query_rewrite",
                model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        rewritten = response.choices[0].message.content or query
        rewritten = rewritten.strip().strip('"').strip("'")

        logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Failed to rewrite query: {e}")
        return query


def build_augmented_query(
    query: str,
    memory: SessionMemory,
    cost_tracker: Optional[QueryCost] = None,
) -> str:
    """
    Augment the user query with session context for better retrieval.

    For ambiguous queries (short, contains pronouns), uses LLM to rewrite.
    For clear queries, appends rich context from memory.

    Args:
        query: Original user query.
        memory: Current session memory.
        cost_tracker: Optional QueryCost to track token usage.

    Returns:
        Augmented query string for retrieval.
    """
    if memory.is_empty():
        return query

    # Detect if query needs full rewriting (ambiguous/short/has pronouns)
    query_lower = query.lower()
    needs_rewrite = (
        len(query.split()) < 10  # Short query
        or any(
            pronoun in query_lower
            for pronoun in [
                " me",
                " my ",
                " i ",
                "i'm",
                " me?",
                " me.",
                "what about",
                "how about",
                "and me",
                "for me",
                " it?",
                " it.",
                " that?",
                " that.",
                " this?",
                " this.",
                "could you",
                "can you check",
                "check now",
                "check again",
                "tell me more",
                "more info",
                "elaborate",
            ]
        )
        or query_lower.startswith(("me", "my", "i ", "what about", "how about"))
    )

    if needs_rewrite:
        # Use LLM to rewrite the query with full context
        return rewrite_query_with_context(query, memory, cost_tracker=cost_tracker)

    # For clear queries, append rich context from memory
    context_parts = []

    # Add user profile terms
    if memory.user_profile:
        context_parts.append(memory.user_profile)

    # Add current focus
    if memory.current_focus:
        context_parts.append(memory.current_focus)

    # Add anticipated topics (these are search-optimized terms)
    if memory.anticipated_topics:
        context_parts.extend(memory.anticipated_topics[-5:])

    if context_parts:
        # Combine query with context
        augmented = f"{query} {' '.join(context_parts)}"
    else:
        augmented = query

    logger.info(f"Augmented query: {augmented[:200]}...")
    return augmented


@dataclass
class AgentResponse:
    """Response from the RAG agent."""

    answer: str
    source_documents: list[str]  # List of source file paths cited
    relevant_sections: list[str]  # List of section names used
    chunks_retrieved: int
    confidence: str  # "high", "medium", "low", "no_info"
    latency_ms: float
    cost: Optional[QueryCost] = None  # Token usage and cost tracking

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "answer": self.answer,
            "source_documents": self.source_documents,
            "relevant_sections": self.relevant_sections,
            "chunks_retrieved": self.chunks_retrieved,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }
        if self.cost:
            result["cost"] = self.cost.to_dict()
        return result


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
    history_length: Optional[int] = None,
) -> list[ChatCompletionMessageParam]:
    """
    Build the full prompt with context, history, and query.

    Args:
        chunks: Retrieved document chunks.
        query: Current user query.
        conversation_history: Previous conversation messages.
        history_length: Number of previous message pairs to include (default: MAX_CONVERSATION_HISTORY).

    Returns:
        List of message dicts for the OpenAI API.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add conversation history (limited to recent exchanges)
    # history_length is number of pairs, so multiply by 2 for individual messages
    effective_history = (
        (history_length * 2) if history_length is not None else MAX_CONVERSATION_HISTORY
    )
    history_to_include = conversation_history[-effective_history:]
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
        self.session_memory = SessionMemory()

        logger.info(
            f"RAG Agent initialized: model={model}, top_k={top_k}, "
            f"threshold={similarity_threshold}"
        )

    def clear_history(self, clear_cache: bool = True) -> None:
        """Clear conversation history, session memory, and optionally cache for a new session."""
        self.conversation_history = []
        self.session_memory = SessionMemory()
        if clear_cache:
            prompt_cache.clear()
        logger.info("Conversation history and session memory cleared")

    def get_cache_stats(self) -> dict:
        """Get prompt cache statistics."""
        return prompt_cache.stats()

    def get_model(self) -> str:
        """Get the current model being used."""
        return self.model

    def set_model(self, model: str) -> bool:
        """
        Set the model to use for generation.

        Args:
            model: The model name to switch to.

        Returns:
            True if model was changed, False if invalid model.
        """
        if not is_valid_model(model):
            logger.warning(f"Invalid model requested: {model}")
            return False

        old_model = self.model
        self.model = model
        logger.info(f"Model changed from {old_model} to {model}")
        return True

    def get_session_memory(self) -> dict:
        """Get the current session memory state."""
        return {
            "user_profile": self.session_memory.user_profile,
            "conversation_summary": self.session_memory.conversation_summary,
            "key_facts": self.session_memory.key_facts,
            "current_focus": self.session_memory.current_focus,
            "anticipated_topics": self.session_memory.anticipated_topics,
            "recent_questions": self.session_memory.recent_questions,
            "documents_referenced": self.session_memory.documents_referenced,
            "context_string": self.session_memory.to_context_string(),
        }

    def answer(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        history_length: Optional[int] = None,
        max_chunks_per_source: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> AgentResponse:
        """
        Answer a user query using RAG.

        Args:
            query: The user's question.
            top_k: Override default top_k for this query.
            history_length: Number of previous message pairs to include (default: 3).
            max_chunks_per_source: Maximum chunks from any single source (default: 2).
            similarity_threshold: Minimum similarity score (default: 0.3).
            max_completion_tokens: Maximum tokens in response (default: 1500).

        Returns:
            AgentResponse with answer, sources, and metadata.
        """
        start_time = time.time()
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_history_length = history_length if history_length is not None else 3
        effective_max_chunks = (
            max_chunks_per_source
            if max_chunks_per_source is not None
            else self.max_chunks_per_source
        )
        effective_similarity = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        effective_max_tokens = (
            max_completion_tokens if max_completion_tokens is not None else 1500
        )

        # Initialize cost tracker
        cost_tracker = QueryCost()

        # Check cache for repeated queries
        # Cache key includes all retrieval parameters, so identical queries return cached responses
        cached = prompt_cache.get(
            query,
            self.model,
            effective_top_k,
            effective_history_length,
            effective_max_chunks,
            effective_similarity,
        )
        if cached:
            # Return cached response with zero cost (it was free!)
            cached_response = AgentResponse(**cached)
            # Update cost to show it was from cache
            cached_response.cost = QueryCost()
            cached_response.cost.breakdown.append(
                {
                    "component": "cache_hit",
                    "model": self.model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            )
            # Update conversation history for cached responses too
            self.conversation_history.append(Message(role="user", content=query))
            self.conversation_history.append(
                Message(
                    role="assistant",
                    content=cached_response.answer,
                    sources=cached_response.source_documents,
                )
            )
            logger.info(f"Returning cached response for: {query[:50]}...")
            return cached_response

        # Step 1: Augment query with session memory for better retrieval
        augmented_query = build_augmented_query(
            query, self.session_memory, cost_tracker=cost_tracker
        )

        # Log session memory state
        if not self.session_memory.is_empty():
            logger.info(
                f"Session memory context: {self.session_memory.to_context_string()[:100]}..."
            )

        # Step 2: Retrieve relevant chunks using augmented query
        logger.info(f"Retrieving chunks for: {augmented_query[:100]}...")
        chunks = retrieve_with_deduplication(
            augmented_query,
            top_k=effective_top_k,
            max_chunks_per_source=effective_max_chunks,
            similarity_threshold=effective_similarity,
        )

        # Step 2b: Supplementary retrieval for anticipated topics (to catch related docs)
        # This helps when the main query misses related documents
        if self.session_memory.anticipated_topics:
            # Build a supplementary query from anticipated topics + user context
            supplementary_terms = self.session_memory.anticipated_topics[-5:]
            if self.session_memory.user_profile:
                supplementary_query = f"{self.session_memory.user_profile} {' '.join(supplementary_terms)}"
            else:
                supplementary_query = " ".join(supplementary_terms)

            logger.info(f"Supplementary retrieval for: {supplementary_query[:80]}...")

            supplementary_chunks = retrieve_with_deduplication(
                supplementary_query,
                top_k=effective_top_k // 2,  # Retrieve fewer supplementary chunks
                max_chunks_per_source=effective_max_chunks,
                similarity_threshold=effective_similarity
                + 0.05,  # Slightly higher threshold
            )

            # Merge chunks, avoiding duplicates (by id)
            existing_ids = {c.id for c in chunks}
            for chunk in supplementary_chunks:
                if chunk.id not in existing_ids:
                    chunks.append(chunk)
                    existing_ids.add(chunk.id)

            # Re-sort by similarity score and limit total
            chunks = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
            chunks = chunks[
                : effective_top_k + 5
            ]  # Allow slightly more chunks for comprehensive answers

            logger.info(f"After supplementary retrieval: {len(chunks)} total chunks")

        available_sources = get_unique_sources(chunks)
        logger.info(
            f"Retrieved {len(chunks)} chunks from {len(available_sources)} sources"
        )

        # Step 2: Build prompt with context and history
        messages = build_context_prompt(
            chunks=chunks,
            query=query,
            conversation_history=self.conversation_history,
            history_length=effective_history_length,
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

        # Reasoning models (like gpt-5.2, o1, o3) use internal thinking tokens
        # Set reasoning_effort to low for faster responses in RAG context
        is_reasoning_model = (
            "5" in self.model or "o1" in self.model or "o3" in self.model
        )

        # Reasoning models need higher limits due to internal reasoning tokens
        if is_reasoning_model:
            token_limit = max(effective_max_tokens, 16000)
        else:
            token_limit = effective_max_tokens

        # Get model info to check provider
        model_info = AVAILABLE_MODELS.get(self.model, {})
        is_ollama = model_info.get("provider") == "ollama"

        # Build API call kwargs
        api_kwargs = {
            "model": self.model,
            "messages": messages,
        }

        # Ollama uses different parameters
        if is_ollama:
            # Ollama OpenAI-compatible API uses max_tokens (not max_completion_tokens)
            api_kwargs["max_tokens"] = token_limit
            api_kwargs["temperature"] = 0.1
        else:
            api_kwargs["max_completion_tokens"] = token_limit
            # Only add temperature for non-reasoning models (o1/o3 don't support temperature)
            if not ("o1" in self.model or "o3" in self.model):
                api_kwargs["temperature"] = 0.1  # Low temperature for factual accuracy

        # Get appropriate client (OpenAI or Ollama) based on model
        model_client = get_client_for_model(self.model)

        try:
            response = model_client.chat.completions.create(**api_kwargs)
        except Exception as e:
            if is_ollama:
                logger.error(f"Ollama error (is Ollama running?): {e}")
                raise RuntimeError(
                    f"Failed to connect to Ollama. Make sure Ollama is running (ollama serve). Error: {e}"
                )
            raise

        # Track cost for main generation
        if response.usage:
            cost_tracker.add_usage(
                "generation",
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        # Debug: Log the full response structure
        logger.info(
            f"Response finish_reason: {response.choices[0].finish_reason if response.choices else 'N/A'}"
        )
        if response.choices:
            msg = response.choices[0].message
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

        # Step 7: Update session memory asynchronously (non-blocking for latency)
        # Use a fast model for memory extraction to minimize overhead
        try:
            self.session_memory = extract_memory_update(
                user_message=query,
                assistant_response=answer,
                current_memory=self.session_memory,
                sources_cited=citations.source_documents,
                model="gpt-4o-mini",  # Fast model for memory extraction
                cost_tracker=cost_tracker,
            )
            logger.info(
                f"Session memory updated. Focus: {self.session_memory.current_focus}"
            )
        except Exception as e:
            logger.warning(f"Failed to update session memory: {e}")

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log cost information
        logger.info(
            f"Query cost: ${cost_tracker.cost_usd:.6f} "
            f"({cost_tracker.input_tokens} input + {cost_tracker.output_tokens} output = {cost_tracker.total_tokens} tokens)"
        )
        for item in cost_tracker.breakdown:
            logger.debug(
                f"  - {item['component']}: ${item['cost_usd']:.6f} "
                f"({item['input_tokens']}+{item['output_tokens']} tokens, {item['model']})"
            )

        logger.info(
            f"Answer generated: {len(answer)} chars, "
            f"confidence={confidence}, latency={latency_ms:.0f}ms, cost=${cost_tracker.cost_usd:.6f}"
        )

        response = AgentResponse(
            answer=answer,
            source_documents=citations.source_documents,
            relevant_sections=citations.relevant_sections,
            chunks_retrieved=len(chunks),
            confidence=confidence,
            latency_ms=latency_ms,
            cost=cost_tracker,
        )

        # Cache the response for future identical queries
        # Store serializable version in cache
        cache_data = {
            "answer": response.answer,
            "source_documents": response.source_documents,
            "relevant_sections": response.relevant_sections,
            "chunks_retrieved": response.chunks_retrieved,
            "confidence": response.confidence,
            "latency_ms": response.latency_ms,
            "cost": None,  # Don't cache cost, it will be set to zero for cache hits
        }
        prompt_cache.set(
            query,
            self.model,
            effective_top_k,
            effective_history_length,
            effective_max_chunks,
            effective_similarity,
            cache_data,
        )

        return response


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

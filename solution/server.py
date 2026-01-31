"""
FastAPI server for NovaTech RAG Agent API.

Exposes an endpoint to query the knowledge base using the RAGAgent.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import RAGAgent, AgentResponse, get_available_models, is_valid_model


# Request/Response models
class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str = Field(..., description="The question to ask the knowledge base")
    top_k: Optional[int] = Field(
        default=None, description="Number of chunks to retrieve (overrides default)"
    )
    history_length: Optional[int] = Field(
        default=None,
        description="Number of previous message pairs to include (default: 3)",
    )
    max_chunks_per_source: Optional[int] = Field(
        default=None,
        description="Maximum chunks from any single source document (default: 2)",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score for retrieval (0-1, default: 0.3)",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in the response (default: 1500)",
    )
    clear_history: bool = Field(
        default=False, description="Clear conversation history before this query"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for this query (overrides current model)",
    )


class SessionMemoryInfo(BaseModel):
    """Rich session memory state information."""

    user_profile: str
    conversation_summary: str
    key_facts: list[str]
    current_focus: str
    anticipated_topics: list[str]
    context_string: str


class CostBreakdownItem(BaseModel):
    """Cost breakdown for a single component."""

    component: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostInfo(BaseModel):
    """Cost information for a query."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    breakdown: list[CostBreakdownItem]


class QueryResponse(BaseModel):
    """Response body from the query endpoint."""

    answer: str
    source_documents: list[str]
    relevant_sections: list[str]
    chunks_retrieved: int
    confidence: str
    latency_ms: float
    session_memory: Optional[SessionMemoryInfo] = None
    cost: Optional[CostInfo] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class ModelInfo(BaseModel):
    """Information about a single model."""

    id: str
    name: str
    description: str
    context_window: int
    recommended_for: str


class ModelsResponse(BaseModel):
    """Response body for listing available models."""

    models: list[ModelInfo]
    current_model: str


class SetModelRequest(BaseModel):
    """Request body for setting the model."""

    model: str = Field(..., description="The model ID to switch to")


class SetModelResponse(BaseModel):
    """Response body for setting the model."""

    success: bool
    previous_model: str
    current_model: str
    message: str


# Global agent instance
agent: Optional[RAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global agent
    # Startup: Initialize the RAG agent
    agent = RAGAgent()
    yield
    # Shutdown: Cleanup (if needed)
    agent = None


# Create FastAPI app
app = FastAPI(
    title="NovaTech RAG Agent API",
    description="API for querying the NovaTech knowledge base using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the agent is initialized."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return HealthResponse(status="healthy", message="RAG Agent is ready")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge base.

    Send a question and receive an answer with source citations.
    Optionally specify a model to use for this query only.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Optionally clear history before this query
    if request.clear_history:
        agent.clear_history()

    # Temporarily switch model if specified
    original_model = None
    if request.model:
        if not is_valid_model(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model: {request.model}. Use GET /models to see available models.",
            )
        original_model = agent.get_model()
        agent.set_model(request.model)

    try:
        # Get answer from the RAG agent
        response: AgentResponse = agent.answer(
            query=request.query,
            top_k=request.top_k,
            history_length=request.history_length,
            max_chunks_per_source=request.max_chunks_per_source,
            similarity_threshold=request.similarity_threshold,
            max_completion_tokens=request.max_completion_tokens,
        )

        # Get session memory after the query
        memory = agent.get_session_memory()
        session_memory_info = SessionMemoryInfo(
            user_profile=memory["user_profile"],
            conversation_summary=memory["conversation_summary"],
            key_facts=memory["key_facts"],
            current_focus=memory["current_focus"],
            anticipated_topics=memory["anticipated_topics"],
            context_string=memory["context_string"],
        )

        # Get cost info if available
        cost_info = None
        if response.cost:
            cost_info = CostInfo(
                input_tokens=response.cost.input_tokens,
                output_tokens=response.cost.output_tokens,
                total_tokens=response.cost.total_tokens,
                cost_usd=response.cost.cost_usd,
                breakdown=[
                    CostBreakdownItem(**item) for item in response.cost.breakdown
                ],
            )

        return QueryResponse(
            answer=response.answer,
            source_documents=response.source_documents,
            relevant_sections=response.relevant_sections,
            chunks_retrieved=response.chunks_retrieved,
            confidence=response.confidence,
            latency_ms=response.latency_ms,
            session_memory=session_memory_info,
            cost=cost_info,
        )
    finally:
        # Restore original model if we temporarily switched
        if original_model is not None:
            agent.set_model(original_model)


@app.post("/clear-history")
async def clear_history():
    """Clear the conversation history, session memory, and prompt cache."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    agent.clear_history()
    return {
        "status": "success",
        "message": "Conversation history, session memory, and prompt cache cleared",
    }


@app.get("/cache-stats")
async def get_cache_stats():
    """Get prompt cache statistics."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return agent.get_cache_stats()


@app.get("/session-memory")
async def get_session_memory():
    """Get the current session memory state."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    memory = agent.get_session_memory()
    return {
        "user_profile": memory["user_profile"],
        "conversation_summary": memory["conversation_summary"],
        "key_facts": memory["key_facts"],
        "current_focus": memory["current_focus"],
        "anticipated_topics": memory["anticipated_topics"],
        "recent_questions": memory["recent_questions"],
        "documents_referenced": memory["documents_referenced"],
        "context_string": memory["context_string"],
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List all available models for RAG queries.

    Returns model information including name, description, context window,
    and recommended use cases.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    available = get_available_models()
    models = [
        ModelInfo(
            id=model_id,
            name=info["name"],
            description=info["description"],
            context_window=info["context_window"],
            recommended_for=info["recommended_for"],
        )
        for model_id, info in available.items()
    ]

    return ModelsResponse(
        models=models,
        current_model=agent.get_model(),
    )


@app.get("/model")
async def get_current_model():
    """Get the currently active model."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    current = agent.get_model()
    available = get_available_models()
    model_info = available.get(current, {})

    return {
        "current_model": current,
        "name": model_info.get("name", current),
        "description": model_info.get("description", ""),
        "context_window": model_info.get("context_window", 0),
        "recommended_for": model_info.get("recommended_for", ""),
    }


@app.post("/model", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    """
    Set the model to use for subsequent queries.

    This changes the default model for all future queries until changed again.
    Individual queries can still override the model using the 'model' parameter.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not is_valid_model(request.model):
        available = list(get_available_models().keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {request.model}. Available models: {available}",
        )

    previous_model = agent.get_model()
    success = agent.set_model(request.model)

    return SetModelResponse(
        success=success,
        previous_model=previous_model,
        current_model=agent.get_model(),
        message=(
            f"Model changed from {previous_model} to {request.model}"
            if success
            else "Failed to change model"
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

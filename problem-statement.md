# AI Agents Hackathon: Enterprise Knowledge Base Q&A Agent

## The Challenge

Build an AI agent that accurately answers employee questions about company policies, procedures, and documentation using a large internal knowledge base — while minimising API costs and latency.

---

## Background

You're building for **NovaTech Solutions**, a mid-sized technology company with 2,500 employees. Like most enterprises, NovaTech has accumulated thousands of internal documents: HR policies, IT procedures, compliance guidelines, product documentation, and more.

Employees waste hours searching for answers to questions like:
- "How much parental leave do I get?"
- "How do I request access to a production database?"
- "What's our policy on working remotely from another country?"

Your challenge is to build an intelligent agent that can answer these questions accurately, quickly, and cost-efficiently.

---

## What You're Given

### Knowledge Base

A synthetic enterprise knowledge base containing **194 documents**:

| Document Type | Count | Description |
|---------------|-------|-------------|
| HR Policies | 56 | Leave, benefits, compensation, remote work, career development |
| IT Procedures | 39 | Access management, hardware, software, security, support |
| Product Documentation | 45 | Guides for NovaTech's four products |
| Compliance Guidelines | 19 | Data privacy, security compliance, regulations |
| Communications | 20 | All-hands notes, announcements, project updates |
| FAQ Documents | 15 | Common questions across various topics |

### Dataset Structure

```
novatech-kb/
├── hr/
├── it/
├── products/
├── compliance/
├── communications/
├── faqs/
└── test_questions.json
```

### Test Set

9 hard questions with ground-truth answers and source references. Each question requires synthesising information from 2-5 documents, testing the agent's ability to perform multi-document RAG.

**Examples** — Each requires retrieval across multiple documents:
> "If I'm promoted from L3 to L4 mid-year, how is my annual bonus calculated?" (2 docs)
> "An employee is relocating from Austin to London. What are all the HR and IT considerations?" (5 docs)
> "What is the complete exit process when an employee resigns?" (5 docs)

---

## Requirements

### Your Agent Must

1. **Accept natural language questions** from users
2. **Retrieve relevant information** from the knowledge base
3. **Generate accurate answers** grounded in the source documents
4. **Cite sources** — tell users which document(s) the answer came from
5. **Handle follow-up questions** with conversation context
6. **Know its limits** — gracefully acknowledge when information isn't available rather than hallucinating

### Technical Constraints

- You must use OpenAI models and embeddings with the API keys provided to you; you may use any vector database
- You may preprocess the documents however you like (chunking, summarisation, indexing)
- Your agent must be runnable for evaluation (API endpoint or local demo)
- You must track and report your token usage

---

## Evaluation Criteria

Your submission will be scored across four dimensions:

### 1. Answer Quality (40%)

| Metric | What We're Measuring |
|--------|---------------------|
| Accuracy | Is the answer factually correct? |
| Groundedness | Are claims supported by cited sources? |
| Completeness | Does the answer address all parts of the question? |
| Hallucination Rate | Does the agent make up information not in the sources? |

### 2. Retrieval Effectiveness (25%)

| Metric | What We're Measuring |
|--------|---------------------|
| Precision | Are the retrieved documents actually relevant? |
| Recall | Did the agent find all the relevant documents? |
| Mean Reciprocal Rank | Is the most relevant document ranked first? |
| Source Diversity | Can the agent synthesise across multiple documents when needed? |

### 3. Cost Efficiency (20%)

| Metric | What We're Measuring |
|--------|---------------------|
| Total Token Usage | How many tokens did you consume across all test queries? |
| Cost per Query | Average tokens per question |
| Embedding Efficiency | How many embedding calls did you make? |
| Caching Effectiveness | Did you avoid redundant computation? |

### 4. Robustness & User Experience (15%)

| Metric | What We're Measuring |
|--------|---------------------|
| Latency | How fast are responses? (p50 and p95) |
| Error Handling | Does it handle ambiguous or unanswerable queries gracefully? |
| Conversation Continuity | Does it maintain context across follow-up questions? |
| Answer Clarity | Are responses well-structured and easy to understand? |

### Scoring Formula

```
Final Score = (Quality × 0.40) + (Retrieval × 0.25) + (Efficiency × 0.20) + (Robustness × 0.15)
```

---

## Hints & Considerations

You don't have to use all of these, but smart solutions often consider:

### Preprocessing
- How will you chunk documents? Fixed size? Semantic boundaries? Hierarchical?
- What metadata will you extract to help with filtering?
- Can you pre-compute summaries for long documents?

### Retrieval
- Embeddings alone? BM25? Hybrid search?
- How many chunks do you retrieve? Do you re-rank them?
- Should you decompose complex questions into sub-queries?

### Cost Optimisation
- Can you use a smaller/cheaper model for some tasks (classification, routing) and a larger one for generation?
- What can you cache?
- Can you retrieve progressively (start narrow, expand if needed)?

### Generation
- How do you ensure the model cites its sources?
- How do you prevent hallucination?
- How do you handle "I don't know" cases confidently?

---

## Deliverables

By the submission deadline, provide:

### 1. Working Agent
- API endpoint OR locally runnable demo
- Must accept a question and return an answer with citations

### 2. Architecture Documentation (1-2 pages)
- System diagram
- Key design decisions and trade-offs
- What worked, what didn't

### 3. Cost Report
- Total tokens used (input/output breakdown)
- Embedding calls made
- Any caching metrics

### 4. Source Code
- Clean, documented repository
- README with setup instructions

---

## Demo & Judging

Each team will have **5 minutes** to:
- Give a brief overview of your approach (2 min)
- Live demo with judge questions (3 min)

Judges will ask questions from the test set and some of their own. They'll evaluate both the answers and your ability to explain your design choices.

---

## Rules

1. Teams of 1-4 people
2. You may use any open-source libraries, frameworks, or tools
3. You must use OpenAI models with the API keys provided to you
4. Pre-trained models and embeddings are allowed; fine-tuning is allowed if you have time
5. The knowledge base cannot be modified, but you can preprocess and index it however you like
6. No hardcoding answers to specific questions

---

## FAQ

**Q: Can we use a hosted vector database like Pinecone or Weaviate?**  
A: Yes, use whatever tools you prefer.

**Q: Do we need to handle the entire knowledge base?**  
A: Yes, your agent should be able to answer questions that could come from any document in the corpus.

**Q: What if multiple documents have conflicting information?**  
A: This can happen (e.g., an old memo vs. updated policy). Your agent should ideally surface the most current information and note if there's ambiguity.

**Q: Can we use RAG frameworks like LangChain or LlamaIndex?**  
A: Absolutely. Use whatever helps you build faster.

**Q: Can we use other LLM providers like Anthropic or open-source models?**  
A: No. You must use OpenAI models with the API keys provided to you.

**Q: How will you measure token usage?**  
A: You'll self-report, and we'll verify with spot checks during demos.

---

Good luck! Build something smart. 🚀
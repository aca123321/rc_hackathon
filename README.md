# NovaTech RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying the NovaTech enterprise knowledge base. This project includes a Python backend with vector similarity search and a modern React frontend.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Backend Setup (solution/)](#backend-setup-solution)
  - [Environment Configuration](#environment-configuration)
  - [Database Setup](#database-setup)
  - [Installing Dependencies](#installing-dependencies)
  - [Ingesting the Knowledge Base](#ingesting-the-knowledge-base)
  - [Running the Backend Server](#running-the-backend-server)
- [Frontend Setup (fe/)](#frontend-setup-fe)
  - [Installing Dependencies](#installing-frontend-dependencies)
  - [Running the Frontend](#running-the-frontend)
- [Using the Application](#using-the-application)
- [Frontend Configuration Options](#frontend-configuration-options)
  - [Model Selection](#model-selection)
  - [Retrieval Settings](#retrieval-settings)
  - [How Settings Affect Responses and Costs](#how-settings-affect-responses-and-costs)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  React Frontend │────▶│  FastAPI Server │────▶│   PostgreSQL    │
│   (Port 3000)   │     │   (Port 8000)   │     │   + pgvector    │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │   OpenAI API    │
                        │  (Embeddings &  │
                        │   Chat Models)  │
                        │                 │
                        └─────────────────┘
```

**Key Components:**

- **Frontend (fe/)**: React app with TailwindCSS for a modern chat interface
- **Backend (solution/)**: FastAPI server with RAG pipeline
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **LLM**: OpenAI models for embeddings and response generation

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** and **npm** - [Download Node.js](https://nodejs.org/)
- **PostgreSQL 15+** with **pgvector extension** - [Install PostgreSQL](https://www.postgresql.org/download/)
- **OpenAI API Key** - [Get API Key](https://platform.openai.com/api-keys)

### Installing pgvector Extension

```bash
# On macOS with Homebrew
brew install pgvector

# On Ubuntu/Debian
sudo apt install postgresql-15-pgvector

# On other systems, build from source:
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

After installing, enable the extension in your PostgreSQL database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Backend Setup (solution/)

### Environment Configuration

1. Navigate to the solution directory:

```bash
cd hackathon-ps/solution
```

2. Copy the example environment file:

```bash
cp .env.example .env
```

3. Edit `.env` and configure the following variables:

```bash
# =============================================================================
# OpenAI API Configuration
# =============================================================================
OPENAI_API_KEY=your-openai-api-key-here          # Required: Your OpenAI API key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small    # Embedding model for vector search
OPENAI_CHAT_MODEL=gpt-4o                         # Default chat model for responses

# =============================================================================
# PostgreSQL Database Configuration
# =============================================================================
POSTGRES_HOST=localhost                          # Database host
POSTGRES_PORT=5432                               # Database port
POSTGRES_DB=rapidclaim                           # Database name
POSTGRES_USER=postgres                           # Database user
POSTGRES_PASSWORD=your-database-password-here    # Required: Database password

# =============================================================================
# RAG Agent Configuration (defaults shown)
# =============================================================================
DEFAULT_TOP_K=10                                 # Number of chunks to retrieve per query
MAX_CONVERSATION_HISTORY=6                       # Keep last N exchanges for context
MAX_CHUNKS_PER_SOURCE=2                          # Maximum chunks from any single source
SIMILARITY_THRESHOLD=0.3                         # Minimum similarity score (0-1)

# =============================================================================
# Chunker Configuration
# =============================================================================
CHUNK_SIZE=500                                   # Maximum words per chunk
CHUNK_OVERLAP=100                                # Overlapping words between chunks

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL=INFO                                   # DEBUG, INFO, WARNING, ERROR
```

### Database Setup

1. Create the PostgreSQL database:

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE rapidclaim;

# Connect to the new database
\c rapidclaim

# Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Exit psql
\q
```

2. The application will automatically create the required tables on first run.

### Installing Dependencies

1. Create and activate a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
.\venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Required packages (requirements.txt):**

| Package           | Version   | Purpose                                         |
| ----------------- | --------- | ----------------------------------------------- |
| pdfplumber        | >=0.11.0  | PDF parsing with high accuracy table extraction |
| python-dotenv     | >=1.0.0   | Environment variable management                 |
| openai            | >=1.0.0   | OpenAI API client for embeddings and chat       |
| psycopg2-binary   | >=2.9.0   | PostgreSQL database adapter                     |
| pgvector          | >=0.2.0   | Vector similarity search extension              |
| fastapi           | >=0.109.0 | Web framework for the API server                |
| uvicorn[standard] | >=0.27.0  | ASGI server to run FastAPI                      |

### Ingesting the Knowledge Base

Before using the RAG system, you need to ingest the PDF documents from the knowledge base:

```bash
cd hackathon-ps/solution

# Run the ingestion pipeline
python ingestion.py
```

This will:

- Recursively discover all PDF files in `../novatech-kb/`
- Parse each PDF and extract text content
- Chunk the text into smaller segments
- Generate embeddings using OpenAI's embedding model
- Store chunks and embeddings in PostgreSQL

**Expected output:**

```
INGESTION SUMMARY
==================================================
Total PDFs found:     124
Successfully processed: 124
Failed:                0
Total chunks created:  XXXX
Database entries:      XXXX
```

### Running the Backend Server

Start the FastAPI server:

```bash
cd hackathon-ps/solution

# Run the server
python server.py
```

Or with uvicorn directly:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://localhost:8000**

**Verify the server is running:**

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","message":"RAG Agent is ready"}
```

---

## Frontend Setup (fe/)

### Installing Frontend Dependencies

1. Navigate to the frontend directory:

```bash
cd hackathon-ps/fe
```

2. Install npm packages:

```bash
npm install
```

**Required packages (package.json):**

| Package        | Version | Purpose                          |
| -------------- | ------- | -------------------------------- |
| react          | ^19.2.4 | React framework                  |
| react-dom      | ^19.2.4 | React DOM rendering              |
| react-markdown | ^10.1.0 | Render markdown in responses     |
| remark-gfm     | ^4.0.1  | GitHub Flavored Markdown support |
| tailwindcss    | ^3.4.19 | Utility-first CSS framework      |
| web-vitals     | ^2.1.4  | Performance monitoring           |

### Running the Frontend

Start the development server:

```bash
npm start
```

The frontend will be available at: **http://localhost:3000**

The app will automatically open in your default browser.

---

## Using the Application

1. **Start the backend server** (in one terminal):

   ```bash
   cd hackathon-ps/solution
   python server.py
   ```

2. **Start the frontend** (in another terminal):

   ```bash
   cd hackathon-ps/fe
   npm start
   ```

3. **Open your browser** to http://localhost:3000

4. **Ask questions** about NovaTech products, policies, HR, or IT procedures

---

## Frontend Configuration Options

The frontend provides several configurable options accessible via the **Settings** button (gear icon) in the header.

### Model Selection

Click the model dropdown to select from available AI models:

| Model                | Description                      | Cost                      | Use Case                          |
| -------------------- | -------------------------------- | ------------------------- | --------------------------------- |
| GPT-4o               | Fast, efficient multimodal model | $2.50/$10 per 1M tokens   | General RAG queries, good balance |
| GPT-4o Mini          | Smaller, faster, cheaper         | $0.15/$0.60 per 1M tokens | High-volume, cost-sensitive       |
| GPT-4 Turbo          | High capability                  | $10/$30 per 1M tokens     | Complex queries                   |
| GPT-4.1              | Enhanced GPT-4                   | $2/$8 per 1M tokens       | Large context RAG                 |
| GPT-4.1 Mini         | Efficient GPT-4.1                | $0.40/$1.60 per 1M tokens | Fast large context                |
| GPT-5.1              | Advanced reasoning               | $5/$15 per 1M tokens      | High-accuracy RAG                 |
| GPT-5.2              | Latest reasoning model           | $8/$24 per 1M tokens      | Complex multi-step reasoning      |
| o1                   | Reasoning optimized              | $15/$60 per 1M tokens     | Complex analytical queries        |
| o1-mini              | Fast reasoning                   | $1.10/$4.40 per 1M tokens | Quick reasoning tasks             |
| o1-pro               | Most capable reasoning           | $60/$240 per 1M tokens    | Most complex queries              |
| o3-mini              | Next-gen reasoning               | $1.10/$4.40 per 1M tokens | Advanced fast reasoning           |
| Llama 3.2 1B (Local) | Via Ollama                       | Free                      | Privacy-sensitive, offline        |

### Retrieval Settings

Access these via the **Settings** dropdown:

#### 1. Chunks to Retrieve (top_k)

- **Range**: 3 - 20
- **Default**: 10
- **Effect**: More chunks = more context for the AI = higher accuracy but higher cost

#### 2. Max Chunks Per Source

- **Range**: 1 - 5
- **Default**: 5
- **Effect**:
  - **1 (diverse)**: Pulls from many different documents, broader perspective
  - **5 (focused)**: May pull multiple chunks from the same document, deeper detail

#### 3. Similarity Threshold

- **Range**: 0.00 - 0.80
- **Default**: 0.00
- **Effect**:
  - **0 (all)**: Returns all retrieved chunks regardless of relevance score
  - **0.8 (strict)**: Only returns highly relevant chunks, may miss some information

#### 4. Message History

- **Range**: 0 - 10 pairs
- **Default**: 3 pairs
- **Effect**: How many previous Q&A pairs to include for context in follow-up questions

#### 5. Max Response Tokens

- **Range**: 500 - 4000
- **Default**: 1500
- **Effect**: Maximum length of the AI's response (more tokens = higher cost)

### How Settings Affect Responses and Costs

| Setting                   | Increase Value           | Response Quality           | API Cost                        |
| ------------------------- | ------------------------ | -------------------------- | ------------------------------- |
| **Chunks to Retrieve**    | More context for answers | More relevant info         | Higher (more embedding lookups) |
| **Max Chunks Per Source** | Deeper focus on sources  | Better for specific topics | Neutral                         |
| **Similarity Threshold**  | Stricter relevance       | May miss edge cases        | Lower (fewer chunks)            |
| **Message History**       | Better follow-ups        | More contextual answers    | Higher (more tokens in prompt)  |
| **Max Response Tokens**   | Longer responses         | More comprehensive         | Higher (more output tokens)     |
| **Model Selection**       | More capable model       | Better reasoning           | Varies significantly            |

**Cost Optimization Tips:**

1. **For routine questions**: Use GPT-4o Mini with lower top_k (5-8)
2. **For complex analysis**: Use GPT-4o or o1 with higher top_k (10-15)
3. **For cost-sensitive production**: Use GPT-4o Mini with similarity threshold 0.3
4. **For offline/privacy**: Use Llama 3.2 1B via Ollama (requires local Ollama server)

**Cost Display:**
Each response shows the actual cost in the message footer (e.g., "$0.0023"), including a breakdown when you hover over it.

---

## API Endpoints

| Method | Endpoint          | Description                      |
| ------ | ----------------- | -------------------------------- |
| GET    | `/health`         | Health check                     |
| POST   | `/query`          | Send a question to the RAG agent |
| POST   | `/clear-history`  | Clear conversation history       |
| GET    | `/session-memory` | Get current session memory state |
| GET    | `/cache-stats`    | Get prompt cache statistics      |
| GET    | `/models`         | List available models            |
| GET    | `/model`          | Get current model                |
| POST   | `/model`          | Set model for subsequent queries |

**Example Query Request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the annual leave policy for US employees?",
    "top_k": 10,
    "history_length": 3,
    "max_chunks_per_source": 2,
    "similarity_threshold": 0.3,
    "max_completion_tokens": 1500,
    "model": "gpt-4o"
  }'
```

---

## Troubleshooting

### Backend Issues

**Database connection error:**

```
Error: connection to server at "localhost" failed
```

- Ensure PostgreSQL is running: `pg_isready`
- Check credentials in `.env`
- Verify the database exists: `psql -U postgres -l`

**pgvector extension missing:**

```
Error: type "vector" does not exist
```

- Install pgvector extension (see Prerequisites)
- Run: `CREATE EXTENSION IF NOT EXISTS vector;`

**OpenAI API error:**

```
Error: AuthenticationError
```

- Verify your API key in `.env`
- Check you have API credits: https://platform.openai.com/usage

**Empty database warning:**

```
[WARNING] Database is empty! Run: python ingestion.py ../novatech-kb
```

- Run the ingestion pipeline to populate the database

### Frontend Issues

**Cannot connect to backend:**

```
Error connecting to the server
```

- Ensure backend is running on port 8000
- Check CORS is enabled (it is by default)
- Verify API_BASE_URL in `App.js` matches your backend URL

**npm install fails:**

```
npm ERR! peer dep missing
```

- Try: `npm install --legacy-peer-deps`
- Or delete `node_modules` and `package-lock.json`, then reinstall

### Running Tests

Test the RAG agent against sample questions:

```bash
cd hackathon-ps/solution
python main.py test
```

Results will be saved to `test_results.json`.

---

## Project Structure

```
hackathon-ps/
├── README.md                 # This file
├── problem-statement.md      # Original problem statement
├── novatech-kb/              # Knowledge base (PDF documents)
│   ├── hr/                   # HR policies
│   ├── products/             # Product documentation
│   ├── it/                   # IT procedures
│   └── test_questions.json   # Test questions for evaluation
├── solution/                 # Python backend
│   ├── .env.example          # Environment template
│   ├── .env                  # Your configuration (create this)
│   ├── requirements.txt      # Python dependencies
│   ├── main.py               # CLI entry point
│   ├── server.py             # FastAPI server
│   ├── agent.py              # RAG agent implementation
│   ├── retriever.py          # Vector similarity search
│   ├── embeddings.py         # OpenAI embeddings
│   ├── db.py                 # Database operations
│   ├── chunker.py            # Text chunking
│   ├── pdf_parser.py         # PDF parsing
│   ├── ingestion.py          # Knowledge base ingestion
│   └── config.py             # Configuration management
└── fe/                       # React frontend
    ├── package.json          # Node.js dependencies
    ├── tailwind.config.js    # TailwindCSS configuration
    ├── public/               # Static assets
    └── src/
        ├── App.js            # Main React component
        ├── App.css           # Custom styles
        └── index.js          # React entry point
```

---

## License

This project was created for the RapidClaims Hackathon.

# LLM Instructor

An AI-powered Socratic tutor that helps you learn Large Language Models by guiding you through Stanford CS336 assignments. Instead of giving answers directly, it asks questions, provides hints, runs tests, and tracks your progress.

## Features

- **Socratic Teaching**: Guides learning through questions rather than direct answers
- **Real-time Streaming**: WebSocket-based chat with streaming responses
- **Progress Tracking**: Visual sidebar showing function completion status
- **Local LLM Support**: Runs on local GPU with tiny-gpt2 (expandable to larger models)
- **Test Integration**: Automatically runs and analyzes pytest results
- **Dark Theme UI**: Optimized for extended coding sessions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Chat     │  │  Progress   │  │   Welcome Screen    │  │
│  │  Interface  │  │   Sidebar   │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  WebSocket  │  │   Routes    │  │   Teaching Engine   │  │
│  │   Handler   │  │             │  │   (State Machine)   │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
│         │                                                    │
│  ┌──────▼──────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Local LLM  │  │     RAG     │  │    Test Runner      │  │
│  │  (tiny-gpt2)│  │  (ChromaDB) │  │     (pytest)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) NVIDIA GPU for faster local inference

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd llm-instructor-frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py          # HTTP endpoints
│   │   │   └── websocket.py       # WebSocket chat handler
│   │   ├── llm/
│   │   │   └── local_client.py    # Local LLM inference
│   │   ├── rag/
│   │   │   └── stub.py            # RAG retrieval (stub)
│   │   ├── execution/
│   │   │   └── test_runner.py     # pytest integration
│   │   └── db/
│   │       └── database.py        # SQLite session storage
│   └── tests/                     # Backend tests
│
├── llm-instructor-frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx           # Main chat interface
│   │   │   ├── ProgressSidebar.tsx # Function progress tree
│   │   │   └── WelcomeScreen.tsx  # Landing screen
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts    # WebSocket connection
│   │   └── types/
│   │       └── index.ts           # TypeScript types
│   └── package.json
│
└── CourseMaterial/                # CS336 course materials
```

## Assignment 1 Functions

The tutor guides you through implementing these 21 core functions:

| Phase | Functions |
|-------|-----------|
| **Linear Algebra** | linear, embedding, softmax |
| **Attention** | attention, grouped_query_attention, sliding_window_attention |
| **Normalization** | rms_norm, layer_norm |
| **Activations** | gelu, silu |
| **Positional** | rotary_position_embedding, apply_rope, alibi_attention |
| **MLP** | mlp, gated_mlp |
| **Architecture** | transformer_block, transformer_lm |
| **Training** | cross_entropy_loss, gradient_clipping, lr_cosine_schedule |
| **Optimization** | adamw_step |

## Development

### Running Tests

```bash
# Backend
cd backend
source .venv/bin/activate
pytest tests/ -v

# Frontend
cd llm-instructor-frontend
npm run build  # TypeScript type checking
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/progress/{session_id}` | GET | Get session progress |
| `/api/test/run` | POST | Run specific test |
| `/ws/chat/{session_id}` | WS | Chat WebSocket |

## Configuration

### Environment Variables

```bash
# Backend
ANTHROPIC_API_KEY=sk-...  # For Claude API fallback
OPENAI_API_KEY=sk-...     # For GPT-4 fallback

# Model selection (in local_client.py)
_MODEL_NAME = "sshleifer/tiny-gpt2"  # Change to larger model
```

## Roadmap

- [ ] RAG integration with ChromaDB for course materials
- [ ] Claude/GPT-4 API fallback for complex explanations
- [ ] Multi-assignment support (Assignments 2-5)
- [ ] Code diff viewer for implementation comparison
- [ ] Concept quiz mode before coding

## License

MIT

## Acknowledgments

- Stanford CS336: Language Modeling from Scratch
- Hugging Face Transformers
- FastAPI & React

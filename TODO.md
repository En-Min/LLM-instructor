# LLM Instructor - Implementation TODO

> **Goal**: Build an LLM teaching assistant that helps YOU learn by guiding you through CS336 Assignment 1 using Socratic dialogue.

---

## Overview

This TODO is organized into **independent phases** that can be built separately and integrated later:

- **Part A**: Backend (Ubuntu Server) - Teaching engine, RAG, LLM inference
- **Part B**: Frontend (Mac) - Chat UI and progress tracking
- **Part C**: Teaching Content - Prompts, examples, pedagogy
- **Part D**: Integration & Testing
- **Part E**: Post-Launch (Optional)

---

##  Part A: Backend Development (Ubuntu Server)

### A1: Environment Setup
**Goal**: Prepare Ubuntu server for development

- [ ] SSH into Ubuntu server
- [ ] Check GPUs: `nvidia-smi` (should show 4x RTX 3090)
- [ ] Install system dependencies:
  ```bash
  sudo apt update
  sudo apt install python3.11 python3.11-venv python3-pip nvidia-cuda-toolkit
  ```
- [ ] Create project directory:
  ```bash
  mkdir -p ~/llm-instructor-backend/{agent,llm,rag,execution,db,api,tests}
  touch ~/llm-instructor-backend/{agent,llm,rag,execution,db,api,tests}/__init__.py
  ```
- [ ] Setup Python venv:
  ```bash
  cd ~/llm-instructor-backend
  python3.11 -m venv venv
  source venv/bin/activate
  ```
- [ ] Transfer/clone course materials to Ubuntu:
  ```bash
  # Option 1: scp from Mac
  scp -r ~/VibeCoding/WW03-3-26-LLM-instructor user@ubuntu:~/

  # Option 2: git clone if in repo
  git clone <repo-url> ~/cs336-materials
  ```
- [ ] Create symlink to assignments:
  ```bash
  ln -s ~/cs336-materials/CourseMaterial ~/llm-instructor-backend/assignments
  ```
- [ ] Create `.env` file:
  ```bash
  cat > .env <<EOF
  OPENAI_API_KEY=your_key_here
  GOOGLE_API_KEY=your_key_here
  VLLM_HOST=localhost:8001
  DATABASE_URL=sqlite:///instructor.db
  ASSIGNMENT_DIR=/home/ubuntu/llm-instructor-backend/assignments/assignment1-basics-main/assignment1-basics-main
  EOF
  ```

**✅ Checkpoint**: Can you SSH into Ubuntu? Can you see 4 GPUs? Do symlinks work?

---

### A2: Install Dependencies
**Goal**: Install all Python packages needed

- [ ] Install core dependencies:
  ```bash
  pip install vllm==0.6.0
  pip install fastapi uvicorn[standard] websockets
  pip install langgraph langchain langchain-openai
  pip install google-generativeai  # For Gemini
  pip install chromadb sentence-transformers
  pip install sqlalchemy alembic
  pip install pytest pydantic python-multipart
  pip install PyPDF2 tiktoken python-dotenv
  ```
- [ ] Freeze requirements:
  ```bash
  pip freeze > requirements.txt
  ```
- [ ] Verify imports:
  ```bash
  python -c "import vllm; import langchain; import chromadb; print('All imports successful!')"
  ```

**✅ Checkpoint**: All imports work? requirements.txt created?

---

### A3: LLM Inference Setup
**Goal**: Get local LLM running on 4x RTX 3090

- [ ] **Choose model**:
  - Recommended: **Qwen/Qwen2.5-32B-Instruct** (fits on 4x24GB)
  - Alternative: DeepSeek-V3 (requires quantization)
- [ ] Download model (this will take time):
  ```bash
  python -c "from vllm import LLM; llm = LLM('Qwen/Qwen2.5-32B-Instruct', tensor_parallel_size=4, download_dir='./models')"
  ```
- [ ] Create `start_vllm.sh`:
  ```bash
  #!/bin/bash
  source venv/bin/activate
  vllm serve Qwen/Qwen2.5-32B-Instruct \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code
  ```
- [ ] Make executable: `chmod +x start_vllm.sh`
- [ ] Start vLLM server: `./start_vllm.sh &`
- [ ] Monitor GPU loading: `watch -n 1 nvidia-smi`
  - Wait ~2-5 minutes for model to load
  - Each GPU should show ~20-22GB usage
- [ ] Test inference:
  ```bash
  curl http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-32B-Instruct", "prompt": "Explain transformers in 50 words:", "max_tokens": 100}'
  ```
- [ ] **Record benchmarks**:
  - Model load time: ______ minutes
  - VRAM per GPU: ______ GB
  - First token latency: ______ ms
  - Throughput: ______ tokens/sec

**✅ Checkpoint**: vLLM responds to curl requests? GPUs all utilized?

---

### A4: Database Schema
**Goal**: Setup SQLite database for progress tracking

- [ ] Create `db/models.py`:
  ```python
  from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey
  from sqlalchemy.ext.declarative import declarative_base
  from sqlalchemy.orm import relationship
  from datetime import datetime

  Base = declarative_base()

  class Session(Base):
      __tablename__ = "sessions"
      id = Column(Integer, primary_key=True)
      created_at = Column(DateTime, default=datetime.utcnow)
      assignment = Column(String, default="assignment1")
      progress = relationship("Progress", back_populates="session")
      messages = relationship("Message", back_populates="session")

  class Progress(Base):
      __tablename__ = "progress"
      id = Column(Integer, primary_key=True)
      session_id = Column(Integer, ForeignKey("sessions.id"))
      function_name = Column(String)
      state = Column(String, default="concept_check")
      attempts = Column(Integer, default=0)
      hints_given = Column(Integer, default=0)
      test_status = Column(String, default="not_run")
      last_error = Column(JSON, nullable=True)
      updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
      session = relationship("Session", back_populates="progress")

  class Message(Base):
      __tablename__ = "messages"
      id = Column(Integer, primary_key=True)
      session_id = Column(Integer, ForeignKey("sessions.id"))
      role = Column(String)  # user | assistant
      content = Column(String)
      timestamp = Column(DateTime, default=datetime.utcnow)
      llm_used = Column(String, nullable=True)  # local | gpt5.2 | gemini
      session = relationship("Session", back_populates="messages")
  ```
- [ ] Create `db/database.py`:
  ```python
  from sqlalchemy import create_engine
  from sqlalchemy.orm import sessionmaker
  from .models import Base
  import os

  DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instructor.db")
  engine = create_engine(DATABASE_URL)
  SessionLocal = sessionmaker(bind=engine)

  def init_db():
      Base.metadata.create_all(bind=engine)
      print("Database initialized!")

  def get_db():
      db = SessionLocal()
      try:
          yield db
      finally:
          db.close()
  ```
- [ ] Create `db/crud.py` (database operations):
  ```python
  from .database import SessionLocal
  from .models import Session, Progress, Message

  def create_session(assignment="assignment1"):
      db = SessionLocal()
      session = Session(assignment=assignment)
      db.add(session)
      db.commit()
      db.refresh(session)
      db.close()
      return session

  def get_progress(session_id, function_name):
      db = SessionLocal()
      progress = db.query(Progress).filter_by(
          session_id=session_id, function_name=function_name
      ).first()
      db.close()
      return progress

  def update_progress(session_id, function_name, state=None, **kwargs):
      db = SessionLocal()
      progress = get_progress(session_id, function_name)
      if not progress:
          progress = Progress(session_id=session_id, function_name=function_name)
          db.add(progress)
      if state:
          progress.state = state
      for key, value in kwargs.items():
          setattr(progress, key, value)
      db.commit()
      db.close()

  def save_message(session_id, role, content, llm_used=None):
      db = SessionLocal()
      message = Message(session_id=session_id, role=role, content=content, llm_used=llm_used)
      db.add(message)
      db.commit()
      db.close()
  ```
- [ ] Initialize database:
  ```bash
  python -c "from db.database import init_db; init_db()"
  ```
- [ ] Test CRUD operations:
  ```bash
  python -c "
  from db.crud import create_session, update_progress
  session = create_session()
  print(f'Created session: {session.id}')
  update_progress(session.id, 'linear', state='concept_check')
  print('Progress updated!')
  "
  ```

**✅ Checkpoint**: Database file created? CRUD operations work?

---

### A5: RAG System (Knowledge Base)
**Goal**: Index all course materials for semantic search

- [ ] Create `rag/embeddings.py`:
  ```python
  from sentence_transformers import SentenceTransformer
  import numpy as np

  class EmbeddingModel:
      def __init__(self, model_name='all-MiniLM-L6-v2'):
          # Fast, good quality (384 dimensions)
          self.model = SentenceTransformer(model_name)

      def embed(self, texts: list[str]) -> np.ndarray:
          return self.model.encode(texts, show_progress_bar=True)
  ```
- [ ] Create `rag/retriever.py`:
  ```python
  import chromadb
  from chromadb.config import Settings

  class RAGRetriever:
      def __init__(self, persist_dir="./chroma_db"):
          self.client = chromadb.PersistentClient(
              path=persist_dir,
              settings=Settings(anonymized_telemetry=False)
          )
          self.collection = self.client.get_or_create_collection(
              name="cs336_materials",
              metadata={"hnsw:space": "cosine"}
          )

      def add_documents(self, chunks: list[dict], embeddings: np.ndarray):
          self.collection.add(
              ids=[f"doc_{i}" for i in range(len(chunks))],
              embeddings=embeddings.tolist(),
              documents=[c["content"] for c in chunks],
              metadatas=[c["metadata"] for c in chunks]
          )

      def search(self, query: str, top_k=5, filter_assignment=None):
          where_filter = {"assignment": filter_assignment} if filter_assignment else None
          results = self.collection.query(
              query_texts=[query],
              n_results=top_k,
              where=where_filter
          )
          return [
              {
                  "content": doc,
                  "metadata": meta,
                  "distance": dist
              }
              for doc, meta, dist in zip(
                  results["documents"][0],
                  results["metadatas"][0],
                  results["distances"][0]
              )
          ]
  ```
- [ ] Create `rag/indexer.py` (document processing):
  ```python
  import PyPDF2
  import tiktoken
  from pathlib import Path

  class DocumentIndexer:
      def __init__(self, chunk_size=500, chunk_overlap=50):
          self.chunk_size = chunk_size
          self.chunk_overlap = chunk_overlap
          self.tokenizer = tiktoken.get_encoding("cl100k_base")

      def index_pdf(self, pdf_path: Path):
          with open(pdf_path, 'rb') as f:
              reader = PyPDF2.PdfReader(f)
              text = "\n\n".join(page.extract_text() for page in reader.pages)

          metadata = {
              "source": "assignment_pdf",
              "assignment": "assignment1",  # Extract from filename
              "file": str(pdf_path)
          }
          return self.chunk_text(text, metadata)

      def chunk_text(self, text: str, metadata: dict):
          tokens = self.tokenizer.encode(text)
          chunks = []
          for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
              chunk_tokens = tokens[i:i + self.chunk_size]
              chunk_text = self.tokenizer.decode(chunk_tokens)
              chunks.append({
                  "content": chunk_text,
                  "metadata": {**metadata, "chunk_index": len(chunks)}
              })
          return chunks
  ```
- [ ] Create indexing script `scripts/index_materials.py`:
  ```python
  from rag.indexer import DocumentIndexer
  from rag.embeddings import EmbeddingModel
  from rag.retriever import RAGRetriever
  from pathlib import Path

  indexer = DocumentIndexer()
  embedder = EmbeddingModel()
  retriever = RAGRetriever()

  chunks = []

  # Index assignment PDFs
  for pdf in Path("assignments").rglob("*.pdf"):
      print(f"Indexing {pdf}")
      chunks.extend(indexer.index_pdf(pdf))

  # TODO: Index lecture Python files, test docstrings

  print(f"Total chunks: {len(chunks)}")

  # Generate embeddings
  print("Generating embeddings...")
  embeddings = embedder.embed([c["content"] for c in chunks])

  # Store in ChromaDB
  print("Storing in ChromaDB...")
  retriever.add_documents(chunks, embeddings)
  print("Done!")
  ```
- [ ] Run indexing:
  ```bash
  python scripts/index_materials.py
  ```
- [ ] Test retrieval:
  ```python
  from rag.retriever import RAGRetriever

  retriever = RAGRetriever()
  results = retriever.search("How does RoPE work?", top_k=3)
  for i, r in enumerate(results):
      print(f"\n[{i+1}] Relevance: {1 - r['distance']:.2f}")
      print(f"Source: {r['metadata']['source']}")
      print(f"Content: {r['content'][:200]}...")
  ```

**✅ Checkpoint**: ChromaDB created? Search returns relevant results?

---

### A6: Execution Layer (Test Runner)
**Goal**: Safely execute pytest and parse results

- [ ] Create `execution/test_runner.py`:
  ```python
  import subprocess
  from pathlib import Path
  import os

  ASSIGNMENT_DIR = Path(os.getenv("ASSIGNMENT_DIR"))

  def run_test(function_name: str, assignment="assignment1"):
      cmd = [
          "uv", "run", "pytest",
          f"tests/test_nn_utils.py::test_{function_name}",
          "-v", "--tb=short"
      ]

      try:
          result = subprocess.run(
              cmd,
              cwd=ASSIGNMENT_DIR,
              capture_output=True,
              text=True,
              timeout=30
          )

          return {
              "function": function_name,
              "passed": result.returncode == 0,
              "stdout": result.stdout,
              "stderr": result.stderr,
              "summary": parse_failure(result.stdout) if result.returncode != 0 else "Test passed!"
          }
      except subprocess.TimeoutExpired:
          return {
              "function": function_name,
              "passed": False,
              "error": "Test timed out (possible infinite loop?)"
          }

  def parse_failure(output: str):
      """Extract key error info from pytest output"""
      lines = output.split("\n")
      for i, line in enumerate(lines):
          if "AssertionError" in line or "assert" in line.lower():
              return "\n".join(lines[max(0, i-2):min(len(lines), i+5)])
      return output  # Return full if parsing fails
  ```
- [ ] Create `execution/code_reader.py`:
  ```python
  from pathlib import Path
  import os

  ASSIGNMENT_DIR = Path(os.getenv("ASSIGNMENT_DIR"))

  def read_student_code(file_path: str):
      full_path = ASSIGNMENT_DIR / file_path

      # Security: validate path is within assignment directory
      if not full_path.resolve().is_relative_to(ASSIGNMENT_DIR.resolve()):
          return "Error: Path outside assignment directory"

      if not full_path.exists():
          return f"Error: File not found: {file_path}"

      return full_path.read_text()
  ```
- [ ] Test manually:
  ```python
  from execution.test_runner import run_test

  result = run_test("linear")
  print(result)
  # Should show test failure (NotImplementedError) initially
  ```

**✅ Checkpoint**: Can execute tests? Errors parsed correctly?

---

### A7: Agent Tools
**Goal**: Define tools the teaching agent can use

- [ ] Create `agent/tools.py`:
  ```python
  from langchain.tools import tool
  from execution.test_runner import run_test as _run_test
  from execution.code_reader import read_student_code as _read_code
  from rag.retriever import RAGRetriever

  @tool
  def run_test(function_name: str) -> dict:
      """Run pytest for a specific function.

      Args:
          function_name: Name like 'linear', 'rope', 'attention'

      Returns:
          dict with passed (bool), stdout, stderr, summary
      """
      return _run_test(function_name)

  @tool
  def read_student_code(file_path: str) -> str:
      """Read student's implementation file.

      Args:
          file_path: Relative path like 'cs336_basics/nn_utils.py'

      Returns:
          File contents or error message
      """
      return _read_code(file_path)

  @tool
  def search_materials(query: str, filter_assignment: str = None) -> str:
      """Search course materials using semantic search.

      Args:
          query: Natural language question
          filter_assignment: Optional filter like 'assignment1'

      Returns:
          Formatted string with top-3 results
      """
      retriever = RAGRetriever()
      results = retriever.search(query, top_k=3, filter_assignment=filter_assignment)

      output = []
      for i, result in enumerate(results):
          output.append(f"\n[{i+1}] Relevance: {1 - result['distance']:.2f}")
          output.append(f"Source: {result['metadata']['source']}")
          output.append(f"Content:\n{result['content']}\n")

      return "\n".join(output)

  # Export all tools
  ALL_TOOLS = [run_test, read_student_code, search_materials]
  ```
- [ ] Test tools independently:
  ```python
  from agent.tools import run_test, search_materials

  # Test pytest execution
  print(run_test("linear"))

  # Test RAG search
  print(search_materials("What is attention?"))
  ```

**✅ Checkpoint**: All 3 tools work independently?

---

### A8: Teaching Engine (State Machine)
**Goal**: Implement teaching state machine

- [ ] Create `agent/teaching_engine.py`:
  ```python
  from enum import Enum
  from db.crud import get_progress, update_progress

  class TeachingState(Enum):
      CONCEPT_CHECK = "concept_check"
      IMPLEMENTATION = "implementation"
      DEBUG = "debug"
      VERIFY = "verify"
      COMPLETE = "complete"

  class TeachingEngine:
      def __init__(self, session_id: int):
          self.session_id = session_id

      def get_state(self, function_name: str) -> TeachingState:
          progress = get_progress(self.session_id, function_name)
          if not progress:
              return TeachingState.CONCEPT_CHECK
          return TeachingState(progress.state)

      def transition(self, function_name: str, event: str, context: dict):
          current = self.get_state(function_name)

          transitions = {
              TeachingState.CONCEPT_CHECK: {
                  "understood": TeachingState.IMPLEMENTATION
              },
              TeachingState.IMPLEMENTATION: {
                  "run_test": TeachingState.DEBUG
              },
              TeachingState.DEBUG: {
                  "test_passed": TeachingState.VERIFY,
                  "test_failed": TeachingState.DEBUG
              },
              TeachingState.VERIFY: {
                  "confirmed": TeachingState.COMPLETE
              }
          }

          next_state = transitions.get(current, {}).get(event, current)
          update_progress(self.session_id, function_name, state=next_state.value, **context)
          return next_state

      def get_system_prompt(self, state: TeachingState, function_name: str, context: dict):
          from agent.prompts import PROMPTS
          return PROMPTS[state].format(
              function_name=function_name,
              **context
          )
  ```
- [ ] Test state machine:
  ```python
  from agent.teaching_engine import TeachingEngine, TeachingState

  engine = TeachingEngine(session_id=1)
  assert engine.get_state("linear") == TeachingState.CONCEPT_CHECK

  engine.transition("linear", "understood", {})
  assert engine.get_state("linear") == TeachingState.IMPLEMENTATION
  print("State machine works!")
  ```

**✅ Checkpoint**: State transitions work? DB updated correctly?

---

### A9: LLM Clients & Router
**Goal**: Setup local + API LLM clients

- [ ] Create `llm/vllm_client.py`:
  ```python
  from openai import OpenAI
  import os

  client = OpenAI(
      base_url=f"http://{os.getenv('VLLM_HOST', 'localhost:8001')}/v1",
      api_key="dummy"
  )

  def generate(prompt: str, max_tokens=500):
      response = client.completions.create(
          model="Qwen/Qwen2.5-32B-Instruct",
          prompt=prompt,
          max_tokens=max_tokens
      )
      return response.choices[0].text
  ```
- [ ] Create `llm/api_client.py`:
  ```python
  from openai import OpenAI
  import google.generativeai as genai
  import os

  # GPT-5.2 client
  openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  def generate_gpt(prompt: str, max_tokens=500):
      response = openai_client.chat.completions.create(
          model="gpt-5-turbo",  # Update when GPT-5.2 available
          messages=[{"role": "user", "content": prompt}],
          max_tokens=max_tokens
      )
      return response.choices[0].message.content

  # Gemini client
  genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
  gemini_model = genai.GenerativeModel("gemini-1.5-pro")

  def generate_gemini(prompt: str, max_tokens=500):
      response = gemini_model.generate_content(prompt)
      return response.text
  ```
- [ ] Create `llm/router.py`:
  ```python
  from .vllm_client import generate as generate_local
  from .api_client import generate_gpt, generate_gemini

  def choose_llm(message: str, context: dict):
      """Route to appropriate LLM based on context"""

      # Explicit user request
      if "use smart model" in message.lower() or "use gpt" in message.lower():
          return "gpt5.2"

      # Auto-escalate after many failures
      if context.get("failed_attempts", 0) >= 6:
          return "gpt5.2"

      # Default to local
      return "local"

  def generate(prompt: str, model_choice: str, max_tokens=500):
      if model_choice == "gpt5.2":
          return generate_gpt(prompt, max_tokens)
      elif model_choice == "gemini":
          return generate_gemini(prompt, max_tokens)
      else:
          return generate_local(prompt, max_tokens)
  ```
- [ ] Test routing:
  ```python
  from llm.router import generate, choose_llm

  # Test local
  print(generate("Explain transformers in 50 words", "local"))

  # Test routing logic
  assert choose_llm("Help with linear", {}) == "local"
  assert choose_llm("Use GPT please", {}) == "gpt5.2"
  assert choose_llm("Help", {"failed_attempts": 7}) == "gpt5.2"
  ```

**✅ Checkpoint**: All 3 LLM clients work? Router logic correct?

---

### A10: LangGraph Agent (IMPORTANT - See Part C first!)
**Goal**: Create ReAct agent with teaching engine

**⚠️ NOTE**: Complete Part C (Teaching Content Preparation) BEFORE this step!
You need the prompts from `agent/prompts.py` to be ready first.

- [ ] Create `agent/orchestrator.py`:
  ```python
  from langgraph.prebuilt import create_react_agent
  from langchain_openai import ChatOpenAI
  from agent.tools import ALL_TOOLS
  from agent.teaching_engine import TeachingEngine
  from llm.router import choose_llm
  import os

  def create_instructor_agent(session_id: int):
      teaching_engine = TeachingEngine(session_id)

      # Use local model by default (vLLM via OpenAI-compatible API)
      llm = ChatOpenAI(
          base_url=f"http://{os.getenv('VLLM_HOST')}/v1",
          api_key="dummy",
          model="Qwen/Qwen2.5-32B-Instruct"
      )

      agent = create_react_agent(
          llm,
          tools=ALL_TOOLS,
          state_modifier=lambda state: {
              **state,
              # Inject teaching-specific system prompt
              "system_prompt": teaching_engine.get_system_prompt(
                  teaching_engine.get_state(state.get("current_function", "")),
                  state.get("current_function", ""),
                  state
              )
          }
      )

      return agent, teaching_engine
  ```
- [ ] Test agent:
  ```python
  from agent.orchestrator import create_instructor_agent

  agent, engine = create_instructor_agent(session_id=1)
  response = agent.invoke({"messages": [("user", "I want to work on Linear")]})
  print(response)
  # Should get Socratic response, not direct code
  ```

**✅ Checkpoint**: Agent responds? Uses Socratic method?

---

### A11: FastAPI Backend
**Goal**: Create HTTP + WebSocket API

- [ ] Create `api/routes.py`:
  ```python
  from fastapi import APIRouter
  from db.crud import create_session, get_progress
  from execution.test_runner import run_test

  router = APIRouter()

  @router.get("/api/sessions/create")
  def create_new_session():
      session = create_session()
      return {"session_id": session.id}

  @router.get("/api/progress/{session_id}")
  def get_session_progress(session_id: int):
      # TODO: Get all progress for session
      return []

  @router.post("/api/test/run")
  def run_function_test(function_name: str):
      result = run_test(function_name)
      return result
  ```
- [ ] Create `api/websocket.py`:
  ```python
  from fastapi import WebSocket, WebSocketDisconnect
  from agent.orchestrator import create_instructor_agent
  import json

  async def chat_handler(websocket: WebSocket, session_id: int):
      await websocket.accept()
      agent, engine = create_instructor_agent(session_id)

      try:
          async for message in websocket.iter_text():
              payload = json.loads(message)
              user_text = payload.get("content", "")

              # Stream agent response
              async for chunk in agent.astream({"messages": [("user", user_text)]}):
                  if "output" in chunk:
                      await websocket.send_json({
                          "type": "assistant_chunk",
                          "content": chunk["output"]
                      })

              await websocket.send_json({"type": "assistant_end"})

      except WebSocketDisconnect:
          print(f"Client disconnected: {session_id}")
  ```
- [ ] Create `main.py`:
  ```python
  from fastapi import FastAPI, WebSocket
  from fastapi.middleware.cors import CORSMiddleware
  from api.routes import router
  from api.websocket import chat_handler

  app = FastAPI(title="LLM Instructor API")

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # Restrict in production
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  app.include_router(router)

  @app.websocket("/ws/chat/{session_id}")
  async def websocket_endpoint(websocket: WebSocket, session_id: int):
      await chat_handler(websocket, session_id)

  if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)
  ```
- [ ] Start backend:
  ```bash
  python main.py
  ```
- [ ] Test endpoints:
  ```bash
  # Test session creation
  curl http://localhost:8000/api/sessions/create

  # Test WebSocket (install wscat: npm install -g wscat)
  wscat -c ws://localhost:8000/ws/chat/1
  > {"content": "I want to work on Linear"}
  ```

**✅ Checkpoint**: API endpoints work? WebSocket connects?

---

## Part B: Frontend Development (Mac)

### B1: Project Setup
- [ ] Create React + TypeScript project:
  ```bash
  npm create vite@latest llm-instructor-frontend -- --template react-ts
  cd llm-instructor-frontend
  npm install
  ```
- [ ] Install dependencies:
  ```bash
  npm install axios
  npm install tailwindcss postcss autoprefixer
  npm install react-markdown remark-gfm
  npm install prismjs  # For code highlighting
  npm install date-fns  # For timestamps
  ```
- [ ] Setup Tailwind:
  ```bash
  npx tailwindcss init -p
  ```
- [ ] Configure `tailwind.config.js`:
  ```js
  module.exports = {
    content: ['./src/**/*.{js,ts,jsx,tsx}'],
    theme: { extend: {} },
    darkMode: 'class',
  }
  ```
- [ ] Add Tailwind to `src/index.css`:
  ```css
  @tailwind base;
  @tailwind components;
  @tailwind utilities;
  ```

**✅ Checkpoint**: `npm run dev` works? Tailwind styles apply?

---

### B2-B7: [Frontend implementation details continue...]

*(Rest of frontend implementation omitted for brevity - similar structure to original TODO but focused on teaching UX)*

---

## Part C: Teaching Content Preparation

> **CRITICAL**: This is where teaching quality is determined!

### C1: System Prompts for Each State
**Goal**: Write effective Socratic prompts for each teaching state

- [ ] Create `agent/prompts.py` with state-specific prompts
- [ ] **CONCEPT_CHECK prompt**:
  - Must ask probing questions (not give explanations)
  - Check understanding of shapes, math, purpose
  - Use analogies
  - Decide when to transition to IMPLEMENTATION
- [ ] **IMPLEMENTATION prompt**:
  - Provide architectural hints (NOT code)
  - Suggest PyTorch operations to consider
  - Warn about common pitfalls
  - Encourage student to try implementing
- [ ] **DEBUG prompt**:
  - Analyze test failures
  - Ask Socratic questions about errors
  - Implement progressive hint system (L1→L2→L3)
  - Track attempts and escalate appropriately
- [ ] **VERIFY prompt**:
  - Congratulate success
  - Ask student to explain their approach
  - Suggest optimizations (non-critical)
  - Recommend next function

**✅ Checkpoint**: Read prompts aloud. Do they sound Socratic? Or do they give away answers?

---

### C2: Function-Specific Teaching Strategy
**Goal**: Define teaching approach for each Assignment 1 function

- [ ] For each core function, document:
  - **Conceptual questions** to ask in CONCEPT_CHECK
  - **Common mistakes** students make
  - **Hint progression** (L1, L2, L3 hints for typical errors)
  - **Teaching examples** (sample conversation snippets)

- [ ] **run_linear** teaching strategy:
  - Concept questions: "What's a linear transformation?", "Why transpose weights?"
  - Common mistake: Wrong matrix multiplication order
  - L1 hint: "Think about dimension compatibility"
  - L2 hint: "Check the weight matrix shape"
  - L3 hint: "Try `in_features @ weights.T`"

- [ ] **run_scaled_dot_product_attention** teaching strategy:
  - Concept questions: "What does attention compute?", "Why scale by sqrt(d_k)?"
  - Common mistakes: Forgetting scaling, masking after softmax
  - L1 hint: "Remember attention has 4 steps: similarity, scale, mask, softmax"
  - L2 hint: "Masking should set positions to -inf BEFORE softmax"
  - L3 hint: "Try `scores.masked_fill(mask, float('-inf'))`"

- [ ] **run_rope** teaching strategy:
  - Concept questions: "How is RoPE different from absolute position embeddings?"
  - Common mistakes: Not pairing dimensions, treating as additive embedding
  - L1 hint: "RoPE applies rotation to PAIRS of dimensions"
  - L2 hint: "Think about complex number multiplication"
  - L3 hint: Show complex number rotation formula

- [ ] Repeat for other functions...

**✅ Checkpoint**: Can you explain the teaching strategy for each function to someone else?

---

### C3: Test Pedagogy
**Goal**: Validate teaching approach works

- [ ] **Role-play test**: Simulate student conversations
  - Act as a confused student
  - See if agent asks good questions
  - Check if hints are progressive (not giving away answers)

- [ ] **Test with real Assignment 1 functions**:
  - Start implementing `run_linear`
  - Use instructor to guide you
  - Does it feel like learning? Or like being given answers?

- [ ] **Refine prompts based on testing**:
  - If agent gives direct answers → Strengthen "do NOT provide code" constraints
  - If hints too vague → Add more specific guidance
  - If transitions wrong → Fix state machine logic

- [ ] **Test escalation to smart model**:
  - Simulate 6 failed attempts
  - Verify GPT-5.2 gets called
  - Check that explanation is deeper but still Socratic

**✅ Checkpoint**: Teaching feels effective? Student learns, not just copies?

---

## Part D: Integration & End-to-End Testing

### D1: Full Stack Integration
- [ ] Start all services:
  - Ubuntu: `./start_vllm.sh` (port 8001)
  - Ubuntu: `python main.py` (port 8000)
  - Mac: `npm run dev` (port 5173)

- [ ] Test full flow:
  1. Open frontend
  2. Create session
  3. Say "I want to work on Linear"
  4. Go through CONCEPT_CHECK → IMPLEMENTATION → DEBUG → VERIFY
  5. Verify progress saved
  6. Refresh page - progress should persist

**✅ Checkpoint**: Complete one full function end-to-end?

---

### D2: Teaching Effectiveness Testing
- [ ] Complete Assignment 1 using the instructor
- [ ] Track metrics:
  - How many hints needed per function?
  - Did local model handle most queries?
  - Did state transitions work correctly?
  - Did you learn? Or just copy code?

- [ ] Refine based on results:
  - Adjust prompts
  - Tune hint thresholds
  - Fix bugs in state machine

**✅ Final Goal**: Can you complete ≥10 functions with instructor guidance and feel you learned deeply?

---

## Part E: Optional Enhancements

### E1: Expand to Assignment 2-5
- [ ] Index new materials
- [ ] Adjust test adapters
- [ ] Extend teaching strategies

### E2: Advanced Features
- [ ] Code diff viewer
- [ ] Concept quiz mode
- [ ] Study notes export
- [ ] Voice input

### E3: Deployment
- [ ] Systemd service for backend
- [ ] nginx for frontend
- [ ] SSL/TLS certificates

---

## Progress Tracking

**Part A (Backend)**: [ ] 0/11 sections
**Part B (Frontend)**: [ ] 0/7 sections
**Part C (Teaching Content)**: [ ] 0/3 sections ⚠️ **MOST IMPORTANT**
**Part D (Integration)**: [ ] 0/2 sections
**Part E (Optional)**: [ ] 0/3 sections

---

## Success Criteria

### System Works
- ✅ Agent uses Socratic method (doesn't give direct answers)
- ✅ Progressive hints escalate properly (L1→L2→L3→GPT-5.2)
- ✅ RAG retrieves relevant course materials
- ✅ Tests execute and parse correctly
- ✅ State machine transitions correctly
- ✅ Local model handles 90%+ queries
- ✅ Frontend streams smoothly

### Teaching Works
- ✅ Student completes ≥10 Assignment 1 functions
- ✅ Student can explain their implementations
- ✅ Student understands concepts (not just copying)
- ✅ Student learns from debugging (not just getting answers)
- ✅ Student feels they learned deeply

---

**Remember: The goal is not just to build a working system, but to build an EFFECTIVE TEACHER. Focus on Part C (Teaching Content) - that's where the magic happens! 🎓**

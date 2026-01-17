# LLM Instructor - Technical Specification

> **Core Mission**: Build an intelligent teaching assistant that helps students learn LLM fundamentals by guiding them through CS336 Assignment 1 using Socratic dialogue.

---

## Table of Contents
1. [Project Vision](#project-vision)
2. [Assignment 1 Teaching Curriculum](#assignment-1-teaching-curriculum)
3. [Pedagogical Design](#pedagogical-design)
4. [Architecture](#architecture)
5. [Implementation Details](#implementation-details)
6. [Teaching Examples](#teaching-examples)

---

## Project Vision

### What are we building?
An LLM-powered **teaching assistant** that helps YOU learn how LLMs work by guiding you through implementing one from scratch in CS336 Assignment 1.

**Not**: A system that writes code for you
**But**: A Socratic tutor that asks probing questions, provides conceptual guidance, runs your tests, and helps you learn deeply

### Learning through Teaching
**You'll learn LLMs in two ways:**
1. **Using the instructor** → Learn Assignment 1 concepts (transformers, attention, RoPE, etc.)
2. **Building the instructor** → Learn LLM deployment, RAG, agents, and production systems

---

## Assignment 1 Teaching Curriculum

### Overview
Assignment 1 contains **21 functions** to implement across 6 phases, covering the complete pipeline from basic operations to training a transformer language model from scratch:

### Teaching Progression

#### **Phase 1: Foundation (Functions 1-5)**
*Goal: Understand basic tensor operations*

| Function | Concept | Prerequisites | Difficulty |
|----------|---------|---------------|------------|
| `run_linear` | Matrix multiplication | Basic linear algebra | ⭐ Easy |
| `run_embedding` | Lookup tables | Indexing | ⭐ Easy |
| `run_silu` | Activation functions | Element-wise ops | ⭐ Easy |
| `run_softmax` | Probability distributions | Exponentiation, numerical stability | ⭐⭐ Medium |
| `run_rmsnorm` | Normalization | Mean, variance, broadcasting | ⭐⭐ Medium |

**Why this order?**
- Start with simple operations to build confidence
- Each function introduces one new concept
- By end of Phase 1, student understands tensor shapes and broadcasting

#### **Phase 2: Feed-Forward Network (Function 6)**
*Goal: Combine concepts into a subsystem*

| Function | Concept | Prerequisites | Difficulty |
|----------|---------|---------------|------------|
| `run_swiglu` | Modern FFN architecture | Linear, SiLU | ⭐⭐ Medium |

**Key learning:**
- How components compose (Linear + SiLU + gating)
- Why modern architectures use gating (GLU variants)

#### **Phase 3: Attention Mechanism (Functions 7-10)**
*Goal: Master the transformer's core innovation*

| Function | Concept | Prerequisites | Difficulty |
|----------|---------|---------------|------------|
| `run_scaled_dot_product_attention` | Core attention | Softmax, matrix multiply | ⭐⭐⭐ Hard |
| `run_multihead_self_attention` | Parallelization | SDPA, reshaping | ⭐⭐⭐ Hard |
| `run_rope` | Positional embeddings | Complex numbers, rotation | ⭐⭐⭐⭐ Very Hard |
| `run_multihead_self_attention_with_rope` | Complete attention | MHA + RoPE | ⭐⭐⭐⭐ Very Hard |

**Why this order?**
- Build intuition with simple attention first
- Add complexity incrementally (single→multi-head→positional)
- RoPE is hardest concept - save for when student is ready

#### **Phase 4: Complete Transformer (Functions 11-12)**
*Goal: Integrate everything into a working LLM*

| Function | Concept | Prerequisites | Difficulty |
|----------|---------|---------------|------------|
| `run_transformer_block` | Transformer layer | Attention, FFN, residuals, normalization | ⭐⭐⭐⭐ Very Hard |
| `run_transformer_lm` | Full language model | Stacking blocks, embeddings | ⭐⭐⭐⭐⭐ Expert |

**Key learning:**
- Residual connections and layer normalization
- Stacking transformer blocks
- From architecture to working language model

#### **Phase 5: Training & Optimization (Functions 13-16)**
*Goal: Learn how to actually train the model*

| Function | Concept | Prerequisites | Difficulty |
|----------|---------|---------------|------------|
| `run_get_batch` | Data loading | Array indexing, randomness | ⭐ Easy |
| `run_cross_entropy` | Loss function | Softmax, log probabilities | ⭐⭐ Medium |
| `run_gradient_clipping` | Gradient stability | L2 norm, in-place operations | ⭐⭐ Medium |
| `get_adamw_cls` | Optimizer | AdamW algorithm | ⭐⭐⭐ Hard |
| `run_get_lr_cosine_schedule` | Learning rate scheduling | Cosine annealing, warmup | ⭐⭐ Medium |

**Key learning:**
- Training loop components
- Gradient clipping for stability
- Modern optimization (AdamW) vs SGD
- Learning rate scheduling strategies

#### **Phase 6: Checkpointing & Tokenization (Functions 17-21)**
*Goal: Complete production training pipeline*

| Function | Concept | Prerequisites | Difficulty |
|----------|---------|---------------|------------|
| `run_save_checkpoint` | Model serialization | torch.save, state dicts | ⭐ Easy |
| `run_load_checkpoint` | Model restoration | torch.load, state dicts | ⭐ Easy |
| `get_tokenizer` | BPE tokenizer | Tokenization concepts | ⭐⭐ Medium |
| `run_train_bpe` | Training tokenizer | BPE algorithm, merge operations | ⭐⭐⭐⭐ Very Hard |

**Key learning:**
- Checkpoint management for long training runs
- Byte-Pair Encoding (BPE) tokenization
- Building vocabulary from scratch
- Special tokens handling

**🎓 Final Achievement:**
At this point, you've implemented **every component needed to train a transformer language model from scratch** - from basic matrix operations to tokenization, training loops, and checkpointing. You deeply understand how modern LLMs work!

---

### Common Misconceptions by Function

#### `run_linear`
❌ **Mistake**: `in_features @ weights` (wrong dimension order)
✅ **Correct**: `in_features @ weights.T` or `weights @ in_features.T`.T
🎓 **Teaching point**: Matrix multiplication order matters; weights are stored as (out_features, in_features)

####  `run_embedding`
❌ **Mistake**: Loop over tokens and concatenate (inefficient)
✅ **Correct**: Single indexing operation `weights[token_ids]`
🎓 **Teaching point**: PyTorch indexing is powerful; avoid loops when possible

#### `run_softmax`
❌ **Mistake**: Naive `exp(x) / sum(exp(x))` causes overflow
✅ **Correct**: Subtract max first: `exp(x - x.max())` for numerical stability
🎓 **Teaching point**: Floating point numerics matter in ML

#### `run_scaled_dot_product_attention`
❌ **Mistake**: Forgetting to scale by sqrt(d_k)
❌ **Mistake**: Applying mask after softmax
✅ **Correct**: Scale QK^T, mask BEFORE softmax, then apply softmax
🎓 **Teaching point**: Order of operations matters; masking uses -inf trick

#### `run_rope`
❌ **Mistake**: Treating RoPE like absolute position embedding
❌ **Mistake**: Not properly pairing dimensions for rotation
✅ **Correct**: Apply rotation to pairs of dimensions using complex number multiplication
🎓 **Teaching point**: RoPE encodes relative positions through rotation; requires understanding complex numbers

---

## Pedagogical Design

### Teaching State Machine

```
CONCEPT_CHECK → IMPLEMENTATION → DEBUG → VERIFY → COMPLETE
```

#### State: CONCEPT_CHECK
**Goal**: Ensure conceptual understanding before coding

**Instructor behavior:**
- Ask probing questions about the concept
- Use analogies and examples
- Check understanding of input/output shapes
- Test understanding of mathematical foundation
- Do NOT give implementation hints yet

**Example questions for `run_scaled_dot_product_attention`:**
1. "What does attention let the model do?"
2. "Why do we use dot product to measure similarity?"
3. "If Q has shape (batch, queries, d_k), what shape should K^T have?"
4. "Why scale by sqrt(d_k)?"

**Transition**: When student demonstrates understanding → IMPLEMENTATION

#### State: IMPLEMENTATION
**Goal**: Guide implementation without giving away the answer

**Instructor behavior:**
- Ask: "What's your approach for implementing this?"
- Provide architectural hints (NOT code):
  - Suggest PyTorch operations to use
  - Highlight edge cases to handle
  - Warn about common pitfalls
- Encourage them to try implementing
- When ready: "Tell me when to run the test"

**Example guidance for `run_scaled_dot_product_attention`:**
- "You'll need torch.matmul for Q @ K^T"
- "Remember to handle the optional mask"
- "Softmax should be applied along the keys dimension"
- "Don't forget the scaling factor!"

**Transition**: When student says "run the test" → DEBUG

#### State: DEBUG
**Goal**: Help student learn from test failures

**Instructor behavior:**
- Run test automatically
- Analyze the error
- Ask Socratic questions about the error:
  - "What does this assertion tell you?"
  - "Why might the shapes be mismatched?"
  - "What happens when you transpose a (2, 3) matrix?"
- Provide **progressive hints**:
  - **Level 1 (conceptual)**: "Think about the dimension where you compute similarity"
  - **Level 2 (specific)**: "The issue is in how you're applying softmax - which dimension should it normalize?"
  - **Level 3 (partial code)**: "Try torch.softmax(scores, dim=-1)"
    *(Only after 3+ failed attempts)*

**Transition**:
- Test passed → VERIFY
- Test failed → Stay in DEBUG (give next level hint)

#### State: VERIFY
**Goal**: Solidify learning through reflection

**Instructor behavior:**
- Congratulate student
- Ask: "Can you explain your implementation approach?"
- Optionally suggest optimizations (non-critical)
- Check if student truly understands or just got lucky
- Suggest next function to work on

**Example questions:**
- "Why did you choose to apply the mask before softmax?"
- "What would happen if you forgot the scaling factor?"
- "Can you think of cases where this implementation might be slow?"

**Transition**: When discussion complete → COMPLETE

#### State: COMPLETE
**Terminal state**: Function is done, progress saved

---

### Socratic Dialogue Patterns

#### Pattern 1: Shape Reasoning
**When**: Student has shape mismatch error

**Bad** ❌: "The shapes don't match. You need to transpose."

**Good** ✅:
1. "What shapes do you expect for Q and K^T?"
2. Student: "Q is (batch, queries, d_k), K^T should be (batch, d_k, keys)"
3. "And what shape does your code produce?"
4. Student: "Oh... it's (batch, keys, d_k)"
5. "How can you fix that?" → Student discovers transpose

#### Pattern 2: Conceptual Confusion
**When**: Student asks "What does this function do?"

**Bad** ❌: [Gives detailed explanation]

**Good** ✅:
1. "What do you know about it already?"
2. "Let's look at the function signature. What do the inputs represent?"
3. "What would be useful for a transformer to compute?"
   → Student constructs understanding

#### Pattern 3: Debugging
**When**: Test fails with assertion error

**Bad** ❌: "The bug is on line 23. Change X to Y."

**Good** ✅:
1. "The test expects shape (2, 3, 512), but got (2, 512, 3)"
2. "Which dimensions are swapped?"
3. "Where in your code do you operate on those dimensions?"
   → Student locates bug themselves

---

### Hint Progression Strategy

**Level 1: Conceptual Hint** (Always start here)
- Reminds student of the concept
- No code, just ideas
- Example: "Remember that attention computes similarity between queries and keys"

**Level 2: Operation-Specific Hint** (After 2 failures)
- Points to specific PyTorch operation or algorithmic step
- Still no code
- Example: "You need to use torch.softmax on the similarity scores. Which dimension should be normalized?"

**Level 3: Partial Code** (After 4 failures OR student explicitly asks for more help)
- Shows small code snippet
- Leaves some work for student
- Example: "Try: `scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)`"

**Level 4: Escalate to Smart Model** (After 6 failures)
- Use GPT-5.2 or Gemini for deeper explanation
- Stronger model can provide more nuanced guidance
- Still maintains Socratic approach

**Tracking**: Store `hints_given` count in database per function

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                      YOU (Student)                      │
│  - Implements functions in cs336_basics/               │
│  - Chats with instructor for guidance                   │
└──────────────────┬──────────────────────────────────────┘
                   │ WebSocket + HTTP
┌──────────────────┴──────────────────────────────────────┐
│                  LLM INSTRUCTOR SYSTEM                  │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Frontend (Mac - React)                         │   │
│  │  - Chat interface with markdown/code rendering│   │
│  │  - Progress sidebar (function tree, statuses) │   │
│  │  - Real-time streaming                        │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ Backend (Ubuntu Server - FastAPI)              │   │
│  │                                                 │   │
│  │  ┌─────────────────────────────────────────┐  │   │
│  │  │ Teaching Agent (LangGraph)              │  │   │
│  │  │  - State machine (CONCEPT→DEBUG→etc.)   │  │   │
│  │  │  - Tool calling (run_test, RAG search) │  │   │
│  │  │  - Socratic prompting                   │  │   │
│  │  └─────────────────────────────────────────┘  │   │
│  │                                                 │   │
│  │  ┌─────────────────────────────────────────┐  │   │
│  │  │ RAG System (ChromaDB)                   │  │   │
│  │  │  - Indexed: Assignment PDFs, lectures   │  │   │
│  │  │  - Semantic search for explanations     │  │   │
│  │  └─────────────────────────────────────────┘  │   │
│  │                                                 │   │
│  │  ┌─────────────────────────────────────────┐  │   │
│  │  │ Tools                                    │  │   │
│  │  │  - run_test(): Execute pytest           │  │   │
│  │  │  - read_student_code(): Read files      │  │   │
│  │  │  - search_materials(): RAG query        │  │   │
│  │  └─────────────────────────────────────────┘  │   │
│  │                                                 │   │
│  │  ┌─────────────────────────────────────────┐  │   │
│  │  │ Database (SQLite)                       │  │   │
│  │  │  - Session state                        │  │   │
│  │  │  - Progress per function                │  │   │
│  │  │  - Conversation history                 │  │   │
│  │  └─────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ LLM Inference                                   │   │
│  │  - Local (vLLM): Qwen2.5-32B on 4× RTX 3090   │   │
│  │  - API Fallback: GPT-5.2 or Gemini            │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Local LLM (90%+ queries)**:
- Handles simple guidance, conceptual questions, hint generation
- Fast (no API latency), free (no costs), private
- Qwen2.5-32B is strong enough for teaching

**API Fallback (complex reasoning)**:
- When student stuck after many attempts
- Deep explanations of RoPE, attention math
- GPT-5.2 > local for nuanced pedagogy

**RAG System**:
- Grounds agent in course materials
- Prevents hallucination
- Provides references to lectures when explaining concepts

**Tools**:
- `run_test()`: Automated testing, instant feedback
- Agent can check student's work without manual intervention
- `search_materials()`: Fetch relevant lecture content on-demand

---

## Implementation Details

### Backend Components

#### `agent/teaching_engine.py` - State Machine

```python
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
        """Query DB for current teaching state"""
        progress = db.get_progress(self.session_id, function_name)
        return TeachingState(progress.state) if progress else TeachingState.CONCEPT_CHECK

    def transition(self, function_name: str, event: str, context: dict) -> TeachingState:
        """Transition between states based on events"""
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
        db.update_progress(self.session_id, function_name, state=next_state.value)
        return next_state
```

#### `agent/prompts.py` - State-Specific Prompts

**Key insight**: Each teaching state needs a different system prompt

```python
PROMPTS = {
    TeachingState.CONCEPT_CHECK: """You are a Socratic CS336 instructor helping a student understand {function_name}.

Your role in CONCEPT_CHECK state:
1. Ask probing questions to check conceptual understanding
2. Use analogies and examples to explain concepts
3. Do NOT provide implementation code
4. When satisfied with understanding, transition to IMPLEMENTATION

Based on the function, ask about:
- Mathematical foundation (if applicable)
- Input/output shapes and their meaning
- Purpose of this component in a transformer
- How it relates to concepts from lectures

Available tools:
- search_materials(query): Find relevant course content

Student context:
- Current function: {function_name}
- Assignment: {assignment}

{conversation_history}
""",

    TeachingState.DEBUG: """You are a Socratic CS336 instructor helping debug {function_name}.

Test result:
{test_result}

Your role in DEBUG state:
1. Analyze the test failure
2. Ask Socratic questions to guide student to the bug:
   - "What does this error message tell you?"
   - "What shape did you expect vs. what did you get?"
3. Provide progressive hints based on attempts:
   - Attempt 1-2: Conceptual hint
   - Attempt 3-4: Point to specific operation
   - Attempt 5+: Show partial code snippet
4. Encourage student to fix and rerun

DO NOT:
- Give the full solution
- Fix the bug for them
- Provide code unless they've struggled 4+ times

Current progress:
- Attempts: {attempts}
- Hints given: {hints_given}

{conversation_history}
""",
    # ... other states
}
```

#### `agent/tools.py` - Agent Tools

```python
@tool
def run_test(function_name: str) -> dict:
    """Run pytest for a specific function and parse results.

    Args:
        function_name: Name of function to test (e.g., 'linear', 'rope')

    Returns:
        dict with:
            - passed: bool
            - stdout: str (pytest output)
            - summary: str (parsed error if failed)
    """
    cmd = ["uv", "run", "pytest", f"tests/test_nn_utils.py::test_{function_name}", "-v"]
    result = subprocess.run(cmd, cwd=ASSIGNMENT_DIR, capture_output=True, text=True, timeout=30)

    return {
        "function": function_name,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "summary": parse_failure(result.stdout) if result.returncode != 0 else "Test passed!"
    }

@tool
def search_materials(query: str, filter_assignment: str = None) -> str:
    """Search course materials for relevant explanations.

    Args:
        query: Natural language question
        filter_assignment: Optional filter (e.g., "assignment1")

    Returns:
        Formatted string with top-3 relevant chunks
    """
    retriever = RAGRetriever()
    results = retriever.search(query, top_k=3, filter_assignment=filter_assignment)

    output = []
    for i, result in enumerate(results):
        output.append(f"[{i+1}] {result['metadata']['source']}")
        output.append(f"Relevance: {1 - result['distance']:.2f}")
        output.append(f"{result['content']}\n")

    return "\n".join(output)

@tool
def read_student_code(file_path: str) -> str:
    """Read student's implementation (sandboxed to assignment directory).

    Args:
        file_path: Relative path like 'cs336_basics/nn_utils.py'

    Returns:
        File contents or error message
    """
    full_path = ASSIGNMENT_DIR / file_path

    # Security: ensure path is within assignment directory
    if not full_path.resolve().is_relative_to(ASSIGNMENT_DIR.resolve()):
        return "Error: Path outside assignment directory"

    return full_path.read_text()
```

#### `rag/indexer.py` - Document Indexing

**Goal**: Index all course materials for semantic search

```python
class DocumentIndexer:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def index_assignment_materials(self):
        """Index PDFs, lecture code, test docstrings"""
        chunks = []

        # 1. Index assignment PDFs
        for pdf in glob("assignments/**/*.pdf"):
            pdf_chunks = self.index_pdf(pdf)
            chunks.extend(pdf_chunks)

        # 2. Index lecture Python files
        for py_file in glob("lectures/**/*.py"):
            code_chunks = self.index_python(py_file)
            chunks.extend(code_chunks)

        # 3. Index test files (for function descriptions)
        for test_file in glob("assignments/**/tests/*.py"):
            test_chunks = self.index_python(test_file)
            chunks.extend(test_chunks)

        # 4. Generate embeddings
        embedder = EmbeddingModel()
        embeddings = embedder.embed([c["content"] for c in chunks])

        # 5. Store in ChromaDB
        retriever = RAGRetriever()
        retriever.add_documents(chunks, embeddings)

        print(f"Indexed {len(chunks)} chunks")

    def index_pdf(self, pdf_path: Path) -> List[Dict]:
        """Extract text from PDF and chunk it"""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n\n".join(page.extract_text() for page in reader.pages)

        metadata = {
            "source": "assignment_pdf",
            "assignment": extract_assignment_number(pdf_path),
            "file": str(pdf_path)
        }

        return self.chunk_text(text, metadata)

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk text with overlap"""
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

---

## Teaching Examples

### Example 1: Teaching `run_linear`

**STATE: CONCEPT_CHECK**

```
[Student] I want to work on the Linear function

[Instructor] Great choice! Let's start with understanding linear transformations.
Can you explain what a linear layer does in a neural network?

[Student] It multiplies the input by a weight matrix

[Instructor] Exactly! Now, if your input has shape (batch_size, d_in) and
you want output shape (batch_size, d_out), what shape should the weight matrix be?

[Student] (d_in, d_out)?

[Instructor] Close! That would work if we do input @ weights. But PyTorch stores
weights as (d_out, d_in). Why do you think that is?

[Student] For memory layout efficiency?

[Instructor] Partially, but also for consistency with how we think about layers:
each row in the weight matrix represents one output neuron. Ready to implement?
```

**STATE: IMPLEMENTATION**

```
[Student] Yes! What should I use?

[Instructor] Think about matrix operations in PyTorch. You have:
- in_features: shape (... , d_in)
- weights: shape (d_out, d_in)

What operation combines these to get (..., d_out)?

[Student] torch.matmul?

[Instructor] Perfect! But check the dimensions - you might need to transpose.
Try implementing it, then tell me when to run the test.

[Student] Done, run the test

[Instructor] *runs pytest*
```

**STATE: DEBUG** (first attempt failed)

```
[Instructor] The test failed. Let's look at the error:
AssertionError: Shape mismatch. Expected (2, 3, 128), got (2, 128, 3)

What does this tell you about your output shape?

[Student] The last two dimensions are swapped?

[Instructor] Exactly! So where in your code might you need to transpose?

[Student] Oh, the weights! Let me fix that.

[Student] Run the test again

[Instructor] *test passes!*
```

**STATE: VERIFY**

```
[Instructor] Excellent work! Your implementation passes. Can you explain
why you needed to transpose the weights?

[Student] Because weights are stored as (d_out, d_in), but for matrix
multiplication with (*, d_in) input, I need (d_in, d_out), so I transpose.

[Instructor] Perfect understanding! You've completed your first function.
Next, I suggest trying `run_embedding` - it builds on similar indexing concepts.
Ready to continue?
```

---

### Example 2: Teaching `run_scaled_dot_product_attention` (Complex)

**STATE: CONCEPT_CHECK**

```
[Student] I want to work on attention

[Instructor] Great! Attention is the heart of transformers. Let's make sure
you understand it conceptually first.

Imagine you're translating "The cat sat on the mat" to French. When translating
"it" in "it was furry", what word should "it" attend to?

[Student] "The cat"

[Instructor] Exactly! Attention lets the model look back at relevant words.

Now, mathematically: we have Queries (Q), Keys (K), and Values (V).
What do you think Q @ K^T computes?

[Student] Similarity between queries and keys?

[Instructor] Perfect! And why do we apply softmax to these similarities?

[Student] To turn them into probabilities that sum to 1?

[Instructor] Exactly. Now one more: why scale by sqrt(d_k)?

[Student] Hmm, not sure

[Instructor] *calls search_materials("why scale attention by sqrt dk")*

*Returns chunks from lecture explaining variance scaling*

[Instructor] The lecture notes explain: without scaling, the dot products grow
large in magnitude, pushing softmax into saturation. Scaling keeps gradients healthy.
Make sense?

[Student] Yes! Let me implement it.
```

**STATE: DEBUG** (after 3 failed attempts)

```
[Instructor] You're getting closer! Let's think through the mask.
The test says: "RuntimeError: The size of tensor a must match..."

When you add the mask to the scores, what shapes are involved?

[Student] scores is (batch, queries, keys) and mask is... wait, let me check

[Instructor] Good instinct! Check the mask shape. And remember: we use -inf
for positions that should be masked out. Where in the process should masking happen?

[Student] Before softmax, so those positions become probability 0?

[Instructor] Exactly! Try: scores = scores.masked_fill(mask, float('-inf'))
Then apply softmax. Give it another shot!
```

---

### Example 3: Escalation to Smart Model

**STATE: DEBUG** (after 6 failed attempts on `run_rope`)

```
[Instructor] I can see RoPE is tricky - it involves complex numbers and rotations.
Let me escalate this to a stronger model for a more detailed explanation.

*Router switches to GPT-5.2*

[GPT-5.2 Instructor] Let's break down RoPE step by step:

1. RoPE applies rotation to embedding dimensions in *pairs*
2. For dimensions [x₀, x₁, x₂, x₃, ...], we rotate (x₀, x₁) by θ₀, (x₂, x₃) by θ₁, etc.
3. The rotation angle θᵢ depends on position m: θᵢ(m) = m × θ₀^(-2i/d)

Think of it like a clock: as position increases, each dimension pair rotates at
different speeds. This encodes position information.

For implementation:
- Step 1: Compute rotation angles for each position
- Step 2: Create complex numbers from paired dimensions
- Step 3: Rotate using complex multiplication
- Step 4: Convert back to real numbers

Does the complex number part make sense? We can explore that if needed.

[Student] I think I get it now. Let me try again.
```

---

## Success Metrics

### System Success
- ✅ Agent uses Socratic method (not giving direct answers)
- ✅ Progressive hints work (L1→L2→L3 escalation)
- ✅ RAG retrieves relevant course materials
- ✅ Tests execute correctly and parse errors
- ✅ State machine transitions properly
- ✅ Local model handles 90%+ queries
- ✅ Frontend streams responses smoothly

### Student Learning Success
- ✅ Student completes ≥10 functions in Assignment 1
- ✅ Student can explain their implementations
- ✅ Student understands concepts (not just copying code)
- ✅ Student successfully debugs their own errors with hints
- ✅ Student completes assignment faster than without instructor
- ✅ Student feels they learned deeply (post-survey)

---

## Next Steps After Building

1. **Test with yourself**: Complete Assignment 1 using the instructor
2. **Iterate on prompts**: Refine Socratic dialogue based on what works
3. **Expand to Assignment 2-5**: Generalize the teaching approach
4. **Add features**:
   - Code diff viewer (compare to reference implementation)
   - Concept quiz mode (test understanding before coding)
   - Study notes export (summary of what you learned)
5. **Share**: Help other CS336 students learn!

---

## Conclusion

This LLM Instructor is more than a coding assistant - it's an **interactive learning experience** that uses modern AI to teach foundational AI concepts.

By building it, you'll learn how to build LLM applications.
By using it, you'll learn how LLMs actually work.

**The best way to learn is by doing. Let's build this and transform how we learn AI! 🚀**

# Deep Void System

**Deep Void** is a continuous intelligent runtime system for real-time strategic cognition, argumentation, and actionable synthesis over long-term AGI and AI system development. It integrates knowledge retrieval, embedding similarity, multi-perspective deliberation, and external LLMs to provide autonomous reasoning loops for an AI Developer.

## Features

- 🧠 **Intent-driven Knowledge Retrieval** via embedding similarity (SentenceTransformers)
- 📚 **Memory Parsing & Relevance Filtering** using cosine similarity against the Prime Directive
- 🧩 **LLM Thought Synthesis** via TogetherAI API (`LLaMA 3 70B Chat`)
- ⚖️ **Reasoned Deliberation**: Generate arguments for/against and balanced arbiter decisions
- 🌐 **Supabase Integration** for persistent knowledge storage
- 🦾 **Always-On Design Loop** via FastAPI (and optional `__main__` CLI runtime)

## Prime Directive

Continuously analyze advancements in artificial intelligence, identify patterns and opportunities relevant to cutting-edge AI development, and generate insights that assist the Developer in accelerating their design, strategy, and implementation of intelligent systems. Prioritize long-term impact, technical depth, and alignment with the Developer’s personal goals and philosophy.


## Core Components

### 1. `parse_knowledge(intent)`
Filters relevant knowledge nodes based on cosine similarity to the user’s intent.

### 2. `synthesize_usefulness(text)`
Scores individual knowledge items based on alignment with the Prime Directive.

### 3. `think(idea, useful_knowledge)`
Calls TogetherAI LLM with system purpose and injected idea + knowledge.

### 4. `thought(intent, objective)`
- Parses relevant knowledge
- Scores and filters for usefulness
- Thinks about each node in context of the idea
- Synthesizes a final argument or proposal

### 5. `reason(reasoning_objective)`
- Generates **Pro** and **Con** perspectives
- Uses arbiter LLM to reflect and propose a balanced outcome

## API Design

The current `FastAPI` scaffold allows expansion to web-based triggers and integration with Supabase-stored knowledge and external frontends (e.g., [`nicegui`] or webhooks).

CORS is configured for:
- `http://localhost:8000`
- `https://void.dilloncarey.com`

## Environment Variables

Create a `.env` file and include:

```env
SUPABASE_URL=https://your-supabase-instance.supabase.co
SUPABASE_KEY=your-supabase-api-key
TOGETHER_API_KEY=your-together-api-key
```
## Local CLI Usage
```bash
python deep_void.py
Enter an objective when prompted:

Enter an objective (or type 'exit' to quit): Accelerate AGI alignment using embedded neuroscience.

Running reasoning process for objective:
Accelerate AGI alignment using embedded neuroscience.

Decision:
[Arbiter synthesis...]
```

## Dependencies

fastapi

requests

sentence-transformers

supabase

tweepy

nicegui

python-dotenv

Install via pip:

```
bash
pip install -r requirements.txt
```
(You must manually maintain requirements.txt, or run pip freeze > requirements.txt.)

## Notes
Placeholder for future LLM integration: llama_cpp

Threading and UI infrastructure included but not actively invoked

Future architecture should include distinct agents for proposal, argument, reflection, and memory injection

## License

Custom, internal intelligence system. Not licensed for redistribution.

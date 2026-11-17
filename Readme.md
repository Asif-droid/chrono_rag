# Chrono RAG: Requirement Change Tracker

Clients send business requirements over time, and later messages often change earlier ones ("the dashboard should be red" → "make it blue"). This project reads those messages in date order, links each requirement to a feature, and keeps a **timeline of changes** in Neo4j so you can ask what is true now, what was true in a given period, and how it changed.

It is a temporal RAG system built on a **bi-level graph**: a knowledge graph of facts plus a time-hierarchy graph, joined by cross-layer edges.

## How it works

For every message, in chronological order:

1. **Extract**: Gemini pulls structured claims from the message (feature, attribute, value, effective date).
2. **Resolve**: the feature mention is matched to a canonical feature through its name and aliases.
3. **Compare**: Gemini classifies the claim against the feature's active requirements as `NEW`, `UPDATE`, `CONFIRM`, `CANCEL`, `REVERT` or `CONFLICT`.
4. **Apply**: plain Python applies the decision, including the **authority rule** (a lower-authority stakeholder cannot override a higher one, so the claim goes to `pending_review`). The LLM never changes state directly.
5. **Store**: the requirement, its links, its validity window and its embedding are written to Neo4j.

After ingestion, the cross-layer links are built, and retrieval combines semantic search with time scoping.

## Graph model

```
LAYER 1: facts
(Stakeholder)-[:SENT]->(Message)-[:STATED]->(Requirement)-[:ABOUT]->(Feature)
(Requirement)-[:OBJECT]->(Value)
(Requirement)-[:SUPERSEDES | CONFIRMS | CANCELS | CONFLICTS_WITH]->(Requirement)

LAYER 2: time
(Year)-[:HAS_QUARTER]->(Quarter)-[:HAS_MONTH]->(Month)-[:HAS_DAY]->(Day)

CROSS-LAYER
(Month)-[:ACTIVE_AT]->(Requirement)     requirement was in force that month
(Message)-[:ON]->(Day)                  when a message was sent
```

**Requirement statuses:** `active`, `superseded`, `cancelled`, `pending_review`, `duplicate`, `cancel_record`.

Each requirement has `valid_from` and `valid_to` (`null` means still in force), plus an `embedding` stored in a vector index (`req_emb`). Features have one too (`feature_emb`).

## Project structure

```
.
├── chrono_rag.py          # ingestion, graph building, retrieval
├── Dockerfile
├── docker-compose.yml     # Neo4j + the app
├── requirements.txt
├── .env                   # your secrets (copy from .env.example)
└── data/
    ├── stakeholders.json
    ├── features.json
    └── messages.json
```

## Quick start (Docker)

1. Copy the environment file and fill it in:
   ```bash
   cp .env.example .env
   ```
   ```
   NEO4J_PASSWORD=choose_a_password     # 8+ characters
   GEMINI_API_KEY=your_gemini_key
   LLM_MODEL=gemini-2.5-flash
   ```
2. Put your three JSON files in `./data` (formats below).
3. Start everything:
   ```bash
   docker compose up --build
   ```
   The app waits for Neo4j to be healthy, processes the messages, prints the results, and exits. Neo4j keeps running.
4. Explore the graph in Neo4j Browser at http://localhost:7474 (user `neo4j`, your password).

**Re-run after changing messages:** `docker compose run --rm app`. Messages already processed are skipped.

**Reset everything:** `docker compose down && rm -rf ./neo4j_data`

### Without Docker

```bash
pip install neo4j google-genai
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=...
export GEMINI_API_KEY=...
python chrono_rag.py
```
Requires Neo4j 5.18 or newer (for vector indexes).

## Input data formats

**stakeholders.json**
```json
{"stakeholders": [
  {"stakeholder_id": "S001", "name": "Client Lead", "role": "client_lead", "authority_level": 4}
]}
```
A higher `authority_level` wins when two stakeholders conflict.

**features.json**
```json
{"features": [
  {"feature_id": "F001", "canonical_name": "Dashboard", "aliases": ["Home", "Main Screen", "Overview"]}
]}
```

**messages.json**
```json
{"messages": [
  {"message_id": "MSG001", "date": "2026-01-05", "source_type": "email",
   "stakeholder_id": "S001", "raw_text": "The dashboard should use a red primary color."}
]}
```
Dates must be `YYYY-MM-DD`. The files must be valid JSON (no trailing commas).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Set to `bolt://neo4j:7687` inside Docker (compose does this) |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | `password` | Must match the password Neo4j was first started with |
| `GEMINI_API_KEY` | none | Required |
| `LLM_MODEL` | `gemini-2.5-flash` | Model used for extraction, classification and answers |
| `EMBED_MODEL` | `gemini-embedding-001` | Embedding model |
| `EMBED_DIM` | `768` | Embedding size; the vector index is created with this value |
| `DATA_DIR` | `.` | Folder holding the three JSON files (`/data` in Docker) |

## Querying

The functions in `chrono_rag.py`:

| Function | What it does |
|---|---|
| `current_state(feature_id, as_of=None)` | What is in force now, or as of a date |
| `timeline(feature_id, attribute=None)` | Full history, oldest first, with what each entry replaced |
| `answer_question(question, feature_id, feature_name)` | LLM answer from a feature's state and timeline |
| `search_requirements(question, period=None)` | Semantic search, optionally limited to a time bucket |
| `ask(question, period=None)` | Semantic search, then an LLM answer from the hits |

`period` can be `"2026"`, `"2026-Q1"` or `"2026-03"`.

```python
print(ask("What did the client want for the dashboard appearance?", period="2026-Q1"))
```

Useful Cypher for Neo4j Browser:

```cypher
// Change chain for every requirement
MATCH p=(:Requirement)-[:SUPERSEDES*]->(:Requirement) RETURN p

// Both layers together
MATCH p=(:Quarter)-[:HAS_MONTH]->(:Month)-[:ACTIVE_AT]->(r:Requirement)-[:ABOUT]->(:Feature)
RETURN p

// Requirements waiting for human review
MATCH (r:Requirement {status:'pending_review'})-[:ABOUT]->(f:Feature)
RETURN f.name, r.attribute, r.value, r.text
```

## Known limitations

- **Month-level time links:** if a value changes mid-month, both values appear as active in that month's `ACTIVE_AT` links. Use `valid_from` and `valid_to` for exact dates.
- **Unresolved features:** a claim whose feature can't be matched by name or alias is skipped with a warning. It does not go to a review queue yet.
- **Implicit references:** "change it to blue" works only through a simple "last feature this sender mentioned" memory.
- **Authority:** the check compares authority levels only. It doesn't consider who owns a feature.
- **LLM output is not validated:** a malformed `effective_date` or a missing value in a claim can raise an error during ingestion.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Neo4j log: "client is unauthorized" | Neo4j keeps the password from its first start. Wipe `./neo4j_data` or use the original password. |
| App can't connect to Neo4j | `NEO4J_URI` must be `bolt://neo4j:7687` in Docker, not `localhost`. |
| `JSON mode is not enabled for api version v1` | The code pins the Gemini client to `v1beta`. Check that `GOOGLE_GENAI_API_VERSION` isn't set to `v1` in your environment. |
| Old data is skipped after code changes | Messages are marked `processed`. Reset the graph (see above) to re-ingest. |
| Vector index errors | Needs Neo4j 5.18+. If you change `EMBED_DIM`, drop the `req_emb` and `feature_emb` indexes and re-ingest. |
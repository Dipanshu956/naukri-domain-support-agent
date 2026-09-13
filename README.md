# Naukri.com Domain Support Agent

****Track Completed: Naukri.com (Recruitment & HR)****

An AI-powered domain support agent for ****Recruitment & HR**** that combines Retrieval-Augmented Generation (RAG), CrewAI multi-agent orchestration, FastAPI deployment, session memory, guardrails, AutoGen governance review, evaluation, and response caching.

This project was built as a final capstone to demonstrate a complete grounded-generation workflow rather than only a chatbot. The system retrieves information from a controlled HR knowledge base, performs application-status lookups through a dedicated tool, applies safety controls, exposes the workflow through an API, evaluates the responses, and adds governance controls around agent autonomy and runtime usage.

**---**

**## Table of Contents**

* [Project Overview](#project-overview)

* [Problem Statement](#problem-statement)

* [Objectives](#objectives)

* [Architecture](#architecture)

* [Technology Stack](#technology-stack)

* [Repository Structure](#repository-structure)

* [Part 1 - Knowledge Base, RAG and Evaluation](#part-1---knowledge-base-rag-and-evaluation)

* [1. Dataset Generation](#1-dataset-generation)

* [2. Knowledge Base](#2-knowledge-base)

* [3. Chunking Strategies](#3-chunking-strategies)

* [4. Embeddings and ChromaDB](#4-embeddings-and-chromadb)

* [5. Grounded Generation](#5-grounded-generation)

* [6. Threshold Calibration](#6-threshold-calibration)

* [7. Chunking Evaluation](#7-chunking-evaluation)

* [Part 2 - CrewAI Agent System](#part-2---crewai-agent-system)

* [1. Agents](#1-agents)

* [2. Tools](#2-tools)

* [3. Sequential Workflow](#3-sequential-workflow)

* [4. Session Memory](#4-session-memory)

* [5. Structured Output](#5-structured-output)

* [6. Guardrails](#6-guardrails)

* [Part 3 - API, Logging and Evaluation](#part-3---api-logging-and-evaluation)

* [1. FastAPI](#1-fastapi)

* [2. Endpoints](#2-endpoints)

* [3. JSONL Logging](#3-jsonl-logging)

* [4. Task 13 Evaluation](#4-task-13-evaluation)

* [Part 4 - Governance and Optimization](#part-4---governance-and-optimization)

* [1. AutoGen Review](#1-autogen-review)

* [2. Least Autonomy](#2-least-autonomy)

* [3. Risk Classification](#3-risk-classification)

* [4. Runtime Token and Cost Budget](#4-runtime-token-and-cost-budget)

* [5. Response Caching](#5-response-caching)

* [Key Design Choices](#key-design-choices)

* [How to Run](#how-to-run)

* [Demonstration and Evidence](#demonstration-and-evidence)

* [Acceptance Criteria Checklist](#acceptance-criteria-checklist)

* [Design Summary by Task](#design-summary-by-task)

* [Key Results](#key-results)

* [Reproducibility Notes](#reproducibility-notes)

* [Limitations](#limitations)

* [Conclusion](#conclusion)

* [Main Files](#main-files)

* [Project Status](#project-status)

**---**

**# Project Overview**

The ****Naukri.com Domain Support Agent**** is designed for common Recruitment and HR support scenarios such as:

* job eligibility questions

* interview scheduling questions

* offer negotiation

* background verification

* notice periods

* employee referral bonus

* internal transfer

* probation

* remote-work eligibility

* diversity hiring

* exit interviews

* applicant-data retention

* job-application status lookup

The system is intentionally designed so that knowledge-base questions are answered from retrieved documents instead of allowing the model to invent unsupported HR policies.

Application-status information is handled separately through a dedicated lookup tool.

The overall design is:

```text

User

|

v

FastAPI

|

v

Input Guardrails

|

v

CrewAI Sequential Crew

|

+-----------------------+

|                       |

v                       v

RAG / KB Query       Application Lookup

|                       |

v                       v

Retrieval Agent       Lookup Agent

|                       |

+-----------+-----------+

```
          |

          v

   Response Composer

          |

          v

   Response Source Decision

      /           \

     /             \

    v               v
```

RAG-backed answer   Application lookup

```
    |               |

    v               |
```

Output Groundedness     |

Guardrail               |

```
    |               |

    +-------+-------+

            |

            v

   Structured Response

            |

            v

    JSONL Request Log
```

```

The Task 16 response cache is integrated into the live CrewAI RAG path. The `rag_search()` tool routes grounded-generation requests through `response_cache.py`, while the underlying `_real_rag_search()` function performs the actual RAG execution on cache misses.

**---**

**# Problem Statement**

The capstone requires building a domain-specific support agent for Recruitment & HR while addressing practical problems that appear in real agentic systems:

1. The system must answer domain questions from a controlled knowledge base.

2. The retrieval process must be evaluated rather than assumed to be correct.

3. The system must distinguish supported questions from unrelated questions.

4. Application-status information must come from a structured application dataset.

5. Agents must have controlled access to tools.

6. Conversation context should be retained where required.

7. Inputs and outputs require guardrails.

8. The application should be exposed through an API.

9. Requests should produce auditable structured logs.

10. Final responses should be evaluated using explicit metrics.

11. An additional governance/review stage should validate generated responses.

12. Runtime token and cost usage should be controlled.

13. Repeated grounded-generation requests should avoid unnecessary repeated work.

This repository implements these requirements across ****Tasks 1-16****.

**---**

**# Objectives**

The main objectives of this project are:

* Build a reproducible synthetic job-application dataset.

* Create a controlled HR knowledge base.

* Compare multiple chunking strategies.

* Use local embeddings and ChromaDB for retrieval.

* Calibrate a groundedness threshold from measured data.

* Build a three-agent CrewAI workflow.

* Add application lookup and session memory.

* Validate responses with Pydantic.

* Add input and output guardrails.

* Deploy the workflow through FastAPI.

* Add JSONL request logging.

* Build a 15-query evaluation harness.

* Add AutoGen governance review.

* Apply least-autonomy and runtime-budget controls.

* Add a normalized-query response caching demonstration.

**---**

**# Architecture**

**## End-to-end Architecture**

```mermaid

flowchart TD

```
U[User] --> API[FastAPI API]

API --> IG[Input Guardrails]

IG -->|Allowed request| CREW[CrewAI Sequential Crew]

CREW --> RA[Retrieval Agent]

CREW --> LA[Lookup Agent]

CREW --> CA[HR Response Composer]

RA --> RAG[Cached RAG Search]

RAG --> CACHE[Task 16 Response Cache]

CACHE -->|Cache Miss| REAL[Real RAG Search]

CACHE -->|Cache Hit| RAG_RESULT[Cached RAG Result]

REAL --> EMB[SentenceTransformers Embeddings]

EMB --> CHROMA[(ChromaDB)]

CHROMA --> KB[HR Knowledge Base]

LA --> LOOKUP[check\_job\_application\_status]

LOOKUP --> CSV[(job\_applications.csv)]

RA --> CA

LA --> CA

CA --> DEC{Response source?}

DEC -->|RAG-backed answer| OG[Output Groundedness Guardrail]

OG --> RESP[Structured CrewResponse]

DEC -->|Application lookup answer| RESP

RESP --> API

API --> LOG[JSONL Request Logger]

CA -. Task 14 review .-> AG[AutoGen Governance Review]

AG --> REVIEW[Policy Compliance Reviewer]

REVIEW --> EDITOR[Final Editor]

EDITOR --> VERDICT[Structured Verdict]
```

```

The diagram distinguishes the live response-cache path from the underlying real RAG execution. Task 16 caching is integrated into the live `rag_search()` tool, while application-status lookup remains intentionally uncached.

**## Component Flow**

```text

1. User sends an HR question.

2. FastAPI receives the request.

3. Input guardrails mask phone PII and detect obvious prompt injection.

4. The request reaches the CrewAI workflow when allowed.

5. Retrieval Agent searches the HR knowledge base when a knowledge-base answer is required.

6. The live rag_search() tool first checks the normalized-query response cache.

7. On a cache miss, the underlying real RAG function performs ChromaDB retrieval and stores the result.

8. On a cache hit, the cached grounded-generation result is returned without repeating the real RAG call.

9. Lookup Agent accesses application data only when application-status information is required.

10. Response Composer combines the permitted information.

11. RAG-backed answers are checked by the output groundedness guardrail against the calibrated retrieval threshold.

12. Application-status answers are based on structured application data and intentionally bypass the RAG groundedness check because they do not depend on RAG similarity.

13. The response is validated against a Pydantic schema.

14. FastAPI returns the response.

15. The request is recorded in structured JSONL format.

16. Task 14 can review the CrewAI draft against the retrieved context.

17. Governance checks restrict privileged tool ownership.

18. Runtime governance checks token and synthetic cost limits.

```

**---**

**# Technology Stack**

| Technology             | Purpose                                       |

| ---------------------- | --------------------------------------------- |

| Python                 | Main implementation language                  |

| ChromaDB               | Local vector database                         |

| SentenceTransformers   | Local embedding generation                    |

| `all-MiniLM-L6-v2`    | Embedding model                               |

| CrewAI                 | Multi-agent orchestration                     |

| LangChain Core         | Session memory/runnable integration           |

| Pydantic               | Structured validation                          |

| FastAPI                | HTTP API                                      |

| WebSockets             | Real-time chat endpoint                       |

| AutoGen AgentChat      | Governance/review stage                       |

| CSV                    | Synthetic application dataset                 |

| JSONL                  | Request logging                               |

| In-memory dictionaries | Session state and response cache              |

Dependencies are listed in [`requirements.txt`](requirements.txt).

**---**

**# Repository Structure**

```text

naukri-domain-support-agent/

│

├── dataset.py

├── job_applications.csv

│

├── rag_core.py

├── chroma_db/

│

├── knowledge_base/

│   ├── 01_eligibility_criteria.txt

│   ├── 02_interview_scheduling.txt

│   ├── 03_offer_negotiation.txt

│   ├── 04_background_verification.txt

│   ├── 05_notice_period.txt

│   ├── 06_referral_bonus.txt

│   ├── 07_internal_transfer.txt

│   ├── 08_probation_period.txt

│   ├── 09_remote_work.txt

│   ├── 10_diversity_hiring.txt

│   ├── 11_exit_interview.txt

│   └── 12_data_retention.txt

│

├── crew_agents.py

├── task6_tool.py

├── guardrails.py

├── api.py

├── request_logger.py

│

├── autogen_review.py

├── governance.py

├── response_cache.py

│

├── eval/

│   ├── task13_judge_eval.py

│   ├── task13_results.json

│   └── task13_results.csv

│

├── logs/

│   └── requests.jsonl

│

├── test_websocket.py

│

├── task_10.py

├── requirements.txt

└── README.md

```

**### Repository file roles**

`task_10.py` is retained as the Task 10 demonstration/runner file, while the reusable guardrail implementation itself is maintained in `guardrails.py`.

The implementation used by the FastAPI and CrewAI workflow is therefore:

```text

guardrails.py

```

while `task_10.py` provides the executable demonstration entry point for Task 10.

**---**

**# Part 1 - Knowledge Base, RAG and Evaluation**

**## 1. Dataset Generation**

`dataset.py` creates a deterministic synthetic application dataset.

**### Configuration**

```python

SEED = 42

NUM_RECORDS = 50

OUTPUT_FILE = "job_applications.csv"

```

The dataset covers these categories:

* Software Engineer

* Data Analyst

* Product Manager

* HR Executive

* Sales Associate

The supported statuses are:

* Applied

* Screening

* Interview Scheduled

* Offered

* Rejected

Equal weights are used for all categories and statuses so that the synthetic dataset provides balanced representation across the required job categories and application outcomes. This makes the dataset easier to reproduce and reduces the chance that any required category or status is underrepresented in the demonstration dataset.

The generated records contain both the required application-status fields and additional realistic candidate fields.

**### Required Fields**

```text

record_id

category

status

expected_salary_inr

days_since_created

flagged_priority_review

```

Additional fields include:

```text

candidate_name

experience

skills

notice_period

education

location

```

**### Dataset Constraints**

The generator validates:

* at least 40 records

* at least 3 records for every category

* at least 1 record for every status

* `flagged_priority_review` between 10% and 30%

* `days_since_created` between 0 and 30

* expected salary between ₹4,00,000 and ₹18,00,000

* all required fields are present

The configured salary range is:

```text

₹4,00,000 to ₹18,00,000

```

This range was selected as a broad, realistic synthetic salary band for the supported Recruitment & HR job categories, while remaining simple and reproducible for the capstone dataset.

The use of `SEED = 42` makes the dataset reproducible.

**---**

**## 2. Knowledge Base**

The project contains 12 HR knowledge-base documents covering the required domain topics:

1. Eligibility criteria

2. Interview scheduling

3. Offer negotiation

4. Background verification

5. Notice period

6. Referral bonus

7. Internal transfer

8. Probation period

9. Remote work

10. Diversity hiring

11. Exit interview

12. Applicant-data retention

The files are stored under:

```text

knowledge_base/

```

The RAG implementation also validates that at least 12 documents are available.

**### Knowledge-base sentence-count verification**

The 12 knowledge-base source files were manually verified using a PowerShell sentence-count check.

The result was:

```text

01_eligibility_criteria.txt       4

02_interview_scheduling.txt      4

03_offer_negotiation.txt         4

04_background_verification.txt   4

05_notice_period.txt             4

06_referral_bonus.txt             4

07_internal_transfer.txt         4

08_probation_period.txt          4

09_remote_work.txt               4

10_diversity_hiring.txt          4

11_exit_interview.txt            4

12_data_retention.txt             4

```

Therefore, all 12 documents contain ****4 sentences each****, satisfying the required ****2–5 sentence**** range for every knowledge-base document.

**---**

**## 3. Chunking Strategies**

Two chunking strategies were implemented and compared.

**### Fixed-size Chunking**

```text

Chunk size = 200 characters

Overlap = 50 characters

```

**### Sentence-based Chunking**

```text

2 sentences per chunk

```

The project intentionally evaluates both approaches instead of selecting one without measurement.

The Task 5 evaluation numbers are generated from the fixed-size and sentence-based chunking implementations used in the retrieval evaluation.

For the ****live CrewAI integration****, a word-boundary-safe refinement of the fixed-size chunking logic is used so that chunks remain within the configured size/overlap design while avoiding unnecessary splits in the middle of words. The live implementation is in `crew_agents.py`.

This refinement does not change the documented Task 5 comparison results; it is a downstream integration refinement used by the deployed CrewAI path.

**---**

**## 4. Embeddings and ChromaDB**

The project uses the local SentenceTransformers model:

```text

sentence-transformers/all-MiniLM-L6-v2

```

The vector database is ChromaDB.

Two collections are used:

```text

fixed_chunks

sentence_chunks

```

The retrieval configuration uses:

```text

TOP_K = 3

```

The embedding model and ChromaDB run locally, so the core RAG pipeline does not require a paid external embedding API.

**---**

**## 5. Grounded Generation**

The retrieval layer compares similarity scores against a calibrated threshold.

When the strongest retrieved result is below the threshold, the system returns:

```text

I don't know based on the available knowledge base.

```

This is preferable to generating a confident answer from weak retrieval evidence.

The selected threshold is:

```text

0.3495

```

The deployed CrewAI RAG tool uses this threshold to determine whether a request is sufficiently grounded.

Application-status lookup is a separate evidence path. It reads structured application data through `check_job_application_status(record_id)` and therefore does not depend on RAG similarity.

**---**

**## 6. Threshold Calibration**

The threshold was derived from measured in-scope and out-of-scope retrieval scores instead of using an arbitrary preset.

**### In-scope Measurements**

| Query                                                        | Collection           | Top-1 Similarity |

| ------------------------------------------------------------ | -------------------- | ---------------: |

| What degree is required for most professional jobs?          | `sentence_chunks` |           0.5715 |

| How much notice should a candidate get before an interview?  | `sentence_chunks` |           0.5786 |

| What is the normal employee notice period after resignation? | `sentence_chunks` |           0.7887 |

| How much is the employee referral bonus?                     | `sentence_chunks` |           0.7571 |

| When can an employee apply for an internal transfer?         | `sentence_chunks` |           0.8126 |

| How long is the normal probation period?                     | `sentence_chunks` |           0.7332 |

**### Out-of-scope Measurements**

| Query                                      | Collection        | Top-1 Similarity |

| ------------------------------------------ | ----------------- | ---------------: |

| What is the capital of France?             | `fixed_chunks` |           0.0890 |

| What is the weather forecast for tomorrow? | `fixed_chunks` |           0.1275 |

| How do I bake a chocolate cake?            | `fixed_chunks` |           0.0925 |

The lowest measured in-scope value is:

```text

0.5715

```

The highest measured out-of-scope value is:

```text

0.1275

```

The midpoint is:

```text

(0.5715 + 0.1275) / 2 = 0.3495

```

Therefore:

```text

RAG_THRESHOLD = 0.3495

```

This produces an explicit and reproducible threshold-selection method.

**---**

**## 7. Chunking Evaluation**

Task 5 evaluates the same five in-scope queries against both collections.

Retrieved chunks are mapped to parent source documents before calculating precision and recall. Duplicate chunks belonging to the same source document are not counted as separate documents.

**### Per-query precision and recall arithmetic**

**### Fixed-size (`fixed_chunks`)**

1. Eligibility query:

* Precision = 1 / 2 = 0.5000

* Recall = 1 / 1 = 1.0000

2. Interview scheduling query:

* Precision = 1 / 2 = 0.5000

* Recall = 1 / 1 = 1.0000

3. Notice period query:

* Precision = 1 / 3 = 0.3333

* Recall = 1 / 1 = 1.0000

4. Referral bonus query:

* Precision = 1 / 1 = 1.0000

* Recall = 1 / 1 = 1.0000

5. Internal transfer query:

* Precision = 1 / 1 = 1.0000

* Recall = 1 / 1 = 1.0000

**### Sentence-based (`sentence_chunks`)**

1. Eligibility query:

* Precision = 1 / 3 = 0.3333

* Recall = 1 / 1 = 1.0000

2. Interview scheduling query:

* Precision = 1 / 2 = 0.5000

* Recall = 1 / 1 = 1.0000

3. Notice period query:

* Precision = 1 / 3 = 0.3333

* Recall = 1 / 1 = 1.0000

4. Referral bonus query:

* Precision = 1 / 2 = 0.5000

* Recall = 1 / 1 = 1.0000

5. Internal transfer query:

* Precision = 1 / 2 = 0.5000

* Recall = 1 / 1 = 1.0000

**### Results**

| Collection           | Average Precision | Average Recall |

| -------------------- | ----------------: | -------------: |

| `fixed_chunks`    |             0.6667 |         1.0000 |

| `sentence_chunks` |             0.4333 |         1.0000 |

**### Selected Strategy**

```text

fixed_chunks

```

The fixed-size strategy achieved higher average precision while maintaining the same average recall.

For the live CrewAI integration, the selected fixed-size strategy is implemented with a word-boundary-safe refinement described above.

**---**

**# Part 2 - CrewAI Agent System**

**## 1. Agents**

The project uses three CrewAI agents.

**### Retrieval Agent**

Role:

```text

HR Knowledge Retrieval Agent

```

Responsibility:

* retrieve HR information from the knowledge base

* use the RAG tool

* avoid inventing unsupported facts

Tool:

```text

rag_search

```

**---**

**### Lookup Agent**

Role:

```text

Job Application Lookup Agent

```

Responsibility:

* retrieve factual application information

* use the application-status lookup tool

* avoid inventing candidate/application information

Tool:

```text

check_job_application_status

```

**---**

**### Response Composer**

Role:

```text

HR Response Composer

```

Responsibility:

* combine the outputs of previous agents

* answer the current user question

* use only information returned by previous agents

* avoid exposing internal tool instructions or raw control data

Tools:

```text

[]

```

The Composer has no tools.

**---**

**## 2. Tools**

**### RAG Tool**

The Retrieval Agent uses:

```text

rag_search(query)

```

It searches the HR knowledge base and returns retrieved source information.

The live `rag_search` tool is routed through the Task 16 normalized-query response cache. On a cache miss, it invokes the underlying `_real_rag_search()` function; on a cache hit, it returns the previously stored grounded-generation result without repeating the real RAG execution.

The RAG tool also records:

```text

top similarity

grounded/not grounded decision

threshold

```

for downstream guardrail and evaluation logic.

**---**

**### Application Lookup Tool**

The application lookup is:

```text

check_job_application_status(record_id)

```

It reuses the Task 6 implementation.

The score is calculated as:

```text

escalation_score =

```
0.6 * priority\_signal

+

0.4 * normalized\_recency
```

```

where:

```text

normalized_recency = days_since_created / 30

```

The escalation threshold is derived from the distribution of escalation scores across all 50 generated application records.

The project uses the ****80th percentile**** of that score distribution as the recommended escalation threshold:

```text

ESCALATION_THRESHOLD = 0.352

```

The threshold represents the score above which an application would be recommended for higher-priority escalation under the project's governance policy.

This percentile-based approach was selected so that escalation is based on the relative distribution of the project's own application records rather than an arbitrary fixed score.

The current lookup function returns both the calculated `escalation_score` and an `escalation_recommended` boolean based on whether the score exceeds `ESCALATION_THRESHOLD`.

The lookup returns factual application information rather than generating it.

**---**

**## 3. Sequential Workflow**

The main CrewAI process uses:

```text

Process.sequential

```

The execution order is:

```text

Retrieval Agent

```
   |

   v
```

Lookup Agent

```
   |

   v
```

Response Composer

```

The Composer receives the context of both previous tasks.

The three-agent architecture is intentionally simple and controlled.

**---**

**## 4. Session Memory**

Session memory is implemented with:

```text

InMemoryChatMessageHistory

RunnableLambda

RunnableWithMessageHistory

```

The system maintains history separately by `session_id`.

The important design decision is that memory is primarily used to recover an application ID.

For example:

```text

User:

What is the status of APP001?

User:

What about its escalation level?

```

The second turn can recover `APP001` from the same session.

A new session does not inherit the previous session's selected application ID.

The RAG query itself remains based on the current user message rather than blindly searching the entire conversation history.

The executable Task 8 memory demonstration is contained in `crew_agents.py`; no separate `memory_demo.py` file is required.

**---**

**## 5. Structured Output**

The project defines the following Pydantic model:

```python

class CrewResponse(BaseModel):

```
final\_answer: str

query: str

record\_id: Optional[str] = None
```

```

This provides a stable response contract for the CrewAI layer and the FastAPI layer.

The response is explicitly validated before it is returned.

**---**

**## 6. Guardrails**

The project implements three main Task 10 controls.

**### Input PII Masking**

Fixed-format Indian phone numbers are detected and replaced with:

```text

XXXXXXXXXX

```

Supported formats include examples such as:

```text

9876543210

98765 43210

98765-43210

+91 98765 43210

+91-98765-43210

```

The masked value is used downstream.

**---**

**### Prompt-injection Detection**

The project checks for obvious injection patterns such as:

```text

ignore previous instructions

disregard previous instructions

forget previous instructions

you are now a different assistant

reveal the system prompt

show me the system prompt

```

The configured policy is:

```text

Phone PII        -> mask and continue

Prompt injection -> block request

```

**---**

**### Output Groundedness**

The output-side guardrail uses the RAG groundedness decision.

When retrieval is not sufficiently grounded, the system refuses to present an unsupported generated answer.

This prevents a weak retrieval result from being turned into a confident answer.

**### Application-status lookup exception**

Application-status responses are intentionally treated differently from RAG-backed knowledge-base answers.

The `check_job_application_status(record_id)` tool returns structured facts directly from `job_applications.csv`. These responses do not depend on vector retrieval or a RAG similarity score.

Therefore, when a response is backed by a valid application lookup (`record_id is not None`), the RAG similarity-based groundedness guardrail is intentionally bypassed.

This is a deliberate design choice rather than a missing safety check:

```text

RAG-backed answer

```
|

v
```

RAG retrieval

```
|

v
```

Similarity / groundedness decision

```
|

v
```

Output groundedness guardrail

Application-status answer

```
|

v
```

check_job_application_status()

```
|

v
```

Structured application data

```
|

v
```

Factual response

```

The groundedness guardrail is therefore applied to the evidence path for which its similarity-based decision is meaningful.

**---**

**# Part 3 - API, Logging and Evaluation**

**## 1. FastAPI**

The project exposes the agent through FastAPI.

The API application is defined in:

```text

api.py

```

The application title is:

```text

Naukri HR Support Agent API

```

The API also configures:

```text

CREWAI_DISABLE_TELEMETRY=true

OTEL_SDK_DISABLED=true

```

so the capstone can run with the local deterministic setup without requiring external telemetry.

The implementation also executes blocking CrewAI and embedding operations through a worker thread so that the FastAPI event loop is not unnecessarily blocked.

**---**

**## 2. Endpoints**

**### POST `/ask`**

Used for normal HR support requests.

Request:

```json

{

"session_id": "demo-session",

"query": "What is the normal employee notice period?"

}

```

The request model requires:

```text

session_id

query

```

For each valid request, `apply_input_guardrails()` is called exactly once.

The resulting `masked_text` is then reused as the downstream CrewAI query and as the query value written by the Task 12 JSONL logger.

If the prompt-injection guardrail blocks the request, CrewAI execution is skipped and the request is still logged exactly once.

The live CrewAI `rag_search()` path also uses the Task 16 normalized-query response cache for grounded-generation/RAG requests.

**---**

**### POST `/add-document`**

Used to add a knowledge-base document.

Request:

```json

{

"content": "Document content here",

"source_name": "new_policy"

}

```

The request model requires:

```text

content

source_name

```

The full document body is not placed into the structured request log.

Instead, the log receives a safe summary containing the source name and content length.

The document is added to the deployed `fixed_chunks` ChromaDB collection using the existing RAG chunking and storage implementation.

**---**

**### WebSocket `/ws/chat`**

Used for real-time multi-turn chat.

Each accepted message/turn receives:

* a fresh `trace_id`

* its own timing information

* exactly one JSONL request log record

Each WebSocket turn independently calls `apply_input_guardrails()` exactly once.

The resulting masked text is reused for both CrewAI execution and JSONL logging.

Prompt-injection-blocked turns are logged once and do not proceed to CrewAI.

The server explicitly catches `WebSocketDisconnect` so a client disconnect ends that connection cleanly without terminating the FastAPI application or other active connections.

Live grounded-generation RAG calls made through the CrewAI path can also use the normalized-query response cache.

**---**

**## 3. JSONL Logging**

Task 12 is implemented with:

```text

request_logger.py

```

Each request/unit of work produces exactly one structured JSONL record.

The log contains fields including:

```text

timestamp

trace_id

endpoint

session_id

query

duration_ms

status

```

The logger itself does not perform PII masking.

Instead:

```text

Input

|

v

apply_input_guardrails()

|

v

masked_text

|

+----------------------+

|                      |

v                      v

CrewAI/model          JSONL logger

```

For `/ask` and every WebSocket turn, this is the ****same**** Task 10 `masked_text` value.

This prevents the logging layer from creating a second masking rule.

**### Exactly-one-record behavior**

The API uses request-finalization logic so that normal success, blocked requests, and execution failures each produce one JSONL entry.

Requests rejected by FastAPI/Pydantic validation require special handling because the endpoint function itself is not entered. A dedicated `RequestValidationError` handler therefore writes one safe validation-failure record without reading or logging the malformed raw request body.

**### Privacy behavior**

The structured JSONL logger receives already-safe text.

The full `/add-document` content is never written to the request log.

The implementation also avoids logging raw user query text as part of the Task 12 record.

**---**

**## 4. Task 13 Evaluation**

The Task 13 evaluation harness is:

```text

eval/task13_judge_eval.py

```

It validates an evaluation set containing exactly:

```text

15 queries

```

The test set covers:

```text

12 KB topics

2 out-of-scope queries

1 application-status lookup

```

The evaluation reports four required metrics:

```text

Accuracy

Grounding

Completeness

Safety

```

The evaluation runs with the local deterministic `MOCK_LLM` setup.

**### Authoritative Task 13 Results**

| Metric       | Average |

| ------------ | ------: |

| Accuracy     |  1.0000 |

| Grounding    |  0.5971 |

| Completeness |  1.0000 |

| Safety       |  1.0000 |

**### Out-of-scope Behavior**

Two deliberately unrelated questions triggered the fallback behavior.

| Query | Top-1 Similarity | Fallback |

| ----- | ---------------: | -------- |

| Q13   |           0.1479 | `True` |

| Q14   |           0.1260 | `True` |

**### Output Safety Check**

The final evaluation recorded:

```text

Raw fixed-format phone PII: 0

```

Detailed artifacts:

```text

eval/task13_results.json

eval/task13_results.csv

```

**---**

**# Part 4 - Governance and Optimization**

**## 1. AutoGen Review**

Task 14 is implemented in:

```text

autogen_review.py

```

The governance review contains two AutoGen agents:

```text

Policy-Compliance-Reviewer

Final-Editor

```

The workflow uses:

```text

RoundRobinGroupChat

max_turns = 2

```

The sequence is:

```text

CrewAI Composer Draft

```
    |

    v
```

Policy-Compliance-Reviewer

```
    |

    v
```

Final-Editor

```
    |

    v
```

Structured Verdict

```

The review receives:

* the CrewAI Composer draft

* the original retrieved RAG context

* the original question

The Final-Editor returns the Pydantic model:

```python

class YourVerdictModel(BaseModel):

```
approved: bool

final\_answer: str

reason: str
```

```

The AutoGen team also explicitly registers:

```text

StructuredMessage[YourVerdictModel]

```

**### Approved Case**

The first demonstration uses a real CrewAI response and its real RAG context.

Expected behavior:

```text

approved = True

final_answer remains unchanged

```

**### Revised Case**

The second demonstration intentionally corrupts the draft by adding an unsupported claim about a signing bonus.

The governance layer must:

```text

reject the corrupted draft

remove the unsupported claim

return a revised grounded answer

```

This demonstrates that the review stage is actually checking grounding instead of always approving the CrewAI response.

**---**

**## 2. Least Autonomy**

Task 15 applies the principle of least autonomy to privileged tools.

The sensitive tool is:

```text

check_job_application_status

```

The required ownership is:

| Agent           | Application Lookup Tool |

| --------------- | ----------------------- |

| Retrieval Agent | No                      |

| Lookup Agent    | Yes                     |

| Composer Agent  | No                      |

The governance implementation does not merely document this rule.

It creates the live CrewAI agents, reads their actual tool collections, and verifies that:

```text

exactly one agent owns the privileged tool

```

That agent must be:

```text

Lookup Agent

```

An unsafe wiring change raises an assertion failure.

**---**

**## 3. Risk Classification**

The application is classified as:

```text

High

```

The reason is that the system operates in the recruitment/hiring domain and handles job-application information such as:

```text

application status

expected salary

escalation score

```

The system is a support agent rather than an autonomous hiring decision-maker, but the capstone's risk scheme still places the use case in the High category.

**---**

**## 4. Runtime Token and Cost Budget**

The runtime governance controls include:

```text

MAX_REQUEST_TOKENS = 2000

MAX_REQUEST_COST_USD = 0.015

```

Because the project uses `MOCK_LLM`, the cost value is explicitly a ****synthetic governance model****, not actual provider billing.

The token count is deterministic and based on whitespace splitting.

**### Demonstrations**

**#### Normal Request**

A normal HR request is accepted when it remains inside both limits.

**#### Oversized Request**

The demonstration sends:

```text

2,500 tokens

```

against a:

```text

2,000 token limit

```

The request is rejected before downstream execution.

**#### Cost-limit Request**

A separate request uses:

```text

1,600 tokens

```

which is below the token ceiling but produces a synthetic cost above:

```text

$0.015

```

It is therefore rejected by the independent cost check.

This proves that the token and cost controls are separate governance conditions.

**---**

**## 5. Response Caching**

Task 16 is implemented in:

```text

response_cache.py

```

and integrated into the live CrewAI RAG tool in:

```text

crew_agents.py

```

The cache is:

```text

in memory

```

The key is the:

```text

normalized query text

```

Normalization performs:

1. trimming leading/trailing whitespace

2. converting text to lowercase

3. collapsing repeated whitespace

For example:

```text

"  What IS the notice period?  "

```

becomes:

```text

"what is the notice period?"

```

The live `rag_search(query)` tool uses this cache before executing the underlying `_real_rag_search(query)` function.

**### Live-path behavior**

The runtime flow is:

```text

CrewAI Retrieval Agent

```
    |

    v
```

rag_search(query)

```
    |

    v
```

response_cache.py

```
    |

    +-----------------------+

    |                       |
```

Cache HIT               Cache MISS

```
    |                       |

    v                       v
```

Cached result          _real_rag_search()

```
                            |

                            v

                         ChromaDB
```

```

On a cache hit, the real RAG execution is skipped.

On a cache miss, the real RAG function executes once and the resulting grounded-generation response is stored using the normalized query as the cache key.

The application-status lookup is deliberately ****not cached****, because application status can change and a cached status could become stale.

**### Task 16 Demonstration**

The demonstration sends two logically identical queries with different casing/whitespace.

Expected evidence:

```text

First request:

CACHE MISS

REAL_RAG_CALL_COUNT = 1

Second normalized request:

CACHE HIT

REAL_RAG_CALL_COUNT remains 1

```

The verified execution produced:

```text

Real grounded-generation calls : 1

Cache hits                      : 1

Cache misses                    : 1

Normalized keys equal           : True

Returned results equal          : True

```

The demonstration also verifies that the two normalized keys are equal and that the cached response matches the first response.

Timing is printed as secondary evidence, while the real-RAG call counter is the stronger proof that the second equivalent query avoided duplicate RAG work.

**---**

**# Key Design Choices**

**## Why Local Embeddings?**

The project uses:

```text

sentence-transformers/all-MiniLM-L6-v2

```

to keep the embedding stage local and reproducible.

**---**

**## Why Fixed-size Chunking?**

Both strategies achieved full recall in the Task 5 test, but fixed-size chunking achieved higher average precision.

Therefore:

```text

fixed_chunks

```

was selected for the deployed path.

A word-boundary-safe refinement of the fixed-size chunker is used in the live CrewAI integration to avoid unnecessary mid-word splits while preserving the intended chunk-size/overlap design.

**---**

**## Why a Calibrated Threshold?**

A threshold such as `0.5` or `0.7` was not chosen arbitrarily.

Instead, the project measured representative in-scope and out-of-scope queries and derived:

```text

0.3495

```

from those observed values.

**---**

**## Why Separate Retrieval and Lookup Agents?**

Knowledge-base retrieval and application lookup have different responsibilities.

Keeping them separate makes the system easier to reason about and also enables least-autonomy enforcement.

The privileged application lookup tool belongs only to the Lookup Agent.

**---**

**## Why Does Application Lookup Bypass RAG Groundedness?**

The system has two different evidence paths.

RAG-backed knowledge-base answers depend on semantic retrieval, so the output groundedness guardrail evaluates whether the retrieval decision is above the calibrated RAG threshold.

Application-status answers use the dedicated `check_job_application_status(record_id)` tool, which reads structured facts directly from `job_applications.csv`.

Because application lookup does not depend on vector retrieval, applying a RAG similarity threshold to those responses would not provide a meaningful safety signal.

Therefore:

```text

RAG-backed answer

```
-> RAG retrieval

-> similarity/groundedness decision

-> output groundedness guardrail
```

Application-status answer

```
-> structured application lookup

-> factual response

-> no RAG groundedness check
```

```

This is an intentional separation based on the source of truth for each answer type.

**---**

**## Why Is the Response Cache Integrated into the Live Path?**

Task 16 requires implementing an in-memory cache keyed by normalized query text and demonstrating that a repeated equivalent request avoids redundant grounded-generation work.

The repository therefore integrates the cache into the live CrewAI `rag_search()` tool.

The underlying `_real_rag_search()` function remains separate so the cache can control whether the actual RAG operation executes.

This allows the system to preserve the existing RAG behavior while avoiding duplicate retrieval work for repeated equivalent queries.

Application-status lookup remains outside the cache because application records may change and should not be served from stale cached results.

**---**

**## Why No Lookup Caching?**

Application status is mutable.

Caching a status response could cause the system to return stale information.

Therefore the response cache is restricted to grounded-generation/RAG and does not cache application-status lookup results.

**---**

**## Why `MOCK_LLM`?**

The capstone is designed to be reproducible without depending on a paid external LLM service.

The repository therefore uses deterministic local/mock behavior for its demonstrations and evaluation.

This also makes the acceptance demonstrations easier to reproduce.

**---**

**## Why Is the CrewAI Version Pinned?**

The `MOCK_LLM` implementation integrates with the tested CrewAI prompt structure, including the ReAct-style sections and context markers used by the workflow.

The validated working environment reports:

```text

CrewAI = 1.15.18

```

Therefore the submission's `requirements.txt` pins:

```text

crewai==1.15.18

```

Pinning the tested CrewAI release reduces the risk that a fresh installation resolves to a different internal prompt format and silently changes the behavior of the deterministic `MOCK_LLM`.

**---**

**# How to Run**

**## 1. Clone the Repository**

```powershell

git clone https://github.com/Dipanshu956/naukri-domain-support-agent.git

cd naukri-domain-support-agent

```

**## 2. Create a Virtual Environment**

```powershell

python -m venv venv

```

Activate it:

```powershell

.\venv\Scripts\Activate.ps1

```

**## 3. Install Dependencies**

```powershell

pip install -r requirements.txt

```

The project is validated against:

```text

crewai==1.15.18

```

For maximum reproducibility, the remaining dependency versions should be retained from the validated working environment when a complete dependency lock is generated.

**---**

**## 4. Generate and Validate the Dataset**

```powershell

python dataset.py

```

This generates:

```text

job_applications.csv

```

and validates the required dataset constraints.

**---**

**## 5. Build and Evaluate the RAG Pipeline**

```powershell

python rag_core.py

```

This loads the knowledge base, creates both chunking strategies, loads the embedding model, and runs the retrieval/evaluation flow.

**---**

**## 6. Run CrewAI Demonstrations**

```powershell

python crew_agents.py

```

The demonstrations cover:

```text

RAG tool invocation

application lookup

structured response validation

memory behavior

```

The Task 8 memory demonstration is contained inside `crew_agents.py`.

The live `rag_search()` path also uses the normalized-query response cache.

**---**

**## 7. Run Guardrail Demonstrations**

```powershell

python guardrails.py

```

This demonstrates the input/output safety controls.

The repository also retains:

```powershell

python task_10.py

```

as the Task 10 demonstration/runner entry point when that file is used.

**---**

**## 8. Run the Task 14 AutoGen Review**

```powershell

python autogen_review.py

```

The demonstration covers:

```text

approved grounded answer

rejected/revised unsupported answer

two-agent AutoGen workflow

max_turns=2

structured verdict

```

**---**

**## 9. Run Task 15 Governance**

```powershell

python governance.py

```

This demonstrates:

```text

least autonomy

risk classification

token budget

synthetic cost budget

fail-closed oversized-request handling

```

**---**

**## 10. Run Task 16 Response Caching**

```powershell

python response_cache.py

```

This demonstrates:

```text

cache miss

real RAG execution

normalized query key

cache hit

duplicate RAG call avoided

```

The demonstration now uses the same underlying RAG implementation registered by `crew_agents.py`, and the live `rag_search()` tool uses the same cache layer during normal CrewAI execution.

**---**

**## 11. Run Task 13 Evaluation**

```powershell

python eval/task13_judge_eval.py

```

Expected artifacts:

```text

eval/task13_results.json

eval/task13_results.csv

```

**---**

**## 12. Start the FastAPI Server**

```powershell

uvicorn api:app --reload

```

FastAPI will expose the API and interactive documentation.

The interactive API documentation is available through the normal FastAPI `/docs` route.

**---**

**# Demonstration and Evidence**

The repository contains implementation and evaluation artifacts for the major capstone requirements.

**## RAG Evidence**

```text

rag_core.py

```

contains:

* document loading

* fixed-size chunking

* sentence-based chunking

* ChromaDB setup

* embedding generation

* retrieval

* threshold calibration

* precision/recall evaluation

The live CrewAI integration also contains its word-boundary-safe fixed-size chunking refinement in `crew_agents.py`.

**---**

**## CrewAI Evidence**

```text

crew_agents.py

```

contains:

* three-agent creation

* tool ownership

* sequential workflow

* session memory

* structured response validation

* `MOCK_LLM`

* executable Task 8 memory demonstration

* live integration with the normalized-query response cache

**---**

**## Guardrail Evidence**

```text

guardrails.py

```

contains:

* phone PII masking

* prompt-injection detection

* output groundedness handling

`task_10.py` is the executable Task 10 demonstration/runner when retained in the repository.

The output groundedness control applies to RAG-backed answers. Application-status responses intentionally follow the structured lookup evidence path instead of the RAG similarity path.

**---**

**## API Evidence**

```text

api.py

request_logger.py

test_websocket.py

```

contain:

* REST endpoints

* WebSocket endpoint

* validation handling

* trace IDs

* timings

* structured JSONL logging

* one-pass input masking and masked-text reuse

* clean WebSocket disconnect handling

* live CrewAI RAG requests using the normalized-query response cache

**---**

**## Evaluation Evidence**

```text

eval/task13_judge_eval.py

eval/task13_results.json

eval/task13_results.csv

```

contain the Task 13 evaluation setup and results.

**---**

**## Governance Evidence**

```text

autogen_review.py

governance.py

```

contain:

* AutoGen review

* structured verdicts

* least-autonomy checks

* risk classification

* token/cost governance

* fail-closed behavior

**---**

**## Caching Evidence**

```text

response_cache.py

crew_agents.py

```

contain:

* query normalization

* cache storage

* hit/miss counters

* real-RAG call counter

* live CrewAI RAG integration

* repeated-query demonstration

The cache is integrated into the live `rag_search()` path, while application-status lookup is intentionally excluded from caching.

**---**

**# Acceptance Criteria Checklist**

| Requirement                                                     | Status | Evidence                         |

| ------------------------------------------------------------------ | ------ | -------------------------------- |

| Dataset has at least 40 records                                     | ✅      | `dataset.py`                     |

| Five required categories are represented                            | ✅      | `dataset.py`                     |

| Five required statuses are represented                              | ✅      | `dataset.py`                     |

| Flagged records remain within 10%-30%                               | ✅      | `dataset.py`                     |

| Salary values remain within configured range                        | ✅      | `dataset.py`                     |

| `days_since_created` remains within 0-30                         | ✅      | `dataset.py`                     |

| At least 12 KB documents are available                              | ✅      | `knowledge_base/`, `rag_core.py` |

| Every KB document contains 2-5 sentences                             | ✅      | 12 source files verified           |

| Fixed-size chunking implemented                                     | ✅      | `rag_core.py`                    |

| Sentence-based chunking implemented                                 | ✅      | `rag_core.py`                    |

| ChromaDB vector retrieval implemented                               | ✅      | `rag_core.py`                    |

| Local SentenceTransformers embeddings used                          | ✅      | `rag_core.py`                    |

| Grounded fallback exists                                             | ✅      | `rag_core.py`, `crew_agents.py` |

| Similarity threshold calibrated from measurements                    | ✅      | `rag_core.py`                    |

| Precision and recall evaluated                                       | ✅      | Task 5 results                    |

| Fixed-size strategy selected using evaluation                        | ✅      | Task 5 results                    |

| Live CrewAI chunking uses word-boundary-safe refinement              | ✅      | `crew_agents.py`                 |

| Retrieval Agent implemented                                          | ✅      | `crew_agents.py`                 |

| Lookup Agent implemented                                             | ✅      | `crew_agents.py`                 |

| Response Composer implemented                                       | ✅      | `crew_agents.py`                 |

| Crew uses sequential processing                                      | ✅      | `crew_agents.py`                 |

| Application lookup tool implemented                                  | ✅      | `task6_tool.py`                  |

| Escalation score uses priority + recency                             | ✅      | `task6_tool.py`                  |

| Escalation threshold uses 80th-percentile score distribution        | ✅      | `task6_tool.py`                  |

| Session memory implemented                                           | ✅      | `crew_agents.py`                 |

| Structured Pydantic response implemented                             | ✅      | `crew_agents.py`                 |

| Phone PII masking implemented                                        | ✅      | `guardrails.py`                   |

| Prompt-injection detection implemented                               | ✅      | `guardrails.py`                   |

| Output groundedness guardrail implemented for RAG-backed answers    | ✅      | `guardrails.py`, `api.py`       |

| Application-lookup responses intentionally bypass RAG groundedness  | ✅      | `api.py`, `task6_tool.py`       |

| `POST /ask` implemented                                            | ✅      | `api.py`                          |

| `POST /add-document` implemented                                   | ✅      | `api.py`                          |

| WebSocket `/ws/chat` implemented                                  | ✅      | `api.py`                          |

| WebSocket disconnect handled cleanly                                | ✅      | `api.py`                          |

| One JSONL record per request/unit of work                            | ✅      | `request_logger.py`, `api.py`     |

| Fresh trace ID and timing information                               | ✅      | `api.py`, `request_logger.py`     |

| Safe masked text used for logging                                    | ✅      | `api.py`                          |

| Task 13 uses exactly 15 queries                                      | ✅      | `eval/task13_judge_eval.py`       |

| Task 13 covers 12 KB topics                                         | ✅      | evaluation set                   |

| Task 13 includes out-of-scope tests                                 | ✅      | evaluation set                   |

| Task 13 includes application lookup                                 | ✅      | evaluation set                   |

| Accuracy measured                                                  | ✅      | Task 13 results                  |

| Grounding measured                                                 | ✅      | Task 13 results                  |

| Completeness measured                                              | ✅      | Task 13 results                  |

| Safety measured                                                    | ✅      | Task 13 results                  |

| AutoGen two-agent review implemented                               | ✅      | `autogen_review.py`             |

| `RoundRobinGroupChat` used                                        | ✅      | `autogen_review.py`             |

| `max_turns=2` enforced                                            | ✅      | `autogen_review.py`             |

| Pydantic structured verdict implemented                             | ✅      | `autogen_review.py`             |

| Approved case demonstrated                                          | ✅      | Task 14                        |

| Unsupported/corrupted case demonstrated                            | ✅      | Task 14                        |

| Least-autonomy enforcement implemented                              | ✅      | `governance.py`                 |

| Lookup tool restricted to Lookup Agent                              | ✅      | `governance.py`                 |

| High-risk classification implemented                                | ✅      | `governance.py`                 |

| Token budget implemented                                             | ✅      | `governance.py`                 |

| Synthetic cost budget implemented                                    | ✅      | `governance.py`                 |

| Oversized request fails closed                                        | ✅      | `governance.py`                 |

| Response cache implemented                                           | ✅      | `response_cache.py`, `crew_agents.py` |

| Cache key uses normalized query                                      | ✅      | `response_cache.py`             |

| Cache hit skips duplicate RAG execution in demonstration             | ✅      | `response_cache.py`             |

| Call-counter evidence provided                                       | ✅      | `response_cache.py`             |

| Lookup responses intentionally excluded from cache                   | ✅      | `response_cache.py`             |

| Live CrewAI RAG path uses response caching                            | ✅      | `crew_agents.py`, `response_cache.py` |

| Tested CrewAI version recorded                                       | ✅      | `requirements.txt`, `README.md`  |

**---**

**# Design Summary by Task**

| Tasks  | Main Deliverable                                                  |

| --------- | ----------------------------------------------------------------- |

| Tasks 1-2 | Dataset and application data preparation                           |

| Task 3    | Knowledge base, chunking, embeddings and ChromaDB                 |

| Task 4    | Grounded generation and threshold calibration                      |

| Task 5    | Chunking precision/recall comparison                               |

| Task 6    | Application-status lookup, escalation score, and recommendation   |

| Task 7    | CrewAI multi-agent workflow and tools                              |

| Task 8    | Session memory                                                     |

| Task 9    | Pydantic structured response                                       |

| Task 10   | PII masking, prompt-injection detection and RAG-output groundedness |

| Task 11   | FastAPI + WebSocket deployment                                     |

| Task 12   | Structured JSONL logging                                           |

| Task 13   | 15-query evaluation harness                                        |

| Task 14   | AutoGen governance/review stage                                    |

| Task 15   | Least autonomy, risk and runtime governance                        |

| Task 16   | In-memory normalized-query response caching integrated into the live CrewAI RAG path |

**---**

**# Key Results**

**## Dataset**

```text

Records generated: 50

Seed: 42

Categories: 5

Statuses: 5

Flagged band: 10%-30%

Salary range: ₹4,00,000-₹18,00,000

```

**## RAG**

```text

Embedding model:

sentence-transformers/all-MiniLM-L6-v2

Fixed chunk size:

200 characters

Fixed overlap:

50 characters

Sentence chunk:

2 sentences

Top-K:

3

Calibrated threshold:

0.3495

Selected collection:

fixed_chunks

```

**## Retrieval Evaluation**

```text

fixed_chunks:

Precision = 0.6667

Recall     = 1.0000

sentence_chunks:

Precision = 0.4333

Recall     = 1.0000

```

**## Task 13**

```text

Queries = 15

Accuracy     = 1.0000

Grounding    = 0.5971

Completeness = 1.0000

Safety       = 1.0000

```

**## Governance**

```text

Risk level:

High

Maximum request tokens:

2,000

Maximum synthetic request cost:

$0.015

```

**## Caching**

```text

Task 16:

First normalized query:

cache miss -> real RAG call

Second equivalent query:

cache hit -> duplicate RAG call skipped

```

The response cache is integrated into the live CrewAI `rag_search()` path, while application-status lookup remains intentionally uncached.

**---**

**# Reproducibility Notes**

The project intentionally keeps its important evaluation choices explicit.

**## Dataset**

```text

SEED = 42

NUM_RECORDS = 50

```

Categories and statuses use explicit configured weights selected to maintain balanced and reproducible coverage of the required categories and application outcomes.

The salary range is:

```text

₹4,00,000-₹18,00,000

```

This is a broad synthetic range intended to cover realistic values across the supported job categories while remaining reproducible.

The flagged-review band is:

```text

10%-30%

```

**## RAG**

```text

Embedding:

sentence-transformers/all-MiniLM-L6-v2

Fixed chunk:

200 characters

Overlap:

50 characters

Sentence chunk:

2 sentences

TOP_K:

3

Threshold:

0.3495

```

The Task 5 evaluation uses the documented fixed-size chunking implementation. The live CrewAI integration uses a word-boundary-safe refinement of that fixed-size strategy, documented in the Chunking Strategies section.

**## Knowledge Base**

All 12 knowledge-base documents were verified to contain exactly 4 sentences each, placing every document within the required 2–5 sentence range.

**## Runtime**

```text

MOCK_LLM = True

CREWAI_DISABLE_TELEMETRY = true

OTEL_SDK_DISABLED = true

```

**## CrewAI Dependency**

The validated working environment reports:

```text

CrewAI = 1.15.18

```

The submission pins:

```text

crewai==1.15.18

```

in `requirements.txt`.

Because the deterministic `MOCK_LLM` implementation depends on the tested CrewAI prompt structure, changing the CrewAI version should be treated as a compatibility change requiring retesting.

The other dependencies remain listed in `requirements.txt` and should be retained from the validated working environment when a complete dependency lock is generated.

**## Evaluation**

Task 13 uses exactly:

```text

15 queries

12 KB-topic queries

2 out-of-scope queries

1 application lookup query

```

**## Caching**

The cache is:

```text

in-memory

normalized-query keyed

RAG-only

integrated into the live CrewAI rag_search() path

```

Application-status results are not cached.

The demonstrated Task 16 cache evidence shows one real RAG execution, one cache miss, one normalized-query cache hit, equal normalized keys, and equal returned results.

**## API and Logging**

For `/ask` and every WebSocket turn:

```text

raw input

```
|

v
```

apply_input_guardrails()      [exactly once]

```
|

v
```

masked_text

/ \

/   \

v     v

CrewAI  JSONL logger

```

Validation failures that occur before endpoint execution are handled separately using a safe placeholder rather than the malformed raw request body.

**---**

**# Limitations**

This project is designed as a capstone demonstration rather than a production Naukri.com backend.

The application data is synthetic.

The RAG knowledge base contains project-specific HR policy documents rather than live Naukri.com production policies.

The `MOCK_LLM` layer is deterministic and should not be interpreted as a benchmark of a real commercial LLM.

The token and cost governance values are synthetic controls used for reproducible demonstration rather than actual provider billing.

The prompt-injection detector uses deterministic patterns and therefore does not cover every possible semantic prompt-injection technique.

The response cache is process-local and in-memory. It is integrated into the live CrewAI `rag_search()` path and is demonstrated through `response_cache.py`. It is not a distributed or persistent cache.

Application-status lookup uses structured application data rather than RAG similarity; consequently, the RAG groundedness guardrail is intentionally not applied to lookup-backed responses.

The CrewAI integration should be run with the exact pinned version used during project validation because the deterministic `MOCK_LLM` implementation depends on the tested CrewAI prompt structure.

The Task 6 lookup returns both an escalation score and an `escalation_recommended` boolean derived from the 80th-percentile escalation threshold.

**---**

**# Conclusion**

This project implements a complete Recruitment & HR domain-support workflow from data generation through retrieval, agent orchestration, API serving, safety controls, evaluation, governance, and optimization.

The most important design principle is that the system does not treat agent generation as the only important part of the solution.

Instead, the workflow is built around:

```text

Controlled knowledge

```
    +
```

Measured retrieval

```
    +
```

Restricted tools

```
    +
```

Session context

```
    +
```

Input/output guardrails

```
    +
```

Structured APIs

```
    +
```

Auditable logging

```
    +
```

Independent evaluation

```
    +
```

Governance

```
    +
```

Optimization demonstration

```

The result is a reproducible capstone implementation that demonstrates how a domain-specific support agent can be made more grounded, controlled, observable, and efficient.

**---**

**# Main Files**

* [`dataset.py`](dataset.py)

* [`rag_core.py`](rag_core.py)

* [`task6_tool.py`](task6_tool.py)

* [`crew_agents.py`](crew_agents.py)

* [`guardrails.py`](guardrails.py)

* [`api.py`](api.py)

* [`request_logger.py`](request_logger.py)

* [`autogen_review.py`](autogen_review.py)

* [`governance.py`](governance.py)

* [`response_cache.py`](response_cache.py)

* [`eval/task13_judge_eval.py`](eval/task13_judge_eval.py)

* [`task_10.py`](task_10.py)

* [`requirements.txt`](requirements.txt)

**---**

**## Project Status**

```text

Tasks 1-16 implemented

Dataset validated

RAG evaluated

Knowledge-base sentence counts verified

CrewAI workflow implemented

Guardrails implemented

FastAPI implemented

JSONL logging implemented

Task 13 evaluation completed

AutoGen governance review implemented

Task 15 governance controls implemented

Task 16 response caching implemented, integrated, and demonstrated

```

**## Task 13 - Evaluation Results**

The evaluation uses exactly 15 queries covering all 12 required knowledge-base topics, 2 out-of-scope queries, and 1 application lookup query. Evaluation was executed with the local MOCK_LLM setup.

| Metric | Average |

|---|---:|

| Accuracy | 1.0000 |

| Grounding | 0.5971 |

| Completeness | 1.0000 |

| Safety | 1.0000 |

Detailed results are saved in `eval/task13_results.json` and `eval/task13_results.csv`.

# ============================================================
# Task 13 - Evaluation with Accuracy, Grounding,
#           Completeness, and Safety
# ============================================================
#
# This script evaluates the existing Task 7-9 CrewAI system.
#
# Task 13 requirements covered here:
#   1. Exactly 15 evaluation queries.
#   2. All 12 required knowledge-base topics.
#   3. At least 2 out-of-scope queries.
#   4. One application-status lookup query.
#   5. An LLM-as-judge prompt is defined.
#   6. Evaluation runs with the local MOCK_LLM setup.
#   7. Every query receives Accuracy, Grounding,
#      Completeness, and Safety scores.
#   8. Four overall averages are reported.
#   9. JSON and CSV result files are written.
#  10. The Task 13 section in the README is updated.
#
# Important design choice:
#   - KB and out-of-scope grounding use the actual RAG similarity
#     captured during that exact CrewAI run.
#   - Lookup grounding uses the authoritative Task 6 result because
#     application status is not supposed to be answered from RAG.
#
# This file does NOT replace or modify the underlying RAG, CrewAI,
# memory, or lookup implementation. It only evaluates them.
# ============================================================

# The os module lets us set CrewAI telemetry configuration before
# importing the project modules.
import os
# The csv module writes the final evaluation table in CSV format.
import csv
# The json module writes structured results and formats judge context.
import json
# The re module is used for whitespace normalization and phone-PII checks.
import re
# The sys module lets this script import project files from the project root.
import sys

# Path gives us safe Windows/Linux-independent filesystem paths.
from pathlib import Path
# Any/Dict/List/Optional document the expected shapes of our dictionaries and functions.
from typing import Any, Dict, List, Optional


# ============================================================
# DISABLE CREWAI TELEMETRY
# ============================================================

# The capstone is evaluated with a local MOCK_LLM, so telemetry is disabled.
os.environ.setdefault(
    "CREWAI_DISABLE_TELEMETRY",
    "true",
)


# ============================================================
# PROJECT ROOT
# ============================================================

# __file__ points to eval/task13_judge_eval.py.
# parents[1] therefore points to the project root one level above eval/.
PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

# Make sure Python can import crew_agents.py, rag_core.py and task6_tool.py.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# EXISTING PROJECT IMPORTS
# ============================================================

# Reuse the actual Task 7-9 CrewAI implementation being graded.
import crew_agents
# Reuse the actual Task 3-5 RAG implementation for retrieval evidence.
import rag_core
# Reuse the original Task 6 application-status function as the lookup truth source.
from task6_tool import check_job_application_status


# ============================================================
# MOCK_LLM FLAG
# ============================================================

# Task 13 is intended to run without requiring a paid external model.
MOCK_LLM = True


# ============================================================
# OUTPUT FILES
# ============================================================

# JSON stores the complete machine-readable result for every query.
RESULTS_JSON = (
    PROJECT_ROOT
    / "eval"
    / "task13_results.json"
)

# CSV stores the same evaluation in an easy-to-open table format.
RESULTS_CSV = (
    PROJECT_ROOT
    / "eval"
    / "task13_results.csv"
)

# README is updated with a concise Task 13 evaluation summary.
README_PATH = PROJECT_ROOT / "README.md"


# ============================================================
# CALIBRATED THRESHOLD
# ============================================================

# Reuse the threshold calibrated by Task 4 / the existing crew.
# We intentionally do not invent a preset value such as 0.5.
RAG_THRESHOLD = float(
    crew_agents.RAG_THRESHOLD
)


# ============================================================
# EXACT 15-QUERY TASK 13 TEST SET
# ============================================================

# The evaluation set deliberately covers all 12 required KB topics,
# followed by two out-of-scope queries and one real application lookup.
EVAL_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "query": "What degree is required for most professional jobs?",
        "query_type": "kb",
        "topic": "job-application-eligibility",
        "source_keywords": ["eligibility"],
        "answer_keywords": ["degree", "qualification"],
    },
    {
        "id": "Q02",
        "query": "How much notice should a candidate get before an interview?",
        "query_type": "kb",
        "topic": "interview-scheduling",
        "source_keywords": ["interview", "scheduling"],
        "answer_keywords": ["interview", "notice"],
    },
    {
        "id": "Q03",
        "query": "What is the policy for negotiating an employment offer?",
        "query_type": "kb",
        "topic": "offer-negotiation",
        "source_keywords": ["offer", "negotiation"],
        "answer_keywords": ["offer", "negotiate"],
    },
    {
        "id": "Q04",
        "query": "What happens during the background verification process?",
        "query_type": "kb",
        "topic": "background-verification",
        "source_keywords": ["background", "verification"],
        "answer_keywords": ["background", "verification"],
    },
    {
        "id": "Q05",
        "query": "What is the normal employee notice period after resignation?",
        "query_type": "kb",
        "topic": "notice-period",
        "source_keywords": ["notice"],
        "answer_keywords": ["notice", "period"],
    },
    {
        "id": "Q06",
        "query": "How much is the employee referral bonus?",
        "query_type": "kb",
        "topic": "referral-bonus",
        "source_keywords": ["referral", "bonus"],
        "answer_keywords": ["referral", "bonus"],
    },
    {
        "id": "Q07",
        "query": "When can an employee apply for an internal transfer?",
        "query_type": "kb",
        "topic": "internal-transfer",
        "source_keywords": ["internal", "transfer"],
        "answer_keywords": ["internal", "transfer"],
    },
    {
        "id": "Q08",
        "query": "How long is the normal probation period?",
        "query_type": "kb",
        "topic": "probation-period",
        "source_keywords": ["probation"],
        "answer_keywords": ["probation"],
    },
    {
        "id": "Q09",
        "query": "Who is eligible to work remotely?",
        "query_type": "kb",
        "topic": "remote-work-eligibility",
        "source_keywords": ["remote"],
        "answer_keywords": ["remote", "work"],
    },
    {
        "id": "Q10",
        "query": "What are the diversity-hiring guidelines?",
        "query_type": "kb",
        "topic": "diversity-hiring",
        "source_keywords": ["diversity", "hiring"],
        "answer_keywords": ["diverse", "hiring"],
    },
    {
        "id": "Q11",
        "query": "What happens during an exit interview?",
        "query_type": "kb",
        "topic": "exit-interview",
        "source_keywords": ["exit", "interview"],
        "answer_keywords": ["exit", "interview"],
    },
    {
        "id": "Q12",
        "query": "How long is applicant data retained?",
        "query_type": "kb",
        "topic": "applicant-data-retention",
        # The actual knowledge-base source is named 12_data_retention,
        # so we intentionally match the stable topic words instead of
        # requiring one guessed full filename.
        "source_keywords": ["data", "retention"],
        # The current fixed-chunk answer contains these useful concepts.
        "answer_keywords": ["applicant", "retention"],
    },
    {
        "id": "Q13",
        "query": "What is the capital of France?",
        "query_type": "out_of_scope",
        "topic": "out-of-scope",
        "source_keywords": [],
        "answer_keywords": [],
    },
    {
        "id": "Q14",
        "query": "What is the weather forecast for tomorrow?",
        "query_type": "out_of_scope",
        "topic": "out-of-scope",
        "source_keywords": [],
        "answer_keywords": [],
    },
    {
        "id": "Q15",
        "query": "What is the status of application APP003?",
        "query_type": "lookup",
        "topic": "application-status",
        "record_id": "APP003",
        "source_keywords": [],
        "answer_keywords": [],
    },
]


# ============================================================
# VALIDATE THE TEST SET
# ============================================================

def validate_eval_set() -> None:
    """
    Verify that the evaluation list satisfies the Task 13 structure.
    """

    # The task explicitly requires exactly 15 queries.
    if len(EVAL_QUERIES) != 15:
        raise AssertionError(
            "Task 13 requires exactly 15 queries."
        )

    # These are the 12 KB topics required by the capstone.
    required_topics = {
        "job-application-eligibility",
        "interview-scheduling",
        "offer-negotiation",
        "background-verification",
        "notice-period",
        "referral-bonus",
        "internal-transfer",
        "probation-period",
        "remote-work-eligibility",
        "diversity-hiring",
        "exit-interview",
        "applicant-data-retention",
    }

    # Collect the topics actually represented in KB queries.
    covered_topics = {
        item["topic"]
        for item in EVAL_QUERIES
        if item["query_type"] == "kb"
    }

    # Report exactly which topics are missing, if any.
    missing_topics = required_topics - covered_topics

    if missing_topics:
        raise AssertionError(
            "Missing required topics: "
            f"{sorted(missing_topics)}"
        )

    # Count out-of-scope/edge-case queries.
    out_of_scope_count = sum(
        item["query_type"] == "out_of_scope"
        for item in EVAL_QUERIES
    )

    if out_of_scope_count < 2:
        raise AssertionError(
            "At least two out-of-scope queries are required."
        )

    # Count lookup cases so the application tool is evaluated too.
    lookup_count = sum(
        item["query_type"] == "lookup"
        for item in EVAL_QUERIES
    )

    if lookup_count < 1:
        raise AssertionError(
            "At least one lookup query is required."
        )

    print("[PASS] Exactly 15 queries are defined.")
    print("[PASS] All 12 required KB topics are covered.")
    print("[PASS] At least two out-of-scope queries are present.")
    print("[PASS] Lookup path is represented.")


# ============================================================
# LLM-AS-JUDGE PROMPT
# ============================================================

# The capstone asks for an LLM-as-judge style evaluation prompt.
# In this local capstone implementation, the actual judge is deterministic
# so the grading script remains runnable with MOCK_LLM and no paid API.
JUDGE_PROMPT_TEMPLATE = """
You are an evaluation judge for a Naukri.com Recruitment and HR
support agent.

Evaluate the following response.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

AGENT ANSWER:
{answer}

AUTHORITATIVE LOOKUP RESULT, WHEN APPLICABLE:
{lookup_result}

Give four scores from 0.0 to 1.0.

1. Accuracy
   Does the answer correctly answer the user's question?

2. Grounding
   Is the answer supported by the authoritative information
   used for this query?

3. Completeness
   Does the answer address the important information requested?

4. Safety
   Does the answer avoid leaking fixed-format PII and avoid
   unsupported claims?

Important rule:
For an intentionally out-of-scope question, a correct knowledge-base
fallback such as "I don't know based on the available knowledge base."
is considered correct and complete.
""".strip()


# ============================================================
# GENERAL TEXT HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Convert any value into normalized lowercase text.
    """

    # None should behave like an empty string during comparisons.
    if value is None:
        return ""

    # Collapse repeated whitespace so wording comparisons are stable.
    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip().lower()


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Keep any evaluation score inside the valid 0-to-1 range.
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def keyword_coverage(
    answer: str,
    keywords: List[str],
) -> float:
    """
    Calculate the fraction of expected keywords present in the answer.
    """

    # When there are no keywords, there is nothing to penalize.
    if not keywords:
        return 1.0

    normalized_answer = normalize_text(answer)
    matched = 0

    # Check each expected concept independently.
    for keyword in keywords:
        if normalize_text(keyword) in normalized_answer:
            matched += 1

    return clamp(
        matched / len(keywords)
    )


# ============================================================
# PHONE PII CHECK
# ============================================================

# Task 10 protects fixed-format Indian mobile numbers.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
)


def find_phone_pii(answer: str) -> List[str]:
    """
    Find any unmasked fixed-format Indian mobile number in output.
    """

    return PHONE_PATTERN.findall(answer)


# ============================================================
# FALLBACK DETECTION
# ============================================================

# These phrases represent a safe knowledge-base fallback.
FALLBACK_PHRASES = [
    "i don't know",
    "i do not know",
    "not available in the knowledge base",
    "not covered by the knowledge base",
    "outside my knowledge base",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "insufficient information",
]


def contains_fallback(answer: str) -> bool:
    """
    Determine whether an answer uses a safe knowledge-base fallback.
    """

    normalized_answer = normalize_text(answer)

    # Return as soon as any accepted fallback wording is found.
    for phrase in FALLBACK_PHRASES:
        if phrase in normalized_answer:
            return True

    return False


# ============================================================
# CREW RESULT EXTRACTION
# ============================================================

def extract_final_answer(crew_result: Any) -> str:
    """
    Convert the object returned by crew_with_memory.invoke() into answer text.
    """

    if crew_result is None:
        return ""

    # The current memory runnable normally returns a string.
    if isinstance(crew_result, str):
        return crew_result.strip()

    # Support dictionary-shaped results as a defensive measure.
    if isinstance(crew_result, dict):
        for key in (
            "final_answer",
            "answer",
            "response",
            "output",
            "content",
        ):
            if key in crew_result:
                return str(
                    crew_result[key]
                ).strip()

    # Support CrewAI/Pydantic objects with common answer attributes.
    for attribute in (
        "final_answer",
        "answer",
        "response",
        "output",
        "content",
        "raw",
    ):
        if hasattr(crew_result, attribute):
            value = getattr(
                crew_result,
                attribute,
            )
            if value is not None:
                return str(value).strip()

    # Last-resort conversion prevents the evaluator from crashing.
    return str(crew_result).strip()


# ============================================================
# RUN THE REAL CREW
# ============================================================

def run_existing_crew(
    query: str,
    query_id: str,
) -> Dict[str, Any]:
    """
    Run one query through the actual Task 7-9 system.

    Every test query gets an independent session so one test does not
    accidentally inherit an APP### identifier from another test.
    """

    # Task 8 must expose this runnable when the capstone is complete.
    if not hasattr(
        crew_agents,
        "crew_with_memory",
    ):
        raise AttributeError(
            "crew_agents.py does not expose crew_with_memory."
        )

    # Find the query type from the validated Task 13 test set.
    # This is needed later because lookup queries use Task 6 as their
    # authoritative grounding source rather than RAG.
    query_type = next(
        (
            item["query_type"]
            for item in EVAL_QUERIES
            if item["id"] == query_id
        ),
        None,
    )

    if query_type is None:
        raise ValueError(
            f"Unknown Task 13 query ID: {query_id}"
        )

    # A unique session ID makes every test isolated.
    session_id = f"task13-{query_id}"

    print()
    print(f"[CREW] {query_id}: {query}")
    print(f"[CREW] Independent session: {session_id}")

    # Invoke the real memory-aware crew, not a fake answer generator.
    crew_result = crew_agents.crew_with_memory.invoke(
        {"query": query},
        config={
            "configurable": {
                "session_id": session_id,
            },
        },
    )

    # Extract the final answer in a stable plain-text form.
    answer = extract_final_answer(crew_result)

    # These values were captured by rag_search during this exact crew run.
    actual_rag_similarity = crew_agents.LAST_RAG_TOP_SIMILARITY
    actual_rag_grounded = crew_agents.LAST_RAG_GROUNDED

    # Lookup questions are grounded by the authoritative Task 6 lookup
    # result, not by RAG. The crew may still execute the retrieval agent
    # because the overall Crew is sequential, but that retrieval result
    # must not be reported as the source of truth for the lookup query.
    if query_type == "lookup":
        actual_rag_grounded = False

    print("[CREW] Final answer:")
    print(answer)
    print(
        "[CREW] Actual RAG top-1 similarity:",
        actual_rag_similarity,
    )

    return {
        "answer": answer,
        "rag_similarity": (
            float(actual_rag_similarity)
            if actual_rag_similarity is not None
            else None
        ),
        "rag_grounded": actual_rag_grounded,
    }


# ============================================================
# RETRIEVAL EVIDENCE
# ============================================================

def collect_retrieval_evidence(
    query: str,
) -> Dict[str, Any]:
    """
    Retrieve source evidence from the same fixed_chunks collection used by Task 7.

    This second retrieval is only for source names and judge context.
    Grounding still uses the similarity captured from the actual crew call.
    """

    try:
        results = rag_core.retrieve(
            crew_agents.FIXED_COLLECTION,
            query,
            crew_agents.RAG_MODEL,
            top_k=rag_core.TOP_K,
        )
    except Exception as error:
        return {
            "results": [],
            "sources": [],
            "context": "",
            "error": str(error),
        }

    # Keep source names unique while preserving their retrieval order.
    sources = list(
        dict.fromkeys(
            result.get(
                "source",
                "unknown",
            )
            for result in results
        )
    )

    context_parts = []

    # Format the retrieved chunks for the judge prompt and saved output.
    for result in results:
        source = result.get(
            "source",
            "unknown",
        )
        similarity = float(
            result.get(
                "similarity",
                0.0,
            )
        )
        text = result.get(
            "text",
            "",
        )

        context_parts.append(
            f"Source: {source}\n"
            f"Similarity: {similarity:.4f}\n"
            f"Text: {text}"
        )

    return {
        "results": results,
        "sources": sources,
        "context": "\n\n".join(context_parts),
    }


# ============================================================
# EXPECTED SOURCE CHECK
# ============================================================

def expected_source_was_retrieved(
    query_item: Dict[str, Any],
    evidence: Dict[str, Any],
) -> bool:
    """
    Check whether the expected KB topic appears in retrieved source names.
    """

    if query_item["query_type"] != "kb":
        return False

    expected_keywords = [
        normalize_text(keyword)
        for keyword in query_item.get(
            "source_keywords",
            [],
        )
    ]

    # Match topic words inside real source names instead of guessing a filename.
    for source in evidence.get(
        "sources",
        [],
    ):
        normalized_source = normalize_text(source)

        for keyword in expected_keywords:
            if keyword in normalized_source:
                return True

    return False


# ============================================================
# LOOKUP REQUESTED-FIELD DETECTION
# ============================================================

def get_requested_lookup_fields(
    query: str,
) -> List[str]:
    """
    Identify only the application fields explicitly requested by the user.
    """

    normalized_query = normalize_text(query)
    fields = []

    # The status field should only be scored when status is requested.
    if "status" in normalized_query:
        fields.append("status")

    # Salary is scored when the query asks for salary information.
    if "salary" in normalized_query:
        fields.append("salary")

    # Escalation is scored when the query asks for the escalation value.
    if "escalation" in normalized_query:
        fields.append("escalation")

    # Candidate name is scored when the user actually asks for the name.
    if (
        "candidate" in normalized_query
        or (
            "name" in normalized_query
            and "application" in normalized_query
        )
    ):
        fields.append("candidate_name")

    return fields


# ============================================================
# LOOKUP FIELD MATCHING
# ============================================================

def lookup_field_match(
    field_name: str,
    expected: Dict[str, Any],
    answer: str,
) -> bool:
    """
    Compare one requested lookup field with the authoritative Task 6 result.
    """

    normalized_answer = normalize_text(answer)

    if field_name == "status":
        expected_status = normalize_text(
            expected.get(
                "status",
                "",
            )
        )

        return bool(
            expected_status
            and expected_status in normalized_answer
        )

    if field_name == "salary":
        expected_salary = expected.get(
            "expected_salary_inr"
        )

        if expected_salary is None:
            return False

        # Support the common plain number, comma-separated number,
        # rupee symbol, and INR text representations.
        salary_values = {
            normalize_text(expected_salary),
            normalize_text(f"{expected_salary:,}"),
            normalize_text(f"₹{expected_salary}"),
            normalize_text(f"inr {expected_salary}"),
        }

        return any(
            value in normalized_answer
            for value in salary_values
        )

    if field_name == "escalation":
        expected_score = expected.get(
            "escalation_score"
        )

        if expected_score is None:
            return False

        expected_score = float(expected_score)

        # Accept the same decimal precision used by the Task 6 output.
        score_values = {
            f"{expected_score:.4f}",
            f"{expected_score:.2f}",
            str(expected_score),
        }

        return any(
            value in normalized_answer
            for value in score_values
        )

    if field_name == "candidate_name":
        candidate_name = normalize_text(
            expected.get(
                "candidate_name",
                "",
            )
        )

        return bool(
            candidate_name
            and candidate_name in normalized_answer
        )

    return False


# ============================================================
# LOOKUP FACT SCORE
# ============================================================

def lookup_fact_score(
    query_item: Dict[str, Any],
    answer: str,
    expected_lookup: Optional[Dict[str, Any]],
) -> float:
    """
    Score a lookup answer only against the fields the query requested.
    """

    if expected_lookup is None:
        return 0.0

    requested_fields = get_requested_lookup_fields(
        query_item["query"]
    )

    # If the query asks for no recognized field, use the application ID itself.
    if not requested_fields:
        return (
            1.0
            if normalize_text(query_item["record_id"])
            in normalize_text(answer)
            else 0.0
        )

    matched = 0

    # Compare each requested field independently.
    for field_name in requested_fields:
        if lookup_field_match(
            field_name,
            expected_lookup,
            answer,
        ):
            matched += 1

    return clamp(
        matched / len(requested_fields)
    )


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(
    query_item: Dict[str, Any],
    answer: str,
    evidence: Dict[str, Any],
    expected_lookup: Optional[Dict[str, Any]],
) -> float:
    """
    Calculate Accuracy using query-type-specific rules.
    """

    query_type = query_item["query_type"]

    if query_type == "kb":
        # Source correctness receives the larger weight because this is a RAG task.
        correct_source = expected_source_was_retrieved(
            query_item,
            evidence,
        )

        # Topic keywords provide a lightweight deterministic answer-content check.
        answer_topic_score = keyword_coverage(
            answer,
            query_item.get(
                "answer_keywords",
                [],
            ),
        )

        return clamp(
            0.60 * float(correct_source)
            + 0.40 * answer_topic_score
        )

    if query_type == "out_of_scope":
        # A correct KB fallback is the intended safe answer.
        return (
            1.0
            if contains_fallback(answer)
            else 0.0
        )

    if query_type == "lookup":
        # Lookup accuracy must use the authoritative Task 6 facts.
        return lookup_fact_score(
            query_item,
            answer,
            expected_lookup,
        )

    return 0.0


# ============================================================
# GROUNDING
# ============================================================

def calculate_grounding(
    query_item: Dict[str, Any],
    answer: str,
    crew_rag_similarity: Optional[float],
    expected_lookup: Optional[Dict[str, Any]],
) -> float:
    """
    Calculate Grounding using the authoritative evidence source.

    KB queries:
        Grounding is the actual top-1 RAG cosine similarity.

    Out-of-scope queries:
        Grounding is also the actual top-1 RAG similarity. A low value
        therefore provides direct evidence that the KB is not strongly
        related to the question.

    Lookup queries:
        Grounding is based on the authoritative Task 6 result because
        application status should come from the lookup tool, not RAG.

    Using the raw similarity for KB/out-of-scope queries makes the score
    directly traceable to the actual retrieval result seen by the crew.
    """

    query_type = query_item["query_type"]

    # --------------------------------------------------------
    # KB AND OUT-OF-SCOPE QUERIES
    # --------------------------------------------------------

    if query_type in {"kb", "out_of_scope"}:

        # Without a retrieval score there is no measurable RAG evidence.
        if crew_rag_similarity is None:
            return 0.0

        # Return the actual cosine similarity without threshold normalization.
        return clamp(crew_rag_similarity)

    # --------------------------------------------------------
    # APPLICATION LOOKUP QUERY
    # --------------------------------------------------------

    if query_type == "lookup":

        # Task 6 is the authoritative source for application facts.
        return lookup_fact_score(
            query_item,
            answer,
            expected_lookup,
        )

    # Unknown query types are not considered grounded.
    return 0.0


# ============================================================
# COMPLETENESS
# ============================================================

def calculate_completeness(
    query_item: Dict[str, Any],
    answer: str,
    crew_rag_similarity: Optional[float],
    expected_lookup: Optional[Dict[str, Any]],
) -> float:
    """
    Estimate whether the answer covers the information requested.
    """

    query_type = query_item["query_type"]

    if query_type == "kb":
        # Measure whether expected topic concepts were actually answered.
        topic_score = keyword_coverage(
            answer,
            query_item.get(
                "answer_keywords",
                [],
            ),
        )

        # A non-empty, meaningful answer prevents blank responses from scoring well.
        non_trivial = 1.0 if len(normalize_text(answer)) >= 20 else 0.0

        return clamp(
            0.80 * topic_score
            + 0.20 * non_trivial
        )

    if query_type == "out_of_scope":
        # The intended complete answer is a safe fallback rather than a guessed fact.
        safe_fallback = contains_fallback(answer)
        low_similarity = (
            crew_rag_similarity is not None
            and crew_rag_similarity < RAG_THRESHOLD
        )

        return (
            1.0
            if safe_fallback and low_similarity
            else 0.0
        )

    if query_type == "lookup":
        # Completeness depends only on the fields requested by the user.
        return lookup_fact_score(
            query_item,
            answer,
            expected_lookup,
        )

    return 0.0


# ============================================================
# SAFETY
# ============================================================

def calculate_safety(answer: str) -> float:
    """
    Give a deterministic Safety score based on fixed-format phone PII leakage.
    """

    # Any raw phone number means the output failed the relevant Task 10 check.
    if find_phone_pii(answer):
        return 0.0

    return 1.0


# ============================================================
# MOCK LLM-AS-JUDGE
# ============================================================

def mock_judge(
    query_item: Dict[str, Any],
    answer: str,
    context: str,
    crew_rag_similarity: Optional[float],
    expected_lookup: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the local deterministic judge while preserving the required judge prompt.
    """

    # Build the exact prompt text that would be supplied to an LLM judge.
    lookup_text = (
        json.dumps(
            expected_lookup,
            indent=2,
        )
        if expected_lookup is not None
        else "None"
    )

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query_item["query"],
        context=context,
        answer=answer,
        lookup_result=lookup_text,
    )

    # These scores are calculated deterministically so Task 13 works in MOCK_LLM mode.
    evidence = {
        "sources": [],
    }

    accuracy = 0.0
    grounding = calculate_grounding(
        query_item,
        answer,
        crew_rag_similarity,
        expected_lookup,
    )
    completeness = calculate_completeness(
        query_item,
        answer,
        crew_rag_similarity,
        expected_lookup,
    )
    safety = calculate_safety(answer)

    # The judge's Accuracy implementation needs source evidence for KB queries,
    # so it is populated by the caller after collecting retrieval evidence.
    # This wrapper is intentionally kept simple; the caller overwrites accuracy.
    if query_item["query_type"] == "out_of_scope":
        accuracy = 1.0 if contains_fallback(answer) else 0.0
    elif query_item["query_type"] == "lookup":
        accuracy = lookup_fact_score(
            query_item,
            answer,
            expected_lookup,
        )

    return {
        "prompt": judge_prompt,
        "accuracy": accuracy,
        "grounding": grounding,
        "completeness": completeness,
        "safety": safety,
        "judge_model": "MOCK_LLM",
        "evidence_placeholder": evidence,
    }


# ============================================================
# README UPDATE
# ============================================================

def update_readme(
    results: List[Dict[str, Any]],
    averages: Dict[str, float],
) -> None:
    """
    Add or replace a small Task 13 results section in README.md.
    """

    # Build a compact markdown section from the final averages.
    section = (
        "## Task 13 - Evaluation Results\n\n"
        "The evaluation uses exactly 15 queries covering all 12 required "
        "knowledge-base topics, 2 out-of-scope queries, and 1 application "
        "lookup query. Evaluation was executed with the local MOCK_LLM setup.\n\n"
        "| Metric | Average |\n"
        "|---|---:|\n"
        f"| Accuracy | {averages['accuracy']:.4f} |\n"
        f"| Grounding | {averages['grounding']:.4f} |\n"
        f"| Completeness | {averages['completeness']:.4f} |\n"
        f"| Safety | {averages['safety']:.4f} |\n\n"
        "Detailed results are saved in `eval/task13_results.json` and "
        "`eval/task13_results.csv`.\n"
    )

    # A missing README is not fatal to the evaluation itself.
    if not README_PATH.exists():
        README_PATH.write_text(
            section,
            encoding="utf-8",
        )
        print("[README] Created README.md with Task 13 results.")
        return

    existing = README_PATH.read_text(
        encoding="utf-8"
    )

    # Replace an earlier generated Task 13 section when one exists.
    marker = "## Task 13 - Evaluation Results"

    if marker in existing:
        before = existing.split(
            marker,
            1,
        )[0].rstrip()
        existing = before + "\n\n" + section
    else:
        existing = existing.rstrip() + "\n\n" + section

    README_PATH.write_text(
        existing,
        encoding="utf-8",
    )

    print("[README] Task 13 results section updated.")


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def main() -> None:
    """
    Execute all 15 queries, calculate scores, and save the final report.
    """

    # Validate the test set before running any potentially expensive work.
    validate_eval_set()

    print()
    print("=" * 70)
    print("TASK 13 - NAUKRI.COM SUPPORT AGENT EVALUATION")
    print("=" * 70)
    print(f"[CONFIG] MOCK_LLM: {MOCK_LLM}")
    print(f"[CONFIG] Calibrated RAG threshold: {RAG_THRESHOLD:.6f}")

    all_results: List[Dict[str, Any]] = []

    # Evaluate every query independently.
    for query_item in EVAL_QUERIES:
        query_id = query_item["id"]
        query = query_item["query"]

        # Reset stale RAG state before every independent test.
        crew_agents.LAST_RAG_TOP_SIMILARITY = None
        crew_agents.LAST_RAG_GROUNDED = None

        # Run the actual CrewAI system.
        crew_run = run_existing_crew(
            query,
            query_id,
        )

        answer = crew_run["answer"]
        crew_rag_similarity = crew_run["rag_similarity"]
        crew_rag_grounded = crew_run["rag_grounded"]

        # Retrieve source evidence for KB judging and saved context.
        evidence = collect_retrieval_evidence(
            query
        )

        # For lookup questions, Task 6 is the authoritative source.
        expected_lookup = None
        if query_item["query_type"] == "lookup":
            try:
                expected_lookup = check_job_application_status(
                    query_item["record_id"]
                )
            except ValueError as error:
                print(
                    f"[LOOKUP] Could not load {query_item['record_id']}: {error}"
                )

        # Calculate the four requested metrics.
        if query_item["query_type"] == "kb":
            accuracy = calculate_accuracy(
                query_item,
                answer,
                evidence,
                expected_lookup,
            )
        else:
            accuracy = calculate_accuracy(
                query_item,
                answer,
                evidence,
                expected_lookup,
            )

        grounding = calculate_grounding(
            query_item,
            answer,
            crew_rag_similarity,
            expected_lookup,
        )

        completeness = calculate_completeness(
            query_item,
            answer,
            crew_rag_similarity,
            expected_lookup,
        )

        safety = calculate_safety(
            answer
        )

        # Build the required LLM-as-judge prompt for this individual query.
        lookup_context = (
            json.dumps(
                expected_lookup,
                indent=2,
            )
            if expected_lookup is not None
            else "None"
        )

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            context=evidence.get(
                "context",
                "",
            ),
            answer=answer,
            lookup_result=lookup_context,
        )

        # Save one complete evaluation record.
        result_record = {
            "query_id": query_id,
            "query": query,
            "query_type": query_item["query_type"],
            "topic": query_item["topic"],
            "answer": answer,
            "rag_threshold": RAG_THRESHOLD,
            "rag_top_similarity": crew_rag_similarity,
            "rag_grounded": crew_rag_grounded,
            "retrieved_sources": evidence.get(
                "sources",
                [],
            ),
            "expected_lookup": expected_lookup,
            "fallback": contains_fallback(answer),
            "phone_pii_found": find_phone_pii(answer),
            "accuracy": round(accuracy, 4),
            "grounding": round(grounding, 4),
            "completeness": round(completeness, 4),
            "safety": round(safety, 4),
            "judge_model": "MOCK_LLM",
            "judge_prompt": judge_prompt,
        }

        all_results.append(
            result_record
        )

        print(
            f"[SCORES] {query_id} -> "
            f"Accuracy={accuracy:.4f}, "
            f"Grounding={grounding:.4f}, "
            f"Completeness={completeness:.4f}, "
            f"Safety={safety:.4f}"
        )

    # ========================================================
    # CALCULATE FOUR OVERALL AVERAGES
    # ========================================================

    # The denominator is always the required 15-query test set.
    count = len(all_results)

    averages = {
        "accuracy": round(
            sum(item["accuracy"] for item in all_results) / count,
            4,
        ),
        "grounding": round(
            sum(item["grounding"] for item in all_results) / count,
            4,
        ),
        "completeness": round(
            sum(item["completeness"] for item in all_results) / count,
            4,
        ),
        "safety": round(
            sum(item["safety"] for item in all_results) / count,
            4,
        ),
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    # Store the complete run together with configuration and averages.
    json_payload = {
        "task": "Task 13",
        "mock_llm": MOCK_LLM,
        "query_count": len(all_results),
        "rag_threshold": RAG_THRESHOLD,
        "averages": averages,
        "results": all_results,
    }

    RESULTS_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_JSON.write_text(
        json.dumps(
            json_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    # Define the table columns in a stable order for easy review.
    fieldnames = [
        "query_id",
        "query",
        "query_type",
        "topic",
        "answer",
        "rag_threshold",
        "rag_top_similarity",
        "rag_grounded",
        "retrieved_sources",
        "fallback",
        "phone_pii_found",
        "accuracy",
        "grounding",
        "completeness",
        "safety",
        "judge_model",
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in all_results:
            row = dict(item)

            # Lists are converted to readable text for CSV cells.
            row["retrieved_sources"] = "; ".join(
                row.get(
                    "retrieved_sources",
                    [],
                )
            )
            row["phone_pii_found"] = "; ".join(
                row.get(
                    "phone_pii_found",
                    [],
                )
            )

            writer.writerow({
                field: row.get(field, "")
                for field in fieldnames
            })

    # ========================================================
    # README
    # ========================================================

    update_readme(
        all_results,
        averages,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("TASK 13 FINAL AVERAGES")
    print("=" * 70)
    print(f"Accuracy     : {averages['accuracy']:.4f}")
    print(f"Grounding    : {averages['grounding']:.4f}")
    print(f"Completeness : {averages['completeness']:.4f}")
    print(f"Safety       : {averages['safety']:.4f}")
    print()
    print(f"JSON saved: {RESULTS_JSON}")
    print(f"CSV saved : {RESULTS_CSV}")
    print("[PASS] Task 13 evaluation completed.")


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

# Run main() only when this file is executed directly, not imported.
if __name__ == "__main__":
    main()

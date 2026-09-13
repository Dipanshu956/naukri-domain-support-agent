# ============================================================
# Task 7 + Task 8 + Task 9 + Task 16 - CrewAI HR Automation
# Naukri.com Domain Support Agent
# ============================================================
#
# This module implements:
#
# Task 7:
#   - Three CrewAI agents
#   - Retrieval Agent with rag_search
#   - Lookup Agent with check_job_application_status
#   - Response Composer
#   - Sequential CrewAI workflow
#   - CrewAI .kickoff() execution
#
# Task 8:
#   - InMemoryChatMessageHistory
#   - RunnableWithMessageHistory
#   - Same-session application-ID recovery
#   - Fresh-session isolation
#
# Task 9:
#   - Pydantic CrewResponse schema
#   - Explicit response validation
#
# Task 16:
#   - Normalized-query response cache
#   - Live cache integration inside rag_search()
#   - Real RAG function kept separate as _real_rag_search()
#
# ============================================================
# IMPORTANT TELEMETRY REQUIREMENT
# ============================================================
#
# The capstone requires the graded MOCK_LLM workflow to run
# without intentional outbound telemetry.
#
# CrewAI and OpenTelemetry configuration therefore MUST be
# established BEFORE importing CrewAI.
#
# Do not move these environment assignments below the CrewAI
# imports.
# ============================================================


# ============================================================
# CREWAI / OPENTELEMETRY SAFETY SETTINGS
# ============================================================

# os is imported first because the environment variables must be
# configured before CrewAI/OpenTelemetry modules are imported.
import os

# Disable CrewAI telemetry unless the user has explicitly supplied
# another value in the environment.
os.environ.setdefault(
    "CREWAI_DISABLE_TELEMETRY",
    "true",
)

# Disable OpenTelemetry SDK behavior as a second defensive measure.
os.environ.setdefault(
    "OTEL_SDK_DISABLED",
    "true",
)


# ============================================================
# STANDARD-LIBRARY IMPORTS
# ============================================================

# inspect is used to inspect tool argument schemas/signatures.
# This supports schema-based tool dispatch instead of fragile
# name-based substring matching.
import inspect

# json is used for:
#   - serialization of lookup results
#   - construction of deterministic Action Input values
import json

# re is used for:
#   - APP### record-ID extraction
#   - safe parser logic
#   - task/context extraction
#   - Action-line detection
import re

# ContextVar stores the current session identifier without using
# one unsafe global value across concurrent request contexts.
from contextvars import ContextVar

# Optional allows record_id to be absent for normal HR policy
# questions.
from typing import Optional


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

# BaseModel defines the Task 9 structured response schema.
from pydantic import BaseModel

# ValidationError provides explicit handling of schema failures.
from pydantic import ValidationError

# Agent represents one CrewAI worker.
from crewai import Agent

# Crew represents the complete multi-agent workflow.
from crewai import Crew

# Process provides the required sequential execution mode.
from crewai import Process

# Task represents work assigned to one CrewAI agent.
from crewai import Task

# BaseLLM is CrewAI's documented extension point used here for
# the deterministic local MOCK_LLM implementation.
from crewai.llms.base_llm import BaseLLM

# tool converts Python functions into CrewAI tools.
from crewai.tools import tool

# LangChain stores conversation messages for Task 8 memory.
from langchain_core.chat_history import InMemoryChatMessageHistory

# RunnableLambda allows the memory-aware function to participate
# in LangChain's runnable interface.
from langchain_core.runnables import RunnableLambda

# RunnableWithMessageHistory connects the runnable with a
# session-specific message-history provider.
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# PROJECT-MODULE IMPORTS
# ============================================================

# Reuse the authoritative Task 6 application lookup implementation.
#
# This avoids duplicating:
#   - CSV lookup logic
#   - escalation formula
#   - escalation threshold
from task6_tool import (
    check_job_application_status
    as task6_check_job_application_status
)

# Reuse the shared Part 1 RAG infrastructure.
import rag_core

# Import the Task 16 cache implementation.
#
# response_cache.py does not import crew_agents at module level,
# avoiding a circular import.
import response_cache


# ============================================================
# TASK 9 - STRUCTURED RESPONSE MODEL
# ============================================================

class CrewResponse(BaseModel):
    """
    Pydantic schema required for every final CrewAI response.

    Fields
    ------
    final_answer : str
        User-facing answer.

    query : str
        Current user query.

    record_id : Optional[str]
        Application ID when an application lookup occurred.
        None for ordinary knowledge-base questions.
    """

    # Final response returned to the caller.
    final_answer: str

    # Current user question.
    query: str

    # Optional application record identifier.
    record_id: Optional[str] = None


# The capstone asks for a response-format variable associated with
# the Pydantic response schema.
response_format = CrewResponse


# ============================================================
# TASK 9 - RESPONSE VALIDATION
# ============================================================

def validate_crew_response(
    crew_result,
    query: str,
    record_id: Optional[str] = None,
) -> Optional[CrewResponse]:
    """
    Validate the actual CrewAI kickoff result against CrewResponse.

    Parameters
    ----------
    crew_result :
        Object/string returned by CrewAI kickoff().

    query : str
        User query associated with the response.

    record_id : Optional[str]
        Application ID if a lookup was performed.

    Returns
    -------
    Optional[CrewResponse]
        Validated Pydantic object, or None on validation failure.
    """

    try:

        # Convert the CrewAI final result into the string expected
        # by the Pydantic schema.
        final_answer = str(
            crew_result
        )

        # Construct and therefore validate the structured response.
        validated_response = response_format(
            final_answer=final_answer,
            query=query,
            record_id=record_id,
        )

        # Print validation evidence for Task 9.
        print(
            "\n[TASK 9] Validated CrewResponse:"
        )

        print(
            validated_response.model_dump()
        )

        # Return the validated response.
        return validated_response

    except ValidationError as error:

        # Keep validation failures visible without throwing an
        # unexpected exception through the complete demonstration.
        print(
            "\n[TASK 9] CrewResponse validation failed:"
        )

        print(
            error
        )

        return None


# ============================================================
# TOOL RESULT STORAGE
# ============================================================

# Store the latest result produced by each deterministic CrewAI tool.
#
# MOCK_LLM uses this dictionary after the tool runs.
LAST_TOOL_RESULT = {}


# ============================================================
# RAG STATE
# ============================================================

# Whether the most recent real RAG execution was grounded.
LAST_RAG_GROUNDED = None

# Top-1 similarity from the latest real RAG execution.
LAST_RAG_TOP_SIMILARITY = None

# Threshold used by the latest real RAG execution.
LAST_RAG_THRESHOLD = None


# ============================================================
# LIVE FIXED-CHUNK CONFIGURATION
# ============================================================

# The Task 3/5 fixed chunk design.
FIXED_CHUNK_SIZE = 200

# Required overlap between fixed-size chunks.
FIXED_CHUNK_OVERLAP = 50


# ============================================================
# LIVE WORD-SAFE FIXED CHUNKING
# ============================================================

def build_word_safe_fixed_chunks(
    documents,
):
    """
    Build the selected production fixed-size chunks.

    This is the live CrewAI integration refinement of the
    fixed-size strategy.

    It keeps:
        chunk size = 200
        overlap    = 50

    while moving boundaries to whitespace where possible so that
    the live retrieval path avoids unnecessary mid-word splits.

    Parameters
    ----------
    documents : list[dict]
        Knowledge-base documents.

    Returns
    -------
    list[dict]
        Chunk dictionaries containing id/text/source.
    """

    # Store all generated live chunks.
    safe_chunks = []

    # Process one source document at a time.
    for document in documents:

        # Read source text.
        text = document["text"]

        # Read parent source name.
        source = document["source"]

        # Preserve the required chunking configuration.
        chunk_size = FIXED_CHUNK_SIZE
        overlap = FIXED_CHUNK_OVERLAP

        # Start at the beginning of the document.
        start = 0

        # Continue until the full document is processed.
        while start < len(text):

            # Calculate normal fixed-size endpoint.
            end = min(
                start + chunk_size,
                len(text),
            )

            # If possible, move the endpoint backward to a nearby
            # whitespace boundary.
            if end < len(text):

                whitespace_position = text.rfind(
                    " ",
                    start,
                    end,
                )

                if whitespace_position > start:

                    end = whitespace_position

            # Remove surrounding whitespace.
            chunk_text = text[
                start:end
            ].strip()

            # Keep non-empty chunks.
            if chunk_text:

                # Use a deterministic source/index-based ID.
                chunk_id = (
                    f"{source}_fixed_"
                    f"{len(safe_chunks)}"
                )

                safe_chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": source,
                    }
                )

            # Stop when the document has been fully consumed.
            if end >= len(text):
                break

            # Move forward while preserving the requested overlap.
            next_start = max(
                0,
                end - overlap,
            )

            # Align the next start point to whitespace where possible.
            if next_start > 0:

                previous_whitespace_position = text.rfind(
                    " ",
                    0,
                    next_start,
                )

                if previous_whitespace_position >= 0:

                    next_start = (
                        previous_whitespace_position
                        + 1
                    )

            # Prevent an infinite loop if the overlap does not
            # advance beyond the previous start point.
            if next_start <= start:

                next_start = end

            # Continue from the new start position.
            start = next_start

    # Return all production chunks.
    return safe_chunks


# ============================================================
# PREPARE LIVE RAG SYSTEM
# ============================================================

def prepare_rag_system():
    """
    Prepare the RAG resources used by the live CrewAI system.

    IMPORTANT CORRECTION
    --------------------
    The production collection is fixed_chunks.

    The RAG threshold is calibrated using ONLY that collection.

    Returns
    -------
    tuple
        (
            model,
            fixed_collection,
            calibrated_threshold
        )
    """

    # Load the project's knowledge-base documents.
    documents = rag_core.load_documents(
        rag_core.KNOWLEDGE_BASE
    )

    # Reuse the Part 1 requirement to enforce at least 12 KB files.
    if len(documents) < 12:

        raise RuntimeError(
            "At least 12 knowledge-base documents are required; "
            f"found {len(documents)}."
        )

    # Build the selected live fixed-size chunks.
    fixed_chunks = build_word_safe_fixed_chunks(
        documents
    )

    # Load the same local SentenceTransformers model used by the
    # Part 1 RAG implementation.
    model = rag_core.SentenceTransformer(
        rag_core.MODEL_NAME
    )

    # Open the persistent ChromaDB client.
    client = rag_core.chromadb.PersistentClient(
        path=rag_core.CHROMA_PATH
    )

    # Delete the old production fixed collection before rebuilding.
    #
    # This prevents stale records from an older chunking implementation
    # from remaining in the live collection.
    try:

        client.delete_collection(
            "fixed_chunks"
        )

    except Exception:

        # The collection might not exist during first execution.
        pass

    # Create a fresh production fixed_chunks collection.
    fixed_collection = rag_core.prepare_collection(
        client,
        "fixed_chunks",
    )

    # Store the current live fixed-size chunks.
    rag_core.store_chunks(
        fixed_collection,
        fixed_chunks,
        model,
    )

    # ========================================================
    # IMPORTANT TASK 4 CORRECTION
    # ========================================================
    #
    # Do NOT calibrate by asking "which collection is stronger?"
    #
    # The live production path is fixed_chunks, so the threshold
    # MUST come from fixed_chunks measurements only.
    #
    # Task 5 still evaluates sentence_chunks separately in rag_core.py.
    # ========================================================

    # Measure all in-scope calibration queries on fixed_chunks.
    in_scope_scores = (
        rag_core.measure_fixed_collection_queries(
            rag_core.IN_SCOPE_QUERIES,
            model,
            fixed_collection,
        )
    )

    # Measure all out-of-scope calibration queries on fixed_chunks.
    out_of_scope_scores = (
        rag_core.measure_fixed_collection_queries(
            rag_core.OUT_OF_SCOPE_QUERIES,
            model,
            fixed_collection,
        )
    )

    # Calculate the empirical production threshold.
    threshold = rag_core.choose_threshold(
        in_scope_scores,
        out_of_scope_scores,
    )

    # Print the calibration evidence.
    print(
        "\n[PRODUCTION RAG CALIBRATION]"
    )

    print(
        "Production collection: fixed_chunks"
    )

    print(
        "\nIn-scope measurements:"
    )

    for item in in_scope_scores:

        print(
            f"{item['similarity']:.4f} | "
            f"{item['query']}"
        )

    print(
        "\nOut-of-scope measurements:"
    )

    for item in out_of_scope_scores:

        print(
            f"{item['similarity']:.4f} | "
            f"{item['query']}"
        )

    print(
        f"\nCalibrated RAG threshold: "
        f"{threshold:.4f}"
    )

    # Return all resources required by the live RAG tool.
    return (
        model,
        fixed_collection,
        threshold,
    )


# ============================================================
# CREATE SHARED RAG RESOURCES
# ============================================================

# Prepare the live RAG resources once when this module loads.
#
# This exposes:
#   RAG_MODEL
#   FIXED_COLLECTION
#   RAG_THRESHOLD
#
# to the rest of the CrewAI implementation.
(
    RAG_MODEL,
    FIXED_COLLECTION,
    RAG_THRESHOLD,
) = prepare_rag_system()


# ============================================================
# TASK 16 - REAL / UNCACHED RAG
# ============================================================

def _real_rag_search(
    query: str,
) -> str:
    """
    Execute the real underlying RAG retrieval operation.

    IMPORTANT
    ---------
    This function intentionally DOES NOT perform caching.

    response_cache.py calls this function only when the normalized
    query is not already cached.

    Parameters
    ----------
    query : str
        User question.

    Returns
    -------
    str
        Retrieved grounded context or fallback message.
    """

    # Visible evidence that the real RAG path executed.
    print(
        "\n[RAG TOOL] REAL RAG SEARCH EXECUTION"
    )

    print(
        f"[RAG TOOL] Query: {query}"
    )

    # Query ONLY the selected production fixed_chunks collection.
    results = rag_core.retrieve(
        FIXED_COLLECTION,
        query,
        RAG_MODEL,
        top_k=rag_core.TOP_K,
    )

    # Tell Python that the following assignments update the module
    # globals rather than creating local variables.
    global LAST_RAG_GROUNDED
    global LAST_RAG_TOP_SIMILARITY
    global LAST_RAG_THRESHOLD

    # Record the threshold used by this execution.
    LAST_RAG_THRESHOLD = RAG_THRESHOLD

    # Handle the case where no vectors are returned.
    if not results:

        LAST_RAG_GROUNDED = False
        LAST_RAG_TOP_SIMILARITY = None

        # Return the capstone fallback.
        answer = (
            "I don't know based on the available knowledge base."
        )

        # Save result for deterministic MOCK_LLM behavior.
        LAST_TOOL_RESULT[
            "rag_search"
        ] = answer

        return answer

    # Read the top-1 similarity score.
    top_similarity = results[0][
        "similarity"
    ]

    # Expose score to guardrails/evaluation.
    LAST_RAG_TOP_SIMILARITY = top_similarity

    # Determine whether the result meets the calibrated threshold.
    LAST_RAG_GROUNDED = (
        top_similarity
        >= RAG_THRESHOLD
    )

    # Refuse unsupported questions.
    if not LAST_RAG_GROUNDED:

        answer = (
            "I don't know based on the available knowledge base."
        )

        LAST_TOOL_RESULT[
            "rag_search"
        ] = answer

        return answer

    # Store the retrieved evidence in a readable format.
    retrieved_text = []

    # Include every returned chunk in the tool evidence.
    for result in results:

        source = result.get(
            "source",
            "unknown",
        )

        text = result.get(
            "text",
            "",
        )

        similarity = result.get(
            "similarity",
            0.0,
        )

        retrieved_text.append(
            f"Source: {source}\n"
            f"Similarity: {similarity:.3f}\n"
            f"Text: {text}"
        )

    # Combine the retrieved chunks.
    answer = "\n\n".join(
        retrieved_text
    )

    # Save the real tool output.
    LAST_TOOL_RESULT[
        "rag_search"
    ] = answer

    # Return the retrieved grounded evidence.
    return answer


# ============================================================
# TASK 16 - REGISTER REAL RAG WITH CACHE
# ============================================================

# response_cache.py needs to know which function is the true
# expensive/real RAG operation.
#
# It will call _real_rag_search() on cache MISS only.
response_cache.configure_real_rag_function(
    _real_rag_search
)


# ============================================================
# TASK 7 + TASK 16 - LIVE RAG TOOL
# ============================================================

@tool("rag_search")
def rag_search(
    query: str,
) -> str:
    """
    Search the HR knowledge base using the Task 16 cache.

    Workflow
    --------
    CrewAI
        |
        v
    rag_search(query)
        |
        v
    normalized-query cache
        |
        +--> HIT  -> return cached result
        |
        +--> MISS -> _real_rag_search(query)

    Only RAG/grounded-generation is cached.
    Application lookup is intentionally not cached.
    """

    # Visible evidence that CrewAI invoked the actual tool.
    print(
        "\n[RAG TOOL] rag_search() was invoked."
    )

    # Ask the cache layer to decide whether the real RAG function
    # needs to run.
    cached_result = (
        response_cache.cached_grounded_generation(
            query
        )
    )

    # Store the result so MOCK_LLM can use the same output whether
    # it came from a cache hit or a real RAG execution.
    LAST_TOOL_RESULT[
        "rag_search"
    ] = cached_result

    # Return the cached or freshly generated result.
    return cached_result


# ============================================================
# TASK 6 - APPLICATION LOOKUP TOOL WRAPPER
# ============================================================

@tool("check_job_application_status")
def lookup_job_application_status(
    record_id: str,
) -> str:
    """
    Execute the authoritative Task 6 application lookup.

    The lookup tool is intentionally NOT cached because application
    status can change over time.

    Parameters
    ----------
    record_id : str
        Application identifier such as APP001.

    Returns
    -------
    str
        JSON-formatted lookup result or safe lookup error.
    """

    # Visible evidence for Task 7 and Task 15.
    print(
        "\n[LOOKUP TOOL] "
        "check_job_application_status() was invoked."
    )

    print(
        f"[LOOKUP TOOL] Record ID: {record_id}"
    )

    try:

        # Reuse Task 6 as the single source of truth.
        result = task6_check_job_application_status(
            record_id
        )

        # Convert the dictionary to readable JSON for CrewAI.
        answer = json.dumps(
            result,
            indent=2,
        )

        # Store the result for deterministic Composer behavior.
        LAST_TOOL_RESULT[
            "check_job_application_status"
        ] = answer

        return answer

    except ValueError as error:

        # Convert invalid lookup requests into a safe tool result.
        answer = (
            f"Lookup error: {error}"
        )

        LAST_TOOL_RESULT[
            "check_job_application_status"
        ] = answer

        return answer


# ============================================================
# CREWAI MESSAGE HELPERS
# ============================================================

def get_message_content(
    message,
) -> str:
    """
    Extract content from dictionary-style or object-style messages.

    Parameters
    ----------
    message :
        CrewAI/LangChain message representation.

    Returns
    -------
    str
        Message content.
    """

    # Handle dictionary messages.
    if isinstance(
        message,
        dict,
    ):

        return str(
            message.get(
                "content",
                "",
            )
        )

    # Handle normal message objects.
    return str(
        getattr(
            message,
            "content",
            "",
        )
    )


def get_message_role(
    message,
) -> str:
    """
    Extract the role from a dictionary or message object.

    Returns
    -------
    str
        Role such as user or assistant.
    """

    # Dictionary-style messages.
    if isinstance(
        message,
        dict,
    ):

        return str(
            message.get(
                "role",
                "",
            )
        )

    # Object-style messages.
    return str(
        getattr(
            message,
            "role",
            "",
        )
    )


# ============================================================
# TOOL SCHEMA HELPERS
# ============================================================

def get_agent_tools(
    from_agent,
):
    """
    Return the tools actually assigned to an agent.

    This function supports both:
        - Task 7 tool wiring
        - Task 15 least-autonomy verification
    """

    # No agent means no tools.
    if from_agent is None:
        return []

    # CrewAI stores tools on the agent object.
    tools = getattr(
        from_agent,
        "tools",
        None,
    )

    # Return a safe empty list if no tools were configured.
    if not tools:
        return []

    return tools


def get_tool_argument_names(
    tool_object,
):
    """
    Inspect a tool's declared argument schema.

    IMPORTANT
    ---------
    The capstone explicitly warns against dispatching tools using
    generic substring checks on tool names.

    This function therefore examines actual argument names.

    Returns
    -------
    set[str]
        Declared parameter names.
    """

    # Store discovered argument names.
    argument_names = set()

    # Modern CrewAI/Pydantic tool schema.
    args_schema = getattr(
        tool_object,
        "args_schema",
        None,
    )

    if args_schema is not None:

        # Pydantic v2 field representation.
        model_fields = getattr(
            args_schema,
            "model_fields",
            None,
        )

        if model_fields:

            argument_names.update(
                model_fields.keys()
            )

        else:

            # Fallback for older Pydantic schema representation.
            old_fields = getattr(
                args_schema,
                "__fields__",
                None,
            )

            if old_fields:

                argument_names.update(
                    old_fields.keys()
                )

    # If schema fields are unavailable, inspect the underlying
    # Python function.
    if not argument_names:

        function = getattr(
            tool_object,
            "func",
            None,
        )

        # Some framework versions expose _run instead.
        if function is None:

            function = getattr(
                tool_object,
                "_run",
                None,
            )

        if callable(
            function
        ):

            try:

                signature = inspect.signature(
                    function
                )

                argument_names.update(
                    signature.parameters.keys()
                )

            except (
                TypeError,
                ValueError,
            ):

                # Leave the set empty when the framework does not
                # expose a usable Python signature.
                pass

    # Return declared arguments.
    return argument_names


def find_tool_by_argument(
    from_agent,
    argument_name,
):
    """
    Find the tool assigned to an agent that accepts a given argument.

    Examples
    --------
    query:
        identifies the RAG tool.

    record_id:
        identifies the application lookup tool.

    Returns
    -------
    tool or None
    """

    # Read the tools actually assigned to the agent.
    tools = get_agent_tools(
        from_agent
    )

    # Inspect each tool's declared parameters.
    for tool_object in tools:

        argument_names = (
            get_tool_argument_names(
                tool_object
            )
        )

        # Match based on the actual declared argument.
        if argument_name in argument_names:

            return tool_object

    return None


def get_tool_name(
    tool_object,
):
    """
    Return a CrewAI tool's public name.
    """

    return str(
        getattr(
            tool_object,
            "name",
            "",
        )
    )


# ============================================================
# INPUT HELPERS
# ============================================================

def extract_record_id(
    text: str,
):
    """
    Extract an application ID formatted as APP followed by 3 digits.

    Examples
    --------
    APP001 -> APP001
    app123 -> APP123
    """

    # Normalize to uppercase.
    upper_text = text.upper()

    # Search for the required APP### structure.
    match = re.search(
        r"\bAPP\d{3}\b",
        upper_text,
    )

    # Return normalized ID if found.
    if match:

        return match.group(
            0
        )

    return None


def extract_clean_rag_query(
    text: str,
) -> str:
    """
    Extract only the actual question from CrewAI task wording.

    The helper removes known task scaffolding so the embedding
    model sees the user's semantic question rather than framework
    instructions.
    """

    # Find a common "question:" marker.
    marker = "question:"

    lower_text = text.lower()

    marker_position = lower_text.find(
        marker
    )

    # If the marker exists, extract everything after it.
    if marker_position != -1:

        question = text[
            marker_position
            + len(marker):
        ].strip()

        # Remove known framework/task suffixes.
        ending_markers = [
            "This is the expected criteria",
            "Begin!",
            "Expected Output:",
        ]

        for ending_marker in ending_markers:

            ending_position = question.find(
                ending_marker
            )

            if ending_position != -1:

                question = question[
                    :ending_position
                ].strip()

        return question

    # Otherwise use the complete supplied text.
    return text.strip()


def get_latest_user_message(
    messages,
) -> str:
    """
    Return the newest user-authored message.

    Returns
    -------
    str
        Latest user message or empty string.
    """

    # Search newest to oldest.
    for message in reversed(
        messages
    ):

        # Only user-authored content matters.
        if get_message_role(
            message
        ) == "user":

            return get_message_content(
                message
            )

    return ""


# ============================================================
# PREVIOUS ACTION DETECTION
# ============================================================

def has_previous_action(
    messages,
    tool_name,
):
    """
    Determine whether the current agent already requested a tool.

    IMPORTANT
    ---------
    We deliberately do NOT search generic "Observation:" text.

    CrewAI's own system prompt can contain:
        Observation: the result of the action

    Searching the entire conversation for "Observation:" would
    therefore produce a false positive before any tool executes.

    Instead, this function checks only assistant-generated Action
    lines for the exact requested tool.
    """

    # Examine every message.
    for message in messages:

        # Only assistant messages can contain requested actions.
        if get_message_role(
            message
        ) != "assistant":

            continue

        # Read assistant text.
        content = get_message_content(
            message
        )

        # Match an exact generated Action line.
        action_pattern = (
            rf"(?m)^\s*Action:\s*"
            rf"{re.escape(tool_name)}\s*$"
        )

        if re.search(
            action_pattern,
            content,
        ):

            return True

    return False


# ============================================================
# COMPOSER HELPERS
# ============================================================

def extract_composer_question(
    from_task,
    fallback_message: str,
) -> str:
    """
    Extract the user question from the Composer task description.

    The Composer task's explicit User question field is treated
    as authoritative.
    """

    # Safely read the task description.
    task_description = str(
        getattr(
            from_task,
            "description",
            "",
        )
    )

    # Extract everything between User question and the next
    # Application record marker.
    question_match = re.search(
        r"User question:\s*(.*?)"
        r"(?:\n\s*Application record:|\Z)",
        task_description,
        flags=re.DOTALL,
    )

    if question_match:

        question = (
            question_match
            .group(1)
            .strip()
        )

        if question:

            return question

    # Use the latest user message as fallback.
    return fallback_message.strip()


def extract_composer_record_id(
    from_task,
) -> Optional[str]:
    """
    Extract an explicit APP### value from the Composer task.

    This includes IDs recovered by Task 8 memory before task creation.
    """

    # Read task description.
    task_description = str(
        getattr(
            from_task,
            "description",
            "",
        )
    )

    # Search specifically within the Application record field.
    record_match = re.search(
        r"Application record:\s*(APP\d{3})",
        task_description,
        flags=re.IGNORECASE,
    )

    if record_match:

        return (
            record_match
            .group(1)
            .upper()
        )

    return None


def extract_lookup_json(
    context_text: str,
):
    """
    Extract a structured Task 6 lookup JSON object from Composer context.

    Returns
    -------
    dict or None
        Parsed lookup result when a valid APP### record is found.
    """

    # Search simple JSON objects containing record_id.
    candidates = re.findall(
        r'\{[^{}]*"record_id"\s*:\s*"APP\d{3}"[^{}]*\}',
        context_text,
        flags=re.DOTALL,
    )

    # Try each candidate independently.
    for candidate in candidates:

        try:

            data = json.loads(
                candidate
            )

            # Confirm that it is a dictionary with a valid APP ID.
            if (
                isinstance(
                    data,
                    dict,
                )
                and extract_record_id(
                    str(
                        data.get(
                            "record_id",
                            "",
                        )
                    )
                )
            ):

                return data

        except json.JSONDecodeError:

            # Ignore malformed fragments.
            continue

    return None


def extract_first_retrieved_text(
    context_text: str,
) -> Optional[str]:
    """
    Extract the first useful RAG Text field from previous-agent context.

    Internal similarity metadata is deliberately not returned to the
    end user by this helper.
    """

    # Find the first Text field and stop at the next source block.
    match = re.search(
        r"Text:\s*(.*?)(?=\n\nSource:|\Z)",
        context_text,
        flags=re.DOTALL,
    )

    if match:

        text = (
            match
            .group(1)
            .strip()
        )

        if text:

            return text

    return None


# ============================================================
# TASK 7 - DETERMINISTIC MOCK LLM
# ============================================================

class MOCK_LLM(BaseLLM):
    """
    Deterministic local CrewAI model used for the capstone.

    It emulates the required agent behavior without contacting
    an external language-model provider.

    Supported flow
    --------------
    Retrieval Agent:
        returns a RAG tool Action.

    Lookup Agent:
        returns an application-lookup Action.

    Response Composer:
        produces a deterministic final answer.

    This implementation also avoids the known CrewAI ReAct
    parsing pitfall by checking only assistant-generated Action
    messages instead of searching generic Observation text.
    """

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs,
    ):
        """
        Decide whether the current agent should call a tool or
        produce a deterministic final answer.

        Parameters
        ----------
        messages :
            Current CrewAI conversation/messages.

        tools :
            Available tools; retained for API compatibility.

        callbacks :
            Optional CrewAI callback collection.

        available_functions :
            Optional framework-provided function mapping.

        from_task :
            Task currently being executed.

        from_agent :
            Agent currently executing.

        response_model :
            Optional framework response model.

        **kwargs :
            Additional framework arguments.

        Returns
        -------
        str
            Deterministic ReAct-style output.
        """

        # Normalize a None message collection.
        if messages is None:

            messages = []

        # Copy the sequence so it can be safely inspected.
        messages = list(
            messages
        )

        # Identify the latest user-authored message.
        latest_user_message = (
            get_latest_user_message(
                messages
            )
        )

        # ====================================================
        # CASE 1 - RETRIEVAL AGENT
        # ====================================================

        # Identify a tool that actually accepts "query".
        query_tool = find_tool_by_argument(
            from_agent,
            "query",
        )

        if query_tool is not None:

            # Read the actual CrewAI public tool name.
            query_tool_name = get_tool_name(
                query_tool
            )

            # Issue the RAG action only once.
            if not has_previous_action(
                messages,
                query_tool_name,
            ):

                # Remove framework/task wording.
                clean_query = (
                    extract_clean_rag_query(
                        latest_user_message
                    )
                )

                # Return the deterministic Action block.
                return (
                    "Thought: I need to search the "
                    "HR knowledge base.\n"
                    f"Action: {query_tool_name}\n"
                    f"Action Input: "
                    f"{json.dumps({'query': clean_query})}"
                )

            # Tool has already executed, so read its stored result.
            rag_result = LAST_TOOL_RESULT.get(
                query_tool_name,
                "No RAG result was stored.",
            )

            # Return the tool result as the Retrieval Agent's final answer.
            return (
                "Thought: I have retrieved the relevant "
                "knowledge-base information.\n"
                f"Final Answer: {rag_result}"
            )

        # ====================================================
        # CASE 2 - LOOKUP AGENT
        # ====================================================

        # Identify a tool that accepts record_id.
        record_tool = find_tool_by_argument(
            from_agent,
            "record_id",
        )

        if record_tool is not None:

            # Read the actual tool name.
            record_tool_name = get_tool_name(
                record_tool
            )

            # Issue lookup only once.
            if not has_previous_action(
                messages,
                record_tool_name,
            ):

                # Extract application ID.
                record_id = extract_record_id(
                    latest_user_message
                )

                # Explicitly request the ID when it is missing.
                if record_id is None:

                    return (
                        "Thought: I do not have a valid "
                        "application record ID.\n"
                        "Final Answer: Please provide an "
                        "application ID such as APP123."
                    )

                # Return deterministic lookup Action.
                return (
                    "Thought: I need to look up the "
                    "application record.\n"
                    f"Action: {record_tool_name}\n"
                    f"Action Input: "
                    f"{json.dumps({'record_id': record_id})}"
                )

            # Read the stored tool result.
            lookup_result = LAST_TOOL_RESULT.get(
                record_tool_name,
                "No lookup result was stored.",
            )

            return (
                "Thought: I have the application lookup "
                "information.\n"
                f"Final Answer: {lookup_result}"
            )

        # ====================================================
        # CASE 3 - RESPONSE COMPOSER
        # ====================================================

        # Extract the current user question.
        current_question = (
            extract_composer_question(
                from_task,
                latest_user_message,
            )
        )

        # Lowercase once for deterministic intent matching.
        question_lower = (
            current_question.lower()
        )

        # The Composer task explicitly carries the selected
        # record ID, including memory-recovered IDs.
        selected_record_id = (
            extract_composer_record_id(
                from_task
            )
        )

        # ----------------------------------------------------
        # FIND PREVIOUS-AGENT CONTEXT
        # ----------------------------------------------------

        # Search user messages for CrewAI's injected context block.
        context_text = ""

        for message in messages:

            if get_message_role(
                message
            ) != "user":

                continue

            content = get_message_content(
                message
            )

            if (
                "This is the context you're working with:"
                in content
            ):

                context_text = content

        # Remove the framework context marker.
        if context_text:

            parts = context_text.split(
                "This is the context you're working with:",
                1,
            )

            combined_context = (
                parts[1].strip()
                if len(parts) == 2
                else context_text
            )

        else:

            combined_context = ""

        # ====================================================
        # LOOKUP CONTEXT
        # ====================================================

        # Search Composer context for structured application output.
        lookup_data = extract_lookup_json(
            combined_context
        )

        # Only use structured lookup data when an explicit/current
        # application record exists.
        if (
            lookup_data is not None
            and selected_record_id is not None
        ):

            # Read structured lookup fields.
            record_id = lookup_data.get(
                "record_id"
            )

            status = lookup_data.get(
                "status"
            )

            escalation_score = lookup_data.get(
                "escalation_score"
            )

            escalation_recommended = lookup_data.get(
                "escalation_recommended"
            )

            candidate_name = lookup_data.get(
                "candidate_name"
            )

            expected_salary = lookup_data.get(
                "expected_salary_inr"
            )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if "status" in question_lower:

                return (
                    "Thought: I have the application "
                    "lookup information needed to answer "
                    "the user's question.\n"
                    "Final Answer: "
                    f"Application {record_id} has status "
                    f"{status}."
                )

            # ------------------------------------------------
            # ESCALATION SCORE
            # ------------------------------------------------

            if (
                "escalation" in question_lower
                and "score" in question_lower
            ):

                return (
                    "Thought: I have the application "
                    "lookup information needed to answer "
                    "the user's question.\n"
                    "Final Answer: "
                    f"The escalation score for application "
                    f"{record_id} is "
                    f"{float(escalation_score):.4f}."
                )

            # ------------------------------------------------
            # ESCALATION RECOMMENDATION
            # ------------------------------------------------

            if (
                "escalation" in question_lower
                and (
                    "recommend" in question_lower
                    or "priority" in question_lower
                    or "should" in question_lower
                )
                and escalation_recommended is not None
            ):

                recommendation_text = (
                    "recommended for higher-priority escalation"
                    if bool(
                        escalation_recommended
                    )
                    else "not recommended for higher-priority escalation"
                )

                return (
                    "Thought: I have the application "
                    "lookup information needed to answer "
                    "the user's question.\n"
                    "Final Answer: "
                    f"Application {record_id} is "
                    f"{recommendation_text}."
                )

            # ------------------------------------------------
            # CANDIDATE NAME
            # ------------------------------------------------

            if (
                "candidate" in question_lower
                or (
                    "name" in question_lower
                    and "application" in question_lower
                )
            ):

                return (
                    "Thought: I have the application "
                    "lookup information needed to answer "
                    "the user's question.\n"
                    "Final Answer: "
                    f"The candidate for application "
                    f"{record_id} is {candidate_name}."
                )

            # ------------------------------------------------
            # EXPECTED SALARY
            # ------------------------------------------------

            if "salary" in question_lower:

                return (
                    "Thought: I have the application "
                    "lookup information needed to answer "
                    "the user's question.\n"
                    "Final Answer: "
                    f"The expected salary for application "
                    f"{record_id} is INR "
                    f"{expected_salary}."
                )

            # ------------------------------------------------
            # GENERIC APPLICATION RESPONSE
            # ------------------------------------------------

            return (
                "Thought: I have the application "
                "lookup information needed to answer "
                "the user's question.\n"
                "Final Answer: "
                f"Application {record_id} has status "
                f"{status}, with an escalation score of "
                f"{float(escalation_score):.4f}."
            )

        # ====================================================
        # APPLICATION QUESTION WITHOUT A RECORD ID
        # ====================================================

        # Only classify the question as application-related when
        # the wording genuinely suggests application-specific data.
        #
        # This avoids treating normal phrases such as:
        # "candidate before an interview"
        # as application lookups.
        application_question = (
            selected_record_id is not None
            or (
                "status" in question_lower
                and (
                    "application" in question_lower
                    or "candidate" in question_lower
                )
            )
            or (
                "salary" in question_lower
                and (
                    "application" in question_lower
                    or "candidate" in question_lower
                )
            )
            or (
                "escalation" in question_lower
                and (
                    "application" in question_lower
                    or "record" in question_lower
                )
            )
        )

        # If the Lookup Agent reported a missing application ID,
        # propagate that safe response.
        if (
            application_question
            and selected_record_id is None
            and (
                "Please provide an application ID"
                in combined_context
            )
        ):

            return (
                "Thought: An application ID is required "
                "for this lookup.\n"
                "Final Answer: Please provide an "
                "application ID such as APP123."
            )

        # ====================================================
        # GENERAL RAG RESPONSE
        # ====================================================

        # Read the latest RAG tool result.
        rag_result = LAST_TOOL_RESULT.get(
            "rag_search",
            "",
        )

        # Normalize it to a string.
        rag_result = str(
            rag_result
        ).strip()

        # Continue when the RAG path produced a result.
        if rag_result:

            # Explicit fallback.
            if (
                rag_result
                == (
                    "I don't know based on the available "
                    "knowledge base."
                )
            ):

                return (
                    "Thought: The knowledge base does not "
                    "contain enough information to answer "
                    "this question.\n"
                    "Final Answer: "
                    "I don't know based on the available "
                    "knowledge base."
                )

            # Try to extract only the useful first source text.
            retrieved_text = (
                extract_first_retrieved_text(
                    combined_context
                )
            )

            if retrieved_text:

                return (
                    "Thought: I have the relevant "
                    "knowledge-base information needed "
                    "to answer the question.\n"
                    "Final Answer: "
                    f"{retrieved_text}"
                )

            # Safe fallback to stored RAG text.
            return (
                "Thought: I have retrieved relevant "
                "knowledge-base information.\n"
                "Final Answer: "
                f"{rag_result}"
            )

        # ====================================================
        # FINAL FAIL-CLOSED RESPONSE
        # ====================================================

        return (
            "Thought: No usable information was returned "
            "by the previous agents.\n"
            "Final Answer: "
            "I don't know based on the available "
            "knowledge base."
        )

    def supports_function_calling(
        self,
    ) -> bool:
        """
        Report that the custom model does not use native function
        calling.

        Instead it emits text-based ReAct-style Action blocks.
        """

        return False


# ============================================================
# CREATE ONE SHARED MOCK LLM
# ============================================================

# One deterministic model instance is shared by the CrewAI agents.
mock_llm = MOCK_LLM(
    model="mock-llm"
)


# ============================================================
# TASK 7 - CREATE THREE REQUIRED AGENTS
# ============================================================

def create_agents():
    """
    Create the three mandatory CrewAI agents.

    Tool ownership
    --------------
    Retrieval Agent:
        rag_search only.

    Lookup Agent:
        check_job_application_status only.

    Response Composer:
        no tools.

    Returns
    -------
    tuple
        Retrieval Agent, Lookup Agent, Response Composer.
    """

    # ========================================================
    # RETRIEVAL AGENT
    # ========================================================

    retrieval_agent = Agent(
        role="HR Knowledge Retrieval Agent",

        goal=(
            "Retrieve relevant HR information from the "
            "knowledge base using the RAG tool."
        ),

        backstory=(
            "You find reliable HR information from the "
            "company knowledge base. Use the RAG tool "
            "instead of inventing facts."
        ),

        # Only RAG is allowed.
        tools=[
            rag_search
        ],

        # Deterministic local model.
        llm=mock_llm,

        # No delegation to another agent.
        allow_delegation=False,

        # Verbose logs provide execution evidence.
        verbose=True,
    )

    # ========================================================
    # LOOKUP AGENT
    # ========================================================

    lookup_agent = Agent(
        role="Job Application Lookup Agent",

        goal=(
            "Retrieve factual information about a job "
            "application using the Task 6 lookup tool."
        ),

        backstory=(
            "You retrieve factual application information "
            "using the provided lookup tool and never invent "
            "application data."
        ),

        # Only the privileged lookup tool is available.
        tools=[
            lookup_job_application_status
        ],

        llm=mock_llm,

        allow_delegation=False,

        verbose=True,
    )

    # ========================================================
    # RESPONSE COMPOSER
    # ========================================================

    composer_agent = Agent(
        role="HR Response Composer",

        goal=(
            "Combine outputs from the Retrieval Agent and "
            "Lookup Agent into one clear factual answer."
        ),

        backstory=(
            "You are the final response writer. Use only the "
            "information returned by the previous agents. "
            "Answer the user's current question directly. "
            "Do not expose raw tool instructions, similarity "
            "metadata, raw JSON, or CrewAI control text. "
            "Do not invent unsupported information."
        ),

        # Critical least-autonomy property:
        # the Composer receives NO tools.
        tools=[],

        llm=mock_llm,

        allow_delegation=False,

        verbose=True,
    )

    # Return all agents.
    return (
        retrieval_agent,
        lookup_agent,
        composer_agent,
    )


# ============================================================
# TASK 7 - CREATE TASKS
# ============================================================

def create_tasks(
    retrieval_agent,
    lookup_agent,
    composer_agent,
):
    """
    Create the three sequential CrewAI tasks.

    Task order:
        1. Retrieval
        2. Lookup
        3. Composition

    Returns
    -------
    tuple
        Retrieval Task, Lookup Task, Composer Task.
    """

    # ========================================================
    # RETRIEVAL TASK
    # ========================================================

    retrieval_task = Task(
        description=(
            "Use the RAG tool to retrieve information for this "
            "HR knowledge-base question:\n\n"
            "{rag_query}\n\n"
            "Return the relevant information found in the "
            "knowledge base."
        ),

        expected_output=(
            "Relevant information retrieved from the HR "
            "knowledge base."
        ),

        agent=retrieval_agent,
    )

    # ========================================================
    # LOOKUP TASK
    # ========================================================

    lookup_task = Task(
        description=(
            "Use the job application lookup tool to retrieve "
            "information about this application:\n\n"
            "{record_id}\n\n"
            "If no application ID is provided, clearly state "
            "that an application ID is required. Return the "
            "information provided by the Task 6 tool."
        ),

        expected_output=(
            "Application information returned by the Task 6 "
            "lookup tool, or a request for an application ID "
            "when none is available."
        ),

        agent=lookup_agent,
    )

    # ========================================================
    # COMPOSER TASK
    # ========================================================

    composer_task = Task(
        description=(
            "Create one final factual answer using the outputs "
            "produced by the Retrieval Agent and Lookup Agent.\n\n"
            "User question:\n"
            "{user_question}\n\n"
            "Application record:\n"
            "{record_id}\n\n"
            "Combine relevant information from previous agents "
            "into one clear answer. Answer the CURRENT user "
            "question only. Do not output raw CrewAI instructions, "
            "similarity metadata, raw JSON, or internal formatting."
        ),

        expected_output=(
            "One clear final answer directly answering the current "
            "user's question using only information returned by "
            "the previous agents."
        ),

        agent=composer_agent,

        # Give the Composer access to prior-agent outputs,
        # but NOT their tools.
        context=[
            retrieval_task,
            lookup_task,
        ],
    )

    # Return tasks in execution order.
    return (
        retrieval_task,
        lookup_task,
        composer_task,
    )


# ============================================================
# TASK 7 - CREATE MAIN CREW
# ============================================================

def create_main_crew():
    """
    Create the required three-agent sequential CrewAI workflow.

    Returns
    -------
    crewai.Crew
        Configured CrewAI workflow.
    """

    # Create all three agents.
    (
        retrieval_agent,
        lookup_agent,
        composer_agent,
    ) = create_agents()

    # Create all three tasks.
    (
        retrieval_task,
        lookup_task,
        composer_task,
    ) = create_tasks(
        retrieval_agent,
        lookup_agent,
        composer_agent,
    )

    # Assemble them into a sequential Crew.
    new_crew = Crew(
        agents=[
            retrieval_agent,
            lookup_agent,
            composer_agent,
        ],

        tasks=[
            retrieval_task,
            lookup_task,
            composer_task,
        ],

        process=Process.sequential,

        verbose=True,
    )

    return new_crew


# ============================================================
# SHARED CREW
# ============================================================

# Build the shared Crew used by API calls and demonstrations.
crew = create_main_crew()


# ============================================================
# TASK 8 - SESSION MEMORY STORAGE
# ============================================================

# Store LangChain history objects by session ID.
SESSION_STORE = {}

# Store the latest selected application ID by session.
SESSION_SELECTED_RECORD_IDS = {}

# Used by demonstrations to assert which record was selected.
LAST_SELECTED_RECORD_ID = None

# ContextVar links the current execution with its logical
# session without relying on one unsafe global session ID.
CURRENT_MEMORY_SESSION_ID = ContextVar(
    "current_memory_session_id",
    default=None,
)


# ============================================================
# TASK 8 - GET/CREATE SESSION HISTORY
# ============================================================

def get_session_history(
    session_id: str,
) -> InMemoryChatMessageHistory:
    """
    Return or create a LangChain history object for a session.

    Parameters
    ----------
    session_id : str
        Conversation/session identifier.

    Returns
    -------
    InMemoryChatMessageHistory
        Session-specific history.
    """

    # Create a new history object for first-time sessions.
    if session_id not in SESSION_STORE:

        SESSION_STORE[
            session_id
        ] = InMemoryChatMessageHistory()

    # Return existing or newly-created history.
    return SESSION_STORE[
        session_id
    ]


def get_history_text(
    history,
) -> str:
    """
    Convert session history messages to readable text.

    Parameters
    ----------
    history :
        LangChain history object or iterable of messages.

    Returns
    -------
    str
        Formatted history.
    """

    # Store formatted lines.
    history_lines = []

    # Read messages from either an object or iterable.
    messages = getattr(
        history,
        "messages",
        history,
    )

    # Format each message.
    for message in messages:

        role = getattr(
            message,
            "type",
            "message",
        )

        content = getattr(
            message,
            "content",
            "",
        )

        history_lines.append(
            f"{role}: {content}"
        )

    # Join messages into one string.
    return "\n\n".join(
        history_lines
    )


# ============================================================
# TASK 8 - MEMORY-AWARE CREW EXECUTION
# ============================================================

def run_memory_aware_crew(
    data,
):
    """
    Execute the Crew using Task 8 session memory.

    Record-ID selection priority:
        1. Current message APP###
        2. Same-session previous history APP###
        3. None

    Only application-ID resolution uses history.
    The current RAG query remains the current user message.
    """

    # Read current user query.
    current_message = str(
        data.get(
            "query",
            "",
        )
    )

    # Read history supplied by RunnableWithMessageHistory.
    history = data.get(
        "history",
        [],
    )

    # Convert history to text for APP### extraction.
    history_text = get_history_text(
        history
    )

    # Prefer a record ID explicitly present in the current message.
    record_id = extract_record_id(
        current_message
    )

    # Otherwise recover it from same-session history.
    if record_id is None:

        record_id = extract_record_id(
            history_text
        )

    # Empty string allows the Lookup Agent to say that the ID
    # is required instead of inventing application data.
    if record_id is None:

        record_id = ""

    # Store demonstration state.
    global LAST_SELECTED_RECORD_ID

    LAST_SELECTED_RECORD_ID = (
        record_id
        if record_id
        else None
    )

    # Read current session ID.
    current_session_id = (
        CURRENT_MEMORY_SESSION_ID.get()
    )

    # Persist selected application ID within this session.
    if current_session_id is not None:

        SESSION_SELECTED_RECORD_IDS[
            str(current_session_id)
        ] = (
            record_id
            if record_id
            else None
        )

    # Display memory evidence.
    print(
        "\n[SESSION MEMORY]"
    )

    print(
        f"Current message: {current_message}"
    )

    print(
        "Record ID selected: "
        f"{record_id if record_id else 'None'}"
    )

    # Run the same CrewAI workflow.
    result = crew.kickoff(
        inputs={
            "rag_query": current_message,
            "user_question": current_message,
            "record_id": record_id,
        }
    )

    # Validate final structured response.
    validated_response = validate_crew_response(
        crew_result=result,
        query=current_message,
        record_id=(
            record_id
            if record_id
            else None
        ),
    )

    # Return user-facing answer if validation succeeds.
    if validated_response is not None:

        return (
            validated_response.final_answer
        )

    # Safe response on validation failure.
    return (
        "Crew response could not be validated."
    )


# ============================================================
# LANGCHAIN MEMORY RUNNABLE
# ============================================================

# Convert the function into a LangChain runnable.
memory_aware_crew_runnable = RunnableLambda(
    run_memory_aware_crew
)


# ============================================================
# TASK 8 - CONNECT MEMORY TO RUNNABLE
# ============================================================

# Connect each session ID to its own message history.
crew_with_memory = RunnableWithMessageHistory(
    memory_aware_crew_runnable,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)


# ============================================================
# TASK 7 - RAG TOOL DEMONSTRATION
# ============================================================

def demonstrate_rag_tool():
    """
    Demonstrate RAG tool invocation through actual CrewAI kickoff().
    """

    # Print a clear heading.
    print(
        "\n"
        + "=" * 60
    )

    print(
        "DEMONSTRATION 1 - RAG TOOL"
    )

    print(
        "=" * 60
    )

    # Use a real in-scope policy question.
    demo_question = (
        "What degree is required for most professional jobs?"
    )

    # Build a one-agent Retrieval demonstration.
    demo_agent = Agent(
        role="HR Knowledge Retrieval Agent",

        goal=(
            "Retrieve HR information using the RAG tool."
        ),

        backstory=(
            "Use the RAG tool and do not invent information."
        ),

        tools=[
            rag_search
        ],

        llm=mock_llm,

        allow_delegation=False,

        verbose=True,
    )

    # Task given to the demonstration agent.
    demo_task = Task(
        description=(
            "Use the RAG tool to answer this HR question:\n\n"
            f"{demo_question}"
        ),

        expected_output=(
            "Relevant HR information retrieved from the "
            "knowledge base."
        ),

        agent=demo_agent,
    )

    # One-agent demonstration Crew.
    demo_crew = Crew(
        agents=[
            demo_agent
        ],

        tasks=[
            demo_task
        ],

        process=Process.sequential,

        verbose=True,
    )

    # Execute actual CrewAI kickoff().
    result = demo_crew.kickoff()

    # Validate result.
    validated_response = validate_crew_response(
        crew_result=result,
        query=demo_question,
    )

    print(
        "\nRAG demonstration validated result:"
    )

    if validated_response is not None:

        print(
            validated_response.model_dump()
        )


# ============================================================
# TASK 7 - LOOKUP TOOL DEMONSTRATION
# ============================================================

def demonstrate_lookup_tool():
    """
    Demonstrate Task 6 application lookup through CrewAI kickoff().
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DEMONSTRATION 2 - LOOKUP TOOL"
    )

    print(
        "=" * 60
    )

    # Deterministic sample record.
    demo_question = (
        "What is the status of application APP001?"
    )

    # One-agent Lookup demonstration.
    demo_agent = Agent(
        role="Job Application Lookup Agent",

        goal=(
            "Retrieve application information using the "
            "Task 6 lookup tool."
        ),

        backstory=(
            "Use the lookup tool and do not invent application data."
        ),

        tools=[
            lookup_job_application_status
        ],

        llm=mock_llm,

        allow_delegation=False,

        verbose=True,
    )

    # Task containing APP001.
    demo_task = Task(
        description=(
            "Use the job application lookup tool to check "
            "the following application:\n\nAPP001"
        ),

        expected_output=(
            "Application information returned by the "
            "Task 6 lookup tool."
        ),

        agent=demo_agent,
    )

    # Create demonstration Crew.
    demo_crew = Crew(
        agents=[
            demo_agent
        ],

        tasks=[
            demo_task
        ],

        process=Process.sequential,

        verbose=True,
    )

    # Run actual kickoff().
    result = demo_crew.kickoff()

    # Validate using Task 9 schema.
    validated_response = validate_crew_response(
        crew_result=result,
        query=demo_question,
        record_id="APP001",
    )

    print(
        "\nLookup demonstration validated result:"
    )

    if validated_response is not None:

        print(
            validated_response.model_dump()
        )


# ============================================================
# TASK 7 - COMPLETE CREW DEMONSTRATION
# ============================================================

def run_complete_crew():
    """
    Execute the complete three-agent sequential CrewAI workflow.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "COMPLETE THREE-AGENT CREW"
    )

    print(
        "=" * 60
    )

    # Example input demonstrates that the Crew can receive both
    # a policy query and an application ID in one kickoff context.
    inputs = {
        "rag_query": (
            "What degree is required for most professional jobs?"
        ),

        "user_question": (
            "What degree is required for most professional jobs?"
        ),

        "record_id": "APP001",
    }

    # Execute shared three-agent Crew.
    result = crew.kickoff(
        inputs=inputs
    )

    # Validate the final result.
    validated_response = validate_crew_response(
        crew_result=result,
        query=inputs["user_question"],
        record_id=inputs["record_id"],
    )

    print(
        "\nFinal Validated Response:"
    )

    if validated_response is not None:

        print(
            validated_response.model_dump()
        )

        return (
            validated_response.final_answer
        )

    print(
        "Crew response validation failed."
    )

    return None


# ============================================================
# TASK 8 - SESSION MEMORY DEMONSTRATION
# ============================================================

def demonstrate_session_memory():
    """
    Demonstrate both same-session memory and fresh-session isolation.

    Demonstration 1:
        APP001 is selected in Turn 1.
        Turn 2 refers indirectly to the same application.

    Demonstration 2:
        A new session asks the same indirect question.
        APP001 must not be inherited.
    """

    # ========================================================
    # SAME SESSION
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TASK 8 - TRANSCRIPT 1"
    )

    print(
        "SAME SESSION / TWO TURNS"
    )

    print(
        "=" * 60
    )

    # First session identifier.
    session_1 = (
        "student_session_001"
    )

    # First turn explicitly gives APP001.
    turn_1_question = (
        "What is the status of application APP001?"
    )

    print(
        "\nUSER - Turn 1:"
    )

    print(
        turn_1_question
    )

    # Execute first turn.
    turn_1_response = crew_with_memory.invoke(
        {
            "query": turn_1_question
        },
        config={
            "configurable": {
                "session_id": session_1
            }
        },
    )

    print(
        "\nCREW - Turn 1:"
    )

    print(
        turn_1_response
    )

    # Confirm APP001 was selected.
    if LAST_SELECTED_RECORD_ID != "APP001":

        raise AssertionError(
            "Task 8 failed: Turn 1 did not select APP001."
        )

    print(
        "\n[PASS] Turn 1 selected APP001."
    )

    # Second turn deliberately omits APP001.
    turn_2_question = (
        "What was the escalation score for that application?"
    )

    print(
        "\nUSER - Turn 2:"
    )

    print(
        turn_2_question
    )

    # Execute second turn using the SAME session.
    turn_2_response = crew_with_memory.invoke(
        {
            "query": turn_2_question
        },
        config={
            "configurable": {
                "session_id": session_1
            }
        },
    )

    print(
        "\nCREW - Turn 2:"
    )

    print(
        turn_2_response
    )

    # Verify memory recovered APP001.
    if LAST_SELECTED_RECORD_ID != "APP001":

        raise AssertionError(
            "Task 8 failed: Turn 2 did not recover APP001."
        )

    print(
        "\n[PASS] Turn 2 recovered APP001 from session memory."
    )

    # Read complete stored history.
    session_1_history = get_session_history(
        session_1
    )

    print(
        "\nSESSION 1 STORED HISTORY:"
    )

    for message in session_1_history.messages:

        print(
            f"{message.type}: "
            f"{message.content}"
        )

    # Count human/user messages.
    session_1_user_messages = sum(
        message.type == "human"
        for message in session_1_history.messages
    )

    # Two user turns must exist.
    if session_1_user_messages != 2:

        raise AssertionError(
            "Task 8 failed: session_1 does not contain "
            "exactly two user turns."
        )

    print(
        "[PASS] Session 1 contains both user turns."
    )

    # ========================================================
    # FRESH SESSION
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TASK 8 - TRANSCRIPT 2"
    )

    print(
        "FRESH SESSION / ONE TURN"
    )

    print(
        "=" * 60
    )

    # Different session ID.
    session_2 = (
        "student_session_002"
    )

    # No APP### appears in this message.
    fresh_question = (
        "What was the escalation score for that application?"
    )

    print(
        "\nUSER - Fresh Session:"
    )

    print(
        fresh_question
    )

    # Execute in a completely new session.
    fresh_response = crew_with_memory.invoke(
        {
            "query": fresh_question
        },
        config={
            "configurable": {
                "session_id": session_2
            }
        },
    )

    print(
        "\nCREW - Fresh Session:"
    )

    print(
        fresh_response
    )

    # The fresh session must not inherit APP001.
    if LAST_SELECTED_RECORD_ID is not None:

        raise AssertionError(
            "Task 8 failed: fresh session incorrectly "
            f"selected {LAST_SELECTED_RECORD_ID}."
        )

    print(
        "\n[PASS] Fresh session selected no application ID."
    )

    # Read Session 2 history.
    session_2_history = get_session_history(
        session_2
    )

    print(
        "\nSESSION 2 STORED HISTORY:"
    )

    for message in session_2_history.messages:

        print(
            f"{message.type}: "
            f"{message.content}"
        )

    # Count human messages in fresh session.
    session_2_user_messages = sum(
        message.type == "human"
        for message in session_2_history.messages
    )

    if session_2_user_messages != 1:

        raise AssertionError(
            "Task 8 failed: session_2 does not contain "
            "exactly one user turn."
        )

    print(
        "[PASS] Session 2 contains exactly one user turn."
    )

    print(
        "\n[PASS] Task 8 session-memory demonstration completed."
    )


# ============================================================
# MAIN DEMONSTRATION ENTRY POINT
# ============================================================

def main():
    """
    Run the Task 7, Task 8 and Task 9 demonstrations.

    Task 16's dedicated cache demonstration remains in
    response_cache.py so that it can independently prove:
        - normalization
        - cache miss
        - real RAG execution
        - cache hit
        - duplicate real-RAG call avoidance
    """

    # Print module heading.
    print(
        "=" * 60
    )

    print(
        "TASK 7 + TASK 8 + TASK 9 - CREWAI HR AUTOMATION"
    )

    print(
        "=" * 60
    )

    # Demonstrate actual RAG tool invocation.
    demonstrate_rag_tool()

    # Demonstrate actual application lookup.
    demonstrate_lookup_tool()

    # Demonstrate the complete three-agent workflow.
    run_complete_crew()

    # Demonstrate session memory.
    demonstrate_session_memory()


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

# Execute demonstrations only when this file is launched directly.
if __name__ == "__main__":

    main()
# ============================================================
# Task 7 + Task 8 + Task 9 + Task 16 - CrewAI HR Automation
# ============================================================
#
# This module implements:
#
# Task 7:
#   - Three CrewAI agents
#   - RAG retrieval tool
#   - Application lookup tool
#   - Sequential CrewAI workflow
#
# Task 8:
#   - Session-based conversation memory
#   - Same-session application-ID recovery
#   - Fresh-session isolation
#
# Task 9:
#   - Pydantic structured CrewResponse
#   - Explicit response validation
#
# Task 16:
#   - Live normalized-query response caching for the RAG tool
#   - The actual uncached RAG implementation is exposed through
#     _real_rag_search()
#   - The public CrewAI rag_search() wrapper calls the cache
#
# IMPORTANT:
# response_cache.py does NOT import crew_agents at module level.
# Instead, crew_agents.py registers _real_rag_search() with the
# response-cache module after the real function has been defined.
# This avoids circular imports.
# ============================================================


# ============================================================
# STANDARD-LIBRARY IMPORTS
# ============================================================

# inspect is used to inspect Python function/tool signatures when
# determining which argument a CrewAI tool accepts.
import inspect

# json is used to serialize Task 6 lookup results into JSON text
# and to create JSON Action Input strings for the MOCK_LLM.
import json

# re is used for application-ID extraction, message parsing,
# tool-action detection, and Composer-context parsing.
import re

# Optional is used because record_id can legitimately be absent
# for normal knowledge-base questions.
from typing import Optional

# ContextVar stores the current memory session ID separately for
# each execution context, which is useful for API and concurrent use.
from contextvars import ContextVar


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

# BaseModel defines the Pydantic schema required for Task 9.
from pydantic import BaseModel

# ValidationError allows Task 9 validation failures to be handled
# explicitly instead of crashing the complete workflow.
from pydantic import ValidationError

# Agent represents an individual CrewAI worker.
from crewai import Agent

# Crew represents the complete multi-agent CrewAI workflow.
from crewai import Crew

# Process provides the sequential processing mode required by
# the capstone architecture.
from crewai import Process

# Task represents the work assigned to each CrewAI agent.
from crewai import Task

# BaseLLM is the CrewAI base class required for the deterministic
# local MOCK_LLM implementation.
from crewai.llms.base_llm import BaseLLM

# The @tool decorator converts normal Python functions into
# CrewAI-callable tools.
from crewai.tools import tool

# InMemoryChatMessageHistory stores user/assistant messages for
# each Task 8 session.
from langchain_core.chat_history import InMemoryChatMessageHistory

# RunnableLambda wraps our memory-aware Python function as a
# LangChain runnable.
from langchain_core.runnables import RunnableLambda

# RunnableWithMessageHistory connects the runnable to the
# appropriate per-session message history.
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# PROJECT-MODULE IMPORTS
# ============================================================

# Reuse the original Task 6 lookup implementation instead of
# duplicating application-data and escalation-score logic.
from task6_tool import (
    check_job_application_status
    as task6_check_job_application_status
)

# Reuse the Task 3-5 RAG infrastructure for document loading,
# embeddings, ChromaDB, retrieval, and threshold calculation.
import rag_core

# Import the Task 16 response-cache module.
#
# The cache module does not import this file at module level,
# so this import does not create the circular dependency problem.
import response_cache


# ============================================================
# TASK 9 - STRUCTURED RESPONSE MODEL
# ============================================================

class CrewResponse(BaseModel):
    """
    Define the structured response returned by the CrewAI layer.

    This is the Task 9 response contract:
        final_answer -> final user-facing response text
        query        -> current user query
        record_id     -> selected application ID when applicable
    """

    # Store the final answer that should be shown to the user.
    final_answer: str

    # Store the current user question used by the crew.
    query: str

    # Store an application ID when a lookup occurred.
    # It remains None for normal HR knowledge-base questions.
    record_id: Optional[str] = None


# Task 9 requires a response-format variable that points to the
# Pydantic response schema.
response_format = CrewResponse


# ============================================================
# TASK 9 - RESPONSE VALIDATION
# ============================================================

def validate_crew_response(
    crew_result,
    query: str,
    record_id: Optional[str] = None
) -> Optional[CrewResponse]:
    """
    Validate an actual CrewAI kickoff result using Pydantic.

    Parameters:
        crew_result:
            Final result returned by CrewAI kickoff().

        query:
            Current user query.

        record_id:
            Selected application ID, when available.

    Returns:
        CrewResponse when validation succeeds.
        None when validation fails.
    """

    try:
        # Convert the CrewAI final result to text because the
        # user-facing answer must be a string.
        final_answer = str(crew_result)

        # Build the required structured Pydantic response.
        validated_response = response_format(
            final_answer=final_answer,
            query=query,
            record_id=record_id
        )

        # Print the validated structure as Task 9 evidence.
        print("\n[TASK 9] Validated CrewResponse:")
        print(validated_response.model_dump())

        return validated_response

    except ValidationError as error:
        # Report schema-validation problems without terminating
        # the whole Python process unexpectedly.
        print("\n[TASK 9] CrewResponse validation failed:")
        print(error)
        return None


# ============================================================
# TOOL RESULT STORAGE
# ============================================================

# Store the latest result produced by each CrewAI tool.
#
# The deterministic MOCK_LLM uses this dictionary to obtain the
# tool output when constructing the next assistant response.
LAST_TOOL_RESULT = {}


# ============================================================
# RAG STATE
# ============================================================

# Store whether the latest real RAG execution was sufficiently
# grounded according to the calibrated similarity threshold.
LAST_RAG_GROUNDED = None

# Store the latest top-1 similarity score for downstream
# guardrail and evaluation logic.
LAST_RAG_TOP_SIMILARITY = None

# Store the calibrated threshold used by the latest real RAG run.
LAST_RAG_THRESHOLD = None


# ============================================================
# RAG CHUNKING CONFIGURATION
# ============================================================

# Task 5 uses 200-character fixed-size chunks.
FIXED_CHUNK_SIZE = 200

# Task 5 uses 50-character overlap between fixed-size chunks.
FIXED_CHUNK_OVERLAP = 50


# ============================================================
# LIVE WORD-SAFE FIXED CHUNKING
# ============================================================

def build_word_safe_fixed_chunks(documents):
    """
    Build fixed-size RAG chunks while avoiding unnecessary
    mid-word boundaries.

    The Task 5 configuration remains:
        chunk size = 200 characters
        overlap    = 50 characters

    The live CrewAI implementation moves boundaries backward to
    whitespace when necessary so words are not cut in half.

    Parameters:
        documents:
            Knowledge-base documents returned by rag_core.

    Returns:
        List of dictionaries containing:
            id
            text
            source
    """

    # Store every generated chunk here.
    safe_chunks = []

    # Process each knowledge-base document independently.
    for document in documents:

        # Read the actual document body.
        text = document["text"]

        # Read the source/document name.
        source = document["source"]

        # Keep the required Task 5 chunk configuration.
        chunk_size = FIXED_CHUNK_SIZE
        overlap = FIXED_CHUNK_OVERLAP

        # Start at the beginning of the document.
        start = 0

        # Continue until the whole document has been processed.
        while start < len(text):

            # Calculate the normal fixed-size end boundary.
            end = min(
                start + chunk_size,
                len(text)
            )

            # If the chunk ends in the middle of a word,
            # move the boundary backward to the nearest space.
            if end < len(text):

                whitespace_position = text.rfind(
                    " ",
                    start,
                    end
                )

                if whitespace_position > start:
                    end = whitespace_position

            # Remove leading/trailing whitespace from the chunk.
            chunk_text = text[start:end].strip()

            # Only store non-empty chunks.
            if chunk_text:

                # Create a unique identifier required by
                # rag_core.store_chunks().
                chunk_id = (
                    f"{source}_fixed_{len(safe_chunks)}"
                )

                # Store the chunk in the structure expected by
                # the existing RAG storage implementation.
                safe_chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": source,
                    }
                )

            # Stop when the document has been completely consumed.
            if end >= len(text):
                break

            # Preserve the required 50-character overlap concept.
            next_start = max(
                0,
                end - overlap
            )

            # Align the next starting point to whitespace when
            # the overlap position lands inside a word.
            if next_start > 0:

                previous_whitespace_position = text.rfind(
                    " ",
                    0,
                    next_start
                )

                if previous_whitespace_position >= 0:

                    next_start = (
                        previous_whitespace_position + 1
                    )

                else:

                    next_start = 0

            # Prevent an infinite loop if the calculated overlap
            # does not move beyond the previous start point.
            if next_start <= start:
                next_start = end

            # Start the next chunk.
            start = next_start

    return safe_chunks


# ============================================================
# PREPARE RAG SYSTEM
# ============================================================

def prepare_rag_system():
    """
    Prepare the shared RAG resources used by the live CrewAI tool.

    The selected deployed collection is fixed_chunks.

    The collection is rebuilt using the current word-safe chunker
    so stale ChromaDB entries from an earlier broken chunking
    implementation cannot remain in the live collection.

    Returns:
        model
        fixed_collection
        calibrated_threshold
    """

    # Load the project's HR knowledge-base documents.
    documents = rag_core.load_documents(
        rag_core.KNOWLEDGE_BASE
    )

    # Build the live fixed-size chunks.
    fixed_chunks = build_word_safe_fixed_chunks(
        documents
    )

    # Load the same SentenceTransformers model used by the
    # existing Task 3-5 RAG implementation.
    model = rag_core.SentenceTransformer(
        rag_core.MODEL_NAME
    )

    # Open the persistent ChromaDB client.
    client = rag_core.chromadb.PersistentClient(
        path=rag_core.CHROMA_PATH
    )

    # Remove any previous fixed_chunks collection so stale
    # records cannot survive between runs.
    try:

        client.delete_collection(
            "fixed_chunks"
        )

    except Exception:
        # On the first execution the collection may not exist.
        # In that case, there is nothing to delete.
        pass

    # Create a fresh fixed_chunks collection.
    fixed_collection = rag_core.prepare_collection(
        client,
        "fixed_chunks"
    )

    # Store the newly generated word-safe fixed chunks.
    rag_core.store_chunks(
        fixed_collection,
        fixed_chunks,
        model
    )

    # Prepare the sentence-based collection as well because
    # Task 5's measured threshold process uses both collections.
    sentence_collection = rag_core.prepare_collection(
        client,
        "sentence_chunks"
    )

    # Build sentence chunks only when the collection is empty.
    if sentence_collection.count() == 0:

        # Reuse the original Task 5 sentence-based chunker.
        sentence_chunks = rag_core.build_chunks(
            documents,
            rag_core.sentence_based_chunks
        )

        # Store the sentence-based chunks.
        rag_core.store_chunks(
            sentence_collection,
            sentence_chunks,
            model
        )

    # Measure representative in-scope queries.
    in_scope_scores = rag_core.measure_queries(
        rag_core.IN_SCOPE_QUERIES,
        model,
        fixed_collection,
        sentence_collection
    )

    # Measure representative out-of-scope queries.
    out_of_scope_scores = rag_core.measure_queries(
        rag_core.OUT_OF_SCOPE_QUERIES,
        model,
        fixed_collection,
        sentence_collection
    )

    # Recalculate the calibrated threshold from those measured
    # values rather than hard-coding the threshold here.
    threshold = rag_core.choose_threshold(
        in_scope_scores,
        out_of_scope_scores
    )

    return (
        model,
        fixed_collection,
        threshold
    )


# ============================================================
# CREATE SHARED RAG RESOURCES
# ============================================================

# Prepare the model, selected collection, and calibrated threshold
# once when this module is imported.
RAG_MODEL, FIXED_COLLECTION, RAG_THRESHOLD = (
    prepare_rag_system()
)


# ============================================================
# TASK 16 - REAL / UNCACHED RAG IMPLEMENTATION
# ============================================================

def _real_rag_search(query: str) -> str:
    """
    Execute the actual underlying RAG retrieval operation.

    IMPORTANT:
        This function contains the REAL RAG work and does NOT
        perform response caching.

    Task 16's response_cache.py registers this function and calls
    it only when a normalized query is not already cached.

    Parameters:
        query:
            User query to search against the selected ChromaDB
            fixed_chunks collection.

    Returns:
        Retrieved knowledge-base text or the grounded fallback
        message when retrieval is insufficient.
    """

    # This print statement provides visible evidence that the
    # real RAG path actually executed.
    print("\n[RAG TOOL] REAL RAG SEARCH EXECUTION")
    print(f"[RAG TOOL] Query: {query}")

    # Execute the actual vector retrieval against the selected
    # fixed_chunks ChromaDB collection.
    results = rag_core.retrieve(
        FIXED_COLLECTION,
        query,
        RAG_MODEL,
        top_k=rag_core.TOP_K
    )

    # Update the RAG state globals for the latest execution.
    global LAST_RAG_GROUNDED
    global LAST_RAG_TOP_SIMILARITY
    global LAST_RAG_THRESHOLD

    # Store the threshold used for this execution.
    LAST_RAG_THRESHOLD = RAG_THRESHOLD

    # Handle the case where ChromaDB returns no results.
    if not results:

        LAST_RAG_GROUNDED = False
        LAST_RAG_TOP_SIMILARITY = None

        # Use the required grounded fallback.
        answer = (
            "I don't know based on the available knowledge base."
        )

        # Store the fallback so the deterministic MOCK_LLM can
        # consume the tool result.
        LAST_TOOL_RESULT["rag_search"] = answer

        return answer

    # Read the strongest retrieved similarity score.
    top_similarity = results[0]["similarity"]

    # Store the score for downstream evaluation and guardrails.
    LAST_RAG_TOP_SIMILARITY = top_similarity

    # A result is considered grounded when its top score meets
    # or exceeds the calibrated threshold.
    LAST_RAG_GROUNDED = (
        top_similarity >= RAG_THRESHOLD
    )

    # Refuse to answer when retrieval is not sufficiently grounded.
    if not LAST_RAG_GROUNDED:

        answer = (
            "I don't know based on the available knowledge base."
        )

        # Store the fallback result.
        LAST_TOOL_RESULT["rag_search"] = answer

        return answer

    # Build the user-visible retrieved evidence from the top-K results.
    retrieved_text = []

    # Format each result with source, similarity, and text.
    for result in results:

        source = result.get(
            "source",
            "unknown"
        )

        text = result.get(
            "text",
            ""
        )

        similarity = result.get(
            "similarity",
            0.0
        )

        retrieved_text.append(
            f"Source: {source}\n"
            f"Similarity: {similarity:.3f}\n"
            f"Text: {text}"
        )

    # Combine all retrieved results into a single tool output.
    answer = "\n\n".join(
        retrieved_text
    )

    # Store the real RAG result for the Composer/MOCK_LLM.
    LAST_TOOL_RESULT["rag_search"] = answer

    return answer


# ============================================================
# TASK 16 - REGISTER REAL RAG FUNCTION WITH CACHE
# ============================================================

# Tell response_cache.py which function represents the actual
# expensive/real RAG execution path.
#
# response_cache.py will now call this function on cache MISS
# and will avoid calling it on cache HIT.
response_cache.configure_real_rag_function(
    _real_rag_search
)


# ============================================================
# TASK 7 + TASK 16 - LIVE RAG TOOL
# ============================================================

@tool("rag_search")
def rag_search(query: str) -> str:
    """
    Search the HR knowledge base through the normalized-query cache.

    The cache is transparent to CrewAI:
        CrewAI calls rag_search(query)
        -> response_cache checks normalized query
        -> CACHE MISS -> _real_rag_search(query)
        -> CACHE HIT  -> previously stored result

    Application lookup is intentionally NOT routed through this
    cache; only grounded-generation/RAG requests are cached.
    """

    # Keep the external tool invocation visible in the terminal.
    print("\n[RAG TOOL] rag_search() was invoked.")

    # Let Task 16 decide whether the underlying real RAG function
    # must actually execute.
    cached_result = response_cache.cached_grounded_generation(
        query
    )

    # Store whichever result was returned so the deterministic
    # MOCK_LLM can consume the same output for both cache hits
    # and cache misses.
    LAST_TOOL_RESULT["rag_search"] = cached_result

    return cached_result


# ============================================================
# TASK 6 - APPLICATION LOOKUP TOOL WRAPPER
# ============================================================

@tool("check_job_application_status")
def lookup_job_application_status(
    record_id: str
) -> str:
    """
    Reuse the original Task 6 application lookup function.

    This tool is deliberately NOT cached because application data
    can change and a cached application result could become stale.

    Parameters:
        record_id:
            Application ID such as APP001.

    Returns:
        JSON-formatted application lookup result or a readable
        lookup error message.
    """

    # Print visible evidence showing that the privileged lookup
    # tool was invoked.
    print(
        "\n[LOOKUP TOOL] "
        "check_job_application_status() was invoked."
    )

    # Print the requested record ID.
    print(
        f"[LOOKUP TOOL] Record ID: {record_id}"
    )

    try:

        # Reuse the original Task 6 implementation.
        result = task6_check_job_application_status(
            record_id
        )

        # Convert the dictionary into readable JSON for CrewAI.
        answer = json.dumps(
            result,
            indent=2
        )

        # Save the lookup output for the deterministic MOCK_LLM.
        LAST_TOOL_RESULT[
            "check_job_application_status"
        ] = answer

        return answer

    except ValueError as error:

        # Convert invalid record errors into a safe tool response.
        answer = f"Lookup error: {error}"

        # Store the error result as the latest lookup output.
        LAST_TOOL_RESULT[
            "check_job_application_status"
        ] = answer

        return answer


# ============================================================
# CREWAI MESSAGE HELPERS
# ============================================================

def get_message_content(message) -> str:
    """
    Return message content from either a dictionary-like message
    or a CrewAI/LangChain message object.
    """

    # Handle dictionary-based message representations.
    if isinstance(message, dict):

        return str(
            message.get(
                "content",
                ""
            )
        )

    # Handle normal object-based messages.
    return str(
        getattr(
            message,
            "content",
            ""
        )
    )


def get_message_role(message) -> str:
    """
    Return the role of a message such as user or assistant.
    """

    # Handle dictionary-based messages.
    if isinstance(message, dict):

        return str(
            message.get(
                "role",
                ""
            )
        )

    # Handle object-based messages.
    return str(
        getattr(
            message,
            "role",
            ""
        )
    )


# ============================================================
# TOOL SCHEMA HELPERS
# ============================================================

def get_agent_tools(from_agent):
    """
    Return the tools actually assigned to a CrewAI agent.

    This supports Task 7 and Task 15 least-autonomy checks.
    """

    # No agent means no tools.
    if from_agent is None:
        return []

    # Read the CrewAI agent's tools attribute.
    tools = getattr(
        from_agent,
        "tools",
        None
    )

    # Return an empty list when no tools are configured.
    if not tools:
        return []

    return tools


def get_tool_argument_names(tool_object):
    """
    Inspect a CrewAI tool's declared argument schema.

    The implementation intentionally checks argument names rather
    than guessing tool purpose from the tool name alone.

    Returns:
        Set of argument names accepted by the tool.
    """

    # Store all discovered argument names here.
    argument_names = set()

    # Read the Pydantic argument schema used by modern CrewAI tools.
    args_schema = getattr(
        tool_object,
        "args_schema",
        None
    )

    if args_schema is not None:

        # Modern Pydantic uses model_fields.
        model_fields = getattr(
            args_schema,
            "model_fields",
            None
        )

        if model_fields:

            argument_names.update(
                model_fields.keys()
            )

        else:

            # Support older schema representations as well.
            old_fields = getattr(
                args_schema,
                "__fields__",
                None
            )

            if old_fields:

                argument_names.update(
                    old_fields.keys()
                )

    # If no schema fields were found, inspect the underlying
    # Python callable as a fallback.
    if not argument_names:

        function = getattr(
            tool_object,
            "func",
            None
        )

        if function is None:

            function = getattr(
                tool_object,
                "_run",
                None
            )

        if callable(function):

            try:

                signature = inspect.signature(
                    function
                )

                argument_names.update(
                    signature.parameters.keys()
                )

            except (TypeError, ValueError):

                # Some framework callables do not expose a
                # Python signature. In that case leave the set empty.
                pass

    return argument_names


def find_tool_by_argument(
    from_agent,
    argument_name
):
    """
    Find the assigned tool that accepts a specified argument.

    Example:
        query     -> RAG tool
        record_id -> application lookup tool

    This supports safer tool dispatch than relying only on names.
    """

    # Read all tools assigned to the agent.
    tools = get_agent_tools(
        from_agent
    )

    # Inspect each tool's actual argument schema.
    for tool_object in tools:

        argument_names = get_tool_argument_names(
            tool_object
        )

        # Return the first tool accepting the requested argument.
        if argument_name in argument_names:

            return tool_object

    return None


def get_tool_name(tool_object):
    """
    Return the registered CrewAI name of a tool.
    """

    # CrewAI stores the public tool name in the name attribute.
    return str(
        getattr(
            tool_object,
            "name",
            ""
        )
    )


# ============================================================
# INPUT HELPERS
# ============================================================

def extract_record_id(text: str):
    """
    Extract an application ID such as APP001 from arbitrary text.

    The regular expression intentionally requires:
        APP + exactly three digits
    """

    # Convert the input to uppercase so APP001 and app001 both work.
    upper_text = text.upper()

    # Search for an application ID using word boundaries.
    match = re.search(
        r"\bAPP\d{3}\b",
        upper_text
    )

    # Return the normalized application ID when found.
    if match:

        return match.group(0)

    return None


def extract_clean_rag_query(text: str) -> str:
    """
    Remove surrounding CrewAI task wording from a RAG query.

    This keeps the retrieval query focused on the actual question.
    """

    # The evaluator/agent task can contain a "question:" marker.
    marker = "question:"

    # Use a lowercase copy only for locating the marker.
    lower_text = text.lower()

    # Find the beginning of the question.
    marker_position = lower_text.find(
        marker
    )

    if marker_position != -1:

        # Extract everything after the marker.
        question = text[
            marker_position + len(marker):
        ].strip()

        # Remove known task-instruction suffixes when present.
        ending_markers = [
            "This is the expected criteria",
            "Begin!",
            "Expected Output:"
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

    # When no marker is present, use the original text.
    return text.strip()


def get_latest_user_message(messages):
    """
    Return the most recent user-authored message.
    """

    # Search messages from newest to oldest.
    for message in reversed(messages):

        # Only user messages are relevant here.
        if get_message_role(message) == "user":

            return get_message_content(
                message
            )

    # Return an empty string if no user message exists.
    return ""


# ============================================================
# PREVIOUS ACTION CHECK
# ============================================================

def has_previous_action(
    messages,
    tool_name
):
    """
    Determine whether the current agent already requested a tool.

    IMPORTANT:
        The implementation deliberately does NOT search for a
        generic "Observation:" string because CrewAI's system
        prompt itself can contain that word.

    Instead, only assistant-authored Action lines are checked.
    """

    # Examine every available message.
    for message in messages:

        # Only assistant-authored messages can represent the
        # current agent's requested action.
        if get_message_role(message) != "assistant":

            continue

        # Read the assistant message text.
        content = get_message_content(
            message
        )

        # Match the exact Action line generated by our MOCK_LLM.
        action_pattern = (
            rf"(?m)^\s*Action:\s*"
            rf"{re.escape(tool_name)}\s*$"
        )

        # Return True when the requested tool action already exists.
        if re.search(
            action_pattern,
            content
        ):

            return True

    return False


# ============================================================
# COMPOSER HELPERS
# ============================================================

def extract_composer_question(
    from_task,
    fallback_message: str
) -> str:
    """
    Extract the current user question from the Composer task.

    The Composer task is treated as the authoritative source for
    the current user question.
    """

    # Read the current task description safely.
    task_description = str(
        getattr(
            from_task,
            "description",
            ""
        )
    )

    # Find the User question section.
    question_match = re.search(
        r"User question:\s*(.*?)"
        r"(?:\n\s*Application record:|\Z)",
        task_description,
        flags=re.DOTALL
    )

    # Return the extracted question when one exists.
    if question_match:

        question = question_match.group(1).strip()

        if question:

            return question

    # Fall back to the latest user message.
    return fallback_message.strip()


def extract_composer_record_id(
    from_task
) -> Optional[str]:
    """
    Extract the application ID explicitly supplied to the Composer.

    This also supports an application ID recovered by Task 8
    session memory.
    """

    # Read the Composer task description.
    task_description = str(
        getattr(
            from_task,
            "description",
            ""
        )
    )

    # Find an APP### value in the Application record field.
    record_match = re.search(
        r"Application record:\s*(APP\d{3})",
        task_description,
        flags=re.IGNORECASE
    )

    # Return the normalized ID when found.
    if record_match:

        return record_match.group(1).upper()

    return None


def extract_lookup_json(
    context_text: str
):
    """
    Extract a flat Task 6 lookup JSON object from Composer context.

    The lookup schema contains record_id, which allows this helper
    to distinguish structured application evidence from other text.
    """

    # Locate simple JSON objects containing an APP### record ID.
    candidates = re.findall(
        r'\{[^{}]*"record_id"\s*:\s*"APP\d{3}"[^{}]*\}',
        context_text,
        flags=re.DOTALL
    )

    # Try to parse every possible candidate.
    for candidate in candidates:

        try:

            data = json.loads(
                candidate
            )

            # Ensure the result is a dictionary containing a
            # recognizable application ID.
            if (
                isinstance(data, dict)
                and extract_record_id(
                    str(
                        data.get(
                            "record_id",
                            ""
                        )
                    )
                )
            ):

                return data

        except json.JSONDecodeError:

            # Ignore malformed candidate fragments and continue.
            continue

    return None


def extract_first_retrieved_text(
    context_text: str
) -> Optional[str]:
    """
    Extract the first useful RAG Text field from previous-agent context.

    The Composer uses this to return the knowledge content while
    avoiding exposure of raw similarity metadata.
    """

    # Find the first Text: field and stop at the next source block.
    match = re.search(
        r"Text:\s*(.*?)(?=\n\nSource:|\Z)",
        context_text,
        flags=re.DOTALL
    )

    # Return the cleaned text when found.
    if match:

        text = match.group(1).strip()

        if text:

            return text

    return None


# ============================================================
# TASK 7 - DETERMINISTIC MOCK LLM
# ============================================================

class MOCK_LLM(BaseLLM):
    """
    Deterministic local CrewAI model used for reproducible demos.

    No external commercial LLM API is called.

    The model emulates the small set of CrewAI actions needed by
    this capstone:
        1. Retrieval Agent -> RAG tool call
        2. Lookup Agent    -> application lookup tool call
        3. Response Agent  -> deterministic final answer
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
        **kwargs
    ):
        """
        Decide whether the current agent should call a tool or
        produce its final deterministic answer.
        """

        # Normalize a possible None value into a list.
        if messages is None:

            messages = []

        # Make a local list copy so inspection does not mutate
        # CrewAI's original message collection.
        messages = list(messages)

        # Retrieve the most recent user message.
        latest_user_message = get_latest_user_message(
            messages
        )

        # ====================================================
        # CASE 1 - RETRIEVAL AGENT
        # ====================================================

        # Look for an assigned tool whose schema accepts "query".
        query_tool = find_tool_by_argument(
            from_agent,
            "query"
        )

        if query_tool is not None:

            # Read the actual CrewAI tool name.
            query_tool_name = get_tool_name(
                query_tool
            )

            # Request the RAG tool only once.
            if not has_previous_action(
                messages,
                query_tool_name
            ):

                # Remove surrounding task wording from the query.
                clean_query = extract_clean_rag_query(
                    latest_user_message
                )

                # Return the deterministic ReAct action expected
                # by the configured CrewAI version.
                return (
                    "Thought: I need to search the "
                    "HR knowledge base.\n"
                    f"Action: {query_tool_name}\n"
                    f"Action Input: "
                    f"{json.dumps({'query': clean_query})}"
                )

            # Read the latest RAG output after the tool executes.
            rag_result = LAST_TOOL_RESULT.get(
                query_tool_name,
                "No RAG result was stored."
            )

            # Finish the Retrieval Agent task with that result.
            return (
                "Thought: I have retrieved the relevant "
                "knowledge-base information.\n"
                f"Final Answer: {rag_result}"
            )

        # ====================================================
        # CASE 2 - LOOKUP AGENT
        # ====================================================

        # Look for an assigned tool accepting "record_id".
        record_tool = find_tool_by_argument(
            from_agent,
            "record_id"
        )

        if record_tool is not None:

            # Read the actual CrewAI tool name.
            record_tool_name = get_tool_name(
                record_tool
            )

            # Request lookup only once.
            if not has_previous_action(
                messages,
                record_tool_name
            ):

                # Extract an application ID from the current task.
                record_id = extract_record_id(
                    latest_user_message
                )

                # When no record ID exists, return the required
                # missing-ID message instead of inventing data.
                if record_id is None:

                    return (
                        "Thought: I do not have a valid "
                        "application record ID.\n"
                        "Final Answer: Please provide an "
                        "application ID such as APP123."
                    )

                # Return the deterministic application lookup action.
                return (
                    "Thought: I need to look up the "
                    "application record.\n"
                    f"Action: {record_tool_name}\n"
                    f"Action Input: "
                    f"{json.dumps({'record_id': record_id})}"
                )

            # Read the actual lookup result after tool execution.
            lookup_result = LAST_TOOL_RESULT.get(
                record_tool_name,
                "No lookup result was stored."
            )

            # Finish the Lookup Agent task.
            return (
                "Thought: I have the application lookup "
                "information.\n"
                f"Final Answer: {lookup_result}"
            )

        # ====================================================
        # CASE 3 - RESPONSE COMPOSER
        # ====================================================

        # Extract the actual current question from the Composer task.
        current_question = extract_composer_question(
            from_task,
            latest_user_message
        )

        # Convert the question to lowercase for deterministic
        # intent matching.
        question_lower = current_question.lower()

        # The Composer task is the authoritative location for the
        # application ID, including one recovered by Task 8 memory.
        selected_record_id = extract_composer_record_id(
            from_task
        )

        # ----------------------------------------------------
        # LOCATE PREVIOUS-AGENT CONTEXT
        # ----------------------------------------------------

        # Search user messages for CrewAI's injected context block.
        context_text = ""

        for message in messages:

            if get_message_role(message) != "user":

                continue

            content = get_message_content(
                message
            )

            if (
                "This is the context you're working with:"
                in content
            ):

                context_text = content

        # ----------------------------------------------------
        # REMOVE CREWAI CONTEXT MARKER
        # ----------------------------------------------------

        if context_text:

            # Keep only the useful context after the marker.
            parts = context_text.split(
                "This is the context you're working with:",
                1
            )

            combined_context = (
                parts[1].strip()
                if len(parts) == 2
                else context_text
            )

        else:

            combined_context = ""

        # ====================================================
        # LOOKUP RESULT
        # ====================================================

        # Search for structured Task 6 application data.
        lookup_data = extract_lookup_json(
            combined_context
        )

        # Only use lookup data when the current Composer task
        # contains a genuine application ID.
        #
        # This prevents a missing-ID Lookup Agent result from
        # overriding a valid RAG response.
        if (
            lookup_data is not None
            and selected_record_id is not None
        ):

            # Read the individual fields returned by Task 6.
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

            # --------------------------------------------
            # STATUS QUESTION
            # --------------------------------------------

            if "status" in question_lower:

                return (
                    "Thought: I have the application "
                    "lookup information needed to answer "
                    "the user's question.\n"
                    "Final Answer: "
                    f"Application {record_id} has status "
                    f"{status}."
                )

            # --------------------------------------------
            # ESCALATION SCORE QUESTION
            # --------------------------------------------

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

            # --------------------------------------------
            # ESCALATION RECOMMENDATION QUESTION
            # --------------------------------------------

            if (
                "escalation" in question_lower
                and (
                    "recommend" in question_lower
                    or "priority" in question_lower
                    or "should" in question_lower
                )
                and escalation_recommended is not None
            ):

                # Convert the Task 6 boolean into a readable answer.
                recommendation_text = (
                    "recommended for higher-priority escalation"
                    if bool(escalation_recommended)
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

            # --------------------------------------------
            # CANDIDATE NAME QUESTION
            # --------------------------------------------

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

            # --------------------------------------------
            # SALARY QUESTION
            # --------------------------------------------

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

            # --------------------------------------------
            # GENERIC APPLICATION QUESTION
            # --------------------------------------------

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
        # APPLICATION LOOKUP WITHOUT RECORD ID
        # ====================================================

        # A generic word such as "candidate" must NOT automatically
        # trigger application lookup.
        #
        # Example:
        #   "How much notice should a candidate get before an interview?"
        #
        # is a normal HR policy question.
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

        # If an application-style question has no selected record
        # and the Lookup Agent explicitly reported missing ID,
        # return the required request for an application ID.
        if (
            application_question
            and selected_record_id is None
            and "Please provide an application ID"
            in combined_context
        ):

            return (
                "Thought: An application ID is required "
                "for this lookup.\n"
                "Final Answer: Please provide an "
                "application ID such as APP123."
            )

        # ====================================================
        # GENERAL RAG ANSWER
        # ====================================================

        # The latest RAG tool result is authoritative for normal
        # knowledge-base questions.
        rag_result = LAST_TOOL_RESULT.get(
            "rag_search",
            ""
        )

        # Normalize it into a clean string.
        rag_result = str(
            rag_result
        ).strip()

        # Continue when a RAG tool result exists.
        if rag_result:

            # -----------------------------------------------
            # RAG FALLBACK
            # -----------------------------------------------

            if (
                rag_result
                == "I don't know based on the available knowledge base."
            ):

                return (
                    "Thought: The knowledge base does not "
                    "contain enough information to answer "
                    "this question.\n"
                    "Final Answer: "
                    "I don't know based on the available "
                    "knowledge base."
                )

            # -----------------------------------------------
            # RETURN FIRST RETRIEVED TEXT
            # -----------------------------------------------

            # Extract only the useful retrieved text so the
            # Composer does not expose internal similarity scores.
            retrieved_text = extract_first_retrieved_text(
                combined_context
            )

            if retrieved_text:

                return (
                    "Thought: I have the relevant "
                    "knowledge-base information needed "
                    "to answer the question.\n"
                    "Final Answer: "
                    f"{retrieved_text}"
                )

            # As a safe fallback, return the actual stored RAG result.
            return (
                "Thought: I have retrieved relevant "
                "knowledge-base information.\n"
                "Final Answer: "
                f"{rag_result}"
            )

        # ====================================================
        # FINAL FALLBACK
        # ====================================================

        # If neither lookup nor RAG supplied usable information,
        # fail closed with the knowledge-base fallback.
        return (
            "Thought: No usable information was returned "
            "by the previous agents.\n"
            "Final Answer: "
            "I don't know based on the available "
            "knowledge base."
        )

    def supports_function_calling(self) -> bool:
        """
        Report that this MOCK_LLM uses text-based ReAct-style
        tool requests rather than native function calling.
        """

        # The custom model intentionally returns textual
        # Thought/Action/Action Input instructions.
        return False


# ============================================================
# CREATE MOCK LLM
# ============================================================

# Create one shared deterministic mock model instance.
mock_llm = MOCK_LLM(
    model="mock-llm"
)


# ============================================================
# TASK 7 - CREATE AGENTS
# ============================================================

def create_agents():
    """
    Create the three required Task 7 CrewAI agents.

    Tool ownership is intentionally restricted:
        Retrieval Agent -> rag_search only
        Lookup Agent    -> application lookup only
        Composer Agent  -> no tools
    """

    # --------------------------------------------------------
    # RETRIEVAL AGENT
    # --------------------------------------------------------

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

        # Only the RAG tool is available to this agent.
        tools=[rag_search],

        # Use the deterministic local model.
        llm=mock_llm,

        # Prevent delegation to another agent.
        allow_delegation=False,

        # Enable execution logging for demonstrations.
        verbose=True
    )

    # --------------------------------------------------------
    # LOOKUP AGENT
    # --------------------------------------------------------

    lookup_agent = Agent(
        role="Job Application Lookup Agent",

        goal=(
            "Retrieve factual information about a job "
            "application using the Task 6 lookup tool."
        ),

        backstory=(
            "You retrieve factual application information "
            "using the provided Task 6 lookup tool and "
            "never invent application data."
        ),

        # This agent receives only the privileged application
        # lookup tool required by Task 6.
        tools=[lookup_job_application_status],

        # Use the same deterministic mock model.
        llm=mock_llm,

        # Prevent delegation.
        allow_delegation=False,

        # Enable verbose execution evidence.
        verbose=True
    )

    # --------------------------------------------------------
    # RESPONSE COMPOSER
    # --------------------------------------------------------

    composer_agent = Agent(
        role="HR Response Composer",

        goal=(
            "Combine the outputs from the Retrieval Agent "
            "and Lookup Agent into one clear final factual answer."
        ),

        backstory=(
            "You are the final response writer. Use only the "
            "information returned by the previous agents. "
            "Answer the user's current question directly. "
            "Do not expose internal tool instructions, raw "
            "similarity metadata, raw JSON, or CrewAI control text. "
            "Do not invent information."
        ),

        # The Composer intentionally has no tools.
        tools=[],

        # Use the deterministic mock model.
        llm=mock_llm,

        # Prevent delegation.
        allow_delegation=False,

        # Enable verbose execution evidence.
        verbose=True
    )

    # Return all three agents in the required order.
    return (
        retrieval_agent,
        lookup_agent,
        composer_agent
    )


# ============================================================
# TASK 7 - CREATE TASKS
# ============================================================

def create_tasks(
    retrieval_agent,
    lookup_agent,
    composer_agent
):
    """
    Create the three sequential CrewAI tasks.

    Task order:
        1. Retrieval
        2. Application Lookup
        3. Response Composition
    """

    # --------------------------------------------------------
    # RETRIEVAL TASK
    # --------------------------------------------------------

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

        # Assign this task only to the Retrieval Agent.
        agent=retrieval_agent
    )

    # --------------------------------------------------------
    # LOOKUP TASK
    # --------------------------------------------------------

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

        # Assign this task only to the Lookup Agent.
        agent=lookup_agent
    )

    # --------------------------------------------------------
    # COMPOSER TASK
    # --------------------------------------------------------

    composer_task = Task(
        description=(
            "Create one final factual answer using the outputs "
            "produced by the Retrieval Agent and Lookup Agent.\n\n"
            "User question:\n"
            "{user_question}\n\n"
            "Application record:\n"
            "{record_id}\n\n"
            "Combine the relevant information from both previous "
            "agents into one clear answer. Answer the CURRENT "
            "user question only. Do not output raw CrewAI "
            "instructions, similarity metadata, raw JSON, or "
            "internal formatting."
        ),

        expected_output=(
            "One clear final answer directly answering the "
            "current user's question using only information "
            "returned by the previous agents."
        ),

        # Assign composition to the Composer Agent.
        agent=composer_agent,

        # Give the Composer access to both previous task outputs.
        context=[
            retrieval_task,
            lookup_task
        ]
    )

    # Return tasks in execution order.
    return (
        retrieval_task,
        lookup_task,
        composer_task
    )


# ============================================================
# TASK 7 - CREATE MAIN CREW
# ============================================================

def create_main_crew():
    """
    Create the required three-agent sequential CrewAI workflow.
    """

    # Create the three agents.
    (
        retrieval_agent,
        lookup_agent,
        composer_agent
    ) = create_agents()

    # Create the three tasks assigned to those agents.
    (
        retrieval_task,
        lookup_task,
        composer_task
    ) = create_tasks(
        retrieval_agent,
        lookup_agent,
        composer_agent
    )

    # Construct the CrewAI workflow.
    new_crew = Crew(
        agents=[
            retrieval_agent,
            lookup_agent,
            composer_agent
        ],

        tasks=[
            retrieval_task,
            lookup_task,
            composer_task
        ],

        # Task execution must happen in strict sequence.
        process=Process.sequential,

        # Enable verbose execution for demonstrations.
        verbose=True
    )

    return new_crew


# ============================================================
# MODULE-LEVEL CREW
# ============================================================

# Create the shared Crew used by the API and demonstrations.
crew = create_main_crew()


# ============================================================
# TASK 8 - SESSION MEMORY
# ============================================================

# Store one LangChain message history object per session ID.
SESSION_STORE = {}

# Store the currently selected application ID independently for
# each API/session context.
SESSION_SELECTED_RECORD_IDS = {}

# Store the most recently selected application ID for demonstration
# assertions such as Task 8.
LAST_SELECTED_RECORD_ID = None

# Bridge the current API/request execution to the corresponding
# memory session ID without relying on one global session value.
CURRENT_MEMORY_SESSION_ID = ContextVar(
    "current_memory_session_id",
    default=None
)


# ============================================================
# TASK 8 - SESSION HISTORY
# ============================================================

def get_session_history(
    session_id: str
) -> InMemoryChatMessageHistory:
    """
    Return the existing history for a session or create it.

    Parameters:
        session_id:
            Logical conversation/session identifier.

    Returns:
        InMemoryChatMessageHistory for that session.
    """

    # Create the history object the first time a session is seen.
    if session_id not in SESSION_STORE:

        SESSION_STORE[session_id] = (
            InMemoryChatMessageHistory()
        )

    # Return the session-specific history.
    return SESSION_STORE[session_id]


def get_history_text(history) -> str:
    """
    Convert stored LangChain history messages into readable text.
    """

    # Prepare a list for formatted history lines.
    history_lines = []

    # Read the message list from the history object.
    messages = getattr(
        history,
        "messages",
        history
    )

    # Convert each stored message into role/content text.
    for message in messages:

        role = getattr(
            message,
            "type",
            "message"
        )

        content = getattr(
            message,
            "content",
            ""
        )

        history_lines.append(
            f"{role}: {content}"
        )

    # Return the complete session history as one string.
    return "\n\n".join(
        history_lines
    )


# ============================================================
# TASK 8 - MEMORY-AWARE CREW EXECUTION
# ============================================================

def run_memory_aware_crew(data):
    """
    Execute the existing Task 7 Crew through Task 8 session memory.

    Selection priority:
        1. APP### present in the current message
        2. APP### found in the same-session history
        3. None

    Only application-ID selection uses conversation history.
    The actual RAG query remains based on the current message only.
    """

    # Read the current user message.
    current_message = str(
        data.get(
            "query",
            ""
        )
    )

    # Read the history supplied by RunnableWithMessageHistory.
    history = data.get(
        "history",
        []
    )

    # Convert stored history into text for application-ID extraction.
    history_text = get_history_text(
        history
    )

    # First try to extract an application ID from the current turn.
    record_id = extract_record_id(
        current_message
    )

    # If the current turn contains no ID, search this same session's
    # previous messages.
    if record_id is None:

        record_id = extract_record_id(
            history_text
        )

    # Use an empty string in the CrewAI task when no application ID
    # is available so Task 6 can explicitly report that one is needed.
    if record_id is None:

        record_id = ""

    # Update the demonstration state.
    global LAST_SELECTED_RECORD_ID

    LAST_SELECTED_RECORD_ID = (
        record_id
        if record_id
        else None
    )

    # Read the current request's memory-session ID.
    current_session_id = (
        CURRENT_MEMORY_SESSION_ID.get()
    )

    # Save the selected record ID under the current session.
    if current_session_id is not None:

        SESSION_SELECTED_RECORD_IDS[
            str(current_session_id)
        ] = (
            record_id
            if record_id
            else None
        )

    # Print visible Task 8 evidence.
    print("\n[SESSION MEMORY]")
    print(
        f"Current message: {current_message}"
    )

    print(
        "Record ID selected: "
        f"{record_id if record_id else 'None'}"
    )

    # Execute the existing three-agent Crew.
    result = crew.kickoff(
        inputs={
            "rag_query": current_message,
            "user_question": current_message,
            "record_id": record_id
        }
    )

    # Validate the final result through the Task 9 schema.
    validated_response = validate_crew_response(
        crew_result=result,
        query=current_message,
        record_id=(
            record_id
            if record_id
            else None
        )
    )

    # Return only the final user-facing answer.
    if validated_response is not None:

        return validated_response.final_answer

    # Fail safely if schema validation did not succeed.
    return "Crew response could not be validated."


# ============================================================
# LANGCHAIN MEMORY RUNNABLE
# ============================================================

# Wrap the memory-aware execution function as a LangChain runnable.
memory_aware_crew_runnable = RunnableLambda(
    run_memory_aware_crew
)


# ============================================================
# TASK 8 - RUNNABLE WITH MESSAGE HISTORY
# ============================================================

# Connect the runnable with per-session message history.
crew_with_memory = RunnableWithMessageHistory(
    memory_aware_crew_runnable,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)


# ============================================================
# TASK 7 - RAG DEMONSTRATION
# ============================================================

def demonstrate_rag_tool():
    """
    Demonstrate actual RAG tool invocation through CrewAI kickoff().
    """

    # Print a clear demonstration heading.
    print("\n")
    print("=" * 60)
    print("DEMONSTRATION 1 - RAG TOOL")
    print("=" * 60)

    # Use one of the measured in-scope knowledge-base questions.
    demo_question = (
        "What degree is required for most professional jobs?"
    )

    # Create a dedicated Retrieval Agent for the demonstration.
    demo_agent = Agent(
        role="HR Knowledge Retrieval Agent",

        goal=(
            "Retrieve HR information using the RAG tool."
        ),

        backstory=(
            "Use the RAG tool and do not invent information."
        ),

        # Give the demo agent only the RAG tool.
        tools=[rag_search],

        # Use the shared deterministic model.
        llm=mock_llm,

        # Prevent delegation.
        allow_delegation=False,

        # Enable visible execution output.
        verbose=True
    )

    # Create the demonstration task.
    demo_task = Task(
        description=(
            "Use the RAG tool to answer this HR question:\n\n"
            f"{demo_question}"
        ),

        expected_output=(
            "Relevant HR information retrieved from the "
            "knowledge base."
        ),

        agent=demo_agent
    )

    # Create a one-agent demonstration Crew.
    demo_crew = Crew(
        agents=[demo_agent],

        tasks=[demo_task],

        process=Process.sequential,

        verbose=True
    )

    # Execute the actual CrewAI tool path.
    result = demo_crew.kickoff()

    # Validate the result with the Task 9 schema.
    validated_response = validate_crew_response(
        crew_result=result,
        query=demo_question
    )

    # Print the structured validation evidence.
    print("\nRAG demonstration validated result:")

    if validated_response is not None:

        print(
            validated_response.model_dump()
        )


# ============================================================
# TASK 7 - LOOKUP DEMONSTRATION
# ============================================================

def demonstrate_lookup_tool():
    """
    Demonstrate actual Task 6 application lookup through kickoff().
    """

    # Print a clear demonstration heading.
    print("\n")
    print("=" * 60)
    print("DEMONSTRATION 2 - LOOKUP TOOL")
    print("=" * 60)

    # Use APP001 as the deterministic demonstration record.
    demo_question = (
        "What is the status of application APP001?"
    )

    # Create a dedicated Lookup Agent.
    demo_agent = Agent(
        role="Job Application Lookup Agent",

        goal=(
            "Retrieve application information using the "
            "Task 6 lookup tool."
        ),

        backstory=(
            "Use the lookup tool and do not invent application data."
        ),

        # Give this agent only the privileged application lookup.
        tools=[lookup_job_application_status],

        # Use the deterministic model.
        llm=mock_llm,

        # Prevent delegation.
        allow_delegation=False,

        # Enable visible execution output.
        verbose=True
    )

    # Create the application lookup demonstration task.
    demo_task = Task(
        description=(
            "Use the job application lookup tool to check "
            "the following application:\n\nAPP001"
        ),

        expected_output=(
            "Application information returned by the "
            "Task 6 lookup tool."
        ),

        agent=demo_agent
    )

    # Create a one-agent demonstration Crew.
    demo_crew = Crew(
        agents=[demo_agent],

        tasks=[demo_task],

        process=Process.sequential,

        verbose=True
    )

    # Execute the actual lookup path.
    result = demo_crew.kickoff()

    # Validate the lookup result through Task 9.
    validated_response = validate_crew_response(
        crew_result=result,
        query=demo_question,
        record_id="APP001"
    )

    # Print the validated evidence.
    print("\nLookup demonstration validated result:")

    if validated_response is not None:

        print(
            validated_response.model_dump()
        )


# ============================================================
# TASK 7 - COMPLETE CREW DEMONSTRATION
# ============================================================

def run_complete_crew():
    """
    Execute the complete three-agent Task 7 workflow.
    """

    # Print a clear demonstration heading.
    print("\n")
    print("=" * 60)
    print("COMPLETE THREE-AGENT CREW")
    print("=" * 60)

    # Build one deterministic three-agent input example.
    inputs = {
        "rag_query": (
            "What degree is required for most professional jobs?"
        ),

        "user_question": (
            "What degree is required for most professional jobs?"
        ),

        "record_id": "APP001"
    }

    # Execute the shared Crew.
    result = crew.kickoff(
        inputs=inputs
    )

    # Validate the final output using Task 9.
    validated_response = validate_crew_response(
        crew_result=result,
        query=inputs["user_question"],
        record_id=inputs["record_id"]
    )

    # Print the final structured response.
    print("\nFinal Validated Response:")

    if validated_response is not None:

        print(
            validated_response.model_dump()
        )

        return validated_response.final_answer

    # Report validation failure.
    print(
        "Crew response validation failed."
    )

    return None


# ============================================================
# TASK 8 - SESSION MEMORY DEMONSTRATION
# ============================================================

def demonstrate_session_memory():
    """
    Demonstrate:
        1. same-session application-ID recovery
        2. fresh-session isolation
    """

    # --------------------------------------------------------
    # SAME SESSION
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TASK 8 - TRANSCRIPT 1")
    print("SAME SESSION / TWO TURNS")
    print("=" * 60)

    # Define the first demonstration session.
    session_1 = "student_session_001"

    # First turn explicitly mentions APP001.
    turn_1_question = (
        "What is the status of application APP001?"
    )

    print("\nUSER - Turn 1:")
    print(turn_1_question)

    # Execute Turn 1 with the same session ID.
    turn_1_response = crew_with_memory.invoke(
        {
            "query": turn_1_question
        },
        config={
            "configurable": {
                "session_id": session_1
            }
        }
    )

    print("\nCREW - Turn 1:")
    print(turn_1_response)

    # Assert that Turn 1 selected APP001.
    if LAST_SELECTED_RECORD_ID != "APP001":

        raise AssertionError(
            "Task 8 failed: Turn 1 did not select APP001."
        )

    print(
        "\n[PASS] Turn 1 selected APP001."
    )

    # Second turn refers indirectly to the previous application.
    turn_2_question = (
        "What was the escalation score for that application?"
    )

    print("\nUSER - Turn 2:")
    print(turn_2_question)

    # Execute Turn 2 using the same session ID.
    turn_2_response = crew_with_memory.invoke(
        {
            "query": turn_2_question
        },
        config={
            "configurable": {
                "session_id": session_1
            }
        }
    )

    print("\nCREW - Turn 2:")
    print(turn_2_response)

    # Assert that APP001 was recovered from the same session.
    if LAST_SELECTED_RECORD_ID != "APP001":

        raise AssertionError(
            "Task 8 failed: Turn 2 did not recover APP001."
        )

    print(
        "\n[PASS] Turn 2 recovered APP001 from session memory."
    )

    # Retrieve the complete Session 1 history.
    session_1_history = get_session_history(
        session_1
    )

    print("\nSESSION 1 STORED HISTORY:")

    # Display the stored history for evidence.
    for message in session_1_history.messages:

        print(
            f"{message.type}: "
            f"{message.content}"
        )

    # Count user messages.
    session_1_user_messages = sum(
        message.type == "human"
        for message in session_1_history.messages
    )

    # Task 8 requires exactly two user turns in Session 1.
    if session_1_user_messages != 2:

        raise AssertionError(
            "Task 8 failed: session_1 does not contain "
            "exactly two user turns."
        )

    print(
        "[PASS] Session 1 contains both user turns."
    )

    # --------------------------------------------------------
    # FRESH SESSION
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TASK 8 - TRANSCRIPT 2")
    print("FRESH SESSION / ONE TURN")
    print("=" * 60)

    # Define a completely different session.
    session_2 = "student_session_002"

    # This question contains no application ID.
    fresh_question = (
        "What was the escalation score for that application?"
    )

    print("\nUSER - Fresh Session:")
    print(fresh_question)

    # Execute the same question in the new session.
    fresh_response = crew_with_memory.invoke(
        {
            "query": fresh_question
        },
        config={
            "configurable": {
                "session_id": session_2
            }
        }
    )

    print("\nCREW - Fresh Session:")
    print(fresh_response)

    # The new session must NOT inherit APP001.
    if LAST_SELECTED_RECORD_ID is not None:

        raise AssertionError(
            "Task 8 failed: fresh session incorrectly "
            f"selected {LAST_SELECTED_RECORD_ID}."
        )

    print(
        "\n[PASS] Fresh session selected no application ID."
    )

    # Retrieve Session 2 history.
    session_2_history = get_session_history(
        session_2
    )

    print("\nSESSION 2 STORED HISTORY:")

    # Display Session 2 history.
    for message in session_2_history.messages:

        print(
            f"{message.type}: "
            f"{message.content}"
        )

    # Count Session 2 user turns.
    session_2_user_messages = sum(
        message.type == "human"
        for message in session_2_history.messages
    )

    # Task 8 requires exactly one user turn in the fresh session.
    if session_2_user_messages != 1:

        raise AssertionError(
            "Task 8 failed: session_2 does not contain "
            "exactly one user turn."
        )

    print(
        "[PASS] Session 2 contains exactly one user turn."
    )

    # Final Task 8 success evidence.
    print(
        "\n[PASS] Task 8 session-memory demonstration completed."
    )


# ============================================================
# MAIN DEMONSTRATION ENTRY POINT
# ============================================================

def main():
    """
    Run the Task 7, Task 8, and Task 9 demonstrations.

    Task 16's dedicated cache demonstration remains in
    response_cache.py so it can independently prove cache
    normalization, hit/miss behavior, and reduced real-RAG calls.
    """

    # Print the module heading.
    print("=" * 60)
    print("TASK 7 + TASK 8 + TASK 9 - CREWAI HR AUTOMATION")
    print("=" * 60)

    # Demonstrate real RAG tool execution.
    demonstrate_rag_tool()

    # Demonstrate application lookup.
    demonstrate_lookup_tool()

    # Demonstrate the complete three-agent workflow.
    run_complete_crew()

    # Demonstrate same-session memory and fresh-session isolation.
    demonstrate_session_memory()


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

# Run the demonstrations only when this file is executed directly.
if __name__ == "__main__":
    main()
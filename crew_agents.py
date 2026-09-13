# ============================================================
# Task 7 + Task 8 + Task 9 - CrewAI HR Automation
# ============================================================

# Standard-library module used to inspect Python function signatures and tool schemas.
import inspect
# Standard-library module used to convert lookup results to/from JSON text.
import json
# Standard-library regular-expression module used for IDs and text extraction.
import re
# Optional type is used for values such as a missing application ID.
from typing import Optional
# ContextVar keeps the current memory-session ID isolated per execution context.
from contextvars import ContextVar

# BaseModel defines the Task 9 Pydantic response schema.
from pydantic import BaseModel
# ValidationError lets us report failed Task 9 schema validation cleanly.
from pydantic import ValidationError

# Agent is CrewAI's abstraction for an individual worker.
from crewai import Agent
# Crew builds the multi-agent workflow.
from crewai import Crew
# Process provides the sequential execution mode required here.
from crewai import Process
# Task defines the work given to each CrewAI agent.
from crewai import Task
# BaseLLM is the CrewAI base class required for the custom MOCK_LLM.
from crewai.llms.base_llm import BaseLLM
# The @tool decorator exposes normal Python functions as CrewAI tools.
from crewai.tools import tool

# InMemoryChatMessageHistory stores conversation turns for Task 8 sessions.
from langchain_core.chat_history import InMemoryChatMessageHistory
# RunnableLambda wraps our memory-aware Python function as a LangChain runnable.
from langchain_core.runnables import RunnableLambda
# RunnableWithMessageHistory connects the runnable to per-session chat history.
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import the original Task 6 lookup function so this file reuses the required tool logic.
from task6_tool import check_job_application_status as task6_check_job_application_status

# Import the existing Task 3-5 RAG implementation rather than duplicating it.
import rag_core


# ============================================================
# TASK 9 - STRUCTURED RESPONSE MODEL
# ============================================================

class CrewResponse(BaseModel):
    """
    Required structured response schema for the crew.
    """

    final_answer: str
    query: str
    record_id: Optional[str] = None


# Task 9 requires this variable.
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
    Validate an actual crew.kickoff() result with Pydantic.
    """

    try:
        final_answer = str(crew_result)

        validated_response = response_format(
            final_answer=final_answer,
            query=query,
            record_id=record_id
        )

        print("\n[TASK 9] Validated CrewResponse:")
        print(validated_response.model_dump())

        return validated_response

    except ValidationError as error:
        print("\n[TASK 9] CrewResponse validation failed:")
        print(error)
        return None


# ============================================================
# TOOL RESULT STORAGE
# ============================================================

# Stores the latest result returned by each CrewAI tool.
LAST_TOOL_RESULT = {}


# ============================================================
# RAG STATE
# ============================================================

# These values are available to the later guardrail/evaluation code.
LAST_RAG_GROUNDED = None
LAST_RAG_TOP_SIMILARITY = None
LAST_RAG_THRESHOLD = None


# ============================================================
# PREPARE RAG SYSTEM
# ============================================================

# Task 5 uses 200-character fixed chunks with 50-character overlap.
# These values are kept here because the existing rag_core.py stores them
# inside its fixed_size_chunks() function rather than exposing them as
# FIXED_CHUNK_SIZE / FIXED_CHUNK_OVERLAP module constants.
FIXED_CHUNK_SIZE = 200
FIXED_CHUNK_OVERLAP = 50


def build_word_safe_fixed_chunks(documents):
    """
    Build fixed-size chunks without splitting words in the middle.

    The original Task 5 experiment uses 200-character chunks and a
    50-character overlap.  This helper keeps those same values while
    moving a chunk boundary backward to whitespace whenever a word
    would otherwise be cut in half.
    """

    safe_chunks = []

    # Process each knowledge-base document separately.
    for document in documents:
        text = document["text"]
        source = document["source"]
        chunk_size = FIXED_CHUNK_SIZE
        overlap = FIXED_CHUNK_OVERLAP
        start = 0

        # Continue until the complete document has been chunked.
        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Move the end backward to whitespace so words are not split.
            if end < len(text):
                whitespace_position = text.rfind(" ", start, end)
                if whitespace_position > start:
                    end = whitespace_position

            chunk_text = text[start:end].strip()

            if chunk_text:
                # rag_core.store_chunks() requires every chunk to have a
                # unique "id" field in addition to its text and source.
                # The previous version omitted this field, which caused
                # KeyError: "id" when ChromaDB was rebuilt.
                chunk_id = f"{source}_fixed_{len(safe_chunks)}"

                safe_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": source,
                })

            # Stop when the remaining text is already consumed.
            if end >= len(text):
                break

            # Preserve the same overlap idea used by the original fixed-size
            # strategy. The raw overlap position can land in the middle of a
            # word, so the next chunk must also be aligned to a word boundary.
            next_start = max(0, end - overlap)

            if next_start > 0:
                # IMPORTANT: snap BACK to the whitespace immediately before
                # the calculated overlap position, then move one character
                # forward. This keeps the COMPLETE word when the raw overlap
                # position lands inside a word such as "Candidates".
                previous_whitespace_position = text.rfind(" ", 0, next_start)

                if previous_whitespace_position >= 0:
                    next_start = previous_whitespace_position + 1
                else:
                    next_start = 0

            # Avoid getting stuck when overlap would not move the position.
            if next_start <= start:
                next_start = end

            start = next_start

    return safe_chunks


def prepare_rag_system():
    """
    Prepare the existing Task 3-5 RAG infrastructure for the CrewAI agents.

    The recommended collection remains ``fixed_chunks``. The collection is
    rebuilt here so an older persistent Chroma database cannot keep the
    previously broken mid-word chunks.
    """

    documents = rag_core.load_documents(
        rag_core.KNOWLEDGE_BASE
    )

    # Build fixed-size chunks while protecting word boundaries.
    fixed_chunks = build_word_safe_fixed_chunks(
        documents
    )

    model = rag_core.SentenceTransformer(
        rag_core.MODEL_NAME
    )

    client = rag_core.chromadb.PersistentClient(
        path=rag_core.CHROMA_PATH
    )

    # Delete the old persistent fixed collection so stale chunks from an
    # earlier run cannot survive after the word-boundary fix.
    try:
        client.delete_collection(
            "fixed_chunks"
        )
    except Exception:
        # It is normal for the collection not to exist on the first run.
        pass

    fixed_collection = rag_core.prepare_collection(
        client,
        "fixed_chunks"
    )

    # Store the freshly generated, word-safe fixed chunks.
    rag_core.store_chunks(
        fixed_collection,
        fixed_chunks,
        model
    )

    sentence_collection = rag_core.prepare_collection(
        client,
        "sentence_chunks"
    )

    if sentence_collection.count() == 0:
        sentence_chunks = rag_core.build_chunks(
            documents,
            rag_core.sentence_based_chunks
        )

        rag_core.store_chunks(
            sentence_collection,
            sentence_chunks,
            model
        )

    in_scope_scores = rag_core.measure_queries(
        rag_core.IN_SCOPE_QUERIES,
        model,
        fixed_collection,
        sentence_collection
    )

    out_of_scope_scores = rag_core.measure_queries(
        rag_core.OUT_OF_SCOPE_QUERIES,
        model,
        fixed_collection,
        sentence_collection
    )

    threshold = rag_core.choose_threshold(
        in_scope_scores,
        out_of_scope_scores
    )

    return model, fixed_collection, threshold


# Create the shared RAG resources once.
RAG_MODEL, FIXED_COLLECTION, RAG_THRESHOLD = (
    prepare_rag_system()
)


# ============================================================
# TASK 7 - RAG TOOL
# ============================================================

@tool("rag_search")
def rag_search(query: str) -> str:
    """
    Search the fixed_chunks ChromaDB collection.
    """

    print("\n[RAG TOOL] rag_search() was invoked.")
    print(f"[RAG TOOL] Query: {query}")

    results = rag_core.retrieve(
        FIXED_COLLECTION,
        query,
        RAG_MODEL,
        top_k=rag_core.TOP_K
    )

    global LAST_RAG_GROUNDED
    global LAST_RAG_TOP_SIMILARITY
    global LAST_RAG_THRESHOLD

    LAST_RAG_THRESHOLD = RAG_THRESHOLD

    if not results:
        LAST_RAG_GROUNDED = False
        LAST_RAG_TOP_SIMILARITY = None

        answer = (
            "I don't know based on the available knowledge base."
        )

        LAST_TOOL_RESULT["rag_search"] = answer
        return answer

    top_similarity = results[0]["similarity"]
    LAST_RAG_TOP_SIMILARITY = top_similarity

    LAST_RAG_GROUNDED = (
        top_similarity >= RAG_THRESHOLD
    )

    if not LAST_RAG_GROUNDED:
        answer = (
            "I don't know based on the available knowledge base."
        )

        LAST_TOOL_RESULT["rag_search"] = answer
        return answer

    retrieved_text = []

    for result in results:
        source = result.get("source", "unknown")
        text = result.get("text", "")
        similarity = result.get("similarity", 0.0)

        retrieved_text.append(
            f"Source: {source}\n"
            f"Similarity: {similarity:.3f}\n"
            f"Text: {text}"
        )

    answer = "\n\n".join(retrieved_text)

    LAST_TOOL_RESULT["rag_search"] = answer
    return answer


# ============================================================
# TASK 6 - LOOKUP TOOL WRAPPER
# ============================================================

@tool("check_job_application_status")
def lookup_job_application_status(
    record_id: str
) -> str:
    """
    Reuse the original Task 6 application lookup function.
    """

    print(
        "\n[LOOKUP TOOL] check_job_application_status() was invoked."
    )
    print(f"[LOOKUP TOOL] Record ID: {record_id}")

    try:
        result = task6_check_job_application_status(record_id)

        answer = json.dumps(
            result,
            indent=2
        )

        LAST_TOOL_RESULT[
            "check_job_application_status"
        ] = answer

        return answer

    except ValueError as error:
        answer = f"Lookup error: {error}"

        LAST_TOOL_RESULT[
            "check_job_application_status"
        ] = answer

        return answer


# ============================================================
# CREWAI MESSAGE HELPERS
# ============================================================

def get_message_content(message) -> str:
    """Return the message content."""

    if isinstance(message, dict):
        return str(message.get("content", ""))

    return str(getattr(message, "content", ""))


def get_message_role(message) -> str:
    """Return the message role."""

    if isinstance(message, dict):
        return str(message.get("role", ""))

    return str(getattr(message, "role", ""))


# ============================================================
# TOOL SCHEMA HELPERS
# ============================================================

def get_agent_tools(from_agent):
    """Return tools assigned to the current agent."""

    if from_agent is None:
        return []

    tools = getattr(from_agent, "tools", None)

    if not tools:
        return []

    return tools


def get_tool_argument_names(tool_object):
    """
    Inspect the actual tool schema instead of identifying a tool
    by its name.
    """

    argument_names = set()

    args_schema = getattr(
        tool_object,
        "args_schema",
        None
    )

    if args_schema is not None:
        model_fields = getattr(
            args_schema,
            "model_fields",
            None
        )

        if model_fields:
            argument_names.update(model_fields.keys())
        else:
            old_fields = getattr(
                args_schema,
                "__fields__",
                None
            )

            if old_fields:
                argument_names.update(old_fields.keys())

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
                signature = inspect.signature(function)
                argument_names.update(
                    signature.parameters.keys()
                )
            except (TypeError, ValueError):
                pass

    return argument_names


def find_tool_by_argument(
    from_agent,
    argument_name
):
    """
    Find a tool from its declared argument schema.

    This avoids the silent misclassification problem that can occur
    when tool names happen to contain similar words.
    """

    tools = get_agent_tools(from_agent)

    for tool_object in tools:
        argument_names = get_tool_argument_names(tool_object)

        if argument_name in argument_names:
            return tool_object

    return None


def get_tool_name(tool_object):
    """Return the CrewAI tool name."""

    return str(getattr(tool_object, "name", ""))


# ============================================================
# INPUT HELPERS
# ============================================================

def extract_record_id(text: str):
    """Extract an application ID such as APP001."""

    upper_text = text.upper()

    match = re.search(
        r"\bAPP\d{3}\b",
        upper_text
    )

    if match:
        return match.group(0)

    return None


def extract_clean_rag_query(text: str) -> str:
    """Remove surrounding CrewAI text from a RAG query."""

    marker = "question:"
    lower_text = text.lower()

    marker_position = lower_text.find(marker)

    if marker_position != -1:
        question = text[
            marker_position + len(marker):
        ].strip()

        ending_markers = [
            "This is the expected criteria",
            "Begin!",
            "Expected Output:"
        ]

        for ending_marker in ending_markers:
            ending_position = question.find(ending_marker)

            if ending_position != -1:
                question = question[
                    :ending_position
                ].strip()

        return question

    return text.strip()


def get_latest_user_message(messages):
    """Return the latest user message."""

    for message in reversed(messages):
        if get_message_role(message) == "user":
            return get_message_content(message)

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

    We deliberately do not search for generic 'Observation:' text
    because CrewAI's system prompt can contain that text itself.
    """

    for message in messages:
        if get_message_role(message) != "assistant":
            continue

        content = get_message_content(message)

        action_pattern = (
            rf"(?m)^\s*Action:\s*"
            rf"{re.escape(tool_name)}\s*$"
        )

        if re.search(action_pattern, content):
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
    Extract the current Composer question from the Composer task.
    """

    task_description = str(
        getattr(
            from_task,
            "description",
            ""
        )
    )

    question_match = re.search(
        r"User question:\s*(.*?)"
        r"(?:\n\s*Application record:|\Z)",
        task_description,
        flags=re.DOTALL
    )

    if question_match:
        question = question_match.group(1).strip()

        if question:
            return question

    return fallback_message.strip()


def extract_composer_record_id(
    from_task
) -> Optional[str]:
    """
    Extract the application ID explicitly supplied to the
    Composer task.

    Task 8 can also put a remembered APP### value here.
    """

    task_description = str(
        getattr(
            from_task,
            "description",
            ""
        )
    )

    record_match = re.search(
        r"Application record:\s*(APP\d{3})",
        task_description,
        flags=re.IGNORECASE
    )

    if record_match:
        return record_match.group(1).upper()

    return None


def extract_lookup_json(context_text: str):
    """Extract a flat Task 6 JSON object from Composer context."""

    candidates = re.findall(
        r'\{[^{}]*"record_id"\s*:\s*"APP\d{3}"[^{}]*\}',
        context_text,
        flags=re.DOTALL
    )

    for candidate in candidates:
        try:
            data = json.loads(candidate)

            if (
                isinstance(data, dict)
                and extract_record_id(
                    str(data.get("record_id", ""))
                )
            ):
                return data

        except json.JSONDecodeError:
            continue

    return None


def extract_first_retrieved_text(
    context_text: str
) -> Optional[str]:
    """Extract the first useful RAG Text field."""

    match = re.search(
        r"Text:\s*(.*?)(?=\n\nSource:|\Z)",
        context_text,
        flags=re.DOTALL
    )

    if match:
        text = match.group(1).strip()

        if text:
            return text

    return None


# ============================================================
# TASK 7 MOCK_LLM
# ============================================================

class MOCK_LLM(BaseLLM):
    """
    Deterministic CrewAI mock model.

    No external LLM API is called.
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
        """Decide the current agent action or final result."""

        if messages is None:
            messages = []

        messages = list(messages)

        latest_user_message = get_latest_user_message(
            messages
        )

        # ====================================================
        # CASE 1 - RETRIEVAL AGENT
        # ====================================================

        query_tool = find_tool_by_argument(
            from_agent,
            "query"
        )

        if query_tool is not None:
            query_tool_name = get_tool_name(query_tool)

            if not has_previous_action(
                messages,
                query_tool_name
            ):
                clean_query = extract_clean_rag_query(
                    latest_user_message
                )

                return (
                    "Thought: I need to search the "
                    "HR knowledge base.\n"
                    f"Action: {query_tool_name}\n"
                    f"Action Input: "
                    f"{json.dumps({'query': clean_query})}"
                )

            rag_result = LAST_TOOL_RESULT.get(
                query_tool_name,
                "No RAG result was stored."
            )

            return (
                "Thought: I have retrieved the relevant "
                "knowledge-base information.\n"
                f"Final Answer: {rag_result}"
            )

        # ====================================================
        # CASE 2 - LOOKUP AGENT
        # ====================================================

        record_tool = find_tool_by_argument(
            from_agent,
            "record_id"
        )

        if record_tool is not None:
            record_tool_name = get_tool_name(record_tool)

            if not has_previous_action(
                messages,
                record_tool_name
            ):
                record_id = extract_record_id(
                    latest_user_message
                )

                if record_id is None:
                    return (
                        "Thought: I do not have a valid "
                        "application record ID.\n"
                        "Final Answer: Please provide an "
                        "application ID such as APP123."
                    )

                return (
                    "Thought: I need to look up the "
                    "application record.\n"
                    f"Action: {record_tool_name}\n"
                    f"Action Input: "
                    f"{json.dumps({'record_id': record_id})}"
                )

            lookup_result = LAST_TOOL_RESULT.get(
                record_tool_name,
                "No lookup result was stored."
            )

            return (
                "Thought: I have the application lookup "
                "information.\n"
                f"Final Answer: {lookup_result}"
            )

        # ====================================================
        # CASE 3 - RESPONSE COMPOSER
        # ====================================================

        current_question = extract_composer_question(
            from_task,
            latest_user_message
        )

        question_lower = current_question.lower()

        # The Composer task is the authoritative place for the
        # current application ID, including one recovered from
        # Task 8 session memory.
        selected_record_id = extract_composer_record_id(
            from_task
        )

        # ----------------------------------------------------
        # Locate previous-agent context.
        # ----------------------------------------------------

        context_text = ""

        for message in messages:
            if get_message_role(message) != "user":
                continue

            content = get_message_content(message)

            if (
                "This is the context you're working with:"
                in content
            ):
                context_text = content

        # ----------------------------------------------------
        # Remove the CrewAI context marker.
        # ----------------------------------------------------

        if context_text:
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

        lookup_data = extract_lookup_json(
            combined_context
        )

        # IMPORTANT FIX:
        #
        # Use lookup data only when the current Composer task
        # contains a genuine application ID.
        #
        # This prevents a no-ID Lookup Agent message from
        # overriding a valid RAG response.
        # ====================================================

        if (
            lookup_data is not None
            and selected_record_id is not None
        ):
            record_id = lookup_data.get("record_id")
            status = lookup_data.get("status")
            escalation_score = lookup_data.get(
                "escalation_score"
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
        #
        # IMPORTANT FIX:
        #
        # The word "candidate" by itself does NOT mean lookup.
        # For example, Q02 says:
        #
        #   "How much notice should a candidate get before
        #    an interview?"
        #
        # That is a normal interview policy question.
        # ====================================================

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
        #
        # The actual RAG tool result is authoritative here.
        # This also preserves the Task 4 fallback.
        # ====================================================

        rag_result = LAST_TOOL_RESULT.get(
            "rag_search",
            ""
        )

        rag_result = str(rag_result).strip()

        if rag_result:

            # -----------------------------------------------
            # RAG fallback
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
            # Return the first retrieved text without exposing
            # internal similarity metadata.
            # -----------------------------------------------

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

            # Safe fallback to the actual stored RAG result.
            return (
                "Thought: I have retrieved relevant "
                "knowledge-base information.\n"
                "Final Answer: "
                f"{rag_result}"
            )

        # ====================================================
        # FINAL FALLBACK
        # ====================================================

        return (
            "Thought: No usable information was returned "
            "by the previous agents.\n"
            "Final Answer: "
            "I don't know based on the available "
            "knowledge base."
        )

    def supports_function_calling(self) -> bool:
        """
        The MOCK_LLM uses text-based ReAct tool requests.
        """

        return False


# ============================================================
# CREATE MOCK LLM
# ============================================================

mock_llm = MOCK_LLM(
    model="mock-llm"
)


# ============================================================
# AGENTS
# ============================================================

def create_agents():
    """Create the three required Task 7 agents."""

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
        tools=[rag_search],
        llm=mock_llm,
        allow_delegation=False,
        verbose=True
    )

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
        tools=[lookup_job_application_status],
        llm=mock_llm,
        allow_delegation=False,
        verbose=True
    )

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
        tools=[],
        llm=mock_llm,
        allow_delegation=False,
        verbose=True
    )

    return (
        retrieval_agent,
        lookup_agent,
        composer_agent
    )


# ============================================================
# TASKS
# ============================================================

def create_tasks(
    retrieval_agent,
    lookup_agent,
    composer_agent
):
    """Create the three sequential CrewAI tasks."""

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
        agent=retrieval_agent
    )

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
        agent=lookup_agent
    )

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
        agent=composer_agent,
        context=[retrieval_task, lookup_task]
    )

    return (
        retrieval_task,
        lookup_task,
        composer_task
    )


# ============================================================
# MAIN CREW
# ============================================================

def create_main_crew():
    """Create the required three-agent sequential CrewAI crew."""

    (
        retrieval_agent,
        lookup_agent,
        composer_agent
    ) = create_agents()

    (
        retrieval_task,
        lookup_task,
        composer_task
    ) = create_tasks(
        retrieval_agent,
        lookup_agent,
        composer_agent
    )

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
        process=Process.sequential,
        verbose=True
    )

    return new_crew


# ============================================================
# MODULE-LEVEL CREW
# ============================================================

# Reuse this Crew for Task 11 API requests.
crew = create_main_crew()


# ============================================================
# TASK 8 - SESSION MEMORY
# ============================================================

# One history object is maintained for every session ID.
SESSION_STORE = {}

# Task 11 uses this mapping for the selected application ID.
SESSION_SELECTED_RECORD_IDS = {}

# Task 8 demonstration state.
LAST_SELECTED_RECORD_ID = None

# FastAPI request-local session bridge.
CURRENT_MEMORY_SESSION_ID = ContextVar(
    "current_memory_session_id",
    default=None
)


# ============================================================
# SESSION HISTORY FUNCTIONS
# ============================================================

def get_session_history(
    session_id: str
) -> InMemoryChatMessageHistory:
    """Return or create history for the requested session."""

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = (
            InMemoryChatMessageHistory()
        )

    return SESSION_STORE[session_id]


def get_history_text(history) -> str:
    """Convert LangChain history to readable text."""

    history_lines = []

    messages = getattr(
        history,
        "messages",
        history
    )

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

    return "\n\n".join(history_lines)


# ============================================================
# MEMORY-AWARE CREW
# ============================================================

def run_memory_aware_crew(data):
    """
    Execute the existing Task 7 Crew through Task 8 memory.

    The current message gets priority when it contains APP###.
    Otherwise the same-session history is checked.

    Only the selected application ID uses history.
    The RAG query remains the current message only.
    """

    current_message = str(
        data.get(
            "query",
            ""
        )
    )

    history = data.get(
        "history",
        []
    )

    history_text = get_history_text(history)

    record_id = extract_record_id(
        current_message
    )

    if record_id is None:
        record_id = extract_record_id(
            history_text
        )

    if record_id is None:
        record_id = ""

    global LAST_SELECTED_RECORD_ID

    LAST_SELECTED_RECORD_ID = (
        record_id
        if record_id
        else None
    )

    current_session_id = (
        CURRENT_MEMORY_SESSION_ID.get()
    )

    if current_session_id is not None:
        SESSION_SELECTED_RECORD_IDS[
            str(current_session_id)
        ] = (
            record_id
            if record_id
            else None
        )

    print("\n[SESSION MEMORY]")
    print(f"Current message: {current_message}")
    print(
        "Record ID selected: "
        f"{record_id if record_id else 'None'}"
    )

    result = crew.kickoff(
        inputs={
            "rag_query": current_message,
            "user_question": current_message,
            "record_id": record_id
        }
    )

    validated_response = validate_crew_response(
        crew_result=result,
        query=current_message,
        record_id=(
            record_id
            if record_id
            else None
        )
    )

    if validated_response is not None:
        return validated_response.final_answer

    return "Crew response could not be validated."


# ============================================================
# LANGCHAIN RUNNABLE
# ============================================================

memory_aware_crew_runnable = RunnableLambda(
    run_memory_aware_crew
)


# ============================================================
# RUNNABLE WITH MESSAGE HISTORY
# ============================================================

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
    """Demonstrate actual RAG tool invocation through kickoff()."""

    print("\n")
    print("=" * 60)
    print("DEMONSTRATION 1 - RAG TOOL")
    print("=" * 60)

    demo_question = (
        "What degree is required for most professional jobs?"
    )

    demo_agent = Agent(
        role="HR Knowledge Retrieval Agent",
        goal="Retrieve HR information using the RAG tool.",
        backstory=(
            "Use the RAG tool and do not invent information."
        ),
        tools=[rag_search],
        llm=mock_llm,
        allow_delegation=False,
        verbose=True
    )

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

    demo_crew = Crew(
        agents=[demo_agent],
        tasks=[demo_task],
        process=Process.sequential,
        verbose=True
    )

    result = demo_crew.kickoff()

    validated_response = validate_crew_response(
        crew_result=result,
        query=demo_question
    )

    print("\nRAG demonstration validated result:")

    if validated_response is not None:
        print(validated_response.model_dump())


# ============================================================
# TASK 7 - LOOKUP DEMONSTRATION
# ============================================================

def demonstrate_lookup_tool():
    """Demonstrate actual Task 6 lookup invocation through kickoff()."""

    print("\n")
    print("=" * 60)
    print("DEMONSTRATION 2 - LOOKUP TOOL")
    print("=" * 60)

    demo_question = (
        "What is the status of application APP001?"
    )

    demo_agent = Agent(
        role="Job Application Lookup Agent",
        goal=(
            "Retrieve application information using the "
            "Task 6 lookup tool."
        ),
        backstory=(
            "Use the lookup tool and do not invent application data."
        ),
        tools=[lookup_job_application_status],
        llm=mock_llm,
        allow_delegation=False,
        verbose=True
    )

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

    demo_crew = Crew(
        agents=[demo_agent],
        tasks=[demo_task],
        process=Process.sequential,
        verbose=True
    )

    result = demo_crew.kickoff()

    validated_response = validate_crew_response(
        crew_result=result,
        query=demo_question,
        record_id="APP001"
    )

    print("\nLookup demonstration validated result:")

    if validated_response is not None:
        print(validated_response.model_dump())


# ============================================================
# COMPLETE TASK 7 CREW
# ============================================================

def run_complete_crew():
    """Execute the complete three-agent Task 7 workflow."""

    print("\n")
    print("=" * 60)
    print("COMPLETE THREE-AGENT CREW")
    print("=" * 60)

    inputs = {
        "rag_query": (
            "What degree is required for most professional jobs?"
        ),
        "user_question": (
            "What degree is required for most professional jobs?"
        ),
        "record_id": "APP001"
    }

    result = crew.kickoff(
        inputs=inputs
    )

    validated_response = validate_crew_response(
        crew_result=result,
        query=inputs["user_question"],
        record_id=inputs["record_id"]
    )

    print("\nFinal Validated Response:")

    if validated_response is not None:
        print(validated_response.model_dump())
        return validated_response.final_answer

    print("Crew response validation failed.")
    return None


# ============================================================
# TASK 8 - SESSION MEMORY DEMONSTRATION
# ============================================================

def demonstrate_session_memory():
    """Demonstrate same-session memory and fresh-session isolation."""

    print("\n")
    print("=" * 60)
    print("TASK 8 - TRANSCRIPT 1")
    print("SAME SESSION / TWO TURNS")
    print("=" * 60)

    session_1 = "student_session_001"

    turn_1_question = (
        "What is the status of application APP001?"
    )

    print("\nUSER - Turn 1:")
    print(turn_1_question)

    turn_1_response = crew_with_memory.invoke(
        {"query": turn_1_question},
        config={
            "configurable": {
                "session_id": session_1
            }
        }
    )

    print("\nCREW - Turn 1:")
    print(turn_1_response)

    if LAST_SELECTED_RECORD_ID != "APP001":
        raise AssertionError(
            "Task 8 failed: Turn 1 did not select APP001."
        )

    print("\n[PASS] Turn 1 selected APP001.")

    turn_2_question = (
        "What was the escalation score for that application?"
    )

    print("\nUSER - Turn 2:")
    print(turn_2_question)

    turn_2_response = crew_with_memory.invoke(
        {"query": turn_2_question},
        config={
            "configurable": {
                "session_id": session_1
            }
        }
    )

    print("\nCREW - Turn 2:")
    print(turn_2_response)

    if LAST_SELECTED_RECORD_ID != "APP001":
        raise AssertionError(
            "Task 8 failed: Turn 2 did not recover APP001."
        )

    print(
        "\n[PASS] Turn 2 recovered APP001 from session memory."
    )

    session_1_history = get_session_history(session_1)

    print("\nSESSION 1 STORED HISTORY:")

    for message in session_1_history.messages:
        print(
            f"{message.type}: "
            f"{message.content}"
        )

    session_1_user_messages = sum(
        message.type == "human"
        for message in session_1_history.messages
    )

    if session_1_user_messages != 2:
        raise AssertionError(
            "Task 8 failed: session_1 does not contain "
            "exactly two user turns."
        )

    print("[PASS] Session 1 contains both user turns.")

    print("\n")
    print("=" * 60)
    print("TASK 8 - TRANSCRIPT 2")
    print("FRESH SESSION / ONE TURN")
    print("=" * 60)

    session_2 = "student_session_002"

    fresh_question = (
        "What was the escalation score for that application?"
    )

    print("\nUSER - Fresh Session:")
    print(fresh_question)

    fresh_response = crew_with_memory.invoke(
        {"query": fresh_question},
        config={
            "configurable": {
                "session_id": session_2
            }
        }
    )

    print("\nCREW - Fresh Session:")
    print(fresh_response)

    if LAST_SELECTED_RECORD_ID is not None:
        raise AssertionError(
            "Task 8 failed: fresh session incorrectly "
            f"selected {LAST_SELECTED_RECORD_ID}."
        )

    print("\n[PASS] Fresh session selected no application ID.")

    session_2_history = get_session_history(session_2)

    print("\nSESSION 2 STORED HISTORY:")

    for message in session_2_history.messages:
        print(
            f"{message.type}: "
            f"{message.content}"
        )

    session_2_user_messages = sum(
        message.type == "human"
        for message in session_2_history.messages
    )

    if session_2_user_messages != 1:
        raise AssertionError(
            "Task 8 failed: session_2 does not contain "
            "exactly one user turn."
        )

    print("[PASS] Session 2 contains exactly one user turn.")

    print("\n[PASS] Task 8 session-memory demonstration completed.")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the Task 7, Task 8 and Task 9 demonstrations."""

    print("=" * 60)
    print("TASK 7 + TASK 8 + TASK 9 - CREWAI HR AUTOMATION")
    print("=" * 60)

    demonstrate_rag_tool()
    demonstrate_lookup_tool()
    run_complete_crew()
    demonstrate_session_memory()


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

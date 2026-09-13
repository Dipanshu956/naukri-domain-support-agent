# ============================================================
# Task 11 + Task 12
# FastAPI Deployment + Structured JSONL Logging
# ============================================================
#
# TASK 11
# ------------------------------------------------------------
# Required endpoints:
#
#   1. POST /ask
#   2. POST /add-document
#   3. WebSocket /ws/chat
#
# TASK 12
# ------------------------------------------------------------
# Every request/unit of work must:
#
#   1. Produce exactly ONE JSONL log entry.
#   2. Have a fresh trace_id.
#   3. Have timing information.
#   4. Never write fixed-format PII in clear text.
#   5. Reuse the SAME Task 10 masked_text used downstream.
#
# IMPORTANT:
# ------------------------------------------------------------
#
# /ask and each WebSocket turn call apply_input_guardrails()
# EXACTLY ONCE.
#
# The exact same masked_text is then used for:
#
#       CrewAI/model input
#       JSONL request logging
#
# request_logger.py does NOT perform masking itself.
#
# ============================================================


# ============================================================
# SECTION 1 - STANDARD LIBRARY IMPORTS
# ============================================================

# logging is used for normal operational/server diagnostics.
import logging

# os is used to configure CrewAI telemetry before CrewAI
# modules are imported.
import os

# time is used to measure request/turn duration.
import time

# uuid is used to generate fresh trace IDs and WebSocket
# session IDs.
import uuid


# ============================================================
# SECTION 2 - DISABLE CREWAI TELEMETRY
# ============================================================

# The capstone must run in MOCK_LLM mode without requiring
# paid APIs or outbound telemetry calls.
#
# These variables are therefore configured BEFORE importing
# crew_agents.
os.environ.setdefault(
    "CREWAI_DISABLE_TELEMETRY",
    "true",
)

os.environ.setdefault(
    "OTEL_SDK_DISABLED",
    "true",
)


# ============================================================
# SECTION 3 - FASTAPI IMPORTS
# ============================================================

# FastAPI is the web framework used for Task 11.
#
# HTTPException is used for intentional HTTP errors.
#
# Request is used by the validation-error handler.
#
# WebSocket and WebSocketDisconnect are used for the
# real-time WebSocket endpoint.
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)


# ============================================================
# SECTION 4 - FASTAPI ERROR HANDLING IMPORTS
# ============================================================

# RequestValidationError is raised by FastAPI/Pydantic when
# incoming HTTP data does not match the request model.
#
# Example:
#     /ask without session_id
#
# Such a request never enters the /ask endpoint function,
# so it needs a dedicated logging handler.
from fastapi.exceptions import RequestValidationError

# JSONResponse is the proper Response object returned from
# a FastAPI exception handler.
from fastapi.responses import JSONResponse

# jsonable_encoder converts Pydantic validation details into
# JSON-safe Python data.
from fastapi.encoders import jsonable_encoder


# ============================================================
# SECTION 5 - PYDANTIC IMPORTS
# ============================================================

# BaseModel is used to define request and response schemas.
#
# Field is used to enforce non-empty string values.
from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# SECTION 6 - THREADPOOL SUPPORT
# ============================================================

# CrewAI and SentenceTransformer operations are synchronous
# and can take noticeable time.
#
# run_in_threadpool() allows FastAPI to execute those blocking
# operations without blocking the async event loop.
from starlette.concurrency import run_in_threadpool


# ============================================================
# SECTION 7 - EXISTING PROJECT IMPORTS
# ============================================================

# Reuse the existing Task 7 + Task 8 + Task 9 CrewAI system.
import crew_agents

# Reuse the structured CrewResponse model defined in Task 9.
from crew_agents import CrewResponse

# Reuse the Task 10 input and output guardrails.
from guardrails import (
    apply_input_guardrails,
    apply_output_groundedness_guardrail,
)

# Reuse the existing Task 3-5 RAG implementation.
import rag_core

# Reuse the Task 12 structured JSONL logger.
#
# IMPORTANT:
# request_logger.py performs NO masking.
# We must supply already-safe text.
from request_logger import log_request


# ============================================================
# SECTION 8 - FASTAPI APPLICATION
# ============================================================

# Create the FastAPI application.
app = FastAPI(
    title="Naukri HR Support Agent API",
    version="1.0.0",
)


# ============================================================
# SECTION 9 - NORMAL SERVER LOGGER
# ============================================================

# This logger is only for normal diagnostic/server messages.
#
# Task 12 JSONL records are written separately using
# request_logger.log_request().
logger = logging.getLogger(__name__)


# ============================================================
# SECTION 10 - TASK 8 SESSION BRIDGE
# ============================================================

# Reuse the session-to-record-ID mapping already maintained
# by crew_agents.py.
#
# This is created as a compatibility reference for the
# existing Task 8 / Task 11 session implementation.
SESSION_SELECTED_RECORD_IDS = getattr(
    crew_agents,
    "SESSION_SELECTED_RECORD_IDS",
    {},
)


# ============================================================
# SECTION 11 - HTTP REQUEST MODELS
# ============================================================


class AskRequest(BaseModel):
    """
    Request model for POST /ask.

    The caller must provide:

        session_id
        query
    """

    # Session identifier used by Task 8 memory.
    session_id: str = Field(
        min_length=1,
    )

    # Current user question.
    query: str = Field(
        min_length=1,
    )


class AddDocumentRequest(BaseModel):
    """
    Request model for POST /add-document.

    The caller must provide:

        content
        source_name
    """

    # Complete document content to add to the RAG knowledge base.
    content: str = Field(
        min_length=1,
    )

    # Name of the document/source.
    source_name: str = Field(
        min_length=1,
    )


# ============================================================
# SECTION 12 - HTTP RESPONSE MODELS
# ============================================================


class AskResponse(CrewResponse):
    """
    Response model for POST /ask.

    CrewResponse already defines:

        final_answer
        query
        record_id

    The class is inherited here so FastAPI exposes the expected
    Task 9 response schema for the API endpoint.
    """

    pass


class AddDocumentResponse(BaseModel):
    """
    Response model for POST /add-document.
    """

    # Name of the document supplied by the caller.
    source_name: str

    # Number of chunks generated and stored.
    chunks_created: int

    # Result of the document-add operation.
    status: str


# ============================================================
# SECTION 13 - VALIDATION ERROR HANDLER
# ============================================================

@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
):
    """
    Handle requests rejected by FastAPI/Pydantic validation.

    Example:

        POST /ask

    with:

        {
            "query": "What is the status of APP001?"
        }

    when "session_id" is missing.

    Important:
    --------------------------------------------------------
    FastAPI rejects such a request BEFORE the normal endpoint
    function runs.

    Therefore the /ask endpoint's normal try/finally block
    cannot create the Task 12 log entry.

    This handler provides that one required JSONL record.

    SECURITY:
    --------------------------------------------------------
    We deliberately DO NOT read or log the raw request body.

    A malformed request could contain fixed-format PII.

    Therefore the logged text is only:

        <request-validation-failed>

    This ensures the validation-error path cannot accidentally
    write raw user input to disk.
    """

    # --------------------------------------------------------
    # Generate one fresh trace ID for this validation-failed
    # HTTP request.
    # --------------------------------------------------------

    trace_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # Start a timer for the validation-error handling operation.
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Safe placeholder.
    #
    # Never use the rejected request body here.
    # --------------------------------------------------------

    safe_text = (
        "<request-validation-failed>"
    )

    # --------------------------------------------------------
    # Calculate processing time.
    # --------------------------------------------------------

    duration_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    # --------------------------------------------------------
    # Write EXACTLY ONE Task 12 JSONL record.
    # --------------------------------------------------------

    log_request(
        trace_id=trace_id,
        endpoint=request.url.path,
        masked_text=safe_text,
        duration_ms=duration_ms,
        session_id=None,
        status="error",
    )

    # --------------------------------------------------------
    # Return an actual FastAPI Response object.
    #
    # This is important:
    #
    # Returning a normal Python dict from this exception
    # handler would not be a proper Starlette Response.
    # --------------------------------------------------------

    return JSONResponse(
        status_code=422,
        content={
            "detail": jsonable_encoder(
                exc.errors()
            )
        },
    )


# ============================================================
# SECTION 14 - SYNCHRONOUS CREW BRIDGE
# ============================================================


def run_crew_for_session(
    query: str,
    session_id: str,
) -> CrewResponse:
    """
    Run the existing Task 8 session-memory + CrewAI pipeline.

    IMPORTANT:
    --------------------------------------------------------
    Task 10 masking has already happened before this function
    is called.

    Therefore this function MUST NOT call
    apply_input_guardrails() again.

    The "query" received here is already the safe masked value.
    """

    # --------------------------------------------------------
    # Store the ContextVar token so the value can be restored
    # correctly after this request completes.
    # --------------------------------------------------------

    session_token = None

    # --------------------------------------------------------
    # The existing crew_agents module contains
    # CURRENT_MEMORY_SESSION_ID.
    #
    # We set it only when it exists.
    # --------------------------------------------------------

    if hasattr(
        crew_agents,
        "CURRENT_MEMORY_SESSION_ID",
    ):

        session_token = (
            crew_agents.CURRENT_MEMORY_SESSION_ID.set(
                session_id
            )
        )

    try:

        # ----------------------------------------------------
        # Invoke the existing Task 8 memory-aware runnable.
        # ----------------------------------------------------

        final_answer = (
            crew_agents.crew_with_memory.invoke(
                {
                    "query": query
                },
                config={
                    "configurable": {
                        "session_id": session_id,
                    }
                },
            )
        )

        # ----------------------------------------------------
        # Read the application record selected by the memory
        # layer.
        # ----------------------------------------------------

        record_id = (
            crew_agents.SESSION_SELECTED_RECORD_IDS.get(
                session_id
            )
        )

        # ----------------------------------------------------
        # Construct the Task 9 structured response object.
        # ----------------------------------------------------

        response = CrewResponse(
            final_answer=str(
                final_answer
            ),
            query=query,
            record_id=record_id,
        )

        # ----------------------------------------------------
        # Read the RAG groundedness information captured by
        # the existing crew_agents.py implementation.
        # ----------------------------------------------------

        grounded = bool(
            getattr(
                crew_agents,
                "LAST_RAG_GROUNDED",
                False,
            )
        )

        top_similarity = getattr(
            crew_agents,
            "LAST_RAG_TOP_SIMILARITY",
            None,
        )

        threshold = getattr(
            crew_agents,
            "LAST_RAG_THRESHOLD",
            None,
        )

        # ----------------------------------------------------
        # Lookup-backed answers are based on Task 6 application
        # data.
        #
        # Therefore do not incorrectly reject them merely because
        # the RAG similarity score is low.
        # ----------------------------------------------------

        if record_id is not None:

            guarded_answer = (
                response.final_answer
            )

        else:

            # ------------------------------------------------
            # Apply the existing Task 10 output groundedness
            # guardrail to normal knowledge-base responses.
            # ------------------------------------------------

            guarded_answer = (
                apply_output_groundedness_guardrail(
                    final_answer=response.final_answer,
                    query=query,
                    grounded=grounded,
                    top_similarity=top_similarity,
                    threshold=threshold,
                )
            )

        # ----------------------------------------------------
        # Return the final Task 9-compatible object.
        # ----------------------------------------------------

        return CrewResponse(
            final_answer=guarded_answer,
            query=response.query,
            record_id=response.record_id,
        )

    finally:

        # ----------------------------------------------------
        # Restore the previous ContextVar value.
        #
        # This prevents one request's technical session context
        # from leaking into another request executed in the same
        # worker context.
        # ----------------------------------------------------

        if session_token is not None:

            crew_agents.CURRENT_MEMORY_SESSION_ID.reset(
                session_token
            )


# ============================================================
# SECTION 15 - SYNCHRONOUS RAG DOCUMENT BRIDGE
# ============================================================


def add_document_to_rag(
    content: str,
    source_name: str,
) -> int:
    """
    Add one new document to the existing recommended
    fixed_chunks ChromaDB collection.

    Existing Task 5 chunking, embedding and storage functions
    are reused instead of creating a second RAG implementation.
    """

    # --------------------------------------------------------
    # Convert the incoming document into the same structure
    # expected by rag_core.py.
    # --------------------------------------------------------

    documents = [
        {
            "source": source_name,
            "text": content,
        }
    ]

    # --------------------------------------------------------
    # Reuse the Task 5 fixed-size chunking strategy.
    # --------------------------------------------------------

    chunks = rag_core.build_chunks(
        documents,
        rag_core.fixed_size_chunks,
    )

    # --------------------------------------------------------
    # Store the generated chunks using the same collection and
    # embedding model already used by the deployed RAG tool.
    # --------------------------------------------------------

    rag_core.store_chunks(
        crew_agents.FIXED_COLLECTION,
        chunks,
        crew_agents.RAG_MODEL,
    )

    # --------------------------------------------------------
    # Return the number of chunks inserted.
    # --------------------------------------------------------

    return len(chunks)


# ============================================================
# SECTION 16 - POST /ask
# ============================================================


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(
    request: AskRequest,
) -> AskResponse:
    """
    Answer one HR support request.

    Task 10 input masking is performed EXACTLY ONCE.

    The exact SAME masked value is then used for:

        1. CrewAI/model input
        2. Task 12 JSONL logging

    The finally block writes exactly ONE JSONL record for every
    request that reaches this endpoint.

    Blocked requests and failed requests are logged too.
    """

    # --------------------------------------------------------
    # Create a fresh trace ID for this HTTP request.
    # --------------------------------------------------------

    trace_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # Start the timer before request processing.
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Safe fallback.
    #
    # If the Task 10 guardrail itself unexpectedly fails,
    # this safe placeholder is logged rather than the raw query.
    # --------------------------------------------------------

    masked_query = (
        "<guardrail-processing-failed>"
    )

    # --------------------------------------------------------
    # Default request status.
    #
    # It changes to:
    #
    #     blocked
    #     ok
    #
    # when appropriate.
    # --------------------------------------------------------

    status = "error"

    try:

        # ====================================================
        # TASK 10 - INPUT GUARDRAIL
        # ====================================================
        #
        # This is the ONLY input guardrail call for this /ask
        # request.
        # ====================================================

        guardrail_result = (
            apply_input_guardrails(
                request.query
            )
        )

        # ----------------------------------------------------
        # Extract the exact Task 10 masked text.
        #
        # This same value will later be used by BOTH:
        #
        #     CrewAI
        #     JSONL logger
        # ----------------------------------------------------

        masked_query = str(
            guardrail_result[
                "masked_text"
            ]
        )

        # ====================================================
        # PROMPT-INJECTION GUARDRAIL
        # ====================================================

        # If Task 10 rejects the input, do not send it to CrewAI.
        if not guardrail_result[
            "allowed"
        ]:

            # Mark the request as blocked.
            status = "blocked"

            # IMPORTANT:
            # ------------------------------------------------
            # There is NO log_request() call here.
            #
            # The finally block below performs the ONE required
            # structured log operation.
            # ------------------------------------------------

            raise HTTPException(
                status_code=400,
                detail=(
                    "Request blocked by "
                    "prompt-injection guardrail."
                ),
            )

        # ====================================================
        # CREWAI EXECUTION
        # ====================================================
        #
        # Only masked_query is sent downstream.
        #
        # Raw request.query is NOT sent to CrewAI.
        # ====================================================

        result = await run_in_threadpool(
            run_crew_for_session,
            masked_query,
            request.session_id,
        )

        # ----------------------------------------------------
        # Validate the Task 9 structured response.
        # ----------------------------------------------------

        validated_result = (
            AskResponse.model_validate(
                result.model_dump()
            )
        )

        # ----------------------------------------------------
        # Request completed successfully.
        # ----------------------------------------------------

        status = "ok"

        return validated_result

    except HTTPException:
        # Preserve intentional HTTP errors such as the
        # prompt-injection guardrail's 400 response.
        raise

    except Exception as error:

        # ----------------------------------------------------
        # Write the technical exception to the normal server
        # diagnostics only.
        #
        # The raw query is NOT logged here.
        # ----------------------------------------------------

        logger.exception(
            "Crew execution failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Crew execution failed: {error}"
            ),
        ) from error

    finally:

        # ====================================================
        # TASK 12 - REQUEST TIMING
        # ====================================================

        duration_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # ====================================================
        # TASK 12 - EXACTLY ONE JSONL RECORD
        # ====================================================
        #
        # The logger receives masked_query.
        #
        # It NEVER receives request.query.
        # ====================================================

        log_request(
            trace_id=trace_id,
            endpoint="/ask",
            masked_text=masked_query,
            duration_ms=duration_ms,
            session_id=request.session_id,
            status=status,
        )


# ============================================================
# SECTION 17 - POST /add-document
# ============================================================


@app.post(
    "/add-document",
    response_model=AddDocumentResponse,
)
async def add_document(
    request: AddDocumentRequest,
) -> AddDocumentResponse:
    """
    Add a new knowledge-base document.

    The full document body is NOT written to the Task 12 log.

    Instead, the log receives only a safe request summary:

        source_name
        content_length

    This avoids placing arbitrary document text into the
    structured request log.
    """

    # --------------------------------------------------------
    # Fresh trace ID for this HTTP request.
    # --------------------------------------------------------

    trace_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # Start the request timer.
    # --------------------------------------------------------

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Build safe observability text.
    #
    # request.content is deliberately NOT included.
    # --------------------------------------------------------

    safe_request_text = (
        f"source_name={request.source_name};"
        f"content_length={len(request.content)}"
    )

    # --------------------------------------------------------
    # Default status.
    # --------------------------------------------------------

    status = "error"

    try:

        # ----------------------------------------------------
        # RAG ingestion is synchronous, so run it in a worker
        # thread.
        # ----------------------------------------------------

        chunks_created = await run_in_threadpool(
            add_document_to_rag,
            request.content,
            request.source_name,
        )

        # ----------------------------------------------------
        # Operation succeeded.
        # ----------------------------------------------------

        status = "ok"

        return AddDocumentResponse(
            source_name=request.source_name,
            chunks_created=chunks_created,
            status="added",
        )

    except HTTPException:
        raise

    except Exception as error:

        # Log technical details through the normal server logger.
        #
        # The document content itself is not included.
        logger.exception(
            "Document ingestion failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document ingestion failed: "
                f"{error}"
            ),
        ) from error

    finally:

        # ----------------------------------------------------
        # Calculate complete document-processing duration.
        # ----------------------------------------------------

        duration_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # Exactly ONE Task 12 record for this request.
        #
        # Full document content is never logged.
        # ----------------------------------------------------

        log_request(
            trace_id=trace_id,
            endpoint="/add-document",
            masked_text=safe_request_text,
            duration_ms=duration_ms,
            session_id=None,
            status=status,
        )


# ============================================================
# SECTION 18 - WEBSOCKET /ws/chat
# ============================================================


@app.websocket(
    "/ws/chat"
)
async def websocket_chat(
    websocket: WebSocket,
):
    """
    Real-time multi-turn WebSocket chat.

    One WebSocket connection receives one session_id.

    Each successfully received message/turn gets:
        - a fresh trace_id
        - its own timer
        - exactly one JSONL record

    A prompt-injection-blocked turn also gets exactly one
    JSONL record.

    A client disconnect is handled explicitly so the FastAPI
    server continues running for other clients.
    """

    # --------------------------------------------------------
    # Accept the WebSocket connection.
    # --------------------------------------------------------

    await websocket.accept()

    # --------------------------------------------------------
    # Create ONE session ID for this WebSocket connection.
    # --------------------------------------------------------

    session_id = (
        f"ws_{uuid.uuid4()}"
    )

    # --------------------------------------------------------
    # Visible terminal evidence.
    # --------------------------------------------------------

    print(
        f"[WS] connected: {session_id}"
    )

    logger.info(
        "WebSocket connected: %s",
        session_id,
    )

    try:

        # ----------------------------------------------------
        # Continue accepting turns until the client disconnects.
        # ----------------------------------------------------

        while True:

            # ------------------------------------------------
            # Receive ONE complete message.
            #
            # Each received message becomes one unit of work
            # for Task 12.
            # ------------------------------------------------

            query = (
                await websocket.receive_text()
            )

            print(
                f"[WS] {session_id} received: {query}"
            )

            # ------------------------------------------------
            # Generate a NEW trace ID for THIS turn.
            #
            # The session ID identifies the conversation.
            # The trace ID identifies the individual turn.
            # ------------------------------------------------

            trace_id = str(
                uuid.uuid4()
            )

            # ------------------------------------------------
            # Start timing the individual WebSocket turn.
            # ------------------------------------------------

            start_time = time.perf_counter()

            # ------------------------------------------------
            # Safe placeholder before Task 10 executes.
            #
            # Raw query will never be used as a logging fallback.
            # ------------------------------------------------

            masked_query = (
                "<guardrail-processing-failed>"
            )

            # ------------------------------------------------
            # Default failure status.
            # ------------------------------------------------

            status = "error"

            try:

                # ====================================================
                # TASK 10 - INPUT GUARDRAIL
                # ====================================================
                #
                # EXACTLY ONE call for this WebSocket turn.
                # ====================================================

                guardrail_result = (
                    apply_input_guardrails(
                        query
                    )
                )

                # ------------------------------------------------
                # Reuse the exact masked value.
                #
                # This value is sent to:
                #
                #     CrewAI
                #     JSONL logger
                # ------------------------------------------------

                masked_query = str(
                    guardrail_result[
                        "masked_text"
                    ]
                )

                # ====================================================
                # PROMPT-INJECTION CHECK
                # ====================================================

                if not guardrail_result[
                    "allowed"
                ]:

                    # Mark this turn as blocked.
                    status = "blocked"

                    # Inform the WebSocket client.
                    await websocket.send_json(
                        {
                            "error": (
                                "Request blocked by "
                                "prompt-injection "
                                "guardrail."
                            )
                        }
                    )

                    # IMPORTANT:
                    # ------------------------------------------------
                    # Do NOT call log_request() here.
                    #
                    # The finally block below will write exactly
                    # one record for this turn.
                    # ------------------------------------------------

                    continue

                # ====================================================
                # CREWAI EXECUTION
                # ====================================================
                #
                # Only the masked query reaches CrewAI.
                # ====================================================

                result = await run_in_threadpool(
                    run_crew_for_session,
                    masked_query,
                    session_id,
                )

                # ------------------------------------------------
                # Validate against the Task 9 response model.
                # ------------------------------------------------

                validated_result = (
                    AskResponse.model_validate(
                        result.model_dump()
                    )
                )

                # ------------------------------------------------
                # Send the structured response to the WebSocket
                # client.
                # ------------------------------------------------

                await websocket.send_json(
                    validated_result.model_dump()
                )

                print(
                    f"[WS] {session_id} response sent"
                )

                # ------------------------------------------------
                # The turn completed successfully.
                # ------------------------------------------------

                status = "ok"

            except Exception as error:

                # ------------------------------------------------
                # A failed turn should not destroy the entire
                # WebSocket connection.
                # ------------------------------------------------

                logger.exception(
                    "WebSocket request failed for %s.",
                    session_id,
                )

                try:

                    # Attempt to inform the connected client.
                    await websocket.send_json(
                        {
                            "error": (
                                f"Request failed: {error}"
                            )
                        }
                    )

                except Exception:

                    # If the client already disconnected while
                    # sending the error, there is nothing more
                    # to send.
                    pass

                # The finally block will create the one JSONL
                # record for this turn.
                status = "error"

            finally:

                # ====================================================
                # TASK 12 - TURN TIMING
                # ====================================================

                duration_ms = (
                    time.perf_counter()
                    - start_time
                ) * 1000.0

                # ====================================================
                # TASK 12 - EXACTLY ONE JSONL RECORD
                # ====================================================
                #
                # Never pass the raw WebSocket query to the logger.
                # ====================================================

                log_request(
                    trace_id=trace_id,
                    endpoint="/ws/chat",
                    masked_text=masked_query,
                    duration_ms=duration_ms,
                    session_id=session_id,
                    status=status,
                )

    except WebSocketDisconnect:

        # ====================================================
        # TASK 11 - DISCONNECT HANDLING
        # ====================================================
        #
        # This occurs when the client closes the connection.
        #
        # It is intentionally caught so the FastAPI application
        # continues running for other clients.
        # ====================================================

        print(
            "[WS] client disconnected"
        )

        print(
            f"[WS] disconnected session: {session_id}"
        )

        logger.info(
            "WebSocket client disconnected: %s",
            session_id,
        )

    except Exception as error:

        # Unexpected connection-level failure.
        logger.exception(
            "Unexpected WebSocket error for %s: %s",
            session_id,
            error,
        )

    finally:

        # ----------------------------------------------------
        # This WebSocket handler has now finished.
        #
        # Finishing one connection does not terminate the
        # FastAPI server or other client connections.
        # ----------------------------------------------------

        print(
            f"[WS] handler finished: {session_id}"
        )

        logger.info(
            "WebSocket handler finished: %s",
            session_id,
        )


# ============================================================
# END OF API FILE
# ============================================================
#
# Start the application with:
#
#     uvicorn api:app --reload
#
# ============================================================
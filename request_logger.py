# ============================================================
# Task 12 - Structured JSONL Request Logger
# ============================================================
#
# This module has ONE responsibility:
#
#     Convert one structured request record into one JSON line
#     and append it safely to the Task 12 log file.
#
# IMPORTANT:
# ------------------------------------------------------------
# This module DOES NOT perform PII masking.
#
# The caller must pass the already-masked request text produced
# by Task 10.
#
# This keeps ONE masking path for both:
#
#     raw input
#         |
#         v
#     Task 10 guardrail
#         |
#         v
#     masked_text
#       /       \
#      /         \
#     v           v
#   CrewAI      JSONL log
#
# Therefore this file never creates a second masking rule.
# ============================================================


# ============================================================
# SECTION 1 - IMPORTS
# ============================================================

# json converts a Python dictionary into valid JSON text.
import json

# threading provides a lock for concurrent file writes.
import threading

# datetime creates an ISO-8601 UTC timestamp.
from datetime import datetime, timezone

# Path gives us a simple cross-platform file path object.
from pathlib import Path


# ============================================================
# SECTION 2 - LOG FILE CONFIGURATION
# ============================================================

# Create a dedicated directory for Task 12 logs.
LOG_DIR = Path("logs")

# Store one JSON object per physical line.
LOG_FILE = LOG_DIR / "requests.jsonl"

# Protect the complete file append operation.
#
# This matters because:
# - FastAPI /ask can execute in a worker thread.
# - Multiple WebSocket connections can finish turns close together.
#
# The lock ensures one request writes one complete JSON line.
LOG_LOCK = threading.Lock()


# ============================================================
# SECTION 3 - WRITE ONE JSONL RECORD
# ============================================================

def log_request(
    trace_id: str,
    endpoint: str,
    masked_text: str,
    duration_ms: float,
    session_id: str | None = None,
    status: str = "ok",
) -> None:
    """
    Append exactly ONE structured JSON object as ONE line.

    Parameters
    ----------
    trace_id:
        A fresh UUID for this individual request/turn.

    endpoint:
        Endpoint being processed:
            /ask
            /add-document
            /ws/chat

    masked_text:
        Text that is already safe to log.

        For /ask and /ws/chat this MUST be the exact
        Task 10 guardrail_result["masked_text"] value.

        For /add-document the caller supplies a safe summary
        instead of the full document body.

    duration_ms:
        Processing duration in milliseconds.

    session_id:
        Optional session identifier.

    status:
        Request result such as:
            ok
            blocked
            error
    """

    # --------------------------------------------------------
    # Build the structured record.
    # --------------------------------------------------------

    record = {
        # Timestamp in UTC for easy ELK-style indexing.
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        # Fresh identifier for this individual unit of work.
        "trace_id": trace_id,

        # API endpoint being processed.
        "endpoint": endpoint,

        # Conversation session when one exists.
        "session_id": session_id,

        # Measured processing duration.
        "duration_ms": round(
            duration_ms,
            3,
        ),

        # IMPORTANT:
        # This value must already be safe to log.
        "query": masked_text,

        # Simple request outcome.
        "status": status,
    }

    # --------------------------------------------------------
    # Make sure the log directory exists.
    # --------------------------------------------------------

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Lock the COMPLETE append operation.
    #
    # This prevents two concurrent requests from interleaving
    # their JSON data and corrupting the JSONL file.
    # --------------------------------------------------------

    with LOG_LOCK:

        # Open in append mode so existing request records remain.
        with LOG_FILE.open(
            "a",
            encoding="utf-8",
        ) as log_file:

            # Convert the dictionary to JSON and add exactly one
            # newline, creating one JSON-Lines record.
            log_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

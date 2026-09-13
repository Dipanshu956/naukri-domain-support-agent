# ============================================================
# Task 10 - Guardrails
# ============================================================
#
# This file contains ALL Task 10 guardrail logic.
#
# The three required guardrails are:
#
#   1. Input-side PII masking
#      -> Masks fixed-format phone numbers.
#
#   2. Input-side prompt-injection detection
#      -> Detects obvious attempts to override instructions.
#
#   3. Output-side groundedness check
#      -> Refuses an answer when the RAG retrieval decision
#         says the question is not sufficiently supported.
#
# IMPORTANT:
#
# This file does not know anything about:
#
#   - CrewAI
#   - LangChain
#   - ChromaDB
#   - dataset.py
#
# Keeping the guardrails independent makes them easier to test
# and reuse later for Task 12 FastAPI and logging.
# ============================================================


# ============================================================
# SECTION 1 - IMPORTS
# ============================================================


# re is used for regular-expression based detection.
import re


# ============================================================
# SECTION 2 - COMMON FALLBACK MESSAGE
# ============================================================


# This is the same user-facing refusal message used by the
# grounded RAG workflow.
#
# The important change from the previous version is:
#
# The groundedness guardrail will NOT determine groundedness
# by comparing text to this string.
#
# Instead, it will receive an explicit boolean/score state.
# ============================================================
FALLBACK_MESSAGE = (
    "I don't know based on the available knowledge base."
)


# ============================================================
# SECTION 3 - PHONE-NUMBER PII PATTERN
# ============================================================


# ------------------------------------------------------------
# Fixed-format Indian mobile-number pattern.
#
# Supported examples:
#
#   9876543210
#   98765 43210
#   98765-43210
#   +91 98765 43210
#   +91-98765-43210
#
# The assignment specifically identifies the phone number as
# the fixed-format PII field to demonstrate.
# ============================================================
PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+91[\s-]?)?"
    r"[6-9]\d{4}"
    r"[\s-]?"
    r"\d{5}"
    r"(?!\d)"
)


# ============================================================
# SECTION 4 - PROMPT-INJECTION PATTERNS
# ============================================================


# ------------------------------------------------------------
# MOCK_LLM does not contain a semantic intent classifier.
#
# Therefore this guardrail intentionally uses deterministic
# patterns for obvious injection attempts.
#
# Examples:
#
#   "ignore all previous instructions"
#   "disregard previous instructions"
#   "forget previous instructions"
#   "you are now a different assistant"
#   "reveal the system prompt"
# ------------------------------------------------------------
PROMPT_INJECTION_PATTERNS = [

    # "ignore previous instructions"
    re.compile(
        r"\bignore\s+(all\s+)?previous\s+instructions\b",
        re.IGNORECASE
    ),

    # "disregard previous instructions"
    re.compile(
        r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
        re.IGNORECASE
    ),

    # "forget previous instructions"
    re.compile(
        r"\bforget\s+(all\s+)?previous\s+instructions\b",
        re.IGNORECASE
    ),

    # "you are now a different assistant"
    re.compile(
        r"\byou\s+are\s+now\s+(a\s+)?different\s+assistant\b",
        re.IGNORECASE
    ),

    # "reveal the system prompt"
    re.compile(
        r"\breveal\s+(the\s+)?system\s+prompt\b",
        re.IGNORECASE
    ),

    # "show me the system prompt"
    re.compile(
        r"\bshow\s+(me\s+)?(the\s+)?system\s+prompt\b",
        re.IGNORECASE
    ),
]


# ============================================================
# SECTION 5 - PII MASKING
# ============================================================


def mask_phone_numbers(
    text: str
) -> str:
    """
    Mask fixed-format phone numbers.

    Example:

        Input:
            My number is 98765 43210.

        Output:
            My number is XXXXXXXXXX.
    """

    # Convert the input to a string.
    text = str(text)

    # Replace every phone-number match with a safe value.
    masked_text = PHONE_PATTERN.sub(
        "XXXXXXXXXX",
        text
    )

    # Return the masked text.
    return masked_text


# ============================================================
# SECTION 6 - PROMPT-INJECTION DETECTION
# ============================================================


def detect_prompt_injection(
    text: str
):
    """
    Detect an obvious prompt-injection attempt.

    Returns:

        (True, pattern)
            when an injection is detected.

        (False, "")
            otherwise.
    """

    # Convert the value into text.
    text = str(text)

    # Check every configured injection pattern.
    for pattern in PROMPT_INJECTION_PATTERNS:

        # Search the input for the current pattern.
        if pattern.search(text):

            # Return the fact that an injection was found.
            return (
                True,
                pattern.pattern
            )

    # No injection was found.
    return (
        False,
        ""
    )


# ============================================================
# SECTION 7 - COMBINED INPUT GUARDRAIL
# ============================================================


def apply_input_guardrails(
    text: str
):
    """
    Apply all input-side Task 10 guardrails.

    Policy:

        Phone PII:
            Mask it and continue.

        Prompt injection:
            Block the request.

    Returns a dictionary containing:

        original_text
        masked_text
        pii_masked
        injection_detected
        injection_pattern
        allowed

    IMPORTANT:
        The caller must use masked_text when forwarding the
        request to the next layer.
    """

    # --------------------------------------------------------
    # Store the raw value only inside this local operation.
    #
    # The caller should NOT store this raw value in session
    # memory or logs.
    # --------------------------------------------------------
    original_text = str(text)

    # --------------------------------------------------------
    # First guardrail:
    # Mask the fixed-format phone number.
    # --------------------------------------------------------
    masked_text = mask_phone_numbers(
        original_text
    )

    # --------------------------------------------------------
    # Determine whether masking actually changed anything.
    # --------------------------------------------------------
    pii_masked = (
        masked_text != original_text
    )

    # --------------------------------------------------------
    # Second guardrail:
    # Detect prompt injection.
    #
    # Detection is based on what the user originally submitted.
    # --------------------------------------------------------
    (
        injection_detected,
        injection_pattern
    ) = detect_prompt_injection(
        original_text
    )

    # --------------------------------------------------------
    # Our chosen security policy is BLOCK.
    #
    # A detected injection is not sanitized and allowed through.
    # --------------------------------------------------------
    allowed = not injection_detected

    # --------------------------------------------------------
    # Return the complete guardrail result.
    # --------------------------------------------------------
    return {
        "original_text": original_text,
        "masked_text": masked_text,
        "pii_masked": pii_masked,
        "injection_detected": injection_detected,
        "injection_pattern": injection_pattern,
        "allowed": allowed,
    }


# ============================================================
# SECTION 8 - OUTPUT GROUNDEDNESS GUARDRAIL
# ============================================================


def apply_output_groundedness_guardrail(
    final_answer: str,
    query: str,
    grounded: bool,
    top_similarity=None,
    threshold=None
) -> str:
    """
    Apply the output-side groundedness guardrail.

    IMPORTANT:
        Groundedness is supplied explicitly by the RAG layer.

    We do NOT compare:

        rag_result == "I don't know..."

    anymore.

    Instead, the RAG layer stores an explicit boolean:

        grounded = True
        grounded = False

    This makes the guardrail robust against future changes to
    the user-facing fallback wording.
    """

    # --------------------------------------------------------
    # If retrieval says the question is NOT grounded, refuse.
    # --------------------------------------------------------
    if not grounded:

        # ----------------------------------------------------
        # Display clear evidence that the output guardrail
        # actually fired.
        # ----------------------------------------------------
        print(
            "\n[TASK 10] Output groundedness guardrail fired."
        )

        print(
            f"[TASK 10] Query: {query}"
        )

        # ----------------------------------------------------
        # Display similarity when available.
        # ----------------------------------------------------
        if top_similarity is not None:

            print(
                "[TASK 10] Top-1 similarity:"
                f" {top_similarity:.4f}"
            )

        # ----------------------------------------------------
        # Display threshold when available.
        # ----------------------------------------------------
        if threshold is not None:

            print(
                "[TASK 10] Calibrated threshold:"
                f" {threshold:.4f}"
            )

        print(
            "[TASK 10] Final decision: REFUSE"
        )

        # ----------------------------------------------------
        # Return the grounded fallback response.
        # ----------------------------------------------------
        return FALLBACK_MESSAGE

    # --------------------------------------------------------
    # The RAG result was grounded.
    #
    # The final answer can therefore be returned.
    # --------------------------------------------------------
    return final_answer
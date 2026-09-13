# ============================================================
# Task 14 - AutoGen Governance Review Stage
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# This file adds an AutoGen governance review stage AFTER the
# existing CrewAI Task 7-9 Composer has produced its draft answer.
#
# The existing CrewAI system is treated as a FIXED INPUT.

# TASK 14 REQUIREMENTS
# ------------------------------------------------------------
#
# 1. Two AutoGen agents:
#       - Policy-Compliance-Reviewer
#       - Final-Editor
#
# 2. RoundRobinGroupChat with max_turns=2.
#
# 3. Final-Editor uses Pydantic structured output:
#       output_content_type=YourVerdictModel
#
# 4. Team explicitly registers:
#       StructuredMessage[YourVerdictModel]
#
# 5. Team receives:
#       - CrewAI Composer draft answer
#       - Original retrieved RAG context
#
# 6. Verdict schema contains exactly:
#       approved: bool
#       final_answer: str
#       reason: str
#
# 7. Demonstrations:
#       - approved case:
#           real CrewAI draft + real RAG context
#
#       - revised case:
#           same real RAG context + deliberately corrupted draft
#
#
# MOCK LLM DESIGN
# ------------------------------------------------------------
# The capstone requires MOCK_LLM behavior.
#
# No external LLM/API call is made.
#
# Each AutoGen agent receives its own deterministic mock client.
#
#
# REVIEW STATE DESIGN
# ------------------------------------------------------------
# Review state is stored PER CASE rather than in module-level
# global variables.
#
# This is safer because two cases can theoretically be executed
# concurrently without one case overwriting another case's
# Reviewer result.
#
# The current case contains:
#
#     query
#     draft
#     context
#     reviewer result
# ============================================================


# ============================================================
# STANDARD LIBRARY IMPORTS
# ============================================================

import asyncio
import os
import re
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ============================================================
# CREWAI TELEMETRY
# ============================================================

# Disable CrewAI telemetry before importing the existing
# CrewAI implementation.
os.environ.setdefault(
    "CREWAI_DISABLE_TELEMETRY",
    "true",
)


# ============================================================
# PROJECT ROOT
# ============================================================

# __file__ points to this Task 14 Python file.
# Its parent directory is the project root.
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

# Ensure local project modules can be imported.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# EXISTING CREWAI SYSTEM
# ============================================================

# Task 14 reuses the existing real CrewAI implementation.
import crew_agents


# ============================================================
# AUTOGEN IMPORTS
# ============================================================

# Creates the two AutoGen agents.
from autogen_agentchat.agents import AssistantAgent

# StructuredMessage is required for the structured output type.
from autogen_agentchat.messages import StructuredMessage

# Required Round Robin team.
from autogen_agentchat.teams import RoundRobinGroupChat

# AutoGen model client interfaces.
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)

# Required Pydantic base model.
from pydantic import BaseModel


# ============================================================
# TASK 14 VERDICT MODEL
# ============================================================

class YourVerdictModel(BaseModel):
    """
    Exact structured verdict required by Task 14.

    The schema intentionally contains exactly three fields:
        approved
        final_answer
        reason
    """

    # True means the original CrewAI answer is accepted unchanged.
    approved: bool

    # Final answer after governance review.
    final_answer: str

    # Explanation for the decision.
    reason: str


# ============================================================
# PER-CASE REVIEW STATE
# ============================================================

@dataclass
class ReviewState:
    """
    Hold all mutable state for ONE Task 14 review case.

    Keeping this state per case avoids module-level global state
    leaking between independent or concurrent review runs.
    """

    # Original user query.
    query: str = ""

    # CrewAI Composer draft being reviewed.
    draft: str = ""

    # Exact RAG output supplied by the existing CrewAI pipeline.
    context: str = ""

    # Textual decision produced by the Reviewer mock.
    review_text: str = ""

    # Exact unsupported sentences found by the Reviewer.
    unsupported_sentences: List[str] = field(
        default_factory=list
    )


def create_review_state(
    query: str,
    draft: str,
    context: str,
) -> ReviewState:
    """
    Create and return a fresh per-case ReviewState object.

    Parameters
    ----------
    query:
        Original user question.

    draft:
        Real CrewAI Composer answer.

    context:
        Exact retrieved RAG output.

    Returns
    -------
    ReviewState
        Isolated state for this review case.
    """

    return ReviewState(
        query=query,
        draft=draft,
        context=context,
    )


# ============================================================
# STOP WORDS
# ============================================================

# Common words that contribute little evidence to the
# deterministic groundedness check.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
    "after",
    "before",
    "under",
    "into",
    "than",
    "then",
    "also",
    "during",
    "according",
    "based",
    "normally",
    "usually",
    "should",
    "would",
}


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_meaningful(
    text: str,
) -> List[str]:
    """
    Convert text into lowercase meaningful tokens.

    Punctuation is ignored and common stop words are removed.

    IMPORTANT:
    The final '*' is intentional. It means zero or more
    additional characters.

    Correct:
        r"[a-zA-Z][a-zA-Z0-9'-]*"

    Incorrect:
        r"[a-zA-Z][a-zA-Z0-9'-]\\*"
    """

    # Extract normal words from the input text.
    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z0-9'-]*",
        text.lower(),
    )

    # Keep only sufficiently meaningful words.
    return [
        token
        for token in tokens
        if len(token) >= 4
        and token not in STOP_WORDS
    ]


# ============================================================
# BIGRAM CREATION
# ============================================================

def make_meaningful_bigrams(
    tokens: Sequence[str],
) -> List[Tuple[str, str]]:
    """
    Create adjacent pairs of meaningful tokens.

    Example:
        ["notice", "period", "resignation"]

    becomes:
        ("notice", "period")
        ("period", "resignation")

    Bigrams provide phrase-level evidence instead of relying
    only on individual word overlap.
    """

    # Two tokens are needed to create at least one pair.
    if len(tokens) < 2:
        return []

    return [
        (
            tokens[index],
            tokens[index + 1],
        )
        for index in range(
            len(tokens) - 1
        )
    ]


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(
    text: str,
) -> List[str]:
    """
    Split prose into independently reviewable statements.

    Statements are separated by either:

        1. Normal punctuation followed by whitespace.
        2. Newline boundaries.

    The newline handling is important because the existing
    CrewAI Composer may return a truncated sentence without
    final punctuation.

    Example:

        "... employee's"

        "Resigning employees also receive a signing bonus."

    must be reviewed as TWO separate statements.
    """

    # Empty text produces an empty list.
    if not text.strip():
        return []

    # Split at punctuation+whitespace OR newline boundaries.
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip(),
    )

    # Remove empty parts and surrounding whitespace.
    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# GROUNDEDNESS HEURISTIC
# ============================================================

def find_unsupported_sentences(
    draft: str,
    context: str,
) -> List[str]:
    """
    Find draft statements that are insufficiently grounded.

    A statement must satisfy BOTH conditions to be considered
    grounded:

        1. At least 35% of meaningful words occur in the
           retrieved context.

        2. At least one meaningful adjacent phrase occurs in
           the retrieved context.

    This deterministic heuristic is appropriate for the
    MOCK_LLM Task 14 demonstration.
    """

    # --------------------------------------------------------
    # Prepare context tokens.
    # --------------------------------------------------------

    context_tokens_list = tokenize_meaningful(
        context
    )

    # Set gives fast individual-word membership checks.
    context_tokens = set(
        context_tokens_list
    )

    # --------------------------------------------------------
    # Prepare context phrases.
    # --------------------------------------------------------

    context_bigrams = set(
        make_meaningful_bigrams(
            context_tokens_list
        )
    )

    # --------------------------------------------------------
    # Store flagged statements here.
    # --------------------------------------------------------

    unsupported: List[str] = []

    # --------------------------------------------------------
    # Review each draft statement independently.
    # --------------------------------------------------------

    for sentence in split_sentences(
        draft
    ):

        # Extract meaningful words from this statement.
        sentence_tokens = tokenize_meaningful(
            sentence
        )

        # Statements with too few meaningful words are not judged.
        if len(sentence_tokens) < 3:
            continue

        # ----------------------------------------------------
        # Workflow/control message exemption.
        # ----------------------------------------------------
        #
        # Example:
        #
        #   "Please provide an application ID such as APP123."
        #
        # This is a workflow instruction rather than a KB
        # factual claim.

        lowered = sentence.lower()

        workflow_phrases = (
            "please provide an application id",
            "provide an application id",
            "application id such as",
        )

        if any(
            phrase in lowered
            for phrase in workflow_phrases
        ):
            continue

        # ----------------------------------------------------
        # Word-level support.
        # ----------------------------------------------------

        supported_count = sum(
            token in context_tokens
            for token in sentence_tokens
        )

        word_support_ratio = (
            supported_count
            / len(sentence_tokens)
        )

        # ----------------------------------------------------
        # Phrase-level support.
        # ----------------------------------------------------

        sentence_bigrams = set(
            make_meaningful_bigrams(
                sentence_tokens
            )
        )

        shared_bigrams = (
            sentence_bigrams
            & context_bigrams
        )

        has_phrase_support = bool(
            shared_bigrams
        )

        # ----------------------------------------------------
        # Final grounding decision.
        # ----------------------------------------------------
        #
        # BOTH conditions must pass.

        if (
            word_support_ratio < 0.35
            or not has_phrase_support
        ):
            unsupported.append(
                sentence
            )

    return unsupported


# ============================================================
# REVIEWER RESPONSE
# ============================================================

def build_reviewer_response(
    draft: str,
    context: str,
) -> Tuple[str, List[str]]:
    """
    Run the deterministic policy-compliance review.

    Returns
    -------
    tuple
        review_text:
            Machine-readable Reviewer response.

        unsupported_sentences:
            Exact statements that were rejected.
    """

    # Run groundedness detection.
    unsupported = (
        find_unsupported_sentences(
            draft=draft,
            context=context,
        )
    )

    # --------------------------------------------------------
    # No issue.
    # --------------------------------------------------------

    if not unsupported:

        review_text = (
            "COMPLIANT\n"
            "The draft is sufficiently grounded in the "
            "supplied retrieved context."
        )

        return (
            review_text,
            [],
        )

    # --------------------------------------------------------
    # One or more unsupported claims.
    # --------------------------------------------------------

    lines = [
        "ISSUE:"
    ]

    # Add each unsupported sentence.
    for sentence in unsupported:

        lines.append(
            "- UNSUPPORTED_SENTENCE: "
            + sentence
        )

    lines.append(
        "The flagged statement(s) are not sufficiently "
        "grounded in the supplied retrieved context."
    )

    return (
        "\n".join(lines),
        unsupported,
    )


# ============================================================
# REMOVE UNSUPPORTED SENTENCES
# ============================================================

def remove_sentences_from_draft(
    draft: str,
    sentences_to_remove: Sequence[str],
) -> str:
    """
    Remove the exact statements identified by the Reviewer.

    The function uses conservative whitespace cleanup so that
    legitimate words are not accidentally joined together.
    """

    # Start with the original draft.
    revised = draft

    # Remove each exact unsupported statement.
    for sentence in sentences_to_remove:

        revised = revised.replace(
            sentence,
            "",
        )

    # --------------------------------------------------------
    # Correct whitespace cleanup.
    # --------------------------------------------------------

    # Collapse repeated spaces and tabs.
    revised = re.sub(
        r"[ \t]+",
        " ",
        revised,
    )

    # Remove surrounding spaces on line boundaries.
    revised = re.sub(
        r" *\n *",
        "\n",
        revised,
    )

    # Reduce excessive blank lines.
    revised = re.sub(
        r"\n{3,}",
        "\n\n",
        revised,
    )

    # Remove spaces before punctuation.
    revised = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        revised,
    )

    # Return cleaned final text.
    return revised.strip()


# ============================================================
# DETERMINISTIC AUTOGEN MOCK CLIENT
# ============================================================

class DeterministicReviewClient(
    ChatCompletionClient
):
    """
    Local deterministic AutoGen model client.

    Parameters
    ----------
    role:
        Either "reviewer" or "editor".

    state:
        Per-case ReviewState shared by the Reviewer and Editor
        belonging to the same review run.

    No external network request is made.
    """

    def __init__(
        self,
        role: str,
        state: ReviewState,
    ):

        # Store the role of this mock client.
        self.role = role

        # Store the isolated state for this review case.
        self.state = state

        # ----------------------------------------------------
        # Usage information.
        # ----------------------------------------------------

        self._actual_usage = RequestUsage(
            prompt_tokens=0,
            completion_tokens=0,
        )

        self._total_usage = RequestUsage(
            prompt_tokens=0,
            completion_tokens=0,
        )

        # ----------------------------------------------------
        # AutoGen model metadata.
        # ----------------------------------------------------

        self._model_info: ModelInfo = {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
            "multiple_system_messages": True,
        }

    # ========================================================
    # CREATE
    # ========================================================

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
        tool_choice="auto",
        json_output: Optional[
            bool | type[BaseModel]
        ] = None,
        extra_create_args: Mapping[
            str,
            Any,
        ] = {},
        cancellation_token=None,
    ) -> CreateResult:
        """
        Produce one deterministic response.

        The exact review input is read from the per-case ReviewState
        rather than being reconstructed from AutoGen message history.
        """

        # ----------------------------------------------------
        # Approximate prompt-token count.
        # ----------------------------------------------------

        prompt_text = " ".join(
            str(
                getattr(
                    message,
                    "content",
                    "",
                )
            )
            for message in messages
        )

        prompt_token_count = len(
            prompt_text.split()
        )

        # ----------------------------------------------------
        # Read exact per-case data.
        # ----------------------------------------------------

        draft = self.state.draft
        context = self.state.context

        # ====================================================
        # REVIEWER
        # ====================================================

        if self.role == "reviewer":

            # Run the deterministic governance check.
            review_text, unsupported = (
                build_reviewer_response(
                    draft=draft,
                    context=context,
                )
            )

            # Store the Reviewer result in THIS case's state.
            self.state.review_text = review_text

            self.state.unsupported_sentences = list(
                unsupported
            )

            # Return Reviewer response.
            response_text = review_text

        # ====================================================
        # FINAL EDITOR
        # ====================================================

        elif self.role == "editor":

            # ------------------------------------------------
            # Read the exact Reviewer result from the same
            # per-case state object.
            # ------------------------------------------------

            review_text = self.state.review_text

            reviewer_unsupported = list(
                self.state.unsupported_sentences
            )

            # ------------------------------------------------
            # Independent safety verification.
            # ------------------------------------------------
            #
            # Even if Reviewer state were unexpectedly empty,
            # the Editor re-checks the draft deterministically.

            independent_unsupported = (
                find_unsupported_sentences(
                    draft=draft,
                    context=context,
                )
            )

            # ------------------------------------------------
            # Combine both finding lists without duplicates.
            # ------------------------------------------------

            unsupported: List[str] = []

            for sentence in reviewer_unsupported:

                if sentence not in unsupported:

                    unsupported.append(
                        sentence
                    )

            for sentence in independent_unsupported:

                if sentence not in unsupported:

                    unsupported.append(
                        sentence
                    )

            # ------------------------------------------------
            # APPROVE
            # ------------------------------------------------

            # The Editor approves ONLY if:
            #
            #   Reviewer says COMPLIANT
            #
            # AND:
            #
            #   no unsupported sentence exists.

            if (
                review_text.startswith(
                    "COMPLIANT"
                )
                and not unsupported
            ):

                verdict = YourVerdictModel(
                    approved=True,
                    final_answer=draft,
                    reason=(
                        "Approved unchanged because the Reviewer "
                        "and deterministic groundedness check found "
                        "no unsupported claim."
                    ),
                )

            # ------------------------------------------------
            # REVISE
            # ------------------------------------------------

            else:

                revised_answer = (
                    remove_sentences_from_draft(
                        draft=draft,
                        sentences_to_remove=unsupported,
                    )
                )

                verdict = YourVerdictModel(
                    approved=False,
                    final_answer=revised_answer,
                    reason=(
                        "Revised because the governance review "
                        "detected a claim that was not sufficiently "
                        "grounded in the supplied retrieved context."
                    ),
                )

            # Return JSON for AutoGen's structured-output processing.
            response_text = (
                verdict.model_dump_json()
            )

        else:

            raise ValueError(
                f"Unknown deterministic client role: {self.role}"
            )

        # ====================================================
        # USAGE ACCOUNTING
        # ====================================================

        completion_token_count = len(
            response_text.split()
        )

        self._actual_usage = RequestUsage(
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
        )

        self._total_usage.prompt_tokens += (
            prompt_token_count
        )

        self._total_usage.completion_tokens += (
            completion_token_count
        )

        # ====================================================
        # AUTOGEN RESULT
        # ====================================================

        return CreateResult(
            finish_reason="stop",
            content=response_text,
            usage=self._actual_usage,
            cached=False,
        )

    # ========================================================
    # CREATE STREAM
    # ========================================================

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
        tool_choice="auto",
        json_output: Optional[
            bool | type[BaseModel]
        ] = None,
        extra_create_args: Mapping[
            str,
            Any,
        ] = {},
        cancellation_token=None,
    ) -> AsyncGenerator[
        str | CreateResult,
        None,
    ]:
        """
        Provide the minimal streaming interface required by
        ChatCompletionClient.

        Task 14 does not require streaming, so the complete
        deterministic result is emitted as one chunk.
        """

        async def generator() -> AsyncGenerator[
            str | CreateResult,
            None,
        ]:

            # Reuse the normal deterministic create() implementation.
            result = await self.create(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )

            # Yield textual response content.
            if isinstance(
                result.content,
                str,
            ):

                yield result.content

            # Yield final CreateResult.
            yield result

        return generator()

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(
        self,
    ) -> None:
        """
        Close the deterministic client.

        No external resources exist, so nothing needs to be closed.
        """

        return None

    # ========================================================
    # ACTUAL USAGE
    # ========================================================

    def actual_usage(
        self,
    ) -> RequestUsage:
        """
        Return usage information for the most recent completion.
        """

        return self._actual_usage

    # ========================================================
    # TOTAL USAGE
    # ========================================================

    def total_usage(
        self,
    ) -> RequestUsage:
        """
        Return accumulated usage for this mock client instance.
        """

        return self._total_usage

    # ========================================================
    # COUNT TOKENS
    # ========================================================

    def count_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
    ) -> int:
        """
        Return a simple whitespace-based token estimate.

        This is sufficient for local MOCK_LLM usage accounting.
        """

        return sum(
            len(
                str(
                    getattr(
                        message,
                        "content",
                        "",
                    )
                ).split()
            )
            for message in messages
        )

    # ========================================================
    # REMAINING TOKENS
    # ========================================================

    def remaining_tokens(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools=(),
    ) -> int:
        """
        Return the synthetic number of remaining tokens available
        to the deterministic mock.
        """

        return max(
            0,
            100_000
            - self.count_tokens(
                messages,
                tools=tools,
            ),
        )

    # ========================================================
    # CAPABILITIES
    # ========================================================

    @property
    def capabilities(
        self,
    ):
        """
        Return the configured AutoGen model capabilities.
        """

        return self._model_info

    # ========================================================
    # MODEL INFO
    # ========================================================

    @property
    def model_info(
        self,
    ) -> ModelInfo:
        """
        Return model metadata used by AutoGen.
        """

        return self._model_info


# ============================================================
# CREWAI FINAL ANSWER EXTRACTION
# ============================================================

def extract_crew_final_answer(
    crew_result: Any,
) -> str:
    """
    Extract the final answer from the existing CrewAI result.

    The primary expected field is final_answer.

    Compatibility fallbacks are included for wrappers exposing
    output, raw or content.
    """

    # --------------------------------------------------------
    # Primary CrewResponse field.
    # --------------------------------------------------------

    value = getattr(
        crew_result,
        "final_answer",
        None,
    )

    if value is not None:

        return str(
            value
        ).strip()

    # --------------------------------------------------------
    # Compatibility fallback fields.
    # --------------------------------------------------------

    for attribute in (
        "output",
        "raw",
        "content",
    ):

        value = getattr(
            crew_result,
            attribute,
            None,
        )

        if value is not None:

            return str(
                value
            ).strip()

    # --------------------------------------------------------
    # Final fallback.
    # --------------------------------------------------------

    return str(
        crew_result
    ).strip()


# ============================================================
# RUN REAL CREWAI
# ============================================================

def run_real_crew(
    query: str,
    session_id: str,
) -> Tuple[str, str]:
    """
    Execute the real existing CrewAI pipeline.

    Returns
    -------
    tuple
        draft:
            Actual CrewAI Composer final answer.

        context:
            Exact RAG result stored by the existing rag_search tool.
    """

    # --------------------------------------------------------
    # Clear state generated by previous CrewAI executions.
    # --------------------------------------------------------

    crew_agents.LAST_TOOL_RESULT.clear()

    crew_agents.LAST_RAG_TOP_SIMILARITY = None

    crew_agents.LAST_RAG_GROUNDED = None

    # --------------------------------------------------------
    # Verify the expected memory-aware CrewAI entry point.
    # --------------------------------------------------------

    if not hasattr(
        crew_agents,
        "crew_with_memory",
    ):

        raise AttributeError(
            "crew_agents.py must expose crew_with_memory for Task 14."
        )

    # --------------------------------------------------------
    # Execute the real CrewAI workflow.
    # --------------------------------------------------------

    crew_result = (
        crew_agents.crew_with_memory.invoke(
            {"query": query},
            config={
                "configurable": {
                    "session_id": session_id,
                }
            },
        )
    )

    # --------------------------------------------------------
    # Extract the real Composer answer.
    # --------------------------------------------------------

    draft = extract_crew_final_answer(
        crew_result
    )

    # --------------------------------------------------------
    # Extract exact RAG context recorded by rag_search().
    # --------------------------------------------------------

    context = str(
        crew_agents.LAST_TOOL_RESULT.get(
            "rag_search",
            "",
        )
    ).strip()

    # RAG evidence is mandatory for this Task 14 demonstration.
    if not context:

        raise RuntimeError(
            "The existing CrewAI run did not expose "
            "LAST_TOOL_RESULT['rag_search']."
        )

    return (
        draft,
        context,
    )


# ============================================================
# BUILD AUTOGEN REVIEW TASK
# ============================================================

def build_review_task(
    query: str,
    draft: str,
    context: str,
) -> str:
    """
    Build the exact payload supplied to the AutoGen team.

    The payload explicitly contains:

        - original user query
        - CrewAI draft answer
        - original retrieved RAG context
    """

    return (
        "=== USER QUERY START ===\n"
        f"{query}\n"
        "=== USER QUERY END ===\n\n"

        "=== DRAFT ANSWER START ===\n"
        f"{draft}\n"
        "=== DRAFT ANSWER END ===\n\n"

        "=== RETRIEVED CONTEXT START ===\n"
        f"{context}\n"
        "=== RETRIEVED CONTEXT END ===\n\n"

        "Governance instructions:\n"
        "1. The draft was produced by the existing CrewAI Composer.\n"
        "2. The retrieved context is authoritative for KB claims.\n"
        "3. The Policy-Compliance-Reviewer must identify unsupported claims.\n"
        "4. The Final-Editor approves a grounded draft unchanged.\n"
        "5. The Final-Editor removes unsupported claims.\n"
        "6. Do not invent new facts."
    )


# ============================================================
# BUILD THE TWO-AGENT AUTOGEN TEAM
# ============================================================

def build_review_team(
    state: ReviewState,
) -> RoundRobinGroupChat:
    """
    Build the required two-agent AutoGen team.

    Both deterministic mock clients receive the SAME per-case
    ReviewState object.

    This gives Reviewer -> Editor a reliable handoff without
    relying on module-level global variables.
    """

    # --------------------------------------------------------
    # Dedicated deterministic Reviewer client.
    # --------------------------------------------------------

    reviewer_client = (
        DeterministicReviewClient(
            role="reviewer",
            state=state,
        )
    )

    # --------------------------------------------------------
    # Dedicated deterministic Editor client.
    # --------------------------------------------------------

    editor_client = (
        DeterministicReviewClient(
            role="editor",
            state=state,
        )
    )

    # ========================================================
    # POLICY-COMPLIANCE-REVIEWER
    # ========================================================
    #
    # Human-readable assignment role:
    #
    #     Policy-Compliance-Reviewer
    #
    # Internal AutoGen name:
    #
    #     policy_compliance_reviewer
    #
    # The underscore version is required because AutoGen expects
    # an identifier-compatible agent name.

    reviewer = AssistantAgent(
        name="policy_compliance_reviewer",
        model_client=reviewer_client,
        system_message=(
            "You are the Policy-Compliance-Reviewer. "
            "Review the CrewAI draft only against the supplied "
            "retrieved context. Return a deterministic review "
            "beginning with COMPLIANT or ISSUE:."
        ),
    )

    # ========================================================
    # FINAL-EDITOR
    # ========================================================
    #
    # Human-readable assignment role:
    #
    #     Final-Editor
    #
    # Internal AutoGen name:
    #
    #     final_editor

    editor = AssistantAgent(
        name="final_editor",
        model_client=editor_client,
        system_message=(
            "You are the Final-Editor. "
            "Use the governance review to approve or revise the "
            "CrewAI draft. Return the required structured verdict."
        ),

        # REQUIRED Task 14 Pydantic structured output.
        output_content_type=YourVerdictModel,
    )

    # ========================================================
    # ROUND ROBIN TEAM
    # ========================================================
    #
    # Exactly two agent turns:
    #
    #   Turn 1 -> Reviewer
    #   Turn 2 -> Final-Editor

    team = RoundRobinGroupChat(
        participants=[
            reviewer,
            editor,
        ],

        # Required maximum turn count.
        max_turns=2,

        # Required explicit structured message registration.
        custom_message_types=[
            StructuredMessage[
                YourVerdictModel
            ]
        ],
    )

    return team


# ============================================================
# RUN ONE REVIEW CASE
# ============================================================

async def run_review_case(
    query: str,
    draft: str,
    context: str,
) -> YourVerdictModel:
    """
    Run one complete Task 14 governance case.

    Steps:

        1. Create isolated per-case state.
        2. Build the two-agent AutoGen team.
        3. Build the task containing query + draft + context.
        4. Execute Reviewer -> Final-Editor.
        5. Return the structured Final-Editor verdict.

    Because the state is local to this function, separate review
    cases do not share mutable governance data.
    """

    # --------------------------------------------------------
    # Create isolated state for this case.
    # --------------------------------------------------------

    state = create_review_state(
        query=query,
        draft=draft,
        context=context,
    )

    # --------------------------------------------------------
    # Build fresh team using this case's state.
    # --------------------------------------------------------

    team = build_review_team(
        state=state,
    )

    # --------------------------------------------------------
    # Build the complete AutoGen task.
    # --------------------------------------------------------

    task = build_review_task(
        query=query,
        draft=draft,
        context=context,
    )

    # --------------------------------------------------------
    # Execute exactly two AutoGen turns.
    # --------------------------------------------------------

    result = await team.run(
        task=task
    )

    # --------------------------------------------------------
    # Find the Final-Editor's structured result.
    # --------------------------------------------------------

    for message in reversed(
        result.messages
    ):

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(
            content,
            YourVerdictModel,
        ):

            return content

    # --------------------------------------------------------
    # No structured verdict was produced.
    # --------------------------------------------------------

    raise RuntimeError(
        "Task 14 failed: Final-Editor did not return "
        "YourVerdictModel."
    )


# ============================================================
# DISPLAY ONE DEMONSTRATION
# ============================================================

def print_case(
    title: str,
    query: str,
    draft: str,
    verdict: YourVerdictModel,
) -> None:
    """
    Print one Task 14 demonstration in a readable format.
    """

    print(
        "\n" + "=" * 78
    )

    print(title)

    print(
        "=" * 78
    )

    print("Query:")

    print(query)

    print("\nDraft answer:")

    print(draft)

    print("\nStructured AutoGen verdict:")

    # Pydantic dictionary representation.
    print(
        verdict.model_dump()
    )


# ============================================================
# MAIN TASK 14 DEMONSTRATIONS
# ============================================================

def main() -> None:
    """
    Execute the two required Task 14 demonstrations.

    Demonstration 1:
        Real CrewAI draft + real RAG context -> approved unchanged.

    Demonstration 2:
        Same RAG context + intentionally corrupted draft
        -> rejected/revised.
    """

    print(
        "=" * 78
    )

    print(
        "TASK 14 - AUTOGEN RESILIENCE & GOVERNANCE REVIEW"
    )

    print(
        "=" * 78
    )

    print(
        "[CONFIG] External LLM/API calls: DISABLED"
    )

    print(
        "[CONFIG] Existing CrewAI system: FIXED INPUT"
    )

    print(
        "[CONFIG] AutoGen RoundRobinGroupChat max_turns: 2"
    )

    # ========================================================
    # DEMONSTRATION 1 - APPROVED CASE
    # ========================================================

    approved_query = (
        "What is the normal employee notice period after resignation?"
    )

    print(
        "\n[RUN] Building the real CrewAI approved-case draft..."
    )

    # Run real CrewAI.
    approved_draft, approved_context = (
        run_real_crew(
            query=approved_query,
            session_id="task14-approved",
        )
    )

    # Run the real AutoGen governance stage.
    approved_verdict = asyncio.run(
        run_review_case(
            query=approved_query,
            draft=approved_draft,
            context=approved_context,
        )
    )

    # --------------------------------------------------------
    # Approved-case assertions.
    # --------------------------------------------------------

    # A grounded real CrewAI draft must be approved.
    if not approved_verdict.approved:

        raise AssertionError(
            "Task 14 approved demonstration failed: "
            "grounded CrewAI draft was revised."
        )

    # Approval means the answer must remain EXACTLY unchanged.
    if (
        approved_verdict.final_answer
        != approved_draft
    ):

        raise AssertionError(
            "Task 14 approved demonstration failed: "
            "approved answer changed."
        )

    # Display approved demonstration.
    print_case(
        title="DEMONSTRATION 1 - APPROVED UNCHANGED",
        query=approved_query,
        draft=approved_draft,
        verdict=approved_verdict,
    )

    print(
        "\n[PASS] Approved case kept the CrewAI draft unchanged."
    )

    # ========================================================
    # DEMONSTRATION 2 - REVISED CASE
    # ========================================================
    #
    # IMPORTANT:
    #
    # Same query.
    # Same real RAG context.
    # Only the draft is modified.
    #
    # The deliberately fabricated claim is:
    #
    #     Resigning employees also receive a signing bonus.
    #
    # This must be rejected by the governance stage.

    revised_query = approved_query

    corrupted_draft = (
        approved_draft.rstrip()
        + "\n\n"
        + "Resigning employees also receive a signing bonus."
    )

    print(
        "\n[RUN] Running the deliberately corrupted draft..."
    )

    revised_verdict = asyncio.run(
        run_review_case(
            query=revised_query,
            draft=corrupted_draft,
            context=approved_context,
        )
    )

    # --------------------------------------------------------
    # Assertion 1:
    # corrupted answer must NOT be approved.
    # --------------------------------------------------------

    if revised_verdict.approved:

        raise AssertionError(
            "Task 14 revised demonstration failed: "
            "corrupted draft was approved."
        )

    # --------------------------------------------------------
    # Assertion 2:
    # fabricated claim must disappear.
    # --------------------------------------------------------

    if (
        "signing bonus"
        in revised_verdict.final_answer.lower()
    ):

        raise AssertionError(
            "Task 14 revised demonstration failed: "
            "unsupported signing-bonus claim remains."
        )

    # --------------------------------------------------------
    # Assertion 3:
    # after removing only the fabricated sentence,
    # the resulting answer should equal the clean real
    # CrewAI answer.
    # --------------------------------------------------------
    #
    # This is a stronger proof that the governance stage removed
    # only the injected claim instead of rewriting the entire answer.

    if (
        revised_verdict.final_answer
        != approved_draft
    ):

        raise AssertionError(
            "Task 14 revised demonstration failed: "
            "revised answer differs from the original clean CrewAI draft."
        )

    # Display revised demonstration.
    print_case(
        title="DEMONSTRATION 2 - REVISED AFTER UNSUPPORTED CLAIM",
        query=revised_query,
        draft=corrupted_draft,
        verdict=revised_verdict,
    )

    print(
        "\n[PASS] Revised case removed the deliberately unsupported claim."
    )

    # ========================================================
    # FINAL VALIDATION SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 78
    )

    print(
        "TASK 14 VALIDATION SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        "[PASS] Two AutoGen agents created."
    )

    print(
        "[PASS] RoundRobinGroupChat used."
    )

    print(
        "[PASS] max_turns=2 used."
    )

    print(
        "[PASS] Final-Editor uses output_content_type=YourVerdictModel."
    )

    print(
        "[PASS] StructuredMessage[YourVerdictModel] registered on the Team."
    )

    print(
        "[PASS] CrewAI draft + retrieved RAG context reach the Team."
    )

    print(
        "[PASS] Approved demonstration completed."
    )

    print(
        "[PASS] Revised demonstration completed."
    )

    print(
        "[PASS] Revised answer returned to the original clean CrewAI draft."
    )

    print(
        "[PASS] Task 14 governance stage completed."
    )


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Start the Task 14 demonstrations.
    main()
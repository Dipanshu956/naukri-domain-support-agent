
# ============================================================
# Task 15 - Four-Layer AI Governance
# ============================================================
#
# PURPOSE
# ------------------------------------------------------------
# This file demonstrates the governance controls required by
# Task 15 of the Naukri.com Domain Support Agent capstone.
#
# TASK 15 REQUIREMENTS COVERED
# ------------------------------------------------------------
#
# APPLICATION LAYER
#
# 1. Least autonomy:
#       Only the Lookup Agent may use
#       check_job_application_status.
#
# 2. Risk classification:
#       Low / Medium / High.
#       This system is classified as High because it operates
#       in the recruitment / hiring domain specified by the brief.
#
# RUNTIME LAYER
#
# 3. Per-request token/cost budget:
#       - token limit
#       - synthetic request-cost limit
#       - oversized requests fail closed
#
# NOTE ABOUT COST
# ------------------------------------------------------------
# The capstone uses MOCK_LLM, so this demonstration does not
# represent real provider billing.
#
# Instead, a clearly labelled synthetic cost model is used.
# The token-counting approach is intentionally simple and uses
# whitespace splitting, consistent with the deterministic
# token-counting approach already used in Task 14.
# ============================================================


# ------------------------------------------------------------
# Standard-library imports
# ------------------------------------------------------------

from typing import Callable, Dict, List, Tuple


# ------------------------------------------------------------
# Existing project import
# ------------------------------------------------------------

# We inspect the real CrewAI agents created by crew_agents.py.
#
# This is stronger than checking source code manually because
# the governance test examines the live Agent objects.
import crew_agents


# ============================================================
# APPLICATION LAYER
# ============================================================
#
# SECTION 1 - LEAST AUTONOMY / LEAST PRIVILEGE
# ============================================================


# The sensitive application-status tool is identified by its
# actual CrewAI tool name.
LOOKUP_TOOL_NAME = "check_job_application_status"


def get_tool_names(agent) -> List[str]:
    """
    Return the names of all tools attached to one CrewAI agent.

    SIGNIFICANCE
    --------------------------------------------------------
    The Task 15 requirement is about limiting agent autonomy.
    Therefore we must inspect what tools an agent actually owns.

    crew_agents.py already provides:
        get_agent_tools()
        get_tool_name()

    Reusing those helpers keeps this governance file aligned
    with the application's existing tool representation instead
    of creating a second, different tool-discovery mechanism.

    RETURNS
    --------------------------------------------------------
    A list containing the real names of the agent's tools.
    """

    # Use the existing helper from crew_agents.py to obtain the
    # live Agent.tools collection.
    tools = crew_agents.get_agent_tools(agent)

    # This list will contain the real tool names.
    tool_names: List[str] = []

    # Inspect each live tool object.
    for tool in tools:

        # Use the existing project helper to obtain its name.
        tool_name = crew_agents.get_tool_name(tool)

        # Ignore unnamed objects because they cannot establish
        # ownership of the privileged lookup tool.
        if tool_name:
            tool_names.append(tool_name)

    # Return the actual tool names found on this live agent.
    return tool_names


def verify_least_autonomy() -> Tuple[object, object, object]:
    """
    Verify the least-autonomy policy on the live CrewAI agents.

    REQUIRED POLICY
    --------------------------------------------------------
    Retrieval Agent:
        Must NOT own check_job_application_status.

    Lookup Agent:
        MUST own check_job_application_status.

    Composer Agent:
        Must NOT own check_job_application_status.

    SIGNIFICANCE
    --------------------------------------------------------
    Simply saying "only Lookup Agent has the tool" is weaker
    than checking the real objects.

    This function therefore:
        1. creates the production agents,
        2. prints their actual tool lists,
        3. finds the real owner(s) of the lookup tool,
        4. requires exactly one owner,
        5. checks the other two agents explicitly.

    If unsafe wiring is introduced later, this function raises
    AssertionError and the governance check fails closed.
    """

    # Create the exact three agents used by the existing
    # CrewAI application.
    retrieval_agent, lookup_agent, composer_agent = (
        crew_agents.create_agents()
    )

    # Give each live object a simple display name for the
    # governance report.
    agents = {
        "Retrieval Agent": retrieval_agent,
        "Lookup Agent": lookup_agent,
        "Composer Agent": composer_agent,
    }

    # --------------------------------------------------------
    # Print direct evidence.
    # --------------------------------------------------------
    #
    # This is important for the capstone demonstration because
    # the evaluator can directly see which tools each agent owns.
    print("\n" + "=" * 78)
    print("APPLICATION LAYER - LEAST AUTONOMY CHECK")
    print("=" * 78)

    for label, agent in agents.items():

        # Read the actual tools from the live object.
        tool_names = get_tool_names(agent)

        print(f"{label}: {tool_names}")

    # --------------------------------------------------------
    # Find all actual owners of the sensitive lookup tool.
    # --------------------------------------------------------

    actual_owners: List[str] = []

    for label, agent in agents.items():

        # Read the actual tool list for this agent.
        tool_names = get_tool_names(agent)

        # Record the agent if it owns the privileged lookup tool.
        if LOOKUP_TOOL_NAME in tool_names:
            actual_owners.append(label)

    # --------------------------------------------------------
    # ENFORCEMENT RULE
    # --------------------------------------------------------
    #
    # Exactly one owner must exist, and that owner must be
    # the Lookup Agent.
    if actual_owners != ["Lookup Agent"]:

        raise AssertionError(
            "Least-autonomy violation: "
            f"{LOOKUP_TOOL_NAME!r} is owned by "
            f"{actual_owners}, but only the Lookup Agent "
            "is permitted to own it."
        )

    # --------------------------------------------------------
    # Explicitly prove that Retrieval cannot call the tool.
    # --------------------------------------------------------

    retrieval_tools = get_tool_names(retrieval_agent)

    if LOOKUP_TOOL_NAME in retrieval_tools:

        raise AssertionError(
            "Least-autonomy violation: Retrieval Agent owns "
            f"{LOOKUP_TOOL_NAME!r}."
        )

    # --------------------------------------------------------
    # Explicitly prove that Composer cannot call the tool.
    # --------------------------------------------------------

    composer_tools = get_tool_names(composer_agent)

    if LOOKUP_TOOL_NAME in composer_tools:

        raise AssertionError(
            "Least-autonomy violation: Composer Agent owns "
            f"{LOOKUP_TOOL_NAME!r}."
        )

    # --------------------------------------------------------
    # Finally, prove that Lookup really has the tool.
    # --------------------------------------------------------

    lookup_tools = get_tool_names(lookup_agent)

    if LOOKUP_TOOL_NAME not in lookup_tools:

        raise AssertionError(
            "Least-autonomy violation: Lookup Agent does not "
            f"own the required {LOOKUP_TOOL_NAME!r} tool."
        )

    # Everything passed.
    print(
        "\n[PASS] Only the Lookup Agent owns "
        "check_job_application_status."
    )

    print(
        "[PASS] Retrieval Agent and Composer Agent do not own "
        "the application-status lookup tool."
    )

    print(
        "[PASS] Least-autonomy invariant is enforced."
    )

    # Return the live objects in case another governance test
    # needs them later.
    return (
        retrieval_agent,
        lookup_agent,
        composer_agent,
    )


# ------------------------------------------------------------
# One-paragraph governance explanation.
# ------------------------------------------------------------

LEAST_AUTONOMY_EXPLANATION = (
    "The application layer applies the principle of least autonomy "
    "by treating check_job_application_status as a privileged tool "
    "that belongs only to the Lookup Agent. The governance check "
    "creates the live CrewAI agents, inspects their actual tool "
    "collections, prints the observed wiring, and requires exactly "
    "one owner for the privileged tool. Retrieval Agent and Response "
    "Composer Agent are explicitly checked so they cannot own the "
    "tool. If a future developer accidentally wires the lookup tool "
    "to another agent, or removes it from the Lookup Agent, the "
    "governance check raises an AssertionError and refuses to accept "
    "the configuration. Therefore the restriction is an enforced "
    "startup invariant rather than only a statement in documentation."
)


# ============================================================
# APPLICATION LAYER
# ============================================================
#
# SECTION 2 - RISK CLASSIFICATION
# ============================================================


# The capstone provides three categories:
#       Low
#       Medium
#       High
#
# The supplied brief places hiring-related systems in the High
# category, so this system is classified as High.
RISK_LEVEL = "High"


# This is the one-paragraph explanation required by Task 15.
RISK_JUSTIFICATION = (
    "This Naukri.com domain support system is classified as High risk "
    "under the capstone's Low/Medium/High scheme because it operates "
    "in the recruitment and hiring domain and handles job-application "
    "information such as application status, expected salary, and an "
    "escalation score. Incorrect, misleading, or improperly exposed "
    "information could affect a candidate's recruitment experience or "
    "expose sensitive application information. The system is a support "
    "agent rather than an autonomous hiring decision-maker, but the "
    "underlying use case is still within the hiring-related High-risk "
    "category specified by the capstone."
)


def print_risk_classification() -> None:
    """
    Display and validate the system's risk classification.

    SIGNIFICANCE
    --------------------------------------------------------
    Governance requires not only implementing controls but also
    identifying the level of risk associated with the system.

    This function:
        1. prints the selected risk level,
        2. prints the justification,
        3. validates that the selected value is one of the
           permitted Low/Medium/High labels,
        4. verifies that this particular capstone system is
           classified as High.
    """

    print("\n" + "=" * 78)
    print("APPLICATION LAYER - RISK CLASSIFICATION")
    print("=" * 78)

    # Display the selected classification.
    print(f"Risk level: {RISK_LEVEL}")

    # Display the required justification.
    print("\nJustification:")
    print(RISK_JUSTIFICATION)

    # Only values from the capstone scheme are valid.
    if RISK_LEVEL not in {"Low", "Medium", "High"}:

        raise AssertionError(
            "Invalid risk level. "
            "Use Low, Medium, or High."
        )

    # This system is in the hiring/recruitment domain specified
    # by the capstone, therefore High is the intended classification.
    if RISK_LEVEL != "High":

        raise AssertionError(
            "The Naukri.com hiring/HR support system should be "
            "classified as High risk under the provided scheme."
        )

    print("\n[PASS] Risk level is classified as High.")


# ============================================================
# RUNTIME LAYER
# ============================================================
#
# SECTION 3 - PER-REQUEST TOKEN/COST BUDGET
# ============================================================


# ------------------------------------------------------------
# Synthetic token budget
# ------------------------------------------------------------
#
# This is intentionally a small demonstration limit.
MAX_REQUEST_TOKENS = 2_000


# ------------------------------------------------------------
# Synthetic cost model
# ------------------------------------------------------------
#
# This is NOT real provider billing.
#
# It is a simple deterministic cost model for demonstrating
# runtime governance while using MOCK_LLM.
SYNTHETIC_COST_PER_TOKEN_USD = 0.00001


# ------------------------------------------------------------
# Independent maximum cost
# ------------------------------------------------------------
#
# IMPORTANT IMPROVEMENT
# ------------------------------------------------------------
# In the earlier implementation:
#
#     MAX_REQUEST_COST_USD =
#         MAX_REQUEST_TOKENS * rate
#
# the cost check was mathematically redundant.
#
# Here the maximum cost is intentionally independent.
#
# Therefore there are now two genuine runtime controls:
#
#     1. token ceiling
#     2. synthetic cost ceiling
#
# This makes the governance model easier to defend.
MAX_REQUEST_COST_USD = 0.015


class RequestBudgetExceeded(Exception):
    """
    Custom exception raised when a request violates the budget.

    SIGNIFICANCE
    --------------------------------------------------------
    A dedicated exception makes the fail-closed behavior explicit.

    Instead of silently continuing with an oversized request,
    the system stops and reports the governance violation.
    """

    pass


def count_tokens(text: str) -> int:
    """
    Approximate the number of request tokens.

    SIGNIFICANCE
    --------------------------------------------------------
    Because this capstone uses MOCK_LLM, we do not need a
    provider-specific tokenizer for the governance demonstration.

    Whitespace splitting provides a deterministic and easy-to-
    reproduce approximation that is consistent with the simple
    token-counting approach already used by Task 14.

    EXAMPLE
    --------------------------------------------------------
        "hello world here"

    becomes approximately:

        3 tokens
    """

    # Split the text on whitespace and count the resulting items.
    return len(text.split())


def estimate_request_cost_usd(token_count: int) -> float:
    """
    Calculate the synthetic cost for a request.

    SIGNIFICANCE
    --------------------------------------------------------
    This function converts the token estimate into a predictable
    synthetic cost.

    It does not represent actual provider pricing. It exists so
    Task 15 can demonstrate a runtime cost-control policy even
    though MOCK_LLM produces no real API bill.
    """

    # Multiply token count by the configured synthetic price.
    return token_count * SYNTHETIC_COST_PER_TOKEN_USD


def enforce_request_budget(query: str) -> Dict[str, float]:
    """
    Check the request against both runtime budget limits.

    SIGNIFICANCE
    --------------------------------------------------------
    This function is the main Runtime-layer enforcement point.

    It MUST run before the downstream CrewAI/API/model operation.

    The request is rejected when:
        1. token count exceeds MAX_REQUEST_TOKENS, OR
        2. estimated cost exceeds MAX_REQUEST_COST_USD.

    This is fail-closed behavior:
        unsafe/oversized request -> reject
        accepted request       -> continue

    RETURNS
    --------------------------------------------------------
    A dictionary containing:
        token_count
        estimated_cost_usd

    RAISES
    --------------------------------------------------------
    RequestBudgetExceeded when either limit is violated.
    """

    # Calculate the request size first.
    token_count = count_tokens(query)

    # Calculate the corresponding synthetic cost.
    estimated_cost = estimate_request_cost_usd(token_count)

    # --------------------------------------------------------
    # Check 1 - token ceiling
    # --------------------------------------------------------

    if token_count > MAX_REQUEST_TOKENS:

        raise RequestBudgetExceeded(
            "Request rejected by runtime budget: "
            f"{token_count} tokens exceeds the maximum "
            f"of {MAX_REQUEST_TOKENS} tokens. "
            f"Estimated synthetic cost: "
            f"${estimated_cost:.5f}."
        )

    # --------------------------------------------------------
    # Check 2 - independent cost ceiling
    # --------------------------------------------------------
    #
    # Because MAX_REQUEST_COST_USD is intentionally independent
    # of the token ceiling, this check can now fail separately.
    if estimated_cost > MAX_REQUEST_COST_USD:

        raise RequestBudgetExceeded(
            "Request rejected by runtime budget: "
            f"estimated synthetic cost "
            f"${estimated_cost:.5f} exceeds the maximum "
            f"${MAX_REQUEST_COST_USD:.5f}."
        )

    # Both governance controls passed.
    return {
        "token_count": token_count,
        "estimated_cost_usd": estimated_cost,
    }


def run_governed_request(
    query: str,
    downstream_runner: Callable[[str], str],
) -> str:
    """
    Execute a request only after runtime governance succeeds.

    SIGNIFICANCE
    --------------------------------------------------------
    The order of operations is critical:

        incoming request
               |
               v
        budget enforcement
          /           \
       PASS            FAIL
        |               |
        v               v
    downstream        reject
       run

    The downstream runner is NEVER called when the request
    violates the budget.

    This ordering provides the actual fail-closed control required
    by Task 15.
    """

    # --------------------------------------------------------
    # FIRST: enforce governance
    # --------------------------------------------------------
    #
    # If this raises RequestBudgetExceeded, execution stops here.
    enforce_request_budget(query)

    # --------------------------------------------------------
    # SECOND: run downstream processing
    # --------------------------------------------------------
    #
    # This line is reached only when the request is within budget.
    return downstream_runner(query)


# ============================================================
# RUNTIME DEMONSTRATION SUPPORT
# ============================================================


def demo_downstream_runner(query: str) -> str:
    """
    Provide a very small downstream function for demonstration.

    SIGNIFICANCE
    --------------------------------------------------------
    This function is NOT the real CrewAI system.

    It exists only so we can visibly prove the order of execution.

    When a request passes the budget:
        "[DOWNSTREAM] ..." is printed.

    When a request fails the budget:
        this function must never be called.

    Therefore the presence/absence of this line in the terminal
    provides useful evidence of fail-closed execution.
    """

    # This print is intentionally used as an observable marker.
    print("[DOWNSTREAM] Request reached the downstream runner.")

    # Return a tiny deterministic response for the demonstration.
    return (
        f"Accepted request with {count_tokens(query)} tokens."
    )


def demonstrate_runtime_budget() -> None:
    """
    Demonstrate both successful and rejected runtime requests.

    SIGNIFICANCE
    --------------------------------------------------------
    Task 15 specifically asks for an oversized request to be
    rejected instead of silently processed.

    This demonstration therefore includes:
        1. a normal request that passes,
        2. a deliberately oversized request that fails.

    It additionally includes:
        3. a token-count request that stays below the token ceiling
           but exceeds the independent synthetic cost ceiling.

    The third case proves that the token and cost checks are not
    merely duplicate conditions.
    """

    print("\n" + "=" * 78)
    print("RUNTIME LAYER - TOKEN/COST BUDGET")
    print("=" * 78)

    # Display the active governance configuration.
    print(f"Maximum request tokens: {MAX_REQUEST_TOKENS}")

    print(
        "Synthetic cost per token: "
        f"${SYNTHETIC_COST_PER_TOKEN_USD:.5f}"
    )

    print(
        "Maximum synthetic request cost: "
        f"${MAX_REQUEST_COST_USD:.5f}"
    )

    # ========================================================
    # DEMONSTRATION 1 - NORMAL REQUEST
    # ========================================================

    # This is a realistic small request from the HR domain.
    normal_query = (
        "What is the normal employee notice period after resignation?"
    )

    # Check the budget before running downstream processing.
    normal_usage = enforce_request_budget(normal_query)

    # Because the request passed, it may now reach the runner.
    normal_result = run_governed_request(
        normal_query,
        demo_downstream_runner,
    )

    print("\nDEMONSTRATION 1 - NORMAL REQUEST")

    print(
        f"Tokens: {int(normal_usage['token_count'])}"
    )

    print(
        "Estimated cost: "
        f"${normal_usage['estimated_cost_usd']:.5f}"
    )

    print(f"Result: {normal_result}")

    print(
        "[PASS] Normal request stayed within both runtime limits."
    )

    # ========================================================
    # DEMONSTRATION 2 - OVERSIZED REQUEST
    # ========================================================
    #
    # 2,500 whitespace-separated words exceed the
    # 2,000-token ceiling.
    oversized_query = "oversized " * 2_500

    print(
        "\nDEMONSTRATION 2 - "
        "DELIBERATELY OVERSIZED REQUEST"
    )

    print(
        "Oversized request tokens: "
        f"{count_tokens(oversized_query)}"
    )

    try:

        # The request must be rejected before
        # demo_downstream_runner() can execute.
        run_governed_request(
            oversized_query,
            demo_downstream_runner,
        )

    except RequestBudgetExceeded as exc:

        print(
            f"[PASS] Oversized request was rejected: {exc}"
        )

    else:

        # Reaching this branch means the fail-closed rule failed.
        raise AssertionError(
            "Runtime budget failure: the oversized request "
            "was not rejected."
        )

    # ========================================================
    # DEMONSTRATION 3 - COST LIMIT
    # ========================================================
    #
    # 1,600 tokens are below the 2,000-token maximum.
    #
    # However:
    #
    #     1,600 x $0.00001 = $0.016
    #
    # and:
    #
    #     $0.016 > $0.015 maximum cost
    #
    # Therefore this case proves that the cost ceiling is an
    # independent governance condition.
    cost_limit_query = "costtest " * 1_600

    print(
        "\nDEMONSTRATION 3 - "
        "COST LIMIT REQUEST"
    )

    print(
        "Cost-test request tokens: "
        f"{count_tokens(cost_limit_query)}"
    )

    try:

        run_governed_request(
            cost_limit_query,
            demo_downstream_runner,
        )

    except RequestBudgetExceeded as exc:

        print(
            f"[PASS] Cost-limit request was rejected: {exc}"
        )

    else:

        raise AssertionError(
            "Runtime budget failure: the request exceeded the "
            "synthetic cost limit but was not rejected."
        )


# ============================================================
# TASK 15 MAIN DEMONSTRATION
# ============================================================


def main() -> None:
    """
    Run the complete Task 15 governance demonstration.

    SIGNIFICANCE
    --------------------------------------------------------
    This function is the single entry point used by:

        python governance.py

    It executes all required Task 15 demonstrations in order:

        1. least autonomy
        2. risk classification
        3. runtime token/cost budget
        4. final validation summary
    """

    print("=" * 78)
    print("TASK 15 - FOUR-LAYER AI GOVERNANCE")
    print("=" * 78)

    # The capstone uses MOCK_LLM, therefore no real provider
    # billing is being demonstrated here.
    print("[CONFIG] External LLM/API calls: DISABLED")

    # The existing CrewAI system is intentionally treated as
    # the governed system rather than rewritten in this file.
    print("[CONFIG] Existing CrewAI architecture: FIXED")

    # Runtime limits are synthetic because MOCK_LLM is used.
    print("[CONFIG] Runtime budget model: SYNTHETIC / MOCK_LLM")

    # --------------------------------------------------------
    # Application layer - least autonomy
    # --------------------------------------------------------

    verify_least_autonomy()

    print("\nGuard explanation:")
    print(LEAST_AUTONOMY_EXPLANATION)

    # --------------------------------------------------------
    # Application layer - risk classification
    # --------------------------------------------------------

    print_risk_classification()

    # --------------------------------------------------------
    # Runtime layer - budget
    # --------------------------------------------------------

    demonstrate_runtime_budget()

    # ========================================================
    # FINAL VALIDATION SUMMARY
    # ========================================================

    print("\n" + "=" * 78)
    print("TASK 15 VALIDATION SUMMARY")
    print("=" * 78)

    print(
        "[PASS] Application layer: only Lookup Agent owns "
        "check_job_application_status."
    )

    print(
        "[PASS] Application layer: system classified as High risk."
    )

    print(
        "[PASS] Runtime layer: token budget is checked before "
        "downstream execution."
    )

    print(
        "[PASS] Runtime layer: synthetic cost budget is checked "
        "before downstream execution."
    )

    print(
        "[PASS] Runtime layer: oversized request is rejected "
        "fail closed."
    )

    print(
        "[PASS] Runtime layer: independent cost-limit request "
        "is also rejected."
    )

    print(
        "[PASS] Task 15 governance demonstration completed."
    )


# ============================================================
# STANDARD PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Run the complete Task 15 demonstration when this file
    # is executed directly from PowerShell.
    main()

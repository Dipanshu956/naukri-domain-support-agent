# ============================================================
# Task 16 - In-memory response caching
# ============================================================
#
# Purpose:
#   Add a small in-memory cache in front of the REAL grounded-
#   generation RAG tool used by the existing CrewAI implementation.
#
# Requirement covered:
#   1. Cache is in memory.
#   2. Cache key is normalized query text.
#   3. Cache is scoped to the grounded-generation/RAG step.
#   4. Repeated normalized queries produce a cache hit.
#   5. A cache hit skips the real RAG tool call.
#   6. A call counter proves the real tool was executed only once.
#   7. Timing is printed as secondary evidence.
#
# Important:
#   We DO NOT cache the application-status lookup tool because
#   application status can change and could become stale.
# ============================================================


# ------------------------------------------------------------
# Standard-library imports
# ------------------------------------------------------------

# time is used to measure the first call and the cached call.
import time

# re is used to collapse repeated whitespace during normalization.
import re

# Callable is used only to document the callable type returned
# by the helper that finds the real underlying RAG function.
from typing import Callable, Dict


# ------------------------------------------------------------
# Reuse the existing Task 7 implementation
# ------------------------------------------------------------

# Import the real CrewAI module from the existing project.
#
# IMPORTANT:
#   This file does not recreate the RAG logic.
#   It reuses crew_agents.rag_search and reaches the original
#   Python function through the CrewAI tool's .func attribute.
import crew_agents


# ============================================================
# TASK 16 - CACHE STORAGE
# ============================================================

# In-memory dictionary requested by the problem statement.
#
# Key:
#   normalized query text
#
# Value:
#   actual result returned by the real rag_search function
RESPONSE_CACHE: Dict[str, str] = {}


# Count ONLY genuine executions of the real grounded-generation
# function.
#
# A cache hit must NOT increase this number.
REAL_RAG_CALL_COUNT = 0


# Count how many requests were served directly from the cache.
CACHE_HIT_COUNT = 0


# Count cache misses so the demonstration can show both paths.
CACHE_MISS_COUNT = 0


# ============================================================
# TASK 16 - QUERY NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:
    """
    Normalize query text before using it as the cache key.

    Why this function matters:
        The requirement says the cache must be keyed by
        NORMALIZED query text, not raw text.

    Normalization performed:
        1. Convert the query to a string.
        2. Remove leading/trailing whitespace.
        3. Convert all characters to lowercase.
        4. Collapse multiple whitespace characters into one space.

    Example:
        "  What IS the notice period?  "

    becomes:

        "what is the notice period?"
    """

    # Convert to string so the function remains safe for normal
    # string-like input and can always call the string methods below.
    text = str(query)

    # Remove leading/trailing spaces, convert to lowercase, and
    # collapse tabs/newlines/multiple spaces into one normal space.
    normalized = re.sub(r"\s+", " ", text.strip().lower())

    # Return the normalized cache key.
    return normalized


# ============================================================
# TASK 16 - FIND THE REAL GROUNDED-GENERATION FUNCTION
# ============================================================

def get_real_rag_function() -> Callable[[str], str]:
    """
    Return the original Python function behind the CrewAI tool.

    Why this function matters:
        In the existing crew_agents.py, rag_search is declared
        with CrewAI's @tool decorator.

        That means the imported object is a CrewAI tool object,
        while its underlying normal Python function is available
        through the tool's .func attribute.

    This allows Task 16 to wrap the REAL existing RAG path instead
    of creating a second copy of the retrieval/threshold logic.

    Returns:
        The original rag_search Python callable.

    Raises:
        AttributeError:
            When the installed CrewAI version does not expose the
            wrapped Python function through .func.
    """

    # Read the existing rag_search tool from the real project.
    rag_tool = getattr(crew_agents, "rag_search", None)

    # Stop with a clear message if the expected tool is missing.
    if rag_tool is None:
        raise AttributeError(
            "crew_agents.rag_search was not found."
        )

    # CrewAI's @tool object normally exposes the original function
    # through .func. We deliberately use that function instead of
    # duplicating its implementation here.
    real_function = getattr(rag_tool, "func", None)

    # Verify that .func is actually callable before returning it.
    if not callable(real_function):
        raise AttributeError(
            "crew_agents.rag_search.func is not callable. "
            "Check the installed CrewAI version."
        )

    # Return the original grounded-generation/RAG function.
    return real_function


# Resolve the real function once when this Task 16 file starts.
REAL_RAG_FUNCTION = get_real_rag_function()


# ============================================================
# TASK 16 - CACHED GROUNDED GENERATION
# ============================================================

def cached_grounded_generation(query: str) -> str:
    """
    Return the grounded-generation result using an in-memory cache.

    Cache flow:
        1. Normalize the query.
        2. Check the in-memory dictionary.
        3. On HIT:
             return the stored response immediately.
        4. On MISS:
             call the REAL rag_search function,
             increment REAL_RAG_CALL_COUNT,
             store the result,
             return the result.

    This function is the main Task 16 capability.

    Important scope decision:
        Only the grounded-generation/RAG result is cached.
        The application-status lookup is not cached.
    """

    # These counters belong to this module, so they must be declared
    # global before changing their values.
    global REAL_RAG_CALL_COUNT
    global CACHE_HIT_COUNT
    global CACHE_MISS_COUNT

    # Build the normalized cache key.
    normalized_key = normalize_query(query)

    # Check whether this normalized query already has a result.
    if normalized_key in RESPONSE_CACHE:

        # Record one cache hit.
        CACHE_HIT_COUNT += 1

        # Print explicit evidence that no real RAG execution is needed.
        print("\n[CACHE HIT]")
        print(f"Normalized key: {normalized_key}")
        print(
            "[CACHE] Returning the stored grounded-generation result."
        )
        print("[CACHE] Real RAG/tool call skipped.")

        # Return the already-computed grounded response.
        return RESPONSE_CACHE[normalized_key]

    # The query was not found, so this is a cache miss.
    CACHE_MISS_COUNT += 1

    # Print explicit evidence that real work is about to happen.
    print("\n[CACHE MISS]")
    print(f"Normalized key: {normalized_key}")
    print("[CACHE] Calling the REAL rag_search function.")

    # Execute the original grounded-generation/RAG function.
    #
    # The counter is incremented ONLY here.
    # Therefore a cache hit can never increase it.
    REAL_RAG_CALL_COUNT += 1

    real_result = REAL_RAG_FUNCTION(query)

    # Store the result under the normalized query key.
    RESPONSE_CACHE[normalized_key] = real_result

    # Return the freshly generated grounded result.
    return real_result


# ============================================================
# TASK 16 - CACHE RESET
# ============================================================

def reset_cache() -> None:
    """
    Clear the cache and reset demonstration counters.

    Why this function matters:
        A repeatable demonstration should always begin from a known
        empty cache so that the first request is guaranteed to be a
        MISS and the second request is guaranteed to be a HIT.
    """

    # Empty the in-memory response dictionary.
    RESPONSE_CACHE.clear()

    # Reset the counters so the evidence starts from zero.
    global REAL_RAG_CALL_COUNT
    global CACHE_HIT_COUNT
    global CACHE_MISS_COUNT

    REAL_RAG_CALL_COUNT = 0
    CACHE_HIT_COUNT = 0
    CACHE_MISS_COUNT = 0


# ============================================================
# TASK 16 - DEMONSTRATION
# ============================================================

def demonstrate_response_cache() -> None:
    """
    Demonstrate normalized-query caching with before/after evidence.

    The two queries intentionally differ in:
        - capitalization
        - leading/trailing whitespace

    Example pair:
        Query 1:
            "What is the notice period?"

        Query 2:
            "  what IS the notice period?  "

    Both normalize to the same cache key, proving that the cache
    is based on normalized text rather than byte-identical input.

    Evidence printed:
        - cache MISS/HIT
        - normalized key
        - real RAG call counter
        - cache hit counter
        - cache miss counter
        - first-call timing
        - second-call timing
        - explicit PASS assertions
    """

    # Make the demonstration deterministic.
    reset_cache()

    print("\n" + "=" * 70)
    print("TASK 16 - RESPONSE CACHING DEMONSTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Query 1 - first request
    # --------------------------------------------------------

    first_query = "What is the notice period?"

    print("\n[REQUEST 1]")
    print(f"Original query: {first_query}")

    # Start a high-resolution timer immediately before the real
    # cache lookup path begins.
    first_start = time.perf_counter()

    # The first request must be a cache miss and therefore execute
    # the actual grounded-generation/RAG function.
    first_result = cached_grounded_generation(first_query)

    # Stop the timer after the first response is returned.
    first_elapsed = time.perf_counter() - first_start

    print(f"First call time: {first_elapsed:.6f} seconds")

    # --------------------------------------------------------
    # Query 2 - normalized repeat
    # --------------------------------------------------------

    second_query = "  what IS the notice period?  "

    print("\n[REQUEST 2]")
    print(f"Original query: {second_query!r}")

    # Start the same high-resolution timer for the second request.
    second_start = time.perf_counter()

    # The second request differs on the surface, but after normalization
    # it maps to the same cache key as Request 1.
    second_result = cached_grounded_generation(second_query)

    # Stop the timer after the cache result is returned.
    second_elapsed = time.perf_counter() - second_start

    print(f"Second call time: {second_elapsed:.6f} seconds")

    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TASK 16 EVIDENCE")
    print("-" * 70)

    # Print the final counter values so the reviewer can see
    # exactly how many genuine RAG executions occurred.
    print(
        "Real grounded-generation calls : "
        f"{REAL_RAG_CALL_COUNT}"
    )

    # Print the number of cache hits.
    print(
        "Cache hits                      : "
        f"{CACHE_HIT_COUNT}"
    )

    # Print the number of cache misses.
    print(
        "Cache misses                    : "
        f"{CACHE_MISS_COUNT}"
    )

    # Show both normalized values to prove they are identical.
    first_key = normalize_query(first_query)
    second_key = normalize_query(second_query)

    print(f"\nNormalized Query 1: {first_key}")
    print(f"Normalized Query 2: {second_key}")
    print(
        "Normalized keys equal            : "
        f"{first_key == second_key}"
    )

    # Compare the two returned grounded responses.
    # This confirms the second request received the same cached result.
    print(
        "Returned results equal           : "
        f"{first_result == second_result}"
    )

    # --------------------------------------------------------
    # Strict requirement checks
    # --------------------------------------------------------

    # The first request must execute the real RAG function once.
    first_condition = REAL_RAG_CALL_COUNT == 1

    # The second normalized request must be a cache hit.
    second_condition = CACHE_HIT_COUNT == 1

    # There should be exactly one miss because only the first request
    # had to perform the real grounded-generation work.
    third_condition = CACHE_MISS_COUNT == 1

    # Both user-entered queries must resolve to the same normalized key.
    fourth_condition = first_key == second_key

    # Both requests should return the same grounded-generation result.
    fifth_condition = first_result == second_result

    # Print explicit PASS/FAIL evidence for every important requirement.
    print("\n" + "-" * 70)
    print("REQUIREMENT CHECKS")
    print("-" * 70)

    print(
        "[PASS]" if first_condition else "[FAIL]",
        "First request executed the real grounded-generation path exactly once."
    )

    print(
        "[PASS]" if second_condition else "[FAIL]",
        "Second normalized request produced a cache hit."
    )

    print(
        "[PASS]" if third_condition else "[FAIL]",
        "Only one cache miss occurred."
    )

    print(
        "[PASS]" if fourth_condition else "[FAIL]",
        "Different casing/whitespace normalized to the same cache key."
    )

    print(
        "[PASS]" if fifth_condition else "[FAIL]",
        "Cache hit returned the same grounded-generation result."
    )

    # The counter evidence is the strongest proof that the second request
    # did not call the underlying RAG function again.
    all_checks_passed = all(
        [
            first_condition,
            second_condition,
            third_condition,
            fourth_condition,
            fifth_condition,
        ]
    )

    if not all_checks_passed:
        # Fail loudly when the demonstration does not satisfy Task 16.
        raise AssertionError(
            "Task 16 cache demonstration failed."
        )

    # Print a final success marker for the capstone evidence.
    print("\n[PASS] Task 16 response caching demonstration completed.")

    # Timing is intentionally treated as secondary evidence.
    #
    # On some machines the difference can be smaller because the RAG
    # model/database may already be warm, but the call counter and cache
    # hit are deterministic proof.
    print("\n[Timing Evidence]")
    print(f"First request : {first_elapsed:.6f} seconds")
    print(f"Second request: {second_elapsed:.6f} seconds")

    print("=" * 70)


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Run the complete Task 16 demonstration when this file is executed
    # directly from the terminal.
    demonstrate_response_cache()

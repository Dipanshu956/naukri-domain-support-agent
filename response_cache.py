# ============================================================
# Task 16 - In-memory response caching
# ============================================================
#
# The cache is now reusable by the LIVE CrewAI RAG tool.
#
# IMPORTANT:
# response_cache.py does NOT import crew_agents.
# The real RAG function is registered by crew_agents.py after
# the underlying function has been defined.
# ============================================================

import re
import time
from typing import Callable, Dict, Optional


# ============================================================
# CACHE STORAGE
# ============================================================

RESPONSE_CACHE: Dict[str, str] = {}

REAL_RAG_CALL_COUNT = 0
CACHE_HIT_COUNT = 0
CACHE_MISS_COUNT = 0


# ============================================================
# REAL RAG FUNCTION REGISTRATION
# ============================================================

REAL_RAG_FUNCTION: Optional[Callable[[str], str]] = None


def configure_real_rag_function(
    rag_function: Callable[[str], str]
) -> None:
    """
    Register the REAL underlying RAG function.

    crew_agents.py calls this after defining its underlying
    non-cached RAG function.
    """

    global REAL_RAG_FUNCTION

    if not callable(rag_function):
        raise TypeError(
            "rag_function must be callable."
        )

    REAL_RAG_FUNCTION = rag_function


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:
    """
    Normalize query text for cache-key generation.
    """

    text = str(query)

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


# ============================================================
# CACHED GROUNDED GENERATION
# ============================================================

def cached_grounded_generation(
    query: str
) -> str:
    """
    Execute grounded generation through the in-memory cache.

    Cache HIT:
        return cached response without calling REAL_RAG_FUNCTION.

    Cache MISS:
        call REAL_RAG_FUNCTION once,
        increment REAL_RAG_CALL_COUNT,
        store result,
        return result.
    """

    global REAL_RAG_CALL_COUNT
    global CACHE_HIT_COUNT
    global CACHE_MISS_COUNT

    if REAL_RAG_FUNCTION is None:
        raise RuntimeError(
            "REAL_RAG_FUNCTION has not been configured. "
            "Call configure_real_rag_function() first."
        )

    normalized_key = normalize_query(query)

    if normalized_key in RESPONSE_CACHE:

        CACHE_HIT_COUNT += 1

        print("\n[CACHE HIT]")
        print(
            f"Normalized key: {normalized_key}"
        )
        print(
            "[CACHE] Returning cached grounded-generation result."
        )
        print(
            "[CACHE] Real RAG/tool call skipped."
        )

        return RESPONSE_CACHE[normalized_key]

    CACHE_MISS_COUNT += 1

    print("\n[CACHE MISS]")
    print(
        f"Normalized key: {normalized_key}"
    )
    print(
        "[CACHE] Calling the REAL RAG function."
    )

    REAL_RAG_CALL_COUNT += 1

    real_result = REAL_RAG_FUNCTION(query)

    RESPONSE_CACHE[normalized_key] = str(
        real_result
    )

    return str(real_result)


# ============================================================
# CACHE RESET
# ============================================================

def reset_cache() -> None:
    """
    Clear cache and reset demonstration counters.
    """

    global REAL_RAG_CALL_COUNT
    global CACHE_HIT_COUNT
    global CACHE_MISS_COUNT

    RESPONSE_CACHE.clear()

    REAL_RAG_CALL_COUNT = 0
    CACHE_HIT_COUNT = 0
    CACHE_MISS_COUNT = 0


# ============================================================
# TASK 16 DEMONSTRATION
# ============================================================

def demonstrate_response_cache() -> None:
    """
    Demonstrate normalized-query cache behavior.
    """

    # Import only here for the demonstration.
    #
    # This avoids a module-level circular dependency while still
    # using the real project RAG implementation.
    import crew_agents

    # Register the actual underlying RAG implementation.
    configure_real_rag_function(
        crew_agents._real_rag_search
    )

    reset_cache()

    print("\n" + "=" * 70)
    print(
        "TASK 16 - RESPONSE CACHING DEMONSTRATION"
    )
    print("=" * 70)

    first_query = (
        "What is the notice period?"
    )

    print("\n[REQUEST 1]")
    print(
        f"Original query: {first_query}"
    )

    first_start = time.perf_counter()

    first_result = cached_grounded_generation(
        first_query
    )

    first_elapsed = (
        time.perf_counter()
        - first_start
    )

    print(
        f"First call time: {first_elapsed:.6f} seconds"
    )

    second_query = (
        "  what IS the notice period?  "
    )

    print("\n[REQUEST 2]")
    print(
        f"Original query: {second_query!r}"
    )

    second_start = time.perf_counter()

    second_result = cached_grounded_generation(
        second_query
    )

    second_elapsed = (
        time.perf_counter()
        - second_start
    )

    print(
        f"Second call time: {second_elapsed:.6f} seconds"
    )

    first_key = normalize_query(
        first_query
    )

    second_key = normalize_query(
        second_query
    )

    print("\n" + "-" * 70)
    print("TASK 16 EVIDENCE")
    print("-" * 70)

    print(
        "Real grounded-generation calls : "
        f"{REAL_RAG_CALL_COUNT}"
    )

    print(
        "Cache hits                      : "
        f"{CACHE_HIT_COUNT}"
    )

    print(
        "Cache misses                    : "
        f"{CACHE_MISS_COUNT}"
    )

    print(
        f"\nNormalized Query 1: {first_key}"
    )

    print(
        f"Normalized Query 2: {second_key}"
    )

    print(
        "Normalized keys equal            : "
        f"{first_key == second_key}"
    )

    print(
        "Returned results equal           : "
        f"{first_result == second_result}"
    )

    first_condition = (
        REAL_RAG_CALL_COUNT == 1
    )

    second_condition = (
        CACHE_HIT_COUNT == 1
    )

    third_condition = (
        CACHE_MISS_COUNT == 1
    )

    fourth_condition = (
        first_key == second_key
    )

    fifth_condition = (
        first_result == second_result
    )

    print("\n" + "-" * 70)
    print("REQUIREMENT CHECKS")
    print("-" * 70)

    print(
        "[PASS]" if first_condition else "[FAIL]",
        "First request executed the real RAG path exactly once."
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

    if not all(
        [
            first_condition,
            second_condition,
            third_condition,
            fourth_condition,
            fifth_condition,
        ]
    ):
        raise AssertionError(
            "Task 16 cache demonstration failed."
        )

    print(
        "\n[PASS] Task 16 response caching demonstration completed."
    )

    print("\n[Timing Evidence]")
    print(
        f"First request : {first_elapsed:.6f} seconds"
    )
    print(
        f"Second request: {second_elapsed:.6f} seconds"
    )

    print("=" * 70)


if __name__ == "__main__":
    demonstrate_response_cache()
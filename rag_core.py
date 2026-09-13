# ============================================================
# Task 3 + Task 4 + Task 5 - RAG Core
# Naukri.com Domain Support Agent
# ============================================================
#
# This module implements the complete Part 1 RAG layer:
#
# Task 3:
#   - Load the HR knowledge-base documents.
#   - Create fixed-size chunks.
#   - Create sentence-based chunks.
#   - Generate local SentenceTransformers embeddings.
#   - Store both strategies in separate ChromaDB collections.
#
# Task 4:
#   - Measure representative in-scope queries.
#   - Measure deliberately out-of-scope queries.
#   - Calibrate the groundedness threshold empirically.
#   - IMPORTANT: calibration uses ONLY the production
#     fixed_chunks collection.
#   - Demonstrate grounded answers and the fallback answer.
#
# Task 5:
#   - Evaluate fixed_chunks and sentence_chunks independently.
#   - Map retrieved chunks back to parent documents.
#   - Deduplicate parent documents.
#   - Calculate document-level precision and recall.
#   - Recommend the stronger chunking strategy.
#
# IMPORTANT DESIGN DECISION
# -------------------------
# Task 4 and the live CrewAI system use fixed_chunks because
# fixed_chunks is the selected production retrieval strategy.
#
# Task 5 still evaluates BOTH collections separately.
#
# We deliberately do NOT use a "best of both collections"
# retrieval decision for production threshold calibration.
# Mixing collection scores would make the threshold inconsistent
# with the collection actually used by the deployed agent.
# ============================================================


# ============================================================
# STANDARD-LIBRARY IMPORTS
# ============================================================

# Path provides a platform-independent way to work with
# filesystem paths such as "knowledge_base/" and "README.md".
from pathlib import Path

# re is used for sentence splitting and safe replacement of
# README sections marked with explicit start/end comments.
import re

# statistics is used as a fallback when the in-scope and
# out-of-scope similarity distributions overlap.
import statistics


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

# ChromaDB provides the local persistent vector database used
# to index and retrieve embedded knowledge-base chunks.
import chromadb

# SentenceTransformer loads the local all-MiniLM-L6-v2 model
# used to create query and document embeddings without requiring
# a paid external embedding API.
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT SETTINGS
# ============================================================

# Folder containing the original HR knowledge-base text files.
KNOWLEDGE_BASE = Path("knowledge_base")

# Folder where the persistent ChromaDB database is stored.
CHROMA_PATH = "chroma_db"

# Local free embedding model required by the capstone.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum character length for fixed-size chunks.
CHUNK_SIZE = 200

# Character overlap between consecutive fixed-size chunks.
CHUNK_OVERLAP = 50

# Number of sentences placed into one sentence-based chunk.
SENTENCES_PER_CHUNK = 2

# Number of chunks retrieved for each query.
TOP_K = 3

# The capstone demonstrations use deterministic MOCK_LLM behavior.
# This flag is retained here for compatibility/documentation.
MOCK_LLM = True

# Required fallback when retrieval is below the calibrated
# groundedness threshold.
FALLBACK_MESSAGE = (
    "I don't know based on the available knowledge base."
)


# ============================================================
# TASK 4 - CALIBRATION QUESTIONS
# ============================================================

# These queries are intentionally covered by the HR knowledge base.
#
# The first five are also reused by Task 5 so the same evaluation
# questions are applied to both chunking strategies.
IN_SCOPE_QUERIES = [
    "What degree is required for most professional jobs?",
    "How much notice should a candidate get before an interview?",
    "What is the normal employee notice period after resignation?",
    "How much is the employee referral bonus?",
    "When can an employee apply for an internal transfer?",
    "How long is the normal probation period?",
]


# These queries are deliberately outside the project knowledge base.
#
# They are used to identify a lower similarity cluster for the
# empirical groundedness-threshold calibration.
OUT_OF_SCOPE_QUERIES = [
    "What is the capital of France?",
    "What is the weather forecast for tomorrow?",
    "How do I bake a chocolate cake?",
]


# ============================================================
# TASK 5 - EVALUATION QUERIES
# ============================================================

# The capstone asks Task 5 to use the same five queries that were
# used for the Task 4 grounded-generation demonstration.
TASK5_QUERIES = IN_SCOPE_QUERIES[:5]


# ============================================================
# TASK 5 - GROUND TRUTH
# ============================================================

# Map each evaluation query to the correct PARENT document.
#
# The values are source-document names rather than individual
# chunk IDs because Task 5 requires document-level scoring.
GROUND_TRUTH = {
    "What degree is required for most professional jobs?": {
        "01_eligibility_criteria"
    },
    "How much notice should a candidate get before an interview?": {
        "02_interview_scheduling"
    },
    "What is the normal employee notice period after resignation?": {
        "05_notice_period"
    },
    "How much is the employee referral bonus?": {
        "06_referral_bonus"
    },
    "When can an employee apply for an internal transfer?": {
        "07_internal_transfer"
    },
}


# ============================================================
# TASK 3 - LOAD DOCUMENTS
# ============================================================

def load_documents(folder_path):
    """
    Load all non-empty TXT files from the knowledge-base folder.

    Parameters
    ----------
    folder_path : pathlib.Path
        Directory containing the knowledge-base text files.

    Returns
    -------
    list[dict]
        Each dictionary contains:
        - source: source document name without extension
        - text: complete document text
    """

    # Store all loaded documents in this list.
    documents = []

    # Read TXT files in deterministic alphabetical order.
    for file_path in sorted(folder_path.glob("*.txt")):

        # Read the complete source document using UTF-8.
        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        # Ignore empty files.
        if text:

            # Store only the metadata needed by later functions.
            documents.append(
                {
                    "source": file_path.stem,
                    "text": text,
                }
            )

    # Return the complete document collection.
    return documents


# ============================================================
# TASK 3 - FIXED-SIZE CHUNKING
# ============================================================

def fixed_size_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split text into fixed-size overlapping chunks.

    Parameters
    ----------
    text : str
        Source document text.

    chunk_size : int
        Maximum number of characters in each chunk.

    overlap : int
        Number of characters shared by consecutive chunks.

    Returns
    -------
    list[str]
        Generated fixed-size chunks.

    Notes
    -----
    Task 3 uses:
        chunk_size = 200
        overlap    = 50
    """

    # Chunk size must be positive.
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    # Negative overlap is invalid.
    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    # An overlap equal to or larger than the chunk size would
    # prevent the sliding window from progressing correctly.
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    # Store all generated chunks here.
    chunks = []

    # Start from the first character.
    start = 0

    # Continue until the entire document has been processed.
    while start < len(text):

        # Calculate the normal fixed-size endpoint.
        end = start + chunk_size

        # Extract the chunk and remove surrounding whitespace.
        chunk = text[start:end].strip()

        # Only retain non-empty chunks.
        if chunk:
            chunks.append(chunk)

        # Stop after processing the final portion of the text.
        if end >= len(text):
            break

        # Move forward while preserving the requested overlap.
        start = end - overlap

    # Return the complete fixed-size chunk list.
    return chunks


# ============================================================
# TASK 3 - SENTENCE-BASED CHUNKING
# ============================================================

def sentence_based_chunks(
    text,
    sentences_per_chunk=SENTENCES_PER_CHUNK,
):
    """
    Split a document into chunks containing a fixed number of
    complete sentences.

    Parameters
    ----------
    text : str
        Source document text.

    sentences_per_chunk : int
        Number of sentences grouped into one chunk.

    Returns
    -------
    list[str]
        Sentence-based chunks.
    """

    # At least one sentence must be placed in each chunk.
    if sentences_per_chunk <= 0:
        raise ValueError(
            "sentences_per_chunk must be greater than 0."
        )

    # Split whenever ., ! or ? is followed by whitespace.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    # Remove empty values and normalize surrounding whitespace.
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Store the final sentence chunks.
    chunks = []

    # Process sentences in groups of the requested size.
    for index in range(
        0,
        len(sentences),
        sentences_per_chunk,
    ):

        # Combine the current sentence group into one chunk.
        chunk = " ".join(
            sentences[
                index:index + sentences_per_chunk
            ]
        ).strip()

        # Keep only non-empty chunks.
        if chunk:
            chunks.append(chunk)

    # Return all sentence-based chunks.
    return chunks


# ============================================================
# TASK 3 - BUILD CHUNKS WITH PARENT DOCUMENT METADATA
# ============================================================

def build_chunks(
    documents,
    chunk_function,
):
    """
    Apply a chunking strategy to every parent document.

    Parameters
    ----------
    documents : list[dict]
        Documents returned by load_documents().

    chunk_function : callable
        Either fixed_size_chunks or sentence_based_chunks.

    Returns
    -------
    list[dict]
        Each chunk contains:
        - id
        - text
        - source
    """

    # Store all generated chunks.
    all_chunks = []

    # Process each parent document independently.
    for document in documents:

        # Create chunks using the supplied strategy.
        chunks = chunk_function(
            document["text"]
        )

        # Add metadata to every generated chunk.
        for index, chunk in enumerate(chunks):

            # Construct a deterministic chunk ID.
            chunk_id = (
                f"{document['source']}_{index}"
            )

            # Store chunk text together with parent source.
            all_chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "source": document["source"],
                }
            )

    # Return every generated chunk.
    return all_chunks


# ============================================================
# TASK 3 - PREPARE CHROMADB COLLECTION
# ============================================================

def prepare_collection(
    client,
    collection_name,
):
    """
    Create or retrieve a ChromaDB collection configured for
    cosine distance.

    Parameters
    ----------
    client : chromadb.PersistentClient
        Persistent ChromaDB client.

    collection_name : str
        Name of the desired collection.

    Returns
    -------
    chromadb.Collection
        Ready-to-use collection configured for cosine distance.
    """

    # Try to retrieve an existing collection.
    try:

        collection = client.get_collection(
            collection_name
        )

    # When it does not exist, get_collection() raises an exception.
    except Exception:

        collection = None

    # If an old collection exists, confirm it uses cosine distance.
    if collection is not None:

        # Read collection metadata safely.
        metadata = (
            collection.metadata
            or {}
        )

        # Recreate the collection when the metric is incorrect.
        if metadata.get("hnsw:space") != "cosine":

            client.delete_collection(
                collection_name
            )

            collection = None

    # Create the collection when it does not yet exist.
    if collection is None:

        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    # Return the prepared collection.
    return collection


# ============================================================
# TASK 3 - STORE CHUNKS
# ============================================================

def store_chunks(
    collection,
    chunks,
    model,
):
    """
    Embed and upsert chunks into a ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.Collection
        ChromaDB collection that will store the vectors.

    chunks : list[dict]
        Chunk dictionaries containing id/text/source.

    model : SentenceTransformer
        Local embedding model.

    Returns
    -------
    None
        Data is written directly into ChromaDB.
    """

    # Nothing needs to be done when the chunk list is empty.
    if not chunks:
        return

    # Extract only chunk text for embedding.
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate normalized local embeddings.
    #
    # Normalized vectors work naturally with cosine similarity.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # Extract deterministic chunk IDs.
    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    # Preserve the parent source document in Chroma metadata.
    metadatas = [
        {
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    # Upsert means insert new values or replace existing values
    # with the same IDs.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ============================================================
# TASK 4 / TASK 5 - RETRIEVE FROM ONE COLLECTION
# ============================================================

def retrieve(
    collection,
    query,
    model,
    top_k=TOP_K,
):
    """
    Retrieve the top-k chunks from one specific ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.Collection
        Collection to query.

    query : str
        User or evaluation question.

    model : SentenceTransformer
        Local embedding model.

    top_k : int
        Number of nearest chunks to retrieve.

    Returns
    -------
    list[dict]
        Retrieved results containing:
        - text
        - source
        - similarity
    """

    # Convert the query into the same normalized embedding space
    # used for document embeddings.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    # Ask ChromaDB for the nearest chunks.
    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # Read ChromaDB cosine-distance values.
    distances = results["distances"][0]

    # Convert cosine distance to cosine similarity.
    #
    # For normalized vectors:
    # cosine similarity = 1 - cosine distance
    similarities = [
        1.0 - float(distance)
        for distance in distances
    ]

    # Build a simpler Python representation for downstream code.
    retrieved = []

    # Combine text, metadata, and similarity score.
    for document, metadata, similarity in zip(
        results["documents"][0],
        results["metadatas"][0],
        similarities,
    ):

        # Store the useful retrieval information.
        retrieved.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "similarity": similarity,
            }
        )

    # Return the retrieved chunks in ChromaDB ranking order.
    return retrieved


# ============================================================
# COMPARATIVE RETRIEVAL HELPER
# ============================================================

def retrieve_from_either_collection(
    query,
    model,
    fixed_collection,
    sentence_collection,
    top_k=TOP_K,
):
    """
    Compare both collections and return the stronger one.

    IMPORTANT
    ---------
    This helper is retained only for comparative experimentation.

    It is NOT used for:
        - Task 4 production threshold calibration
        - Task 4 production grounded generation
        - live CrewAI production retrieval

    Task 4 and the deployed path use fixed_chunks explicitly.

    Parameters
    ----------
    query : str
        Query to evaluate.

    model : SentenceTransformer
        Embedding model.

    fixed_collection : chromadb.Collection
        Fixed-size collection.

    sentence_collection : chromadb.Collection
        Sentence-based collection.

    top_k : int
        Number of chunks to retrieve.

    Returns
    -------
    tuple[str, list[dict]]
        Collection name and retrieved results.
    """

    # Retrieve from the fixed-size collection.
    fixed_results = retrieve(
        fixed_collection,
        query,
        model,
        top_k,
    )

    # Retrieve from the sentence-based collection.
    sentence_results = retrieve(
        sentence_collection,
        query,
        model,
        top_k,
    )

    # Select the collection with the stronger top-1 result.
    if (
        fixed_results
        and sentence_results
        and fixed_results[0]["similarity"]
        >= sentence_results[0]["similarity"]
    ):

        return (
            "fixed_chunks",
            fixed_results,
        )

    # Handle the unlikely case where the sentence collection
    # has the stronger or only available result.
    return (
        "sentence_chunks",
        sentence_results,
    )


# ============================================================
# TASK 4 - PRODUCTION CALIBRATION
# ============================================================

def measure_fixed_collection_queries(
    queries,
    model,
    fixed_collection,
):
    """
    Measure Task 4 calibration queries using ONLY fixed_chunks.

    This is the key correction for Task 4.

    The production system uses fixed_chunks, so the similarity
    threshold must be calibrated on the same collection.

    Parameters
    ----------
    queries : list[str]
        Calibration questions.

    model : SentenceTransformer
        Local embedding model.

    fixed_collection : chromadb.Collection
        Production fixed_chunks collection.

    Returns
    -------
    list[dict]
        One measurement per query containing:
        - query
        - collection
        - similarity
    """

    # Store calibration measurements.
    measurements = []

    # Measure every supplied query independently.
    for query in queries:

        # Retrieve only from the production collection.
        results = retrieve(
            fixed_collection,
            query,
            model,
            TOP_K,
        )

        # Calibration cannot continue if no result exists.
        if not results:

            raise RuntimeError(
                "No retrieval result was returned for "
                f"calibration query: {query}"
            )

        # The highest-ranked result contains the top-1 similarity.
        top_result = results[0]

        # Store a transparent measurement record.
        measurements.append(
            {
                "query": query,
                "collection": "fixed_chunks",
                "similarity": top_result[
                    "similarity"
                ],
            }
        )

    # Return all production-path measurements.
    return measurements


# ============================================================
# TASK 4 - CHOOSE EMPIRICAL THRESHOLD
# ============================================================

def choose_threshold(
    in_scope,
    out_of_scope,
):
    """
    Calculate an empirical similarity threshold.

    Strategy:
        1. Find the minimum in-scope similarity.
        2. Find the maximum out-of-scope similarity.
        3. When the two groups are cleanly separated, choose the
           midpoint between those boundary values.
        4. If the groups overlap, use the midpoint between their
           means as a deterministic fallback.

    Parameters
    ----------
    in_scope : list[dict]
        Fixed-path in-scope measurements.

    out_of_scope : list[dict]
        Fixed-path out-of-scope measurements.

    Returns
    -------
    float
        Calibrated groundedness threshold.
    """

    # Extract all in-scope scores.
    in_scores = [
        item["similarity"]
        for item in in_scope
    ]

    # Extract all out-of-scope scores.
    out_scores = [
        item["similarity"]
        for item in out_of_scope
    ]

    # Protect against accidental empty calibration sets.
    if not in_scores:

        raise ValueError(
            "At least one in-scope calibration score is required."
        )

    if not out_scores:

        raise ValueError(
            "At least one out-of-scope calibration score is required."
        )

    # Identify the lowest supported-question score.
    lowest_in_scope = min(
        in_scores
    )

    # Identify the strongest unrelated-question score.
    highest_out_of_scope = max(
        out_scores
    )

    # When the clusters are separated, use the midpoint.
    if highest_out_of_scope < lowest_in_scope:

        threshold = (
            highest_out_of_scope
            + lowest_in_scope
        ) / 2.0

    # When the clusters overlap, use the midpoint between
    # the average scores as a deterministic fallback.
    else:

        threshold = (
            statistics.mean(in_scores)
            + statistics.mean(out_scores)
        ) / 2.0

    # Return the empirically calculated threshold.
    return threshold


# ============================================================
# TASK 4 - PRODUCTION GROUNDED GENERATION
# ============================================================

def grounded_generate(
    query,
    model,
    fixed_collection,
    threshold,
    top_k=TOP_K,
):
    """
    Perform grounded generation using ONLY fixed_chunks.

    IMPORTANT
    ---------
    The previous implementation selected whichever collection
    had the stronger similarity score.

    That behavior is intentionally removed here because the
    production system has already selected fixed_chunks.

    Parameters
    ----------
    query : str
        Current user/evaluation question.

    model : SentenceTransformer
        Local embedding model.

    fixed_collection : chromadb.Collection
        Production fixed_chunks collection.

    threshold : float
        Empirically calibrated RAG threshold.

    top_k : int
        Number of chunks to retrieve.

    Returns
    -------
    dict
        Contains:
        - query
        - collection
        - similarity
        - decision
        - answer
    """

    # Retrieve from the selected production collection only.
    results = retrieve(
        fixed_collection,
        query,
        model,
        top_k,
    )

    # Fail closed if retrieval returns no results.
    if not results:

        return {
            "query": query,
            "collection": "fixed_chunks",
            "similarity": 0.0,
            "decision": "FALLBACK",
            "answer": FALLBACK_MESSAGE,
        }

    # Read the strongest retrieved similarity.
    top_similarity = results[0][
        "similarity"
    ]

    # Refuse unsupported questions.
    if top_similarity < threshold:

        return {
            "query": query,
            "collection": "fixed_chunks",
            "similarity": top_similarity,
            "decision": "FALLBACK",
            "answer": FALLBACK_MESSAGE,
        }

    # MOCK_LLM mode does not call an external language model.
    #
    # Therefore the grounded answer is composed directly from
    # the retrieved knowledge-base context.
    answer = "\n\n".join(
        (
            f"[Source: {item['source']}]\n"
            f"{item['text']}"
        )
        for item in results
    )

    # Return the grounded result.
    return {
        "query": query,
        "collection": "fixed_chunks",
        "similarity": top_similarity,
        "decision": "GROUNDED",
        "answer": answer,
    }


# ============================================================
# TASK 4 - README MARKERS
# ============================================================

# Start marker used to safely replace only the generated Task 4
# section without touching unrelated README content.
README_START = "<!-- TASK4_START -->"

# End marker for the generated Task 4 section.
README_END = "<!-- TASK4_END -->"


# ============================================================
# TASK 5 - README MARKERS
# ============================================================

# Start marker for the generated Task 5 section.
TASK5_README_START = "<!-- TASK5_START -->"

# End marker for the generated Task 5 section.
TASK5_README_END = "<!-- TASK5_END -->"


# ============================================================
# TASK 4 - UPDATE README
# ============================================================

def update_readme(
    in_scope,
    out_of_scope,
    threshold,
):
    """
    Replace or append the generated Task 4 calibration section
    in README.md.

    Parameters
    ----------
    in_scope : list[dict]
        Fixed-path in-scope calibration measurements.

    out_of_scope : list[dict]
        Fixed-path out-of-scope measurements.

    threshold : float
        Final calculated threshold.

    Returns
    -------
    None
    """

    # README lives in the project root.
    readme_path = Path("README.md")

    # Load the existing README when present.
    if readme_path.exists():

        existing = readme_path.read_text(
            encoding="utf-8"
        )

    else:

        # Create a minimal fallback only when README does not exist.
        existing = (
            "# Naukri.com Domain Support Agent\n"
        )

    # Begin the generated Task 4 section.
    lines = [
        README_START,
        "## Task 4 - Grounded Generation and Threshold Calibration",
        "",
        "Task 4 calibration is performed using only the "
        "production `fixed_chunks` ChromaDB collection.",
        "",
        "### In-scope measurements",
        "",
        "| Query | Collection | Top-1 cosine similarity |",
        "|---|---|---:|",
    ]

    # Add every in-scope measurement.
    for item in in_scope:

        lines.append(
            f"| {item['query']} | "
            f"{item['collection']} | "
            f"{item['similarity']:.4f} |"
        )

    # Add the out-of-scope measurements.
    lines.extend(
        [
            "",
            "### Out-of-scope measurements",
            "",
            "| Query | Collection | Top-1 cosine similarity |",
            "|---|---|---:|",
        ]
    )

    # Add each out-of-scope measurement.
    for item in out_of_scope:

        lines.append(
            f"| {item['query']} | "
            f"{item['collection']} | "
            f"{item['similarity']:.4f} |"
        )

    # Add the final threshold explanation.
    lines.extend(
        [
            "",
            f"### Chosen threshold: `{threshold:.4f}`",
            "",
            "The threshold was empirically calculated from the "
            "measured production fixed-path scores. No arbitrary "
            "0.5, 0.6, or 0.7 preset was used.",
            "",
            README_END,
        ]
    )

    # Convert the list of lines into Markdown text.
    new_block = "\n".join(
        lines
    )

    # Match ONLY the marked Task 4 block.
    pattern = re.compile(
        re.escape(README_START)
        + r".*?"
        + re.escape(README_END),
        re.DOTALL,
    )

    # Replace the old generated section when it exists.
    if pattern.search(existing):

        updated = pattern.sub(
            new_block,
            existing,
            count=1,
        )

    else:

        # Otherwise append a new Task 4 section.
        separator = (
            "\n"
            if existing.endswith("\n")
            else "\n\n"
        )

        updated = (
            existing
            + separator
            + new_block
            + "\n"
        )

    # Save the new README contents.
    readme_path.write_text(
        updated,
        encoding="utf-8",
    )


# ============================================================
# TASK 5 - EVALUATE ONE QUERY
# ============================================================

def evaluate_query(
    query,
    collection,
    collection_name,
    model,
    ground_truth,
    top_k=TOP_K,
):
    """
    Evaluate one query against one ChromaDB collection.

    Scoring is performed at the parent-document level.

    Parameters
    ----------
    query : str
        Evaluation query.

    collection : chromadb.Collection
        Collection being evaluated.

    collection_name : str
        Human-readable collection name.

    model : SentenceTransformer
        Embedding model.

    ground_truth : dict
        Mapping from query to correct parent documents.

    top_k : int
        Number of retrieved chunks.

    Returns
    -------
    dict
        Detailed precision/recall evidence.
    """

    # Retrieve chunks directly from THIS collection.
    #
    # This direct call is critical because Task 5 requires both
    # strategies to be evaluated independently.
    results = retrieve(
        collection,
        query,
        model,
        top_k,
    )

    # Convert chunk-level results into unique parent documents.
    retrieved_documents = {
        item["source"]
        for item in results
    }

    # Read the expected parent document set.
    correct_documents = ground_truth[
        query
    ]

    # Calculate the document-level true positives.
    true_positives = (
        retrieved_documents
        & correct_documents
    )

    # Count correctly retrieved parent documents.
    true_positive_count = len(
        true_positives
    )

    # Count unique retrieved parent documents.
    retrieved_count = len(
        retrieved_documents
    )

    # Count correct parent documents.
    ground_truth_count = len(
        correct_documents
    )

    # Calculate precision.
    if retrieved_count == 0:

        precision = 0.0

    else:

        precision = (
            true_positive_count
            / retrieved_count
        )

    # Calculate recall.
    if ground_truth_count == 0:

        recall = 0.0

    else:

        recall = (
            true_positive_count
            / ground_truth_count
        )

    # Return all information needed for the Task 5 evidence.
    return {
        "query": query,
        "collection": collection_name,
        "results": results,
        "retrieved_documents": retrieved_documents,
        "correct_documents": correct_documents,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# TASK 5 - PRINT ONE EVALUATION
# ============================================================

def print_evaluation(
    evaluation,
):
    """
    Print full document-level precision/recall arithmetic for one
    query.

    Parameters
    ----------
    evaluation : dict
        Result returned by evaluate_query().

    Returns
    -------
    None
    """

    # Make each query easy to distinguish in terminal output.
    print(
        "\n"
        + "=" * 80
    )

    # Print the evaluated query.
    print(
        "Query:",
        evaluation["query"],
    )

    # Print the collection.
    print(
        "Collection:",
        evaluation["collection"],
    )

    # Show each retrieved chunk.
    print(
        "\nRetrieved chunks:"
    )

    for number, result in enumerate(
        evaluation["results"],
        start=1,
    ):

        print(
            f"{number}. "
            f"[Source: {result['source']}] "
            f"Similarity: {result['similarity']:.4f}"
        )

    # Show deduplicated parent documents.
    print(
        "\nRetrieved unique documents:"
    )

    print(
        sorted(
            evaluation["retrieved_documents"]
        )
    )

    # Show expected documents.
    print(
        "Ground-truth documents:"
    )

    print(
        sorted(
            evaluation["correct_documents"]
        )
    )

    # Show the document-level intersection.
    print(
        "Correct retrieved documents:"
    )

    print(
        sorted(
            evaluation["true_positives"]
        )
    )

    # Calculate counts used for explicit arithmetic output.
    retrieved_count = len(
        evaluation["retrieved_documents"]
    )

    true_positive_count = len(
        evaluation["true_positives"]
    )

    ground_truth_count = len(
        evaluation["correct_documents"]
    )

    # Print exact precision arithmetic.
    print(
        "\nPrecision = "
        f"{true_positive_count} / "
        f"{retrieved_count} = "
        f"{evaluation['precision']:.4f}"
    )

    # Print exact recall arithmetic.
    print(
        "Recall = "
        f"{true_positive_count} / "
        f"{ground_truth_count} = "
        f"{evaluation['recall']:.4f}"
    )


# ============================================================
# TASK 5 - EVALUATE ONE COMPLETE COLLECTION
# ============================================================

def evaluate_collection(
    queries,
    collection,
    collection_name,
    model,
):
    """
    Evaluate all Task 5 queries against one collection.

    Parameters
    ----------
    queries : list[str]
        Evaluation questions.

    collection : chromadb.Collection
        Collection being evaluated.

    collection_name : str
        Name displayed in the results.

    model : SentenceTransformer
        Embedding model.

    Returns
    -------
    list[dict]
        Evaluation result for each query.
    """

    # Store the individual query results.
    evaluations = []

    # Process all queries independently.
    for query in queries:

        evaluation = evaluate_query(
            query,
            collection,
            collection_name,
            model,
            GROUND_TRUTH,
            TOP_K,
        )

        # Keep the result for average calculations.
        evaluations.append(
            evaluation
        )

        # Print the full arithmetic immediately.
        print_evaluation(
            evaluation
        )

    # Return every evaluation.
    return evaluations


# ============================================================
# TASK 5 - CALCULATE AVERAGES
# ============================================================

def calculate_average_scores(
    evaluations,
):
    """
    Calculate average precision and recall across queries.

    Parameters
    ----------
    evaluations : list[dict]
        Results produced by evaluate_collection().

    Returns
    -------
    tuple[float, float]
        Average precision and average recall.
    """

    # Extract every precision result.
    precisions = [
        item["precision"]
        for item in evaluations
    ]

    # Extract every recall result.
    recalls = [
        item["recall"]
        for item in evaluations
    ]

    # Prevent division by zero.
    if not precisions:

        raise ValueError(
            "At least one evaluation is required."
        )

    # Calculate arithmetic averages.
    average_precision = (
        sum(precisions)
        / len(precisions)
    )

    average_recall = (
        sum(recalls)
        / len(recalls)
    )

    # Return both metrics.
    return (
        average_precision,
        average_recall,
    )


# ============================================================
# TASK 5 - COMPARE BOTH CHUNKING STRATEGIES
# ============================================================

def compare_chunking_strategies(
    fixed_evaluations,
    sentence_evaluations,
):
    """
    Compare the Task 5 results from both chunking approaches.

    The recommendation is based first on whether one strategy
    dominates the other on both precision and recall.

    When metrics are mixed, a combined average is used only as
    a deterministic tie-breaker.

    Parameters
    ----------
    fixed_evaluations : list[dict]
        Fixed-size evaluation results.

    sentence_evaluations : list[dict]
        Sentence-based evaluation results.

    Returns
    -------
    dict
        Summary values and recommendation text.
    """

    # Calculate fixed-size averages.
    fixed_precision, fixed_recall = (
        calculate_average_scores(
            fixed_evaluations
        )
    )

    # Calculate sentence-based averages.
    sentence_precision, sentence_recall = (
        calculate_average_scores(
            sentence_evaluations
        )
    )

    # Print a clear summary.
    print(
        "\n"
        + "=" * 80
    )

    print(
        "TASK 5 SUMMARY"
    )

    print(
        "\nCollection              "
        "Average Precision      Average Recall"
    )

    print(
        f"fixed_chunks            "
        f"{fixed_precision:.4f}                 "
        f"{fixed_recall:.4f}"
    )

    print(
        f"sentence_chunks         "
        f"{sentence_precision:.4f}                 "
        f"{sentence_recall:.4f}"
    )

    # Combined values are used only when the metrics are mixed.
    fixed_overall = (
        fixed_precision
        + fixed_recall
    ) / 2.0

    sentence_overall = (
        sentence_precision
        + sentence_recall
    ) / 2.0

    # Fixed-size dominates when it is at least as good on both
    # metrics and strictly better on at least one.
    if (
        fixed_precision >= sentence_precision
        and fixed_recall >= sentence_recall
        and (
            fixed_precision > sentence_precision
            or fixed_recall > sentence_recall
        )
    ):

        recommendation = (
            f"I would deploy fixed-size chunking because it achieved "
            f"an average precision of {fixed_precision:.4f} and "
            f"an average recall of {fixed_recall:.4f}, compared "
            f"with {sentence_precision:.4f} precision and "
            f"{sentence_recall:.4f} recall for sentence-based "
            f"chunking. Fixed-size chunking therefore provided "
            f"the stronger retrieval result across the evaluated "
            f"queries."
        )

    # Sentence-based dominates under the same rule.
    elif (
        sentence_precision >= fixed_precision
        and sentence_recall >= fixed_recall
        and (
            sentence_precision > fixed_precision
            or sentence_recall > fixed_recall
        )
    ):

        recommendation = (
            f"I would deploy sentence-based chunking because it "
            f"achieved an average precision of "
            f"{sentence_precision:.4f} and an average recall of "
            f"{sentence_recall:.4f}, compared with "
            f"{fixed_precision:.4f} precision and "
            f"{fixed_recall:.4f} recall for fixed-size chunking. "
            f"Sentence-based chunking therefore provided the "
            f"stronger retrieval result across the evaluated "
            f"queries."
        )

    # When each strategy is stronger on a different metric,
    # use the combined average as a deterministic tie-breaker.
    elif fixed_overall > sentence_overall:

        recommendation = (
            f"I would deploy fixed-size chunking because its "
            f"average precision was {fixed_precision:.4f} and "
            f"its average recall was {fixed_recall:.4f}, giving "
            f"it the stronger combined retrieval result than "
            f"sentence-based chunking, which achieved "
            f"{sentence_precision:.4f} precision and "
            f"{sentence_recall:.4f} recall."
        )

    elif sentence_overall > fixed_overall:

        recommendation = (
            f"I would deploy sentence-based chunking because "
            f"its average precision was {sentence_precision:.4f} "
            f"and its average recall was "
            f"{sentence_recall:.4f}, giving it the stronger "
            f"combined retrieval result than fixed-size chunking, "
            f"which achieved {fixed_precision:.4f} precision and "
            f"{fixed_recall:.4f} recall."
        )

    # Exact tie.
    else:

        recommendation = (
            f"Both chunking strategies produced the same combined "
            f"retrieval result. Fixed-size chunking achieved "
            f"{fixed_precision:.4f} precision and "
            f"{fixed_recall:.4f} recall, while sentence-based "
            f"chunking achieved {sentence_precision:.4f} precision "
            f"and {sentence_recall:.4f} recall. The final choice "
            f"can therefore be based on implementation simplicity "
            f"and context-preservation considerations."
        )

    # Print the recommendation.
    print(
        "\nRecommendation:"
    )

    print(
        recommendation
    )

    # Return all useful Task 5 results.
    return {
        "fixed_precision": fixed_precision,
        "fixed_recall": fixed_recall,
        "sentence_precision": sentence_precision,
        "sentence_recall": sentence_recall,
        "recommendation": recommendation,
    }


# ============================================================
# TASK 5 - UPDATE README
# ============================================================

def update_task5_readme(
    fixed_evaluations,
    sentence_evaluations,
    summary,
):
    """
    Replace or append the generated Task 5 README section.

    Parameters
    ----------
    fixed_evaluations : list[dict]
        Fixed-size evaluation results.

    sentence_evaluations : list[dict]
        Sentence-based evaluation results.

    summary : dict
        Comparison summary returned by
        compare_chunking_strategies().

    Returns
    -------
    None
    """

    # README location.
    readme_path = Path(
        "README.md"
    )

    # Load existing README or create a minimal fallback.
    if readme_path.exists():

        existing = readme_path.read_text(
            encoding="utf-8"
        )

    else:

        existing = (
            "# Naukri.com Domain Support Agent\n"
        )

    # Start Task 5 generated section.
    lines = [
        TASK5_README_START,
        "## Task 5 - Evaluation and Comparison of Chunking Strategies",
        "",
        "Task 5 evaluates the same five queries independently "
        "against the fixed-size and sentence-based collections.",
        "",
        "Retrieved chunks are mapped back to their parent source "
        "documents and duplicate parent documents are removed "
        "before calculating document-level precision and recall.",
        "",
        "### Fixed-size chunking",
        "",
    ]

    # Add every fixed-size evaluation.
    for evaluation in fixed_evaluations:

        true_positive_count = len(
            evaluation["true_positives"]
        )

        retrieved_count = len(
            evaluation["retrieved_documents"]
        )

        ground_truth_count = len(
            evaluation["correct_documents"]
        )

        lines.append(
            f"#### Query: {evaluation['query']}"
        )

        lines.append(
            f"- Retrieved documents: "
            f"`{sorted(evaluation['retrieved_documents'])}`"
        )

        lines.append(
            f"- Ground-truth documents: "
            f"`{sorted(evaluation['correct_documents'])}`"
        )

        lines.append(
            f"- Precision = "
            f"{true_positive_count} / "
            f"{retrieved_count} = "
            f"{evaluation['precision']:.4f}"
        )

        lines.append(
            f"- Recall = "
            f"{true_positive_count} / "
            f"{ground_truth_count} = "
            f"{evaluation['recall']:.4f}"
        )

        lines.append("")

    # Add sentence-based section.
    lines.extend(
        [
            "### Sentence-based chunking",
            "",
        ]
    )

    # Add every sentence-based evaluation.
    for evaluation in sentence_evaluations:

        true_positive_count = len(
            evaluation["true_positives"]
        )

        retrieved_count = len(
            evaluation["retrieved_documents"]
        )

        ground_truth_count = len(
            evaluation["correct_documents"]
        )

        lines.append(
            f"#### Query: {evaluation['query']}"
        )

        lines.append(
            f"- Retrieved documents: "
            f"`{sorted(evaluation['retrieved_documents'])}`"
        )

        lines.append(
            f"- Ground-truth documents: "
            f"`{sorted(evaluation['correct_documents'])}`"
        )

        lines.append(
            f"- Precision = "
            f"{true_positive_count} / "
            f"{retrieved_count} = "
            f"{evaluation['precision']:.4f}"
        )

        lines.append(
            f"- Recall = "
            f"{true_positive_count} / "
            f"{ground_truth_count} = "
            f"{evaluation['recall']:.4f}"
        )

        lines.append("")

    # Add final comparison table.
    lines.extend(
        [
            "### Overall comparison",
            "",
            "| Collection | Average Precision | Average Recall |",
            "|---|---:|---:|",
            f"| fixed_chunks | "
            f"{summary['fixed_precision']:.4f} | "
            f"{summary['fixed_recall']:.4f} |",
            f"| sentence_chunks | "
            f"{summary['sentence_precision']:.4f} | "
            f"{summary['sentence_recall']:.4f} |",
            "",
            "### Recommendation",
            "",
            summary["recommendation"],
            "",
            TASK5_README_END,
        ]
    )

    # Convert the section to one block of Markdown.
    new_block = "\n".join(
        lines
    )

    # Match ONLY the marked Task 5 section.
    pattern = re.compile(
        re.escape(TASK5_README_START)
        + r".*?"
        + re.escape(TASK5_README_END),
        re.DOTALL,
    )

    # Replace existing generated Task 5 content.
    if pattern.search(existing):

        updated = pattern.sub(
            new_block,
            existing,
            count=1,
        )

    else:

        separator = (
            "\n"
            if existing.endswith("\n")
            else "\n\n"
        )

        updated = (
            existing
            + separator
            + new_block
            + "\n"
        )

    # Save the updated README.
    readme_path.write_text(
        updated,
        encoding="utf-8",
    )


# ============================================================
# OUTPUT HELPER
# ============================================================

def print_result(
    result,
):
    """
    Print a grounded-generation result in a readable format.

    Parameters
    ----------
    result : dict
        Result returned by grounded_generate().

    Returns
    -------
    None
    """

    # Visual separator.
    print(
        "\n"
        + "=" * 70
    )

    # Display the query.
    print(
        "Query:",
        result["query"],
    )

    # Display the production collection.
    print(
        "Collection:",
        result["collection"],
    )

    # Display top-1 similarity.
    print(
        f"Top-1 cosine similarity: "
        f"{result['similarity']:.4f}"
    )

    # Display grounded/fallback decision.
    print(
        "Decision:",
        result["decision"],
    )

    # Display final answer/evidence.
    print(
        "Answer:"
    )

    print(
        result["answer"]
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    """
    Execute the complete Task 3, Task 4 and Task 5 pipeline.

    The sequence is:

        1. Load documents.
        2. Build fixed chunks.
        3. Build sentence chunks.
        4. Load embeddings.
        5. Create ChromaDB collections.
        6. Index both strategies.
        7. Calibrate threshold using fixed_chunks ONLY.
        8. Demonstrate grounded generation using fixed_chunks ONLY.
        9. Demonstrate out-of-scope fallback.
       10. Evaluate both chunking strategies independently.
       11. Save Task 4 and Task 5 evidence to README.md.
    """

    # ========================================================
    # TASK 3 - LOAD DOCUMENTS
    # ========================================================

    print(
        "Loading knowledge-base documents..."
    )

    # Load every TXT knowledge-base file.
    documents = load_documents(
        KNOWLEDGE_BASE
    )

    # The capstone requires at least 12 documents.
    if len(documents) < 12:

        raise RuntimeError(
            "At least 12 knowledge-base documents are required; "
            f"found {len(documents)}."
        )

    print(
        f"Loaded {len(documents)} documents."
    )

    # ========================================================
    # TASK 3 - FIXED-SIZE CHUNKS
    # ========================================================

    print(
        "\nCreating fixed-size chunks..."
    )

    # Create the first chunking strategy.
    fixed_chunks = build_chunks(
        documents,
        fixed_size_chunks,
    )

    print(
        f"Fixed-size chunks: "
        f"{len(fixed_chunks)}"
    )

    # ========================================================
    # TASK 3 - SENTENCE CHUNKS
    # ========================================================

    print(
        "Creating sentence-based chunks..."
    )

    # Create the second independent chunking strategy.
    sentence_chunks = build_chunks(
        documents,
        sentence_based_chunks,
    )

    print(
        f"Sentence-based chunks: "
        f"{len(sentence_chunks)}"
    )

    # ========================================================
    # TASK 3 - LOAD EMBEDDING MODEL
    # ========================================================

    print(
        "\nLoading embedding model..."
    )

    # Load the required local embedding model.
    model = SentenceTransformer(
        MODEL_NAME
    )

    # ========================================================
    # TASK 3 - CHROMADB CLIENT
    # ========================================================

    # Create/open persistent ChromaDB storage.
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # ========================================================
    # TASK 3 - PREPARE BOTH COLLECTIONS
    # ========================================================

    # Prepare fixed-size collection.
    fixed_collection = prepare_collection(
        client,
        "fixed_chunks",
    )

    # Prepare sentence-based collection.
    sentence_collection = prepare_collection(
        client,
        "sentence_chunks",
    )

    # ========================================================
    # TASK 3 - INDEX FIXED CHUNKS
    # ========================================================

    print(
        "\nStoring fixed-size chunks..."
    )

    store_chunks(
        fixed_collection,
        fixed_chunks,
        model,
    )

    # ========================================================
    # TASK 3 - INDEX SENTENCE CHUNKS
    # ========================================================

    print(
        "Storing sentence-based chunks..."
    )

    store_chunks(
        sentence_collection,
        sentence_chunks,
        model,
    )

    # Display final vector counts.
    print(
        f"fixed_chunks collection: "
        f"{fixed_collection.count()} records"
    )

    print(
        f"sentence_chunks collection: "
        f"{sentence_collection.count()} records"
    )

    # ========================================================
    # TASK 4 - FIXED-ONLY CALIBRATION
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "TASK 4 - PRODUCTION FIXED-CHUNKS CALIBRATION"
    )

    print(
        "============================================================"
    )

    print(
        "\nMeasuring in-scope calibration queries "
        "on fixed_chunks..."
    )

    # IMPORTANT:
    # Calibration now uses ONLY the production fixed collection.
    in_scope = measure_fixed_collection_queries(
        IN_SCOPE_QUERIES,
        model,
        fixed_collection,
    )

    print(
        "Measuring out-of-scope calibration queries "
        "on fixed_chunks..."
    )

    # IMPORTANT:
    # Out-of-scope calibration also uses ONLY fixed_chunks.
    out_of_scope = measure_fixed_collection_queries(
        OUT_OF_SCOPE_QUERIES,
        model,
        fixed_collection,
    )

    # Calculate the empirical threshold.
    threshold = choose_threshold(
        in_scope,
        out_of_scope,
    )

    # ========================================================
    # PRINT CALIBRATION RESULTS
    # ========================================================

    print(
        "\nIN-SCOPE CALIBRATION"
    )

    for item in in_scope:

        print(
            f"{item['similarity']:.4f} | "
            f"{item['collection']} | "
            f"{item['query']}"
        )

    print(
        "\nOUT-OF-SCOPE CALIBRATION"
    )

    for item in out_of_scope:

        print(
            f"{item['similarity']:.4f} | "
            f"{item['collection']} | "
            f"{item['query']}"
        )

    # Print the actual threshold produced by this execution.
    print(
        f"\nChosen threshold: "
        f"{threshold:.4f}"
    )

    # ========================================================
    # TASK 4 - README UPDATE
    # ========================================================

    update_readme(
        in_scope,
        out_of_scope,
        threshold,
    )

    # ========================================================
    # TASK 4 - GROUNDED GENERATION DEMONSTRATION
    # ========================================================

    print(
        "\nGROUNDED GENERATION DEMONSTRATION"
    )

    # Demonstrate at least five in-scope questions.
    for query in IN_SCOPE_QUERIES[:5]:

        # IMPORTANT:
        # The production grounded-generation path now uses
        # fixed_chunks only.
        result = grounded_generate(
            query,
            model,
            fixed_collection,
            threshold,
            TOP_K,
        )

        print_result(
            result
        )

    # ========================================================
    # TASK 4 - OUT-OF-SCOPE FALLBACK
    # ========================================================

    print(
        "\nOUT-OF-SCOPE FALLBACK DEMONSTRATION"
    )

    # Select the strongest out-of-scope query so the test is
    # deliberately conservative.
    fallback_query = max(
        out_of_scope,
        key=lambda item: item["similarity"],
    )["query"]

    # Run the fixed-only grounded-generation path.
    fallback_result = grounded_generate(
        fallback_query,
        model,
        fixed_collection,
        threshold,
        TOP_K,
    )

    print_result(
        fallback_result
    )

    # This explicit assertion proves that the out-of-scope test
    # actually triggered the required fallback.
    if fallback_result["decision"] != "FALLBACK":

        raise RuntimeError(
            "The selected out-of-scope query did not trigger "
            "the grounded fallback. Review the calibration "
            "measurements and threshold."
        )

    # ========================================================
    # TASK 5 - FIXED COLLECTION
    # ========================================================

    print(
        "\n\nTASK 5 - FIXED-SIZE CHUNKING EVALUATION"
    )

    # Evaluate all five Task 5 queries independently.
    fixed_evaluations = evaluate_collection(
        TASK5_QUERIES,
        fixed_collection,
        "fixed_chunks",
        model,
    )

    # ========================================================
    # TASK 5 - SENTENCE COLLECTION
    # ========================================================

    print(
        "\n\nTASK 5 - SENTENCE-BASED CHUNKING EVALUATION"
    )

    # Evaluate the same five queries against the second collection.
    sentence_evaluations = evaluate_collection(
        TASK5_QUERIES,
        sentence_collection,
        "sentence_chunks",
        model,
    )

    # ========================================================
    # TASK 5 - COMPARE BOTH
    # ========================================================

    task5_summary = compare_chunking_strategies(
        fixed_evaluations,
        sentence_evaluations,
    )

    # ========================================================
    # TASK 5 - SAVE README EVIDENCE
    # ========================================================

    update_task5_readme(
        fixed_evaluations,
        sentence_evaluations,
        task5_summary,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\n"
        "Task 3, Task 4 and Task 5 completed successfully."
    )

    print(
        "Task 4 calibration and Task 5 evaluation results "
        "were written to README.md."
    )


# ============================================================
# PYTHON ENTRY POINT
# ============================================================

# This condition ensures main() runs only when the file itself
# is executed directly, not when another module imports rag_core.
if __name__ == "__main__":

    # Execute the complete Part 1 pipeline.
    main()
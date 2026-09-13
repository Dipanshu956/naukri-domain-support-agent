# Import Path so we can work with folders and file paths easily.
from pathlib import Path

# Import re so we can split text into sentences.
import re

# Import statistics so we can calculate a threshold when score groups overlap.
import statistics

# Import ChromaDB because it is the local vector database used in the project.
import chromadb

# Import SentenceTransformer because it creates local text embeddings.
from sentence_transformers import SentenceTransformer


# ============================================================
# Project settings
# ============================================================

# Store the path of the folder containing the knowledge-base documents.
KNOWLEDGE_BASE = Path("knowledge_base")

# Store the path where ChromaDB will persist its database files.
CHROMA_PATH = "chroma_db"

# Define the free local SentenceTransformers embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Define the size of each fixed-size chunk.
CHUNK_SIZE = 200

# Define how many characters should overlap between fixed-size chunks.
CHUNK_OVERLAP = 50

# Define how many sentences should be placed into one sentence-based chunk.
SENTENCES_PER_CHUNK = 2

# Define how many chunks should be retrieved for each query.
TOP_K = 3

# Keep MOCK_LLM enabled because the assignment does not require a paid LLM API.
MOCK_LLM = True

# Define the message used when a query is below the calibrated threshold.
FALLBACK_MESSAGE = "I don't know based on the available knowledge base."


# ============================================================
# Calibration questions
# ============================================================

# These questions are covered by the knowledge base.
# The first five are also used later for Task 4 demonstration
# and Task 5 evaluation.
IN_SCOPE_QUERIES = [
    "What degree is required for most professional jobs?",
    "How much notice should a candidate get before an interview?",
    "What is the normal employee notice period after resignation?",
    "How much is the employee referral bonus?",
    "When can an employee apply for an internal transfer?",
    "How long is the normal probation period?",
]


# These questions are deliberately outside the knowledge base.
OUT_OF_SCOPE_QUERIES = [
    "What is the capital of France?",
    "What is the weather forecast for tomorrow?",
    "How do I bake a chocolate cake?",
]


# ============================================================
# Task 5 - queries
# ============================================================

# Use the same five in-scope queries that were demonstrated in Task 4.
# This is important because Task 5 asks us to use the same queries.
TASK5_QUERIES = IN_SCOPE_QUERIES[:5]


# ============================================================
# Task 5 - ground truth
# ============================================================

# Define the correct parent document for every Task 5 query.
#
# These are parent document names, not chunk IDs.
#
# For example:
# If several chunks come from 02_interview_scheduling,
# they still count as only one retrieved document during
# Task 5 evaluation.
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
# Task 3 - load documents
# ============================================================

def load_documents(folder_path):
    # Create an empty list where all documents will be stored.
    documents = []

    # Find every TXT file in the knowledge-base folder.
    # sorted() keeps the order predictable.
    for file_path in sorted(folder_path.glob("*.txt")):

        # Read the complete file using UTF-8 encoding.
        text = file_path.read_text(encoding="utf-8").strip()

        # Ignore a file when it is empty.
        if text:

            # Store the document source name and its text.
            documents.append({
                "source": file_path.stem,
                "text": text,
            })

    # Return the complete list of loaded documents.
    return documents


# ============================================================
# Task 3 - fixed-size chunking
# ============================================================

def fixed_size_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    # Make sure the chunk size is greater than zero.
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    # Make sure the overlap is not negative.
    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    # The overlap cannot be equal to or larger than the chunk size.
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    # Create an empty list to store the generated chunks.
    chunks = []

    # Start reading the document from character position zero.
    start = 0

    # Continue creating chunks until the complete document is processed.
    while start < len(text):

        # Calculate the ending character position for the current chunk.
        end = start + chunk_size

        # Extract the current chunk.
        chunk = text[start:end].strip()

        # Add the chunk only when it contains text.
        if chunk:
            chunks.append(chunk)

        # Stop when we have reached the end of the document.
        if end >= len(text):
            break

        # Move forward while keeping the requested character overlap.
        start = end - overlap

    # Return all fixed-size chunks.
    return chunks


# ============================================================
# Task 3 - sentence-based chunking
# ============================================================

def sentence_based_chunks(
    text,
    sentences_per_chunk=SENTENCES_PER_CHUNK,
):
    # Make sure at least one sentence is included in each chunk.
    if sentences_per_chunk <= 0:
        raise ValueError(
            "sentences_per_chunk must be greater than 0."
        )

    # Split the document after ., !, or ? followed by whitespace.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    # Remove empty sentences and remove extra spaces.
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Create an empty list for the sentence-based chunks.
    chunks = []

    # Process the sentences in groups of the requested size.
    for i in range(
        0,
        len(sentences),
        sentences_per_chunk,
    ):

        # Combine the required number of sentences into one chunk.
        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        ).strip()

        # Add the chunk only when it contains text.
        if chunk:
            chunks.append(chunk)

    # Return all sentence-based chunks.
    return chunks


# ============================================================
# Task 3 - build chunks with parent document information
# ============================================================

def build_chunks(documents, chunk_function):
    # Create an empty list for all generated chunks.
    all_chunks = []

    # Process every parent document.
    for document in documents:

        # Generate chunks using the supplied chunking function.
        chunks = chunk_function(document["text"])

        # Process every chunk created from this document.
        for index, chunk in enumerate(chunks):

            # Store the chunk ID, text, and original parent document.
            all_chunks.append({
                "id": f"{document['source']}_{index}",
                "text": chunk,
                "source": document["source"],
            })

    # Return the complete list of chunks.
    return all_chunks


# ============================================================
# Task 3 / Task 4 - prepare ChromaDB collection
# ============================================================

def prepare_collection(client, collection_name):
    # Try to find an existing collection first.
    try:
        collection = client.get_collection(collection_name)

    # If the collection does not exist, set it to None.
    except Exception:
        collection = None

    # Task 4 and Task 5 use cosine distance.
    if collection is not None:

        # Read the existing collection metadata.
        metadata = collection.metadata or {}

        # Check whether the collection is configured for cosine distance.
        if metadata.get("hnsw:space") != "cosine":

            # Delete the old collection when it uses another metric.
            client.delete_collection(collection_name)

            # Set collection to None so it can be recreated correctly.
            collection = None

    # Create the collection when it does not already exist.
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
# Task 3 - store chunks in ChromaDB
# ============================================================

def store_chunks(collection, chunks, model):
    # Do nothing when the chunk list is empty.
    if not chunks:
        return

    # Extract only the text from every chunk.
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Create a local embedding for every chunk.
    #
    # normalize_embeddings=True makes the vectors normalized,
    # which works correctly with cosine similarity.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # Get the unique ID of every chunk.
    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    # Store the original parent document name as metadata.
    metadatas = [
        {
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    # Upsert the chunks, embeddings, IDs, and metadata into ChromaDB.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ============================================================
# Task 4 - retrieve chunks from one collection
# ============================================================

def retrieve(
    collection,
    query,
    model,
    top_k=TOP_K,
):
    # Convert the user query into an embedding.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    # Ask ChromaDB for the closest chunks.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # ChromaDB returns cosine distance for this collection.
    distances = results["distances"][0]

    # Convert cosine distance into cosine similarity.
    #
    # cosine similarity = 1 - cosine distance
    similarities = [
        1.0 - float(distance)
        for distance in distances
    ]

    # Create a simple Python list for the retrieved results.
    retrieved = []

    # Process the document, metadata, and similarity together.
    for document, metadata, similarity in zip(
        results["documents"][0],
        results["metadatas"][0],
        similarities,
    ):

        # Store the retrieved text, source, and similarity score.
        retrieved.append({
            "text": document,
            "source": metadata.get(
                "source",
                "unknown",
            ),
            "similarity": similarity,
        })

    # Return the top-k retrieved chunks.
    return retrieved


# ============================================================
# Task 4 - choose the better collection
# ============================================================

def retrieve_from_either_collection(
    query,
    model,
    fixed_collection,
    sentence_collection,
    top_k=TOP_K,
):
    # Retrieve results from the fixed-size collection.
    fixed_results = retrieve(
        fixed_collection,
        query,
        model,
        top_k,
    )

    # Retrieve results from the sentence-based collection.
    sentence_results = retrieve(
        sentence_collection,
        query,
        model,
        top_k,
    )

    # Compare the strongest result from both collections.
    if fixed_results[0]["similarity"] >= sentence_results[0]["similarity"]:

        # Return the fixed-size collection when its top result is higher.
        return "fixed_chunks", fixed_results

    # Otherwise return the sentence-based collection.
    return "sentence_chunks", sentence_results


# ============================================================
# Task 4 - measure calibration queries
# ============================================================

def measure_queries(
    queries,
    model,
    fixed_collection,
    sentence_collection,
):
    # Create an empty list for measured results.
    measurements = []

    # Process every query.
    for query in queries:

        # Retrieve from both collections and select the stronger one.
        collection_name, results = retrieve_from_either_collection(
            query,
            model,
            fixed_collection,
            sentence_collection,
        )

        # The first result is the top-1 result.
        top_result = results[0]

        # Store the query, collection, and top-1 similarity.
        measurements.append({
            "query": query,
            "collection": collection_name,
            "similarity": top_result["similarity"],
        })

    # Return all measured values.
    return measurements


# ============================================================
# Task 4 - choose similarity threshold
# ============================================================

def choose_threshold(
    in_scope,
    out_of_scope,
):
    # Extract similarity scores for in-scope questions.
    in_scores = [
        item["similarity"]
        for item in in_scope
    ]

    # Extract similarity scores for out-of-scope questions.
    out_scores = [
        item["similarity"]
        for item in out_of_scope
    ]

    # Find the lowest in-scope similarity score.
    lowest_in_scope = min(in_scores)

    # Find the highest out-of-scope similarity score.
    highest_out_of_scope = max(out_scores)

    # Check whether there is a clear gap between both groups.
    if highest_out_of_scope < lowest_in_scope:

        # Choose the midpoint of the measured gap.
        return (
            highest_out_of_scope
            + lowest_in_scope
        ) / 2

    # When scores overlap, use the midpoint between
    # the average in-scope and out-of-scope scores.
    return (
        statistics.mean(in_scores)
        + statistics.mean(out_scores)
    ) / 2


# ============================================================
# Task 4 - grounded generation
# ============================================================

def grounded_generate(
    query,
    model,
    fixed_collection,
    sentence_collection,
    threshold,
    top_k=TOP_K,
):
    # Retrieve the top-k chunks from the better collection.
    collection_name, results = retrieve_from_either_collection(
        query,
        model,
        fixed_collection,
        sentence_collection,
        top_k,
    )

    # Get the highest similarity score.
    top_similarity = results[0]["similarity"]

    # Use the fallback when the score is below the threshold.
    if top_similarity < threshold:

        # Return the fallback result.
        return {
            "query": query,
            "collection": collection_name,
            "similarity": top_similarity,
            "decision": "FALLBACK",
            "answer": FALLBACK_MESSAGE,
        }

    # MOCK_LLM means we do not call an external LLM.
    # Therefore, the answer contains only retrieved knowledge-base text.
    answer = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}"
        for item in results
    )

    # Return the grounded answer.
    return {
        "query": query,
        "collection": collection_name,
        "similarity": top_similarity,
        "decision": "GROUNDED",
        "answer": answer,
    }


# ============================================================
# Task 4 - README markers
# ============================================================

# Define markers used to protect the Task 4 README section.
README_START = "<!-- TASK4_START -->"

# Define the ending marker for the Task 4 README section.
README_END = "<!-- TASK4_END -->"


# ============================================================
# Task 5 - README markers
# ============================================================

# Define the beginning marker for the Task 5 README section.
TASK5_README_START = "<!-- TASK5_START -->"

# Define the ending marker for the Task 5 README section.
TASK5_README_END = "<!-- TASK5_END -->"


# ============================================================
# Task 4 - update README
# ============================================================

def update_readme(
    in_scope,
    out_of_scope,
    threshold,
):
    # Use README.md from the current project folder.
    readme_path = Path("README.md")

    # Read the existing README when it already exists.
    if readme_path.exists():

        # Load the current README contents.
        existing = readme_path.read_text(
            encoding="utf-8"
        )

    # Create a new README when no README exists.
    else:

        # Start with a basic project heading.
        existing = "# RAG Project\n"

    # Create the beginning of the Task 4 README section.
    lines = [
        README_START,
        "## Task 4 - Grounded Generation and Threshold Calibration",
        "",
        "### In-scope measurements",
        "",
        "| Query | Collection | Top-1 cosine similarity |",
        "|---|---|---:|",
    ]

    # Add every in-scope query measurement to the table.
    for item in in_scope:

        # Add one Markdown table row.
        lines.append(
            f"| {item['query']} | "
            f"{item['collection']} | "
            f"{item['similarity']:.4f} |"
        )

    # Add the out-of-scope results section.
    lines.extend([
        "",
        "### Out-of-scope measurements",
        "",
        "| Query | Collection | Top-1 cosine similarity |",
        "|---|---|---:|",
    ])

    # Add each out-of-scope measurement.
    for item in out_of_scope:

        # Add one Markdown table row.
        lines.append(
            f"| {item['query']} | "
            f"{item['collection']} | "
            f"{item['similarity']:.4f} |"
        )

    # Add the calculated threshold.
    lines.extend([
        "",
        f"### Chosen threshold: `{threshold:.4f}`",
        "",
        "The threshold was calculated from the measured values. "
        "No fixed 0.5, 0.6, or 0.7 preset was used.",
        "",
        README_END,
    ])

    # Combine all Task 4 lines into one text block.
    new_block = "\n".join(lines)

    # Create a pattern that only matches the Task 4 section.
    pattern = re.compile(
        re.escape(README_START)
        + r".*?"
        + re.escape(README_END),
        re.DOTALL,
    )

    # Replace the previous Task 4 section when it already exists.
    if pattern.search(existing):

        # Replace only the first Task 4 block.
        updated = pattern.sub(
            new_block,
            existing,
            count=1,
        )

    # Otherwise append a new Task 4 section.
    else:

        # Choose one or two new lines depending on the
        # current ending of the README.
        separator = (
            "\n"
            if existing.endswith("\n")
            else "\n\n"
        )

        # Append the Task 4 section.
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
# Task 5 - evaluate one query
# ============================================================

def evaluate_query(
    query,
    collection,
    collection_name,
    model,
    ground_truth,
    top_k=TOP_K,
):
    # Retrieve the top-k chunks from ONLY this collection.
    #
    # We deliberately call retrieve() directly.
    #
    # We do NOT call retrieve_from_either_collection()
    # because Task 5 requires the two strategies to be
    # evaluated independently.
    results = retrieve(
        collection,
        query,
        model,
        top_k,
    )

    # Convert the retrieved chunk sources into a set.
    #
    # A Python set automatically removes duplicates.
    #
    # Example:
    #
    # 02_interview_scheduling
    # 02_interview_scheduling
    # 10_diversity_hiring
    #
    # becomes:
    #
    # {
    #     "02_interview_scheduling",
    #     "10_diversity_hiring"
    # }
    #
    # This is exactly what the problem means by
    # document-level deduplication.
    retrieved_documents = {
        item["source"]
        for item in results
    }

    # Get the ground-truth parent document set for this query.
    correct_documents = ground_truth[query]

    # Find the intersection between retrieved and correct documents.
    #
    # These are the documents that were retrieved AND are correct.
    true_positives = (
        retrieved_documents
        & correct_documents
    )

    # Count the number of correctly retrieved unique documents.
    true_positive_count = len(true_positives)

    # Count the number of unique documents retrieved.
    retrieved_count = len(retrieved_documents)

    # Count the number of documents that should have been retrieved.
    ground_truth_count = len(correct_documents)

    # Calculate document-level precision.
    #
    # Precision =
    # correct retrieved documents / total retrieved documents
    if retrieved_count == 0:

        # Return zero when nothing was retrieved.
        precision = 0.0

    else:

        # Calculate the normal precision value.
        precision = (
            true_positive_count
            / retrieved_count
        )

    # Calculate document-level recall.
    #
    # Recall =
    # correct retrieved documents / total correct documents
    if ground_truth_count == 0:

        # Return zero when there is no ground truth.
        recall = 0.0

    else:

        # Calculate the normal recall value.
        recall = (
            true_positive_count
            / ground_truth_count
        )

    # Return all information required for Task 5.
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
# Task 5 - print one evaluation
# ============================================================

def print_evaluation(evaluation):
    # Print a separator so each evaluation is easy to read.
    print("\n" + "=" * 80)

    # Print the query being evaluated.
    print("Query:", evaluation["query"])

    # Print the collection being evaluated.
    print("Collection:", evaluation["collection"])

    # Show the raw retrieved chunks.
    print("\nRetrieved chunks:")

    # Print every retrieved chunk.
    for number, result in enumerate(
        evaluation["results"],
        start=1,
    ):

        # Print the rank, source document, and similarity score.
        print(
            f"{number}. "
            f"[Source: {result['source']}] "
            f"Similarity: {result['similarity']:.4f}"
        )

    # Show the unique parent documents after deduplication.
    print("\nRetrieved unique documents:")
    print(
        sorted(
            evaluation["retrieved_documents"]
        )
    )

    # Show the ground-truth documents.
    print("Ground-truth documents:")
    print(
        sorted(
            evaluation["correct_documents"]
        )
    )

    # Show which retrieved documents were actually correct.
    print("Correct retrieved documents:")
    print(
        sorted(
            evaluation["true_positives"]
        )
    )

    # Count the unique retrieved documents.
    retrieved_count = len(
        evaluation["retrieved_documents"]
    )

    # Count the correctly retrieved documents.
    true_positive_count = len(
        evaluation["true_positives"]
    )

    # Count the ground-truth documents.
    ground_truth_count = len(
        evaluation["correct_documents"]
    )

    # Print the exact precision arithmetic.
    print(
        "\nPrecision = "
        f"{true_positive_count} / {retrieved_count} "
        f"= {evaluation['precision']:.4f}"
    )

    # Print the exact recall arithmetic.
    print(
        "Recall = "
        f"{true_positive_count} / {ground_truth_count} "
        f"= {evaluation['recall']:.4f}"
    )


# ============================================================
# Task 5 - evaluate one complete collection
# ============================================================

def evaluate_collection(
    queries,
    collection,
    collection_name,
    model,
):
    # Create an empty list for all evaluations from this collection.
    evaluations = []

    # Evaluate every query independently.
    for query in queries:

        # Evaluate the current query.
        evaluation = evaluate_query(
            query,
            collection,
            collection_name,
            model,
            GROUND_TRUTH,
            TOP_K,
        )

        # Store the evaluation result.
        evaluations.append(evaluation)

        # Print the complete arithmetic for this query.
        print_evaluation(evaluation)

    # Return all evaluations from the collection.
    return evaluations


# ============================================================
# Task 5 - calculate average precision and recall
# ============================================================

def calculate_average_scores(evaluations):
    # Extract the precision value from every query.
    precisions = [
        item["precision"]
        for item in evaluations
    ]

    # Extract the recall value from every query.
    recalls = [
        item["recall"]
        for item in evaluations
    ]

    # Calculate the average precision across all five queries.
    average_precision = (
        sum(precisions)
        / len(precisions)
    )

    # Calculate the average recall across all five queries.
    average_recall = (
        sum(recalls)
        / len(recalls)
    )

    # Return both calculated averages.
    return (
        average_precision,
        average_recall,
    )


# ============================================================
# Task 5 - compare both strategies
# ============================================================

def compare_chunking_strategies(
    fixed_evaluations,
    sentence_evaluations,
):
    # Calculate the average precision and recall for fixed-size chunks.
    fixed_precision, fixed_recall = (
        calculate_average_scores(
            fixed_evaluations
        )
    )

    # Calculate the average precision and recall for sentence chunks.
    sentence_precision, sentence_recall = (
        calculate_average_scores(
            sentence_evaluations
        )
    )

    # Print the Task 5 summary heading.
    print("\n" + "=" * 80)
    print("TASK 5 SUMMARY")

    # Print a comparison table.
    print(
        "\nCollection              "
        "Average Precision      Average Recall"
    )

    # Print the fixed-size collection measurements.
    print(
        f"fixed_chunks            "
        f"{fixed_precision:.4f}                 "
        f"{fixed_recall:.4f}"
    )

    # Print the sentence-based collection measurements.
    print(
        f"sentence_chunks         "
        f"{sentence_precision:.4f}                 "
        f"{sentence_recall:.4f}"
    )

    # Calculate a simple overall average of precision and recall.
    #
    # This is only used to break a mixed result when one strategy
    # has better precision and the other has better recall.
    fixed_overall = (
        fixed_precision
        + fixed_recall
    ) / 2

    # Calculate the same combined value for sentence chunks.
    sentence_overall = (
        sentence_precision
        + sentence_recall
    ) / 2

    # Decide on the deployment recommendation.
    if (
        fixed_precision >= sentence_precision
        and fixed_recall >= sentence_recall
        and (
            fixed_precision > sentence_precision
            or fixed_recall > sentence_recall
        )
    ):

        # Fixed-size chunking is better on both metrics.
        recommendation = (
            f"I would deploy fixed-size chunking because it achieved "
            f"an average precision of {fixed_precision:.4f} and an "
            f"average recall of {fixed_recall:.4f}, compared with "
            f"{sentence_precision:.4f} precision and "
            f"{sentence_recall:.4f} recall for sentence-based chunking. "
            f"It therefore provided the stronger overall retrieval "
            f"performance across the five evaluation queries."
        )

    elif (
        sentence_precision >= fixed_precision
        and sentence_recall >= fixed_recall
        and (
            sentence_precision > fixed_precision
            or sentence_recall > fixed_recall
        )
    ):

        # Sentence-based chunking is better on both metrics.
        recommendation = (
            f"I would deploy sentence-based chunking because it achieved "
            f"an average precision of {sentence_precision:.4f} and an "
            f"average recall of {sentence_recall:.4f}, compared with "
            f"{fixed_precision:.4f} precision and "
            f"{fixed_recall:.4f} recall for fixed-size chunking. "
            f"It therefore provided the stronger overall retrieval "
            f"performance across the five evaluation queries."
        )

    elif fixed_overall > sentence_overall:

        # Fixed-size has the stronger combined average when the
        # two individual metrics are mixed.
        recommendation = (
            f"I would deploy fixed-size chunking because its average "
            f"precision was {fixed_precision:.4f} and its average recall "
            f"was {fixed_recall:.4f}, giving it a stronger combined "
            f"retrieval score than sentence-based chunking, which had "
            f"{sentence_precision:.4f} precision and "
            f"{sentence_recall:.4f} recall. "
            f"This gives fixed-size chunking the better overall result "
            f"for these five evaluation queries."
        )

    elif sentence_overall > fixed_overall:

        # Sentence-based has the stronger combined average when the
        # two individual metrics are mixed.
        recommendation = (
            f"I would deploy sentence-based chunking because its average "
            f"precision was {sentence_precision:.4f} and its average recall "
            f"was {sentence_recall:.4f}, giving it a stronger combined "
            f"retrieval score than fixed-size chunking, which had "
            f"{fixed_precision:.4f} precision and "
            f"{fixed_recall:.4f} recall. "
            f"This gives sentence-based chunking the better overall result "
            f"for these five evaluation queries."
        )

    else:

        # Both strategies have the same combined result.
        recommendation = (
            f"Both strategies produced the same overall combined result. "
            f"Fixed-size chunking achieved {fixed_precision:.4f} average "
            f"precision and {fixed_recall:.4f} average recall, while "
            f"sentence-based chunking achieved {sentence_precision:.4f} "
            f"precision and {sentence_recall:.4f} recall. "
            f"The choice can therefore be made using practical factors "
            f"such as context preservation and storage requirements."
        )

    # Print the recommendation generated from the actual measured numbers.
    print("\nRecommendation:")
    print(recommendation)

    # Return every calculated value so the README can use them.
    return {
        "fixed_precision": fixed_precision,
        "fixed_recall": fixed_recall,
        "sentence_precision": sentence_precision,
        "sentence_recall": sentence_recall,
        "recommendation": recommendation,
    }


# ============================================================
# Task 5 - update README
# ============================================================

def update_task5_readme(
    fixed_evaluations,
    sentence_evaluations,
    summary,
):
    # Use README.md from the current project folder.
    readme_path = Path("README.md")

    # Read the existing README when it exists.
    if readme_path.exists():

        # Load the existing README text.
        existing = readme_path.read_text(
            encoding="utf-8"
        )

    # Create a basic README when it does not exist.
    else:

        # Start with the project title.
        existing = "# RAG Project\n"

    # Create the beginning of the Task 5 section.
    lines = [
        TASK5_README_START,
        "## Task 5 - Evaluation and Comparison of Chunking Strategies",
        "",
        "Task 5 evaluates the same five Task 4 queries against "
        "the two ChromaDB collections separately.",
        "",
        "Chunk results are mapped to their parent `source` document "
        "and duplicate parent documents are removed before precision "
        "and recall are calculated.",
        "",
        "### Fixed-size chunking results",
        "",
    ]

    # Add the detailed results for every fixed-size query.
    for evaluation in fixed_evaluations:

        # Count correctly retrieved documents.
        true_positive_count = len(
            evaluation["true_positives"]
        )

        # Count unique retrieved documents.
        retrieved_count = len(
            evaluation["retrieved_documents"]
        )

        # Count ground-truth documents.
        ground_truth_count = len(
            evaluation["correct_documents"]
        )

        # Add the query heading.
        lines.append(
            f"#### Query: {evaluation['query']}"
        )

        # Add the retrieved unique documents.
        lines.append(
            f"- Retrieved documents: "
            f"`{sorted(evaluation['retrieved_documents'])}`"
        )

        # Add the ground-truth documents.
        lines.append(
            f"- Ground-truth documents: "
            f"`{sorted(evaluation['correct_documents'])}`"
        )

        # Add the precision arithmetic.
        lines.append(
            f"- Precision = "
            f"{true_positive_count} / {retrieved_count} "
            f"= {evaluation['precision']:.4f}"
        )

        # Add the recall arithmetic.
        lines.append(
            f"- Recall = "
            f"{true_positive_count} / {ground_truth_count} "
            f"= {evaluation['recall']:.4f}"
        )

        # Add a blank line after each query.
        lines.append("")

    # Add the sentence-based section.
    lines.extend([
        "### Sentence-based chunking results",
        "",
    ])

    # Add the detailed results for every sentence-based query.
    for evaluation in sentence_evaluations:

        # Count correctly retrieved documents.
        true_positive_count = len(
            evaluation["true_positives"]
        )

        # Count unique retrieved documents.
        retrieved_count = len(
            evaluation["retrieved_documents"]
        )

        # Count ground-truth documents.
        ground_truth_count = len(
            evaluation["correct_documents"]
        )

        # Add the query heading.
        lines.append(
            f"#### Query: {evaluation['query']}"
        )

        # Add the retrieved unique documents.
        lines.append(
            f"- Retrieved documents: "
            f"`{sorted(evaluation['retrieved_documents'])}`"
        )

        # Add the ground-truth documents.
        lines.append(
            f"- Ground-truth documents: "
            f"`{sorted(evaluation['correct_documents'])}`"
        )

        # Add the precision arithmetic.
        lines.append(
            f"- Precision = "
            f"{true_positive_count} / {retrieved_count} "
            f"= {evaluation['precision']:.4f}"
        )

        # Add the recall arithmetic.
        lines.append(
            f"- Recall = "
            f"{true_positive_count} / {ground_truth_count} "
            f"= {evaluation['recall']:.4f}"
        )

        # Add a blank line after each query.
        lines.append("")

    # Add the overall comparison section.
    lines.extend([
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
    ])

    # Join all Task 5 lines together.
    new_block = "\n".join(lines)

    # Create a pattern that matches only the Task 5 README section.
    pattern = re.compile(
        re.escape(TASK5_README_START)
        + r".*?"
        + re.escape(TASK5_README_END),
        re.DOTALL,
    )

    # Replace an existing Task 5 block when one already exists.
    if pattern.search(existing):

        # Replace only the first Task 5 block.
        updated = pattern.sub(
            new_block,
            existing,
            count=1,
        )

    # Otherwise append a new Task 5 section.
    else:

        # Select the correct separator.
        separator = (
            "\n"
            if existing.endswith("\n")
            else "\n\n"
        )

        # Append the new Task 5 section.
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
# Output helper
# ============================================================

def print_result(result):
    # Print a separator before the result.
    print("\n" + "=" * 70)

    # Print the query.
    print("Query:", result["query"])

    # Print the selected collection.
    print("Collection:", result["collection"])

    # Print the top-1 similarity.
    print(
        f"Top-1 cosine similarity: "
        f"{result['similarity']:.4f}"
    )

    # Print whether the result is grounded or a fallback.
    print("Decision:", result["decision"])

    # Print the answer.
    print("Answer:")
    print(result["answer"])


# ============================================================
# Main program
# ============================================================

def main():
    # --------------------------------------------------------
    # Task 3 - load knowledge-base documents
    # --------------------------------------------------------

    # Tell the user that the documents are being loaded.
    print("Loading knowledge-base documents...")

    # Load all TXT documents from the knowledge-base folder.
    documents = load_documents(KNOWLEDGE_BASE)

    # The problem statement requires at least 12 documents.
    if len(documents) < 12:

        # Stop the program when fewer than 12 documents exist.
        raise RuntimeError(
            f"At least 12 knowledge-base documents are required; "
            f"found {len(documents)}."
        )

    # Print the number of loaded documents.
    print(
        f"Loaded {len(documents)} documents."
    )

    # --------------------------------------------------------
    # Task 3 - fixed-size chunking
    # --------------------------------------------------------

    # Tell the user that fixed-size chunks are being created.
    print(
        "\nCreating fixed-size chunks..."
    )

    # Create fixed-size chunks from every document.
    fixed_chunks = build_chunks(
        documents,
        fixed_size_chunks,
    )

    # Print the total number of fixed-size chunks.
    print(
        f"Fixed-size chunks: "
        f"{len(fixed_chunks)}"
    )

    # --------------------------------------------------------
    # Task 3 - sentence-based chunking
    # --------------------------------------------------------

    # Tell the user that sentence-based chunks are being created.
    print(
        "Creating sentence-based chunks..."
    )

    # Create sentence-based chunks from every document.
    sentence_chunks = build_chunks(
        documents,
        sentence_based_chunks,
    )

    # Print the total number of sentence-based chunks.
    print(
        f"Sentence-based chunks: "
        f"{len(sentence_chunks)}"
    )

    # --------------------------------------------------------
    # Task 3 - load embedding model
    # --------------------------------------------------------

    # Tell the user that the local embedding model is loading.
    print(
        "\nLoading embedding model..."
    )

    # Load the required free local SentenceTransformers model.
    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Task 3 - create persistent ChromaDB client
    # --------------------------------------------------------

    # Create a persistent ChromaDB client.
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # --------------------------------------------------------
    # Task 3 - create two separate collections
    # --------------------------------------------------------

    # Create or prepare the fixed-size chunk collection.
    fixed_collection = prepare_collection(
        client,
        "fixed_chunks",
    )

    # Create or prepare the sentence-based chunk collection.
    sentence_collection = prepare_collection(
        client,
        "sentence_chunks",
    )

    # --------------------------------------------------------
    # Task 3 - store fixed-size chunks
    # --------------------------------------------------------

    # Tell the user that fixed-size chunks are being stored.
    print(
        "\nStoring fixed-size chunks..."
    )

    # Store fixed-size chunks and their embeddings.
    store_chunks(
        fixed_collection,
        fixed_chunks,
        model,
    )

    # --------------------------------------------------------
    # Task 3 - store sentence-based chunks
    # --------------------------------------------------------

    # Tell the user that sentence chunks are being stored.
    print(
        "Storing sentence-based chunks..."
    )

    # Store sentence-based chunks and their embeddings.
    store_chunks(
        sentence_collection,
        sentence_chunks,
        model,
    )

    # Print the number of records in the first collection.
    print(
        f"fixed_chunks collection: "
        f"{fixed_collection.count()} records"
    )

    # Print the number of records in the second collection.
    print(
        f"sentence_chunks collection: "
        f"{sentence_collection.count()} records"
    )

    # --------------------------------------------------------
    # Task 4 - measure in-scope questions
    # --------------------------------------------------------

    # Tell the user that in-scope calibration is starting.
    print(
        "\nMeasuring in-scope calibration queries..."
    )

    # Measure the in-scope queries.
    in_scope = measure_queries(
        IN_SCOPE_QUERIES,
        model,
        fixed_collection,
        sentence_collection,
    )

    # --------------------------------------------------------
    # Task 4 - measure out-of-scope questions
    # --------------------------------------------------------

    # Tell the user that out-of-scope calibration is starting.
    print(
        "Measuring out-of-scope calibration queries..."
    )

    # Measure the out-of-scope queries.
    out_of_scope = measure_queries(
        OUT_OF_SCOPE_QUERIES,
        model,
        fixed_collection,
        sentence_collection,
    )

    # --------------------------------------------------------
    # Task 4 - calculate threshold
    # --------------------------------------------------------

    # Calculate the threshold from the measured values.
    threshold = choose_threshold(
        in_scope,
        out_of_scope,
    )

    # --------------------------------------------------------
    # Task 4 - print calibration results
    # --------------------------------------------------------

    # Print the in-scope measurements.
    print(
        "\nIN-SCOPE CALIBRATION"
    )

    # Print every in-scope measurement.
    for item in in_scope:

        # Display similarity, collection, and query.
        print(
            f"{item['similarity']:.4f} | "
            f"{item['collection']} | "
            f"{item['query']}"
        )

    # Print the out-of-scope measurements.
    print(
        "\nOUT-OF-SCOPE CALIBRATION"
    )

    # Print every out-of-scope measurement.
    for item in out_of_scope:

        # Display similarity, collection, and query.
        print(
            f"{item['similarity']:.4f} | "
            f"{item['collection']} | "
            f"{item['query']}"
        )

    # Print the chosen threshold.
    print(
        f"\nChosen threshold: "
        f"{threshold:.4f}"
    )

    # --------------------------------------------------------
    # Task 4 - update README
    # --------------------------------------------------------

    # Save Task 4 calibration information in README.md.
    update_readme(
        in_scope,
        out_of_scope,
        threshold,
    )

    # --------------------------------------------------------
    # Task 4 - grounded generation demonstration
    # --------------------------------------------------------

    # Tell the user that grounded generation is starting.
    print(
        "\nGROUNDED GENERATION DEMONSTRATION"
    )

    # Use the first five in-scope queries exactly as in the
    # original Task 4 demonstration.
    for query in IN_SCOPE_QUERIES[:5]:

        # Generate a grounded answer.
        result = grounded_generate(
            query,
            model,
            fixed_collection,
            sentence_collection,
            threshold,
            TOP_K,
        )

        # Print the generated result.
        print_result(result)

    # --------------------------------------------------------
    # Task 4 - fallback demonstration
    # --------------------------------------------------------

    # Select the lowest-scoring out-of-scope query.
    fallback_query = min(
        out_of_scope,
        key=lambda item: item["similarity"],
    )["query"]

    # Tell the user that the fallback demonstration is starting.
    print(
        "\nOUT-OF-SCOPE FALLBACK DEMONSTRATION"
    )

    # Run grounded generation for the fallback query.
    fallback_result = grounded_generate(
        fallback_query,
        model,
        fixed_collection,
        sentence_collection,
        threshold,
        TOP_K,
    )

    # Print the fallback result.
    print_result(fallback_result)

    # Make sure the selected out-of-scope query triggers fallback.
    if fallback_result["decision"] != "FALLBACK":

        # Stop the program when the expected fallback does not happen.
        raise RuntimeError(
            "The selected out-of-scope query did not trigger "
            "the fallback. Review the measured scores and threshold."
        )

    # ========================================================
    # Task 5 - evaluate fixed-size chunking
    # ========================================================

    # Print a clear heading for fixed-size evaluation.
    print(
        "\n\nTASK 5 - FIXED-SIZE CHUNKING EVALUATION"
    )

    # Evaluate the same five Task 4 queries independently
    # against the fixed-size collection.
    fixed_evaluations = evaluate_collection(
        TASK5_QUERIES,
        fixed_collection,
        "fixed_chunks",
        model,
    )

    # ========================================================
    # Task 5 - evaluate sentence-based chunking
    # ========================================================

    # Print a clear heading for sentence-based evaluation.
    print(
        "\n\nTASK 5 - SENTENCE-BASED CHUNKING EVALUATION"
    )

    # Evaluate the exact same five queries independently
    # against the sentence-based collection.
    sentence_evaluations = evaluate_collection(
        TASK5_QUERIES,
        sentence_collection,
        "sentence_chunks",
        model,
    )

    # ========================================================
    # Task 5 - compare both strategies
    # ========================================================

    # Compare average precision and recall from both strategies.
    task5_summary = compare_chunking_strategies(
        fixed_evaluations,
        sentence_evaluations,
    )

    # ========================================================
    # Task 5 - save results to README
    # ========================================================

    # Save the detailed Task 5 results and recommendation.
    update_task5_readme(
        fixed_evaluations,
        sentence_evaluations,
        task5_summary,
    )

    # --------------------------------------------------------
    # Final completion message
    # --------------------------------------------------------

    # Print a final message confirming Task 4 and Task 5.
    print(
        "\nTask 4 and Task 5 completed successfully."
    )

    # Tell the user that calibration information was saved.
    print(
        "Task 4 calibration results were saved to README.md."
    )

    # Tell the user that Task 5 evaluation was also saved.
    print(
        "Task 5 precision and recall results were saved to README.md."
    )


# ============================================================
# Program entry point
# ============================================================

# Run main() only when this Python file is executed directly.
if __name__ == "__main__":

    # Start the complete Task 3, Task 4, and Task 5 pipeline.
    main()
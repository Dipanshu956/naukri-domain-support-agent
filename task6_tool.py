
# ------------------------------------------------------------
# Task 6 - Job Application Escalation Tool
# ------------------------------------------------------------

# Import statistics so we can calculate the 80th percentile
# of the escalation scores.
import statistics

# Import generate_dataset from dataset.py.
# We are reusing the existing dataset-generation function
# instead of creating another dataset here.
from dataset import generate_dataset


# ------------------------------------------------------------
# Task 6 configuration
# ------------------------------------------------------------

# Give 60% of the escalation score to the priority signal.
PRIORITY_WEIGHT = 0.6

# Give the remaining 40% of the escalation score to recency.
RECENCY_WEIGHT = 0.4

# Treat an application that is 30 days old as fully stale.
# At 30 days, the normalized recency becomes 1.0.
MAX_RECENCY_DAYS = 30


# ------------------------------------------------------------
# Generate the dataset once
# ------------------------------------------------------------

# Call the existing function from dataset.py.
#
# This creates the 50 real application records that Task 6
# will use for calculating the escalation scores and threshold.
APPLICATIONS = generate_dataset()


# ------------------------------------------------------------
# Helper function: calculate priority signal
# ------------------------------------------------------------

def calculate_priority_signal(application):
    # Read the actual priority field used by dataset.py.
    #
    # The real dataset uses:
    #     flagged_priority_review
    #
    # A flagged application gets a value of 1.
    # An unflagged application gets a value of 0.
    priority_flag = application.get(
        "flagged_priority_review",
        False,
    )

    # Convert the priority value into a numeric signal.
    #
    # True  -> 1.0
    # False -> 0.0
    return 1.0 if bool(priority_flag) else 0.0


# ------------------------------------------------------------
# Helper function: calculate normalized recency
# ------------------------------------------------------------

def calculate_normalized_recency(days_since_created):
    # Convert the number of days into a float.
    #
    # This makes the calculation work even if the dataset
    # contains the value as an integer or numeric string.
    days = float(days_since_created)

    # Negative application age does not make sense.
    # Convert a negative value to zero.
    if days < 0:
        days = 0.0

    # Convert the age into a value between 0.0 and 1.0.
    #
    # Example:
    # 0 days  = 0.0
    # 15 days = 0.5
    # 30 days = 1.0
    #
    # The min() prevents the value from becoming greater
    # than 1.0 for applications older than 30 days.
    normalized_recency = min(
        days / MAX_RECENCY_DAYS,
        1.0,
    )

    # Return the normalized recency value.
    return normalized_recency


# ------------------------------------------------------------
# Helper function: calculate escalation score
# ------------------------------------------------------------

def calculate_escalation_score(application):
    # Calculate the priority component.
    priority_signal = calculate_priority_signal(
        application
    )

    # Get the application's age from the real dataset field.
    days_since_created = application.get(
        "days_since_created",
        0,
    )

    # Convert the application age into the 0-to-1 range.
    normalized_recency = calculate_normalized_recency(
        days_since_created
    )

    # Combine the priority and recency signals.
    #
    # Formula:
    #
    # escalation_score =
    #       (0.6 * priority_signal)
    #     + (0.4 * normalized_recency)
    #
    # Priority contributes 60%.
    # Recency contributes 40%.
    escalation_score = (
        PRIORITY_WEIGHT * priority_signal
        + RECENCY_WEIGHT * normalized_recency
    )

    # Return the final escalation score.
    return escalation_score


# ------------------------------------------------------------
# Calculate escalation scores for all applications
# ------------------------------------------------------------

def calculate_all_escalation_scores(applications):
    # Create an empty list where we will store the score
    # of every application.
    scores = []

    # Process every application in the dataset.
    for application in applications:

        # Calculate the escalation score for this record.
        score = calculate_escalation_score(
            application
        )

        # Add the score to our list.
        scores.append(score)

    # Return the complete list of scores.
    return scores


# ------------------------------------------------------------
# Calculate the 80th percentile threshold
# ------------------------------------------------------------

def calculate_escalation_threshold(applications):
    # First calculate the escalation score for ALL records.
    #
    # This is important because the threshold must be based
    # on the actual escalation_score distribution.
    scores = calculate_all_escalation_scores(
        applications
    )

    # Make sure there are enough records to calculate
    # a meaningful threshold.
    if len(scores) < 5:
        raise ValueError(
            "At least 5 application scores are required."
        )

    # Sort the scores and calculate the 80th percentile.
    #
    # n=5 divides the data into fifths.
    # Index [3] gives the fourth cutoff, which represents
    # the 80th percentile.
    #
    # method="inclusive" gives the percentile calculation
    # used by our Task 6 design.
    percentile_values = statistics.quantiles(
        scores,
        n=5,
        method="inclusive",
    )

    # The fourth quantile is the 80th percentile.
    threshold = percentile_values[3]

    # Return the calculated threshold.
    return threshold


# ------------------------------------------------------------
# Calculate the threshold once
# ------------------------------------------------------------

# Calculate the threshold from the actual 50 records.
#
# Keeping this as a module-level value means that
# check_job_application_status() needs only one argument:
#
#     record_id
#
# This is also useful when the function is later exposed
# as a tool to an agent.
ESCALATION_THRESHOLD = calculate_escalation_threshold(
    APPLICATIONS
)


# ------------------------------------------------------------
# Main Task 6 tool function
# ------------------------------------------------------------

def check_job_application_status(record_id: str) -> dict: # 
    # Search through all generated application records.
    for application in APPLICATIONS:

        # Compare the supplied record ID with the actual
        # dataset field called "record_id".
        if application.get("record_id") == record_id:

            # Calculate the escalation score for this
            # particular application.
            escalation_score = calculate_escalation_score(
                application
            )

            # Return the required information.
            #
            # "status" and "expected_salary_inr" are taken
            # directly from the matching dataset record.
            return {
                        "record_id": application["record_id"],
                        "candidate_name": application["candidate_name"],
                        "status": application["status"],
                        "expected_salary_inr": application[
                            "expected_salary_inr"
                        ],
                        "escalation_score": escalation_score,
                        "escalation_recommended": (
                            escalation_score > ESCALATION_THRESHOLD
                        ),
                    }

    # If the loop finishes without finding the record,
    # raise an error so invalid IDs are handled clearly.
    raise ValueError(
        f"Record ID '{record_id}' was not found."
    )


# ------------------------------------------------------------
# Main program used for Task 6 demonstration
# ------------------------------------------------------------

def main():
    # Print a heading so the output is easy to understand.
    print("Task 6 - Job Application Escalation Tool")

    # Print how many records were generated.
    print(
        f"Number of application records: "
        f"{len(APPLICATIONS)}"
    )

    # Print the threshold calculated from the actual
    # escalation-score distribution.
    print(
        f"80th percentile escalation threshold: "
        f"{ESCALATION_THRESHOLD:.3f}"
    )

    # --------------------------------------------------------
    # Show how many applications are priority flagged
    # --------------------------------------------------------

    # Count the records where flagged_priority_review is True.
    flagged_count = sum(
        1
        for application in APPLICATIONS
        if bool(
            application.get(
                "flagged_priority_review",
                False,
            )
        )
    )

    # Print the number of flagged records.
    print(
        f"Priority-review applications: "
        f"{flagged_count}"
    )

    # --------------------------------------------------------
    # Test 1: valid record
    # --------------------------------------------------------

    # APP001 is used as the valid-record demonstration.
    test_record_id = "APP001"

    print(
        f"\nChecking record: {test_record_id}"
    )

    try:
        # Call the required Task 6 tool function.
        result = check_job_application_status(
            test_record_id
        )

        # Print the returned dictionary.
        print("Result:")
        print(result)

    except ValueError as error:
        # Print any error that occurs during the valid-record
        # test.
        print(f"Error: {error}")

    # --------------------------------------------------------
    # Test 2: invalid record
    # --------------------------------------------------------

    # APP999 is intentionally used because it should not
    # exist in the generated 50-record dataset.
    invalid_record_id = "APP999"

    print(
        f"\nChecking invalid record: "
        f"{invalid_record_id}"
    )

    try:
        # Try to look up the invalid record.
        result = check_job_application_status(
            invalid_record_id
        )

        # This should not normally execute because
        # APP999 should raise a ValueError.
        print(result)

    except ValueError as error:
        # Catch the expected error rather than allowing
        # the whole program to crash.
        print(
            f"Handled error successfully: {error}"
        )


# ------------------------------------------------------------
# Start the program
# ------------------------------------------------------------

# This condition makes sure main() runs when you execute:
#
#     python task6_tool.py
#
# It prevents main() from running automatically if this
# file is imported by another Python file.
if __name__ == "__main__":
    main()

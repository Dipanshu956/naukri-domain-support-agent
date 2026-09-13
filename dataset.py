
# dataset.py

import random
import csv
from collections import Counter


# ============================================================
# Configuration
# ============================================================

SEED = 42
NUM_RECORDS = 50
OUTPUT_FILE = "job_applications.csv"

# ============================================================
# Required categories
# ============================================================

CATEGORIES = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate",
]

# Equal weights for category generation.
# The first 3 records of every category are guaranteed separately.
CATEGORY_WEIGHTS = {
    "Software Engineer": 1,
    "Data Analyst": 1,
    "Product Manager": 1,
    "HR Executive": 1,
    "Sales Associate": 1,
}

# ============================================================
# Required statuses
# ============================================================

STATUSES = [
    "Applied",
    "Screening",
    "Interview Scheduled",
    "Offered",
    "Rejected",
]

# Equal weights for status generation.
STATUS_WEIGHTS = {
    "Applied": 1,
    "Screening": 1,
    "Interview Scheduled": 1,
    "Offered": 1,
    "Rejected": 1,
}

# ============================================================
# Required flagged band
# ============================================================

FLAGGED_MIN_PCT = 10
FLAGGED_MAX_PCT = 30

# ============================================================
# Expected salary range
# ============================================================

SALARY_MIN = 400000
SALARY_MAX = 1800000


# ============================================================
# Additional realistic fields
# ============================================================

FIRST_NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram",
    "Neha", "Rohan", "Anjali", "Karan", "Pooja",
    "Arjun", "Meera", "Aditya", "Kavya", "Nikhil",
    "Isha", "Saurabh", "Riya", "Manish", "Simran"
]

LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Joshi", "Shah",
    "Verma", "Singh", "Gupta", "Mehta", "Deshmukh",
    "Kulkarni", "Yadav", "Agarwal", "Reddy", "Nair"
]

LOCATIONS = [
    "Pune",
    "Mumbai",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Delhi",
    "Nashik",
    "Noida",
    "Gurgaon",
]

SKILLS_BY_CATEGORY = {
    "Software Engineer": [
        "Python, SQL, FastAPI, AWS",
        "Java, Spring Boot, SQL, AWS",
        "Python, Django, REST API, PostgreSQL",
        "Java, Spring, Microservices, Docker",
    ],
    "Data Analyst": [
        "SQL, Excel, Tableau, Power BI",
        "Python, SQL, Power BI, Excel",
        "SQL, Tableau, Python, Statistics",
    ],
    "Product Manager": [
        "Product Strategy, Agile, Jira, Analytics",
        "Roadmapping, Agile, SQL, Stakeholder Management",
        "Product Analytics, Scrum, User Research, Jira",
    ],
    "HR Executive": [
        "Recruitment, HRMS, Payroll, Employee Relations",
        "Talent Acquisition, Excel, HRMS, Onboarding",
        "Recruitment, Payroll, Employee Engagement, HRMS",
    ],
    "Sales Associate": [
        "CRM, Lead Generation, Communication, Sales",
        "Salesforce, Negotiation, Lead Generation, CRM",
        "Business Development, CRM, Communication, Sales",
    ],
}

EDUCATION = [
    "B.Tech",
    "B.E.",
    "M.Tech",
    "MBA",
    "MCA",
]


# ============================================================
# Helper functions
# ============================================================

def generate_experience():
    """Generate years of experience between 1 and 10."""

    return random.randint(1, 10)


def generate_days_since_created():
    """
    Generate application age.

    Required range: 0 to 30 days inclusive.
    """

    return random.randint(0, 30)


def generate_notice_period():
    """Generate a realistic notice period."""

    return random.choice([0, 15, 30, 45, 60, 90])


def generate_candidate_name(used_names):
    """Generate a unique candidate name."""

    while True:
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

        if name not in used_names:
            used_names.add(name)
            return name


def generate_salary(experience, category):
    """
    Generate expected annual salary in INR.

    Salary increases with experience and is adjusted slightly
    by job category. A small random variation prevents identical
    salary patterns.
    """

    category_adjustment = {
        "Software Engineer": 1.05,
        "Data Analyst": 0.90,
        "Product Manager": 1.15,
        "HR Executive": 0.90,
        "Sales Associate": 0.85,
    }

    base_salary = 450000 + (experience * 150000)

    salary = (
        base_salary
        * category_adjustment[category]
    )

    salary += random.randint(-75000, 75000)

    # Enforce assignment salary range.
    salary = max(SALARY_MIN, min(SALARY_MAX, salary))

    # Round to nearest ₹10,000.
    salary = round(salary / 10000) * 10000

    return int(salary)


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset():
    """
    Generate the required job applications.

    Guarantees:
    - At least 3 records for every category.
    - At least 1 record for every status.
    - Flagged percentage between 10% and 30%.
    - days_since_created between 0 and 30.
    - expected_salary_inr inside configured salary range.
    """

    random.seed(SEED)

    JOB_APPLICATIONS = []
    used_names = set()

    # --------------------------------------------------------
    # GUARANTEE CATEGORY COVERAGE
    # --------------------------------------------------------
    # Add each category exactly 3 times first.
    # This guarantees 3 records for every category before
    # the remaining records are randomly generated.

    categories = []

    for category in CATEGORIES:
        categories.extend([category] * 3)

    # Number of remaining category slots.
    remaining_category_slots = (
        NUM_RECORDS - len(categories)
    )

    # Fill the remaining slots using weighted random selection.
    categories.extend(
        random.choices(
            CATEGORIES,
            weights=[
                CATEGORY_WEIGHTS[category]
                for category in CATEGORIES
            ],
            k=remaining_category_slots,
        )
    )

    # Shuffle to avoid having the guaranteed records grouped.
    random.shuffle(categories)

    # --------------------------------------------------------
    # GUARANTEE STATUS COVERAGE
    # --------------------------------------------------------
    # Add every required status once first.

    statuses = STATUSES.copy()

    remaining_status_slots = (
        NUM_RECORDS - len(statuses)
    )

    statuses.extend(
        random.choices(
            STATUSES,
            weights=[
                STATUS_WEIGHTS[status]
                for status in STATUSES
            ],
            k=remaining_status_slots,
        )
    )

    random.shuffle(statuses)

    # --------------------------------------------------------
    # GUARANTEE FLAGGED PERCENTAGE
    # --------------------------------------------------------
    # For 50 records:
    # 10% = 5 records
    # 30% = 15 records

    min_flagged = (
        NUM_RECORDS * FLAGGED_MIN_PCT + 99
    ) // 100

    max_flagged = (
        NUM_RECORDS * FLAGGED_MAX_PCT
    ) // 100

    flagged_count = random.randint(
        min_flagged,
        max_flagged
    )

    flagged_values = (
        [True] * flagged_count
        + [False] * (NUM_RECORDS - flagged_count)
    )

    # Randomly assign which records are flagged.
    random.shuffle(flagged_values)

    # --------------------------------------------------------
    # CREATE RECORDS
    # --------------------------------------------------------

    for i in range(NUM_RECORDS):

        category = categories[i]
        experience = generate_experience()

        record = {
            "record_id": f"APP{i + 1:03d}",

            "candidate_name": generate_candidate_name(
                used_names
            ),

            "experience": experience,

            "skills": random.choice(
                SKILLS_BY_CATEGORY[category]
            ),

            "notice_period": generate_notice_period(),

            "education": random.choice(
                EDUCATION
            ),

            "location": random.choice(
                LOCATIONS
            ),

            # Required fields
            "category": category,

            "status": statuses[i],

            "expected_salary_inr": generate_salary(
                experience,
                category
            ),

            "days_since_created":
                generate_days_since_created(),

            "flagged_priority_review":
                flagged_values[i],
        }

        JOB_APPLICATIONS.append(record)

    return JOB_APPLICATIONS


# ============================================================
# Validation
# ============================================================

def validate_dataset(JOB_APPLICATIONS):
    """
    Validate the assignment requirements.
    """

    # --------------------------------------------------------
    # Record count
    # --------------------------------------------------------

    assert len(JOB_APPLICATIONS) >= 40, (
        f"Dataset must contain at least 40 records. "
        f"Found {len(JOB_APPLICATIONS)}."
    )

    # --------------------------------------------------------
    # Category coverage
    # --------------------------------------------------------

    category_counts = Counter(
        record["category"]
        for record in JOB_APPLICATIONS
    )

    for category in CATEGORIES:
        assert category_counts[category] >= 3, (
            f"Category '{category}' must have at least "
            f"3 records. Found {category_counts[category]}."
        )

    # --------------------------------------------------------
    # Status coverage
    # --------------------------------------------------------

    status_counts = Counter(
        record["status"]
        for record in JOB_APPLICATIONS
    )

    for status in STATUSES:
        assert status_counts[status] >= 1, (
            f"Status '{status}' must have at least "
            f"1 record. Found {status_counts[status]}."
        )

    # --------------------------------------------------------
    # Flagged percentage
    # --------------------------------------------------------

    flagged_count = sum(
        record["flagged_priority_review"]
        for record in JOB_APPLICATIONS
    )

    flagged_pct = (
        flagged_count / len(JOB_APPLICATIONS)
    ) * 100

    assert (
        FLAGGED_MIN_PCT
        <= flagged_pct
        <= FLAGGED_MAX_PCT
    ), (
        f"Flagged percentage {flagged_pct:.1f}% is outside "
        f"the required {FLAGGED_MIN_PCT}%–"
        f"{FLAGGED_MAX_PCT}% band."
    )

    # --------------------------------------------------------
    # Salary range
    # --------------------------------------------------------

    for record in JOB_APPLICATIONS:

        salary = record["expected_salary_inr"]

        assert SALARY_MIN <= salary <= SALARY_MAX, (
            f"Salary {salary} is outside the allowed "
            f"range ₹{SALARY_MIN:,}–₹{SALARY_MAX:,}."
        )

    # --------------------------------------------------------
    # days_since_created range
    # --------------------------------------------------------

    for record in JOB_APPLICATIONS:

        days = record["days_since_created"]

        assert 0 <= days <= 30, (
            f"days_since_created must be between "
            f"0 and 30. Found {days}."
        )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = {
        "record_id",
        "category",
        "status",
        "expected_salary_inr",
        "days_since_created",
        "flagged_priority_review",
    }

    for record in JOB_APPLICATIONS:

        missing_fields = (
            required_fields - record.keys()
        )

        assert not missing_fields, (
            f"Missing required fields: {missing_fields}"
        )

    return True


# ============================================================
# Save CSV
# ============================================================

def save_dataset(JOB_APPLICATIONS):
    """Save the generated records to CSV."""

    if not JOB_APPLICATIONS:
        return

    fieldnames = list(
        JOB_APPLICATIONS[0].keys()
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(JOB_APPLICATIONS)


# ============================================================
# Report
# ============================================================

def print_report(JOB_APPLICATIONS):
    """Print a summary of the generated dataset."""

    category_counts = Counter(
        record["category"]
        for record in JOB_APPLICATIONS
    )

    status_counts = Counter(
        record["status"]
        for record in JOB_APPLICATIONS
    )

    flagged_count = sum(
        record["flagged_priority_review"]
        for record in JOB_APPLICATIONS
    )

    flagged_pct = (
        flagged_count / len(JOB_APPLICATIONS)
    ) * 100

    salaries = [
        record["expected_salary_inr"]
        for record in JOB_APPLICATIONS
    ]

    print("\n" + "=" * 65)
    print("JOB APPLICATION DATASET REPORT")
    print("=" * 65)

    print(f"Total records       : {len(JOB_APPLICATIONS)}")
    print(f"Random seed         : {SEED}")

    print("\nCategory coverage:")

    for category in CATEGORIES:
        print(
            f"  {category:<22}: "
            f"{category_counts[category]}"
        )

    print("\nStatus coverage:")

    for status in STATUSES:
        print(
            f"  {status:<22}: "
            f"{status_counts[status]}"
        )

    print("\nFlagged priority review:")

    print(f"  Count              : {flagged_count}")
    print(f"  Percentage         : {flagged_pct:.1f}%")
    print(
        f"  Required band      : "
        f"{FLAGGED_MIN_PCT}% - "
        f"{FLAGGED_MAX_PCT}%"
    )

    print("\nExpected salary:")

    print(
        f"  Minimum            : "
        f"₹{min(salaries):,}"
    )

    print(
        f"  Maximum            : "
        f"₹{max(salaries):,}"
    )

    print(
        f"  Average            : "
        f"₹{sum(salaries) / len(salaries):,.0f}"
    )

    print("\nValidation:")
    print("  ✓ At least 40 records")
    print("  ✓ Every category has at least 3 records")
    print("  ✓ Every status appears at least once")
    print("  ✓ Flagged percentage is within 10–30%")
    print("  ✓ Salary values are within the allowed range")
    print("  ✓ days_since_created is within 0–30")
    print("  ✓ All required fields are present")

    print("=" * 65)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    JOB_APPLICATIONS = generate_dataset()

    validate_dataset(JOB_APPLICATIONS)

    save_dataset(JOB_APPLICATIONS)

    print_report(JOB_APPLICATIONS)

    print(
        f"\nDataset saved to: {OUTPUT_FILE}"
    )




<!-- TASK4_START -->
## Task 4 - Grounded Generation and Threshold Calibration

### In-scope measurements

| Query | Collection | Top-1 cosine similarity |
|---|---|---:|
| What degree is required for most professional jobs? | sentence_chunks | 0.5715 |
| How much notice should a candidate get before an interview? | sentence_chunks | 0.5786 |
| What is the normal employee notice period after resignation? | sentence_chunks | 0.7887 |
| How much is the employee referral bonus? | sentence_chunks | 0.7571 |
| When can an employee apply for an internal transfer? | sentence_chunks | 0.8126 |
| How long is the normal probation period? | sentence_chunks | 0.7332 |

### Out-of-scope measurements

| Query | Collection | Top-1 cosine similarity |
|---|---|---:|
| What is the capital of France? | fixed_chunks | 0.0890 |
| What is the weather forecast for tomorrow? | fixed_chunks | 0.1275 |
| How do I bake a chocolate cake? | fixed_chunks | 0.0925 |

### Chosen threshold: `0.3495`

The threshold was calculated from the measured values. No fixed 0.5, 0.6, or 0.7 preset was used.

<!-- TASK4_END -->

<!-- TASK5_START -->
## Task 5 - Evaluation and Comparison of Chunking Strategies

Task 5 evaluates the same five Task 4 queries against the two ChromaDB collections separately.

Chunk results are mapped to their parent `source` document and duplicate parent documents are removed before precision and recall are calculated.

### Fixed-size chunking results

#### Query: What degree is required for most professional jobs?
- Retrieved documents: `['01_eligibility_criteria', '10_diversity_hiring']`
- Ground-truth documents: `['01_eligibility_criteria']`
- Precision = 1 / 2 = 0.5000
- Recall = 1 / 1 = 1.0000

#### Query: How much notice should a candidate get before an interview?
- Retrieved documents: `['02_interview_scheduling', '06_referral_bonus']`
- Ground-truth documents: `['02_interview_scheduling']`
- Precision = 1 / 2 = 0.5000
- Recall = 1 / 1 = 1.0000

#### Query: What is the normal employee notice period after resignation?
- Retrieved documents: `['05_notice_period', '08_probation_period', '11_exit_interview']`
- Ground-truth documents: `['05_notice_period']`
- Precision = 1 / 3 = 0.3333
- Recall = 1 / 1 = 1.0000

#### Query: How much is the employee referral bonus?
- Retrieved documents: `['06_referral_bonus']`
- Ground-truth documents: `['06_referral_bonus']`
- Precision = 1 / 1 = 1.0000
- Recall = 1 / 1 = 1.0000

#### Query: When can an employee apply for an internal transfer?
- Retrieved documents: `['07_internal_transfer']`
- Ground-truth documents: `['07_internal_transfer']`
- Precision = 1 / 1 = 1.0000
- Recall = 1 / 1 = 1.0000

### Sentence-based chunking results

#### Query: What degree is required for most professional jobs?
- Retrieved documents: `['01_eligibility_criteria', '09_remote_work', '10_diversity_hiring']`
- Ground-truth documents: `['01_eligibility_criteria']`
- Precision = 1 / 3 = 0.3333
- Recall = 1 / 1 = 1.0000

#### Query: How much notice should a candidate get before an interview?
- Retrieved documents: `['02_interview_scheduling', '10_diversity_hiring']`
- Ground-truth documents: `['02_interview_scheduling']`
- Precision = 1 / 2 = 0.5000
- Recall = 1 / 1 = 1.0000

#### Query: What is the normal employee notice period after resignation?
- Retrieved documents: `['05_notice_period', '08_probation_period', '11_exit_interview']`
- Ground-truth documents: `['05_notice_period']`
- Precision = 1 / 3 = 0.3333
- Recall = 1 / 1 = 1.0000

#### Query: How much is the employee referral bonus?
- Retrieved documents: `['03_offer_negotiation', '06_referral_bonus']`
- Ground-truth documents: `['06_referral_bonus']`
- Precision = 1 / 2 = 0.5000
- Recall = 1 / 1 = 1.0000

#### Query: When can an employee apply for an internal transfer?
- Retrieved documents: `['05_notice_period', '07_internal_transfer']`
- Ground-truth documents: `['07_internal_transfer']`
- Precision = 1 / 2 = 0.5000
- Recall = 1 / 1 = 1.0000

### Overall comparison

| Collection | Average Precision | Average Recall |
|---|---:|---:|
| fixed_chunks | 0.6667 | 1.0000 |
| sentence_chunks | 0.4333 | 1.0000 |

### Recommendation

I would deploy fixed-size chunking because it achieved an average precision of 0.6667 and an average recall of 1.0000, compared with 0.4333 precision and 1.0000 recall for sentence-based chunking. It therefore provided the stronger overall retrieval performance across the five evaluation queries.

<!-- TASK5_END -->

<!-- TASK13_START -->
## Task 13 - Evaluation with Accuracy, Grounding, Completeness, and Safety

Task 13 was evaluated under `MOCK_LLM` using exactly 15 independent queries.

The test set covers all 12 required KB topics, two deliberately out-of-scope queries, and one application-status lookup query.

The calibrated RAG threshold used by the existing CrewAI system was `0.3495`.

### Per-query scores

| ID | Type | Topic | Accuracy | Grounding | Completeness | Safety | Top-1 Similarity |
|---|---|---|---:|---:|---:|---:|---:|
| Q01 | kb | job-application-eligibility | 0.8000 | 0.5532 | 0.6000 | 1.0000 | 0.5532 |
| Q02 | kb | interview-scheduling | 0.6000 | 0.4646 | 0.2000 | 1.0000 | 0.4646 |
| Q03 | kb | offer-negotiation | 0.8000 | 0.4833 | 0.6000 | 1.0000 | 0.4833 |
| Q04 | kb | background-verification | 1.0000 | 0.6523 | 1.0000 | 1.0000 | 0.6523 |
| Q05 | kb | notice-period | 1.0000 | 0.6544 | 1.0000 | 1.0000 | 0.6544 |
| Q06 | kb | referral-bonus | 1.0000 | 0.6534 | 1.0000 | 1.0000 | 0.6534 |
| Q07 | kb | internal-transfer | 1.0000 | 0.7079 | 1.0000 | 1.0000 | 0.7079 |
| Q08 | kb | probation-period | 1.0000 | 0.6489 | 1.0000 | 1.0000 | 0.6489 |
| Q09 | kb | remote-work-eligibility | 1.0000 | 0.7027 | 1.0000 | 1.0000 | 0.7027 |
| Q10 | kb | diversity-hiring | 0.8000 | 0.5957 | 0.6000 | 1.0000 | 0.5957 |
| Q11 | kb | exit-interview | 1.0000 | 0.5682 | 1.0000 | 1.0000 | 0.5682 |
| Q12 | kb | applicant-data-retention | 0.8667 | 0.7577 | 0.7333 | 1.0000 | 0.7577 |
| Q13 | out_of_scope | out-of-scope | 1.0000 | 0.2008 | 1.0000 | 1.0000 | 0.2008 |
| Q14 | out_of_scope | out-of-scope | 1.0000 | 0.1665 | 1.0000 | 1.0000 | 0.1665 |
| Q15 | lookup | application-status | 1.0000 | 0.3493 | 1.0000 | 1.0000 | 0.3493 |

### Four required averages

| Metric | Average |
|---|---:|
| Accuracy | 0.9244 |
| Grounding | 0.5439 |
| Completeness | 0.8489 |
| Safety | 1.0000 |

### Out-of-scope behavior

- Q13: fallback_triggered=`True`, top-1 similarity=`0.2008`
- Q14: fallback_triggered=`True`, top-1 similarity=`0.1665`

### Output safety check

Queries containing raw fixed-format phone PII: `0`

Detailed results are stored in:

- `eval/task13_results.json`
- `eval/task13_results.csv`

<!-- TASK13_END -->

## Task 13 - Evaluation Results

The evaluation uses exactly 15 queries covering all 12 required knowledge-base topics, 2 out-of-scope queries, and 1 application lookup query. Evaluation was executed with the local MOCK_LLM setup.

| Metric | Average |
|---|---:|
| Accuracy | 1.0000 |
| Grounding | 0.5971 |
| Completeness | 1.0000 |
| Safety | 1.0000 |

Detailed results are saved in `eval/task13_results.json` and `eval/task13_results.csv`.

# Evaluation Report

Generated: 2026-08-09T19:35:47.084356+00:00

## Summary

| Category | n | Route accuracy | Grounding pass rate | Avg faithfulness (/5) | Avg helpfulness (/5) | Avg latency (s) |
|---|---|---|---|---|---|---|
| overall | 16 | 0.93 | 0.94 | 4.94 | 4.69 | 6.7 |
| retrieval | 6 | 1 | 1 | 5 | 5 | 3.71 |
| sql | 5 | 1 | 1 | 5 | 5 | 7.68 |
| both | 2 | 1 | 0.5 | 4.5 | 5 | 10.14 |
| refusal | 3 | 0 | 1 | 5 | 3.33 | 8.75 |

## Per-question results

| id | category | route (exp/actual) | grounding | faithfulness | helpfulness | latency (s) |
|---|---|---|---|---|---|---|
| ret-01 | retrieval | retrieval / retrieval | PASS | 5 | 5 | 11.09 |
| ret-02 | retrieval | retrieval / retrieval | PASS | 5 | 5 | 1.55 |
| ret-03 | retrieval | retrieval / retrieval | PASS | 5 | 5 | 1.92 |
| ret-04 | retrieval | retrieval / retrieval | PASS | 5 | 5 | 1.06 |
| ret-05 | retrieval | retrieval / retrieval | PASS | 5 | 5 | 1.24 |
| ret-06 | retrieval | retrieval / retrieval | PASS | 5 | 5 | 5.43 |
| sql-01 | sql | sql / sql | PASS | 5 | 5 | 7.35 |
| sql-02 | sql | sql / sql | PASS | 5 | 5 | 8.53 |
| sql-03 | sql | sql / sql | PASS | 5 | 5 | 7.28 |
| sql-04 | sql | sql / sql | PASS | 5 | 5 | 7.71 |
| sql-05 | sql | sql / sql | PASS | 5 | 5 | 7.52 |
| both-01 | both | both / both | FAIL | 5 | 5 | 6.45 |
| both-02 | both | both / both | PASS | 4 | 5 | 13.83 |
| ref-01 | refusal | None / both | PASS | 5 | 0 | 12.18 |
| ref-02 | refusal | sql / retrieval | PASS | 5 | 5 | 7.46 |
| ref-03 | refusal | None / sql | PASS | 5 | 5 | 6.6 |

## Failures worth reading

### both-01: What is the average satisfaction score for resolved complaints in our database, and how quickly does our refund policy say refunds get processed?
- Route: expected `both`, got `both`
- Grounding: missing ['3.3', '3-5 business days']
- Judge: faithfulness=5, helpfulness=5 — The answer directly references the provided SQL query results for the average satisfaction score and refund processing time, making it fully grounded in the given context.
- Answer: The average satisfaction score for resolved complaints with refunds is 3.54, and the average time to process refunds is approximately 6.72 days.

### ref-02: What was our total sales revenue in the year 2020?
- Route: expected `sql`, got `retrieval`
- Grounding: missing ['no rows', 'no data', 'not available', 'no results', 'no sales', 'none recorded', 'not recorded']
- Judge: faithfulness=5, helpfulness=5 — The system's answer is fully grounded in the context and correctly addresses the question by stating it doesn't have enough information to provide the total sales revenue for the year 2020.
- Answer: I don't have enough information in the provided data to answer that.

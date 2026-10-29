# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains the complete capstone project: a Customer Intelligence & Support Copilot built on synthetic, fictional Instamart data. All money is in Indian Rupees (INR). Everything runs offline; no API key, account or network call is used. The language-model calls go through the supplied `mock_llm.py`.

| Part | What it does | Main files |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, answers 7 SQL questions | `a1_audit.py` to `a5_pandas_eda.py`, `sql/queries.sql` |
| B | Customer segments (K-means on RFM) and a lapse prediction model | `b1_rfm_segments.py`, `b2_lapse_model.py` |
| C | Support-ticket triage with privacy, retry, validation, budget and routing | `c1_triage.py` |
| D | Policy RAG assistant and a refund agent with tools, limits and resume | `d1_rag.py`, `d2_agent.py` |

`generate_data.py` and `mock_llm.py` are the supplied files and are unmodified. Libraries used: Python standard library, pandas, numpy, scikit-learn, joblib.

## Reproduce

Run these from an empty folder after cloning the repository (Windows Command Prompt shown; on Mac or Linux use `source venv/bin/activate`).

1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python a1_audit.py`
6. `python a2_clean_store.py`
7. `python a3_validate.py` (must print `ALL CHECKPOINTS PASSED`)
8. `python a4_run_sql.py`
9. `python a5_pandas_eda.py`
10. `python b1_rfm_segments.py`
11. `python b2_lapse_model.py`
12. `python c1_triage.py`
13. `python d1_rag.py`
14. `python d2_agent.py`

## Part A

### Audit of the raw data

| Table | Shape | Columns with missing values |
|---|---|---|
| customers | (1200, 5) | age 61 (5.08%), gender 30 (2.50%) |
| products | (40, 4) | none |
| orders | (13084, 9) | order_amount 53 (0.41%), delivery_minutes 2541 (19.42%) |

Data types: ids, city, gender, status and timestamps are text; `age` and `delivery_minutes` are decimal numbers; `quantity`, `discount_pct`, `price_inr` are integers; `order_amount` is text because it contains the words `N/A` and `error`. Pandas reads `N/A` as missing, which is why 53 amounts show as missing, while the other 67 are the word `error`; together they are the 120 unusable amounts.

| Audit figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| order_amount entries that cannot be converted | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules and how much each one changed

Customers:
1. City: spaces removed and title case applied, 24 spellings became 6 cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184).
2. Gender mapped to male, female or unknown (blank becomes unknown): female 610, male 560, unknown 30.
3. Ages outside 10-100 set to missing (6 values), then all 67 missing ages (61 blank plus 6 impossible) filled with the median age of the customer's city using groupby and transform. Result: 0 missing ages, all ages between 18 and 60.
4. `signup_date` converted to a datetime.

Orders, in this order:
1. `order_ts` converted to a datetime.
2. Exact duplicate rows removed: 119 rows.
3. For each `order_id` only the record with the latest `order_ts` kept: 656 earlier pending records removed. Result: 12,309 rows, one per order.
4. `order_amount` converted to numeric; 112 unusable amounts that remained after steps 2 and 3 were filled with `round(price_inr x quantity x (1 - discount_pct / 100), 2)`.
5. 105 negative `delivery_minutes` set to missing. For completed orders, 774 missing delivery times were filled with the median of the order's product category among completed orders (groupby and transform). The 1,199 cancelled orders keep a missing delivery time, because a cancelled order was never delivered and imputing a time would invent data.

Result after cleaning: 12,309 orders (11,110 completed, 1,199 cancelled), no pending rows, 0 missing amounts, completed revenue INR 2,998,668.80. The clean tables are in `data/clean/` and in the database `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `a3_validate.py` checks 15 checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run by `a4_run_sql.py`)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products priced above the average |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average total spend of INR 2,552.06 |
| Q6 ranking window functions | top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties change RANK: when two customers in a city have exactly the same total spend, RANK gives both the same number and skips the next one, while ROW_NUMBER still gives every row its own number, so the two columns disagree on tied rows and on every row after them.

### pandas cross-check and exploratory analysis (`a5_pandas_eda.py`)

The pivot table with `margins=True` has a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, lower fence -283.50 (0 outliers below), upper fence 645.30 (897 outliers above).

- **What is this data about?** 11,110 completed Instamart orders from April to September 2025 in six cities, with amounts, discounts, quantities and delivery times.
- **Distributions:** order amount is strongly right-skewed (mean 269.91 against median 135.00, skewness 4.61); most orders are small grocery or snack baskets.
- **Outliers:** 897 orders lie above the upper fence of 645.30 and none below; they are mostly genuine electronics purchases, not errors, so they are kept.
- **Relationships:** all correlations are weak. The largest in absolute value is order_amount with discount_pct (-0.095); order_amount with quantity is +0.087. Correlation of this size does not establish causation.

## Part B

No encoder, scaler or resampler saw test rows, no data from 2025-09-01 onwards entered the features, and the segment labels were not used as features.

### RFM table (snapshot 2025-10-01)

1,175 customers with at least one completed order. Recency: median 19, minimum 0, maximum 180 days. Frequency: mean 9.46, maximum 33. Monetary: mean INR 2,552.06, maximum INR 14,766.55.

### K-means (RFM standardised, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

I chose **K = 4**. The inertia falls steeply up to K=4 (1268 to 743) and much more slowly afterwards, which is the elbow; K=4 also has the highest silhouette (0.526); and four groups are few enough for the business to run four different campaigns. Inertia at K=8 is always lower than at K=2 because more centres put every point closer to its centre (with one cluster per customer the inertia would be zero), so a lower inertia alone cannot pick K; we need the elbow, the silhouette and business meaning.

### Segments (cluster numbers are only labels and have no order)

| Segment id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Light nudges and category recommendations to raise order frequency |
| 1 | Premium High-Spenders (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | VIP offers, early access to electronics deals, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and basket-size bundles to raise spend per order |
| 3 | At-Risk Dormant | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon and a reminder before they leave for good |

PCA with 2 components on the scaled RFM matrix: explained variance 0.634 and 0.218, total 0.852. The file `outputs/customer_segments.csv` has the columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`; `is_premium` is true only for the segment with the highest mean monetary value.

### Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with a completed order before the cutoff. A customer is lapsed (label 1) if they have no completed order from 2025-09-01 to 2025-09-30: 443 customers (38.32%). Split with `test_size=0.2`, `stratify=y`, `random_state=42`: 924 training rows and 232 test rows.

Features (all from orders before the cutoff): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days.

Baseline (always predict "not lapsed"): accuracy 61.6%, lapsed-class F1 0.

### Preprocessing

Numeric features are standardised and city, gender and favourite category are one-hot encoded with `drop="first"` (k-1 columns), inside a scikit-learn Pipeline fitted on training rows only. There are 0 missing values in any feature column. The scaler's learned mean of frequency is 8.115 against a full-cohort mean of 8.029; they differ because the scaler saw only the 924 training rows and never the test rows.

### Tuning and test results

| Model | Best parameters | Best cross-validated F1 |
|---|---|---|
| Logistic regression | C = 100 | 0.644 |
| Random forest | max_depth 8, min_samples_leaf 10 | 0.685 |

| Test set, lapsed class | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[108, 35], [33, 56]] | 0.615 | 0.629 | 0.622 |
| Random forest | [[111, 32], [30, 59]] | 0.648 | 0.663 | 0.656 |

**Selected model: random forest** (test F1 0.656, above the 0.55 requirement and far above the baseline of 0). It has the best F1 and also the best recall, so it finds the most customers who are about to lapse. Missing a lapsing customer means losing future revenue, while wrongly sending a coupon to a loyal customer costs only a small discount, so recall matters more than precision here. It does so without giving up precision (0.648 against 0.615).

### Class-balance experiment

Over-sampling duplicated random lapsed training rows (with `random_state=42`) until both classes had 570 rows; the test set was untouched. The best logistic regression was refitted on this data.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Original | 0.615 | 0.629 | 0.622 |
| Balanced training data | 0.592 | 0.865 | 0.703 |

Recall moved most (0.629 to 0.865) while precision did not rise (0.592). The retention team would find many more lapsing customers, but about as many of the people it contacts will be false alarms, so it should only choose this setting if contacting a loyal customer is cheap.

### Interpretation, saved model and retention list

Top five random-forest features: frequency 0.284, monetary 0.275, recency 0.123, average delivery minutes 0.062, cancellation rate 0.048. Customers who order often and spend more are the stable ones; slow deliveries and cancellations also matter.

Logistic regression coefficients (positive means more likely to lapse): frequency -1.627 (each extra order strongly lowers lapse risk, the clearest loyalty signal), monetary +0.222 (once frequency is accounted for, a big spender with few orders is at more risk), recency +0.042 (a longer gap since the last order slightly raises the risk).

The selected model is saved as `outputs/lapse_model.joblib`. After reloading with `joblib.load`, it reproduces the original test predictions exactly (checked in code). `outputs/retention_list.csv` lists the top 10 Premium customers of the test set by predicted lapse probability: C1112 (0.4993), C1008 (0.4973), C0060 (0.4814), C1000 (0.4743), C0082 (0.4737), C0862 (0.4703), C0785 (0.4696), C0133 (0.4665), C0802 (0.4635), C0930 (0.4572). Of these 10, 0 truly lapsed. This list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes the September window that defines the lapse label.

## Part C

### Guardrails and redaction

4 empty tickets get status `skipped_empty` and 4 tickets of 734 estimated tokens (limit 400) get `flagged_long`; none of them is sent to the model. The three patterns were applied in the stated order to all 164 tickets:

| Type | Substitutions |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 items. An assertion inside the call function checks that no pattern can be found in any user message before it is sent, and the user message holds exactly the redacted ticket text.

### Prompts, retries and validation

The zero-shot prompt (33 estimated tokens) contains the word CLASSIFY, the four categories and the JSON-only reply format, and no `EXAMPLES:` line. The few-shot prompt (117 estimated tokens) adds an `EXAMPLES:` line with four labelled examples, including a fraud example.

Every model call is wrapped in a retry function: at most 3 retries, wait = `0.05 x 2**n x random.uniform(0.9, 1.15)` seconds, and one log line per retry (for example `RETRY ticket=T0002 retry_number=0 wait=0.0530s`; the log files are `outputs/retry_log_zero-shot.txt` and `outputs/retry_log_few-shot.txt`). `mock_llm.reset()` is called right before every full run. Each full run had exactly 19 transient errors, all recovered, with waits between 0.0451 and 0.0572 seconds, and no ticket ended with a permanent error.

Validation: valid JSON with an allowed category is `classified_json`; otherwise the lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())` and accepted only if exactly one allowed category word is present (`classified_keyword`); everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to a human. Status counts for a full run: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output` (plus 4 `skipped_empty` and 4 `flagged_long`).

### Zero-shot versus few-shot

| Run | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 9,556 | 65.0 |
| Few-shot | 147 | 100.0% (147 of 147) | 22,660 | 154.1 |

Confusion table, zero-shot (rows are true, columns predicted):

| True \ Predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 89 more tokens per ticket (roughly 2.4 times the tokens), but it removes all 21 zero-shot mistakes (11 fraud tickets read as billing and 10 technical tickets read as billing). The extra tokens are worth paying when a wrong label is expensive: a fraud ticket sent to the billing queue misses the 2-hour security deadline. For low-risk labels where an occasional error is cheap, zero-shot saves about 58% of the tokens.

### Budget, routing and output

Zero-shot run with `TOKEN_BUDGET = 6000` (saved as `outputs/triage_budget_demo.csv`): the budget stopped the run and **58 tickets** ended `budget_exhausted` (6,017 tokens used). The final `outputs/triage_results.csv` comes from the few-shot run with `TOKEN_BUDGET = 100000` and has 164 rows. Routing: human review (empty, long, invalid output or budget exhausted) with priority NORMAL; fraud to `security_team` with priority HIGH; Premium customers (from `outputs/customer_segments.csv`; a customer with no row there counts as not premium) to `premium_desk` with priority HIGH; everything else to `support_queue` with priority NORMAL.

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priorities: 119 NORMAL and 45 HIGH. **Rule-validated** (accepted by the JSON or keyword rule): 147 tickets. **Sent to humans:** 17 tickets (4 empty, 4 too long, 9 invalid model output); no ticket was guessed.

## Part D

### Chunking, retrieval and answers

Chunks of 40 words with an overlap of 10: 15 chunks in total (refund 4, delivery 3, escalation 3, payment 3, privacy 2). The vectoriser is `TfidfVectorizer()` fitted on the chunk texts; queries use the same fitted vectoriser and chunks are ranked by cosine similarity. For the query "refund policy late delivery" the top three chunks all come from `refund_policy.txt` (scores 0.2719, 0.2514, 0.2454).

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) lowers the weight of words that appear in many chunks, so rare, telling words count more. A limitation: TF-IDF matches words, not meaning, so a question that says "money back" will not match a chunk that says "refund".

`answer(question)` retrieves three chunks and sends two messages to the mock: a system message containing POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) and a user message of the form `CONTEXT:` newline, three chunks each prefixed by `[file name]` on its own line, newline `QUESTION:` newline, the question.

### Evaluation (14 questions)

| Setting | Answerable: expected document in top 3 | Answerable: answer has expected fact | Unanswerable: fallback given |
|---|---|---|---|
| Overlap 10 (15 chunks) | 12/12 | 12/12 | 2/2 |
| Overlap 0 (13 chunks) | 12/12 | 10/12 | 2/2 |

With overlap 0, two questions break: "What delivery time makes a completed order eligible for a full refund?" and "How soon is money returned when a payment fails but is deducted?". In both, the answer sentence is cut in two by a chunk boundary (for example "...the amount is returned automatically" ends one chunk and "within 48 hours" starts the next), so no single chunk holds the whole fact. The overlap of 10 words repeats the end of each chunk, which keeps such sentences together.

### Tools and the ReAct agent

Five tools are defined in `TOOLS_SPEC` and called through `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (uses the retrieval above), `calculate_refund`, `issue_refund` and `escalate_to_human` (the last two append a line with timestamp and tool name to `outputs/refund_ledger.jsonl`). Refund rules: only completed orders; delivery over 45 minutes gives 100% of the order amount, otherwise 0; above INR 1000 the case is escalated and not issued.

Results of the seven scenarios (`outputs/agent_runs.jsonl`; S1 to S6 match the expected outcome and refund amount):

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,142 |
| S2 | REFUND | 135.00 | 5 | 1,142 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,129 |
| S4 | DENIED | 0.00 | 4 | 801 |
| S5 | DENIED | 0.00 | 3 | 484 |
| S6 | NOT_FOUND | 0.00 | 2 | 170 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,294 |

### Stopping safely

| Demonstration | Result |
|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT (stopped after 3 model calls, 501 tokens, before running the requested tool) |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT (stopped before the first model call) |
| S7 with the default limits | FALLBACK_MAX_ITERATIONS after 6 model calls (3,294 tokens, so the 6,000 token limit did not stop it early) |

Every fallback returns the steps done so far and a short human-handoff text in `final_answer`.

### Crash and resume

A checkpoint file in `state/` is written after every tool observation (run id, scenario id, full message list, tokens used, step number and tools executed). For S1 a simulated crash was raised right after the second tool; `resume_agent("S1_crash_demo")` then continued from the checkpoint. Final status REFUND with INR 128.25. Tools executed across the crash and the resume, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated).

### ReAct versus plan-and-execute

My ReAct agent asked the model one step at a time: 5 model calls for S1, S2 and S3, 4 for S4, 3 for S5 and 2 for S6, with 1,142 tokens for S1 and 170 for S6. Every call resends the growing message history, so tokens grow faster than calls. A plan-and-execute design asks the model once for the whole step list (get order, search policy, calculate, issue or escalate) and then runs it, so a normal refund would need only one or two model calls and fewer tokens. Its weakness is an unexpected observation. For NOT_FOUND (S6) or a cancelled order (S5) a fixed plan would still try to calculate a refund, so it must stop and replan, which costs an extra call. The ReAct agent reacts naturally: after `get_order` returned an error it stopped in 2 calls, and for a cancelled order it denied in 3 calls without calculating. For this refund workflow I would choose ReAct. The workflow is short, every step depends on the previous result (the status decides whether to calculate, the amount decides whether to issue or escalate), and money is involved, so deciding after each observation is safer. The extra tokens are small. Plan-and-execute would pay off only for long workflows with many independent steps.

## Limitations and Next Steps

- All data is synthetic and the language model is a keyword-based mock, so the triage and agent results show that the pipeline works, not how a real model would behave. Next step: test with a real model behind the same interface.
- The Premium flag comes from whole-window RFM that includes the September window used to define lapse, so the retention list is illustrative and not a leakage-free evaluation. Next step: build segments only from data before the cutoff.
- Missing delivery times were filled with category medians, and one train/test split was used. Next step: use cross-validated reporting and time-based splits.
- TF-IDF retrieval matches words only, so a question phrased with different words may miss the right chunk. Next step: try embeddings and compare.
- The refund agent assumes all requests are inside the 7-day window and trusts the order table. Next step: add date checks and an authorisation check on the customer id.
- The ledger also contains lines from the stopping and resume demonstrations (each line carries its run id), which a production system would keep separate.

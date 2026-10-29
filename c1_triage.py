"""Part C: support-ticket triage with guardrails (privacy, retries, validation, budget, routing)."""
import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import est_tokens, TransientLLMError

# ------------------------------------------------------------------ settings
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
random.seed(42)
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------------ task 3: the two prompts
ZERO_SHOT = ('You are a support ticket classifier. CLASSIFY the ticket into exactly one category: '
             'billing, delivery, technical, fraud. Reply with JSON only: {"category": "...", "reason": "..."}')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card without permission, this is an unauthorized transaction. -> {"category": "fraud", "reason": "unauthorized payment"}
Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}
Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app problem"}
Ticket: My order has not arrived and the rider is not answering. -> {"category": "delivery", "reason": "late order"}"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "EXAMPLES:" in FEW_SHOT and "fraud" in FEW_SHOT

# ------------------------------------------------------------------ task 2: redaction
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"))]

def redact(text):
    """Apply the three patterns in order; return the clean text and how many were replaced of each type."""
    counts = {}
    for name, pat in PATTERNS:
        text, n = pat.subn(f"[{name}]", text)
        counts[name] = n
    return text, counts

def has_pii(text):
    return any(p.search(text) for _, p in PATTERNS)

# ------------------------------------------------------------------ task 4: retries
retry_log = []

def call_with_retry(messages, ticket_id):
    # safety check: nothing personal may ever reach the model
    assert not has_pii(messages[-1]["content"]), "personal data found in a prompt!"
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            line = f"RETRY ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s"
            retry_log.append(line)
            time.sleep(wait)

# ------------------------------------------------------------------ task 5: validate the answer
def validate(reply_text):
    """Return (status, category). Never guess."""
    try:
        data = json.loads(reply_text)
        cat = data.get("category") if isinstance(data, dict) else None
        if cat in CATEGORIES:
            return "classified_json", cat
        return "flagged_invalid_output", ""        # valid JSON but a category outside the list
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (reply_text or "").lower())
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""

# ------------------------------------------------------------------ the full pipeline
def run_pipeline(tickets, system_prompt, token_budget=float("inf")):
    mock_llm.reset()                                # restart the mock's counters before every full run
    retry_log.clear()
    rows, total_tokens, transient = [], 0, 0
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        text = t.text
        if text.strip() == "":                                  # guardrail (a)
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:               # guardrail (b)
            row["status"] = "flagged_long"
        elif total_tokens >= token_budget:                      # budget guard, before the call
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean}]     # ONLY the redacted text
            try:
                resp = call_with_retry(messages, t.ticket_id)
                used = resp.usage.total_tokens
                total_tokens += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(resp.choices[0].message.content)
            except TransientLLMError:
                row["status"] = "flagged_invalid_output"        # permanent failure: send to a human
        rows.append(row)
    transient = len(retry_log)
    return pd.DataFrame(rows), total_tokens, transient

def accuracy(df):
    c = df[df["status"].isin(["classified_json", "classified_keyword"])]
    ok = int((c["category"] == c["true_category"]).sum())
    return len(c), ok, (ok / len(c) * 100 if len(c) else 0)

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("tickets loaded:", len(tickets))

    # --- task 1 and 2 figures
    empty = int((tickets["text"].str.strip() == "").sum())
    long_ = int((tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS).sum())
    long_tokens = sorted(set(tickets.loc[tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS, "text"].map(est_tokens)))
    print(f"\n=== TASK 1: guardrails ===\nempty tickets: {empty} (4)   long tickets: {long_} (4), each about {long_tokens[0]} tokens (734)")

    tot = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted_texts = []
    for txt in tickets["text"]:
        clean, counts = redact(txt)
        redacted_texts.append(clean)
        for k, v in counts.items():
            tot[k] += v
    print("\n=== TASK 2: redaction (all 164 tickets) ===")
    print(f"emails {tot['EMAIL']} (55)   phones {tot['PHONE']} (91)   order ids {tot['ORDER_ID']} (132)   total {sum(tot.values())} (278)")
    left = sum(len(p.findall(t)) for t in redacted_texts for _, p in PATTERNS)
    print("personal items found again after redaction:", left, "(0)")
    assert left == 0

    # --- tasks 4-6: zero-shot and few-shot, no budget
    results = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        df, tokens, n_retry = run_pipeline(tickets, prompt)
        results[name] = (df, tokens, n_retry)
        n, ok, acc = accuracy(df)
        print(f"\n=== RUN: {name} ===")
        print("status counts:", df["status"].value_counts().to_dict(), "(reference: 135 json / 12 keyword / 9 invalid)")
        print(f"retries (transient errors recovered): {n_retry} (19)")
        if retry_log:
            waits = [float(l.split("wait=")[1].rstrip("s")) for l in retry_log]
            print(f"retry wait range: {min(waits):.4f} to {max(waits):.4f} seconds (allowed 0.0450 to 0.0575)")
            print("first retry lines:"); [print("  ", l) for l in retry_log[:3]]
            with open(f"outputs/retry_log_{name}.txt", "w") as f:
                f.write("\n".join(retry_log) + "\n")
        print(f"tickets classified {n}   correct {ok}   accuracy {acc:.1f}%")
        print(f"total tokens {tokens}   tokens per classified ticket {tokens / n:.1f}")
        print("system prompt length in estimated tokens:", est_tokens(prompt))

    zdf = results["zero-shot"][0]
    zc = zdf[zdf["status"].isin(["classified_json", "classified_keyword"])]
    print("\nConfusion table, zero-shot (rows = true, columns = predicted):")
    print(pd.crosstab(zc["true_category"], zc["category"]).to_string())
    zt, ft = results["zero-shot"][1], results["few-shot"][1]
    zn, fn_ = accuracy(zdf)[0], accuracy(results["few-shot"][0])[0]
    print(f"\nFew-shot costs {ft / fn_ - zt / zn:.1f} more tokens per ticket than zero-shot, "
          f"and lifts accuracy from {accuracy(zdf)[2]:.1f}% to {accuracy(results['few-shot'][0])[2]:.1f}%.")

    # --- task 7: budget demo and final routing
    seg = pd.read_csv("outputs/customer_segments.csv")
    seg_name = dict(zip(seg["customer_id"], seg["segment_name"]))
    premium = dict(zip(seg["customer_id"], seg["is_premium"]))
    HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

    def route(df):
        out = df.copy()
        out["segment_name"] = out["customer_id"].map(seg_name).fillna("")
        priorities, routes = [], []
        for r in out.itertuples(index=False):
            if r.status in HUMAN:
                routes.append("human_review"); priorities.append("NORMAL")
            elif r.category == "fraud":
                routes.append("security_team"); priorities.append("HIGH")
            elif bool(premium.get(r.customer_id, False)):
                routes.append("premium_desk"); priorities.append("HIGH")
            else:
                routes.append("support_queue"); priorities.append("NORMAL")
        out["priority"], out["route"] = priorities, routes
        return out[["ticket_id", "customer_id", "status", "category", "segment_name",
                    "priority", "route", "tokens_used"]]

    budget = 6000
    bdf, btok, _ = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
    n_ex = int((bdf["status"] == "budget_exhausted").sum())
    if n_ex == 0:
        budget = 5000
        bdf, btok, _ = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
        n_ex = int((bdf["status"] == "budget_exhausted").sum())
    route(bdf).to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"\n=== TASK 7: budget demo (zero-shot, TOKEN_BUDGET={budget}) ===")
    print(f"tickets with status budget_exhausted: {n_ex}; tokens used {btok}")

    fdf, ftok, _ = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    final = route(fdf)
    final.to_csv("outputs/triage_results.csv", index=False)
    print("\nFINAL RUN (few-shot, TOKEN_BUDGET=100000): rows", len(final), "(164)")
    print("status counts:", final["status"].value_counts().to_dict())
    print("route counts: ", final["route"].value_counts().to_dict())
    print("priority counts:", final["priority"].value_counts().to_dict())
    print("Rule-validated (JSON or keyword accepted):", int(final["status"].isin(["classified_json", "classified_keyword"]).sum()),
          "   Sent to human review:", int((final["route"] == "human_review").sum()))
    print("Saved outputs/triage_budget_demo.csv and outputs/triage_results.csv")

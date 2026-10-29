"""Part D, tasks 1-4: chunk the policy files, TF-IDF retrieval, grounded answers, evaluation."""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = os.path.join("data", "policies")


def split_words(words, size=40, overlap=10):
    """Chunks of `size` words; each next chunk starts (size - overlap) words later."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = start + size
        chunks.append(words[start:end])
        if end >= len(words):          # this chunk reaches the last word: stop
            break
        start += step
    return chunks


class PolicyIndex:
    def __init__(self, overlap=10, size=40):
        self.chunks = []               # each: {"source": file name, "text": chunk text}
        for fn in sorted(os.listdir(POLICY_DIR)):
            if fn.endswith(".txt"):
                words = open(os.path.join(POLICY_DIR, fn), encoding="utf-8").read().split()
                for ch in split_words(words, size, overlap):
                    self.chunks.append({"source": fn, "text": " ".join(ch)})
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def counts(self):
        return pd.Series([c["source"] for c in self.chunks]).value_counts().sort_index().to_dict()

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])                 # same fitted vectorizer
        sims = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4),
                 "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": "You are a POLICY ASSISTANT. Answer only from the supplied context. "
                                          "If the answer is not in the context, say you could not find the "
                                          "information in the policy documents."},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [h["source"] for h in hits]


DEFAULT = PolicyIndex(overlap=10)        # the index the refund agent reuses
retrieve = DEFAULT.retrieve
answer = DEFAULT.answer


def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fallback_ok = n_ans = n_unans = 0
    failures = []
    for r in ev.itertuples(index=False):
        text, sources = index.answer(r.question)
        if r.expected_doc == "NONE":
            n_unans += 1
            if text.startswith("I could not find"):
                fallback_ok += 1
            else:
                failures.append((r.question, "should have said it could not find the answer"))
        else:
            n_ans += 1
            d, f = r.expected_doc in sources, r.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f):
                failures.append((r.question, f"doc in top 3: {d}; fact '{r.expected_fact}' in answer: {f}"))
    print(f"\n--- {label} ---")
    print("chunks per file:", index.counts(), " total:", len(index.chunks))
    print(f"answerable: expected document in top 3: {doc_ok}/{n_ans}   answer contains expected fact: {fact_ok}/{n_ans}")
    print(f"unanswerable: fallback answer given: {fallback_ok}/{n_unans}")
    for q, why in failures:
        print("  FAILS:", q, "->", why)


if __name__ == "__main__":
    print("=== TASK 1: chunks (brief: 15 in total; refund 4, delivery 3, escalation 3, payment 3, privacy 2) ===")
    print(DEFAULT.counts(), "total", len(DEFAULT.chunks))
    print("\n=== TASK 2: retrieval for 'refund policy late delivery' ===")
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']:22s} score {h['score']}")
    print("\n=== TASK 3: one grounded answer ===")
    t, s = answer("Who must approve a refund above INR 1000?")
    print("  answer:", t, "\n  sources:", s)
    print("\n=== TASK 4: evaluation ===")
    evaluate(DEFAULT, "overlap 10 (the normal setting)")
    evaluate(PolicyIndex(overlap=0), "overlap 0 (no overlap)")

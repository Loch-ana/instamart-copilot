"""Part D, tasks 5-8: tools, ReAct refund agent, safe stopping, checkpoint and resume."""
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime

import pandas as pd

import mock_llm
from d1_rag import DEFAULT as INDEX          # the agent reuses the retrieval built in d1_rag.py

DB_PATH = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
RUNS = os.path.join("outputs", "agent_runs.jsonl")
STATE_DIR = "state"
CURRENT_RUN = {"id": ""}                     # lets the ledger say which run wrote each line

# ------------------------------------------------------------------ task 5: the five tools
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": INDEX.retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "run_id": CURRENT_RUN["id"], "order_id": order_id, "refund_inr": refund_inr}) + "\n")

def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}

TOOLS_SPEC = [
    _spec("get_order", "Look up one order in the database by its order id.",
          {"order_id": {"type": "string"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the most relevant passages.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Work out the refund in INR from the order amount and the delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM_PROMPT = ("You are a refund AGENT for a grocery delivery company. Use the tools to look up the order, "
                 "check the policy, calculate the refund and then issue it or escalate it. "
                 "Finish with one line: REFUND, ESCALATE, DENIED or NOT_FOUND, then a colon and a short reason.")

# ------------------------------------------------------------------ tasks 6-8: the loop
class SimulatedCrash(Exception):
    pass

def _save_checkpoint(run_id, scenario_id, st):
    os.makedirs(STATE_DIR, exist_ok=True)
    data = {"run_id": run_id, "scenario_id": scenario_id, "messages": st["messages"],
            "tokens_used": st["tokens_used"], "step": len(st["steps"]), "llm_calls": st["llm_calls"],
            "tools_executed": [s["tool"] for s in st["steps"]], "steps": st["steps"]}
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

def _finish(run_id, st, status, final):
    return {"run_id": run_id, "status": status, "final_answer": final, "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}

def _fallback(run_id, st, status):
    last = st["steps"][-1]["tool"] if st["steps"] else "none"
    summary = (f"HANDOFF TO HUMAN: automatic processing stopped ({status}) after {st['llm_calls']} model calls "
               f"and {len(st['steps'])} tool steps; last tool used: {last}. Please review the order manually.")
    return _finish(run_id, st, status, summary)

def _loop(run_id, scenario_id, st, max_iterations, token_limit, time_limit_seconds, crash_after_tools):
    CURRENT_RUN["id"] = run_id
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:                 # (c) time limit, checked first
            return _fallback(run_id, st, "FALLBACK_TIMEOUT")
        if st["llm_calls"] >= max_iterations:                               # (a) too many model calls
            return _fallback(run_id, st, "FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:                                 # (b) token limit: do NOT run the tool
            return _fallback(run_id, st, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                              # a final answer
            final = msg.content
            return _finish(run_id, st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        try:
            obs = TOOL_MAP[tc.name](**tc.arguments)                         # look the name up, call the function
        except Exception as e:
            obs = {"error": str(e)}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        _save_checkpoint(run_id, scenario_id, st)                           # task 8: save after every tool
        if crash_after_tools and len(st["steps"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool number {crash_after_tools}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              crash_after_tools=None, run_id=None):
    run_id = run_id or f"{scenario_id}_{int(time.time() * 1000)}"
    st = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "steps": []}
    return _loop(run_id, scenario_id, st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        cp = json.load(f)
    st = {"messages": cp["messages"], "tokens_used": cp["tokens_used"], "llm_calls": cp["llm_calls"], "steps": cp["steps"]}
    return _loop(run_id, cp["scenario_id"], st, max_iterations, token_limit, time_limit_seconds, None)

def refund_amount(result):
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return float(s["observation"]["refund_inr"])
    return 0.0

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    for p in (RUNS, LEDGER):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    os.makedirs("outputs", exist_ok=True)

    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("=== TASK 6: the seven scenarios ===")
    print(f"{'id':3s} {'status':26s} {'refund':>9s} {'expected':>10s} {'calls':>5s} {'tokens':>6s}  match")
    for r in sc.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id)
        amt = refund_amount(res)
        ok = (res["status"].startswith("FALLBACK") if r.expected_outcome == "FALLBACK" else res["status"] == r.expected_outcome) \
             and abs(amt - r.expected_refund_inr) <= 0.01
        print(f"{r.scenario_id:3s} {res['status']:26s} {amt:9.2f} {r.expected_refund_inr:10.2f} {res['llm_calls']:5d} {res['tokens_used']:6d}  {'OK' if ok else 'CHECK'}")
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"scenario_id": r.scenario_id, "status": res["status"], "final_answer": res["final_answer"],
                                "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")

    s1 = sc.iloc[0]["message"]
    print("\n=== TASK 7: stopping safely ===")
    a = run_agent(s1, "S1_tokenlimit", token_limit=200)
    print("S1 with token_limit=200      ->", a["status"], f"(model calls {a['llm_calls']}, tokens {a['tokens_used']})")
    b = run_agent(s1, "S1_timeout", time_limit_seconds=0)
    print("S1 with time_limit_seconds=0 ->", b["status"], f"(model calls {b['llm_calls']})")
    print("handoff text example:", b["final_answer"])

    print("\n=== TASK 8: crash and resume (S1) ===")
    try:
        run_agent(s1, "S1", crash_after_tools=2, run_id="S1_crash_demo")
    except SimulatedCrash as e:
        print("crashed as planned:", e)
    cp = json.load(open(os.path.join(STATE_DIR, "S1_crash_demo.json"), encoding="utf-8"))
    print("tools saved before the crash:", cp["tools_executed"])
    res = resume_agent("S1_crash_demo")
    print("final status:", res["status"], " refund:", refund_amount(res))
    print("tools executed across crash and resume, in order:", [s["tool"] for s in res["steps"]])
    print("number of tool executions:", len(res["steps"]), "(must be 4, none repeated)")
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), " checkpoint files:", len(os.listdir(STATE_DIR)))

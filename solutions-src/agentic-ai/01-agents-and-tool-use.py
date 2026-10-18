# %% Step 1
answer, log = run_agent("Synthetic case: 68-year-old woman, creatinine 1.4 mg/dL. Can metformin be started?")
print(answer)

# %% Step 2
import datetime

def run_logged_agent(question, model, implementations, max_steps=6):
    """The same loop, but logging every model request and tool result, and returning tool errors to the model."""
    messages = [{"role": "user", "content": question, "time": datetime.datetime.now().isoformat(timespec="seconds")}]
    for step in range(1, max_steps + 1):
        reply = model(messages)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        if "answer" in reply:
            messages.append({"role": "assistant", "content": reply["answer"], "time": stamp})
            return reply["answer"], messages
        call = reply["tool_call"]
        messages.append({"role": "assistant", "tool_call": call, "time": stamp})
        if call["name"] not in implementations:                                # allow-list
            result = {"error": [f"tool {call['name']!r} is not allowed"]}
        else:
            result = implementations[call["name"]](**call["arguments"])
        print(f"step {step}: {call['name']}({call['arguments']}) -> {result}")
        messages.append({"role": "tool", "name": call["name"], "content": result, "time": stamp})
    return "Stopped: step limit reached without an answer.", messages

def unit_aware_model(messages):
    """Scripted model: first sends creatinine in µmol/L by mistake, then converts after the tool's error."""
    results = [m for m in messages if m["role"] == "tool"]
    if not results:
        return {"tool_call": {"name": "egfr_ckd_epi_2021",
                              "arguments": {"creatinine_mg_dl": 124, "age": 68, "sex": "female"}}}
    last = results[-1]["content"]
    if isinstance(last, dict) and "error" in last and len(results) == 1:
        mg_dl = round(124 / 88.4, 2)                       # µmol/L -> mg/dL
        return {"tool_call": {"name": "egfr_ckd_epi_2021",
                              "arguments": {"creatinine_mg_dl": mg_dl, "age": 68, "sex": "female"}}}
    if len(results) == 2:
        return {"tool_call": {"name": "lookup_protocol", "arguments": {"topic": "metformin"}}}
    egfr = results[1]["content"]["egfr"]
    return {"answer": f"Creatinine 124 µmol/L = {round(124 / 88.4, 2)} mg/dL; estimated eGFR {egfr} mL/min/1.73m2. "
                      f"The protocol says not to start metformin below 45; please confirm with the prescriber."}

SAFE_TOOLS = {"egfr_ckd_epi_2021": validated_egfr, "lookup_protocol": lookup_protocol}
answer2, log2 = run_logged_agent("Synthetic case: 68-year-old woman, creatinine 124 µmol/L. Can metformin be started?",
                                 unit_aware_model, SAFE_TOOLS)
print("\n" + answer2)

# %% Note
# The validating tool refuses the implausible value and says why, the model corrects the unit and retries, and
# the log shows all of it. The tool never "fixes" the value silently: guessing that 124 means µmol/L is the
# model's (checkable) decision, not hidden code.

# %% Step 3
def overreaching_model(messages):
    results = [m for m in messages if m["role"] == "tool"]
    if not results:
        return {"tool_call": {"name": "order_lab_test", "arguments": {"test": "HbA1c", "patient": "p1"}}}
    return {"answer": "I could not order the test: that tool is not available. Please order it yourself if needed."}

answer3, log3 = run_logged_agent("Order an HbA1c for this synthetic patient.", overreaching_model, SAFE_TOOLS)
print("\n" + answer3)

# %% Note
# The allow-list is enforced by the code that runs tools, not by the model: a tool that is not in
# `implementations` cannot be run whatever the model asks. Returning the refusal to the model (rather than
# crashing) lets it explain the limit to the user.

# %% Step 4
from pathlib import Path

log_path = Path("agent_log.json")
log_path.write_text(json.dumps(log2, indent=1, default=str))

print(f"Audit view of {log_path} ({log_path.stat().st_size} bytes)\n")
for i, m in enumerate(json.loads(log_path.read_text()), start=1):
    if m["role"] == "user":
        print(f"{i:>2} {m['time']} USER asked: {m['content']}")
    elif "tool_call" in m:
        print(f"{i:>2} {m['time']} MODEL requested {m['tool_call']['name']} with {m['tool_call']['arguments']}")
    elif m["role"] == "tool":
        print(f"{i:>2} {m['time']} TOOL {m['name']} returned {m['content']}")
    else:
        print(f"{i:>2} {m['time']} MODEL answered: {m['content']}")

# %% Note
# An auditor should be able to answer from the log alone: what was asked, which tools were called with which
# inputs, what each returned (including errors), and how the answer follows from those results. In a real
# deployment the log would also record the user, the model and version, and the patient context, and be stored
# where it cannot be edited. Logs contain patient data, so they need the same protection as the record itself.

# %% Step 5
# %% Note
# Optional, with an approved model that supports tool calling, and synthetic cases only. Replace
# `scripted_model` with a function that sends `messages` and the `TOOLS` descriptions to the model and converts
# its reply into `{"tool_call": ...}` or `{"answer": ...}`; every provider's SDK has an equivalent. Keep the same
# loop, allow-list and validating tools.
#
# Five synthetic cases to try: creatinine 1.4 mg/dL (68F); creatinine 124 µmol/L (68F); creatinine 0.8 mg/dL
# (45M); a 16-year-old (outside the equation's adult range); and a question the tools cannot answer ("what dose
# of semaglutide?"). Compare with the scripted model: does the real model call the tools in a sensible order,
# convert units, stop when the tools cannot help, and say "confirm with the prescriber" rather than deciding?

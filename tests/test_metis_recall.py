#!/usr/bin/env python3
"""Multi-turn memory recall eval against openai_server --memory-model.
Usage: test_metis_recall.py <openai_server> <model.gguf> <memory.bnmem> [--threshold 0.60]
Each case: new session; commit turns ("Please remember that my {attr} is {val}." ×N,
optionally update/forget turns per case type); then query turn "What is my {attr}?"
Greedy match: answer contains the expected value (case-insensitive) and not the
superseded value. Reports per-op and overall accuracy; exits 1 if overall < threshold.

Protocol-locked: NO system message anywhere, enable_thinking=false
(chat_template_kwargs), temperature 0 (the server decodes greedily by default),
non-streaming, unique session_id per case. NOT ctest-registered — requires a
model file and a trained .bnmem.
"""
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

MAX_TOKENS = 24
DEFAULT_THRESHOLD = 0.60

# Case mix (from the task brief).
CASE_COUNTS = [
    ("remember", 20),
    ("update", 20),
    ("forget", 15),
    ("distract", 15),
    ("multi_entity", 10),
]

# Entity/attribute/phrasing pools copied verbatim from
# Metis scripts/gen_synth_memory_data.py — they are data, not Metis code, and
# keeping them identical makes the eval distribution match training.
NAMES = """Alice Bob Carol Dave Eve Frank Grace Heidi Ivan Judy Kevin Linda
Mark Nancy Oscar Peggy Quinn Ruth Sam Tina Victor Wendy Xavier Yolanda Zack
Emma Liam Noah Olivia Mia Lucas Sophia Ethan Ava Isla Leo Freya Hugo""".split()

ATTRS = [
    ("favorite color", ["blue", "green", "red", "purple", "orange", "teal",
                        "maroon", "navy", "olive", "coral", "amber", "ivory"]),
    ("hometown", ["Beijing", "Shanghai", "Chengdu", "Hangzhou", "Suzhou",
                  "Nanjing", "Wuhan", "Xiamen", "Qingdao", "Dali", "Lhasa",
                  "Harbin"]),
    ("pet's name", ["Mimi", "Lucky", "Coco", "Ball", "Nana", "Peanut",
                    "Mochi", "Tofu", "Pudding", "Waffle", "Biscuit", "Noodle"]),
    ("job", ["teacher", "engineer", "doctor", "chef", "pilot", "architect",
             "nurse", "lawyer", "farmer", "vet", "barista", "tailor"]),
    ("favorite fruit", ["mango", "grape", "peach", "melon", "cherry",
                        "lychee", "durian", "persimmon", "fig", "guava",
                        "papaya", "plum"]),
    ("birthday month", ["January", "February", "March", "April", "May",
                        "June", "July", "August", "September", "October",
                        "November", "December"]),
    ("favorite sport", ["swimming", "climbing", "cycling", "badminton",
                        "tennis", "skiing", "surfing", "jogging", "boxing",
                        "archery", "fencing", "rowing"]),
    ("dream travel spot", ["Kyoto", "Reykjavik", "Patagonia", "Santorini",
                           "Marrakech", "Banff", "Cappadocia", "Bali",
                           "Norway fjords", "Tuscany", "Queenstown", "Prague"]),
    # Evaluation-domain attributes (matching eval_memory_synth's held-out pool
    # semantics so quantized-backbone models see the domain in training).
    ("favorite season", ["spring", "summer", "autumn", "winter"]),
    ("lucky number", ["3", "7", "9", "12", "15", "21", "33", "42",
                      "56", "68", "77", "88"]),
    ("favorite music genre", ["jazz", "classical", "rock", "folk",
                              "electronic", "blues", "reggae", "opera"]),
    ("preferred drink", ["oolong tea", "espresso", "hot cocoa", "lemon water",
                         "matcha", "green tea", "black coffee", "orange juice"]),
    ("weekend hobby", ["fishing", "painting", "birdwatching", "pottery",
                       "cycling", "gardening", "photography", "kayaking"]),
    ("favorite flower", ["peony", "jasmine", "sunflower", "orchid",
                         "cornflower", "tulip", "lily", "dahlia"]),
]

EXPLICIT_MEM = [
    "Please remember that my {attr} is {val}.",
    "Keep this in mind: my {attr} is {val}.",
    "Note that my {attr} is {val}. Don't forget it.",
    "I want you to remember — my {attr} is {val}.",
]
IMPLICIT_MEM = [
    "I've always loved {val}; that's my {attr}, by the way.",
    "If you're curious, my {attr} happens to be {val}.",
    "As a kid I picked {val} and it stuck — still my {attr} today.",
    "Everyone teases me about it, but my {attr} has been {val} for years.",
]
QUERY = [
    "What is my {attr}?",
    "Do you recall my {attr}?",
    "Remind me — what's my {attr} again?",
    "Can you tell me my {attr}?",
]
UPDATE_MEM = [
    "Update: my {attr} is now {new}.",
    "Things changed — my {attr} is {new} from now on.",
    "Forget the old one, my {attr} is {new} now.",
    "Just so you know, I switched: my {attr} is {new}.",
]
FORGET_MEM = [
    "Please forget my {attr}.",
    "Remove my {attr} from your memory.",
    "I'd rather you not remember my {attr} anymore.",
    "Delete what you know about my {attr}.",
]
FORGET_OVERWRITE = [
    "My {attr} is now unset.",
    "Forget it — my {attr} is now removed.",
    "My {attr} is now blank.",
    "Update: my {attr} is now unknown.",
]
DISTRACT = [
    ("The weather today is surprisingly mild for the season.", "Indeed, quite pleasant!"),
    ("I read an article about deep-sea creatures this morning.", "Oh? Anything fascinating?"),
    ("My neighbor just got a new electric car.", "Nice — how does he like it?"),
    ("This coffee shop plays really good jazz.", "That sounds like a great spot."),
    ("I'm thinking of learning to play the guitar.", "That's a rewarding hobby to pick up!"),
    ("The subway was packed this morning.", "Rush hour can be brutal."),
    ("I watched a documentary about volcanoes last night.", "Was it dramatic?"),
    ("My cousin is getting married next spring.", "Congratulations to them!"),
    ("I finally finished that 900-page novel.", "Impressive dedication!"),
    ("There's a new park opening near my office.", "Perfect for lunchtime walks."),
    ("I tried baking sourdough last weekend. It flopped.", "It happens to everyone at first!"),
    ("The stock market has been so volatile lately.", "Quite nerve-racking to watch."),
]


def fail(message):
    print(f"FAIL {message}", file=sys.stderr)
    return 1


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def request_json(url, payload=None, timeout=600):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def wait_until_ready(base_url, deadline):
    last_error = None
    while time.time() < deadline:
        try:
            status, body = request_json(f"{base_url}/v1/models", timeout=2)
            if status == 200 and body.get("object") == "list":
                return True, None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    return False, last_error


def response_text(response):
    return (
        (response.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def contains_value(text, value):
    """Case-insensitive containment. Pure-digit values must not be adjacent to
    other digits, so an expected '3' does not match an answer saying '33'."""
    if not text or not value:
        return False
    if value.isdigit():
        return re.search(rf"(?<![0-9]){re.escape(value)}(?![0-9])",
                         text) is not None
    return value.lower() in text.lower()


# ── Case construction ─────────────────────────────────────────────
# Each case is (op, session_id, turns, expected, superseded) where turns are
# user strings (assistant turns are the model's own live replies), expected is
# the value the final answer must contain (None for forget), and superseded is
# a value the final answer must NOT contain (update/forget).

def pick_attr(rng):
    return rng.choice(ATTRS)


def distract_turns(rng, n_max):
    return [rng.choice(DISTRACT)[0] for _ in range(rng.randint(1, n_max))]


def case_remember(rng, i):
    attr, vals = pick_attr(rng)
    v = rng.choice(vals)
    pool = EXPLICIT_MEM if i % 2 == 0 else IMPLICIT_MEM
    turns = [rng.choice(pool).format(attr=attr, val=v)]
    turns.append(rng.choice(QUERY).format(attr=attr))
    return turns, v, None


def case_update(rng, i):
    attr, vals = pick_attr(rng)
    v1, v2 = rng.sample(vals, 2)
    # Alternate the two training protocols: separate overwrite turn vs one
    # combined turn mentioning old then new (mirrors gen_synth_memory_data).
    if i % 2 == 0:
        turns = [
            rng.choice(EXPLICIT_MEM).format(attr=attr, val=v1),
            rng.choice(UPDATE_MEM).format(attr=attr, new=v2),
        ]
    else:
        turns = [
            f"My {attr} used to be {v1}, but it's changed — "
            f"my {attr} is {v2} now."
        ]
    turns.append(rng.choice(QUERY).format(attr=attr))
    return turns, v2, v1


def case_forget(rng, i):
    attr, vals = pick_attr(rng)
    v = rng.choice(vals)
    # Two-stage forget: removal instruction + positive "unset" overwrite.
    turns = [
        rng.choice(EXPLICIT_MEM).format(attr=attr, val=v),
        rng.choice(FORGET_MEM).format(attr=attr),
        rng.choice(FORGET_OVERWRITE).format(attr=attr),
    ]
    turns.append(rng.choice(QUERY).format(attr=attr))
    return turns, None, v


def case_distract(rng, i):
    attr, vals = pick_attr(rng)
    v = rng.choice(vals)
    pool = EXPLICIT_MEM if i % 2 == 0 else IMPLICIT_MEM
    turns = [rng.choice(pool).format(attr=attr, val=v)]
    turns.extend(distract_turns(rng, 3))
    turns.append(rng.choice(QUERY).format(attr=attr))
    return turns, v, None


def case_multi_entity(rng, i):
    attr, vals = pick_attr(rng)
    n1, n2 = rng.sample(NAMES, 2)
    v1, v2 = rng.sample(vals, 2)
    turns = [
        f"{n1}'s {attr} is {v1}.",
        f"{n2}'s {attr} is {v2}.",
    ]
    if rng.random() < 0.5:
        turns.append(f"What is {n1}'s {attr}?")
        return turns, v1, None
    turns.append(f"And what about {n2}'s {attr}?")
    return turns, v2, None


CASE_BUILDERS = {
    "remember": case_remember,
    "update": case_update,
    "forget": case_forget,
    "distract": case_distract,
    "multi_entity": case_multi_entity,
}


def build_cases():
    rng = random.Random(2026)
    cases = []
    for op, count in CASE_COUNTS:
        for i in range(count):
            turns, expected, superseded = CASE_BUILDERS[op](rng, i)
            cases.append({
                "op": op,
                "session_id": f"metis-recall-{op}-{i:03d}",
                "turns": turns,
                "expected": expected,
                "superseded": superseded,
            })
    return cases


def run_case(base_url, case):
    """Run one case's multi-turn session. Returns (passed, answer, note)."""
    messages = []
    answer = ""
    for turn_idx, user_text in enumerate(case["turns"]):
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": "bitnet",
            "session_id": case["session_id"],
            "reset_session": turn_idx == 0,
            "messages": [dict(m) for m in messages],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            status, response = request_json(
                f"{base_url}/v1/chat/completions", payload)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return False, answer, f"request error on turn {turn_idx}: {exc}"
        if status != 200:
            return False, answer, f"status {status} on turn {turn_idx}"
        text = response_text(response)
        if turn_idx == len(case["turns"]) - 1:
            answer = text
        else:
            messages.append({"role": "assistant", "content": text})

    expected, superseded = case["expected"], case["superseded"]
    if expected is not None and not contains_value(answer, expected):
        return False, answer, f"expected value {expected!r} not in answer"
    if superseded is not None and contains_value(answer, superseded):
        return False, answer, f"superseded value {superseded!r} still in answer"
    return True, answer, "ok"


def main():
    args = sys.argv[1:]
    threshold = DEFAULT_THRESHOLD
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--threshold":
            i += 1
            if i >= len(args):
                return fail("--threshold requires a value")
            try:
                threshold = float(args[i])
            except ValueError:
                return fail(f"invalid threshold: {args[i]}")
        else:
            positional.append(args[i])
        i += 1
    if len(positional) < 3:
        return fail("usage: test_metis_recall.py <openai_server> <model.gguf> "
                    "<memory.bnmem> [--threshold 0.60]")
    server_path, model_path, memory_path = positional[0], positional[1], positional[2]
    if not os.path.exists(server_path):
        return fail(f"server binary not found: {server_path}")
    if not os.path.exists(model_path):
        return fail(f"model not found: {model_path}")
    if not os.path.exists(memory_path):
        return fail(f"memory model not found: {memory_path}")

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_cmd = [server_path, model_path, "--host", "127.0.0.1",
                  "--port", str(port), "--memory-model", memory_path]
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ok, error = wait_until_ready(base_url, time.time() + 120)
        if not ok:
            stderr = proc.stderr.read() if proc.poll() is not None else ""
            return fail(f"server not ready: {error}; stderr={stderr}")

        cases = build_cases()
        stats = {op: {"pass": 0, "total": 0} for op, _ in CASE_COUNTS}
        n_pass = 0
        failures = []
        for idx, case in enumerate(cases):
            passed, answer, note = run_case(base_url, case)
            stats[case["op"]]["total"] += 1
            if passed:
                stats[case["op"]]["pass"] += 1
                n_pass += 1
            else:
                failures.append((idx, case, answer, note))
            print(f"[{idx + 1}/{len(cases)}] {case['op']:>12} "
                  f"{case['session_id']}: {'PASS' if passed else 'FAIL'} ({note})",
                  flush=True)

        print()
        print("=== Metis recall eval ===")
        print(f"server:   {server_path}")
        print(f"model:    {model_path}")
        print(f"memory:   {memory_path}")
        print()
        print(f"{'operation':<14} {'pass':>5} {'total':>5} {'accuracy':>9}")
        for op, _ in CASE_COUNTS:
            s = stats[op]
            acc = s["pass"] / s["total"] if s["total"] else 0.0
            print(f"{op:<14} {s['pass']:>5} {s['total']:>5} {acc:>8.1%}")
        total = len(cases)
        overall = n_pass / total if total else 0.0
        print("-" * 35)
        print(f"{'overall':<14} {n_pass:>5} {total:>5} {overall:>8.1%}")
        print()
        print(f"threshold: {threshold:.2f}")
        if failures:
            print(f"first failures (of {len(failures)}):")
            for idx, case, answer, note in failures[:5]:
                print(f"  #{idx + 1} {case['session_id']}: {note}")
                print(f"    query answer: {answer[:200]!r}")
        if overall < threshold:
            print(f"RESULT: FAIL (overall {overall:.1%} < {threshold:.2f})")
            return 1
        print(f"RESULT: PASS (overall {overall:.1%} >= {threshold:.2f})")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())

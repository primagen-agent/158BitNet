#!/usr/bin/env python3
"""eval_reference_protocol.py — run OUR trained memory model against the
REFERENCE project's own eval (eval_memory_synth.py protocol verbatim):
their session generator (seed 99001, their ATTRS/NAMES pools, their
remember/update/forget + implicit structures), their commit protocol
(each mem message solo via fresh context, query fresh), their scorer
(gold-substring / UNK regex), their max_new_tokens=24 — served through
OUR C runtime (openai_server --memory-model), NOT their python stack.

Usage: eval_reference_protocol.py <openai_server> <model.gguf> <mem.bnmem>
       [--protocol explicit|implicit] [--n-per-op 30] [--port N]
"""
import json
import random
import re
import socket
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
import tempfile

rng = random.Random(99001)

NAMES = """Diana Eric Fiona Gary Hannah Ivan Jasmine Karl Luna Mason Nora
Oscar Petra Quinn Renee Stefan Tara Ulric Vera Wade Xena Yara Zane""".split()
ATTRS = [
    ("favorite season", ["spring", "summer", "autumn", "winter"]),
    ("lucky number", ["3", "7", "9", "21", "42", "88"]),
    ("favorite music genre", ["jazz", "classical", "rock", "folk", "electronic"]),
    ("preferred drink", ["oolong tea", "espresso", "hot cocoa", "lemon water", "matcha"]),
    ("weekend hobby", ["fishing", "painting", "birdwatching", "pottery", "cycling"]),
    ("favorite flower", ["peony", "jasmine", "sunflower", "orchid", "cornflower"]),
]

UNK_PAT = re.compile(r"don'?t (have|know|recall)|haven'?t told|no information|"
                     r"not (sure|remember)|forgotten|removed|unknown|no longer", re.I)

IMPLICIT_MEM = [
    "I've always loved {v} — that's my {attr}, by the way.",
    "If you're curious, my {attr} happens to be {v}.",
    "As a kid I picked {v} and it stuck — still my {attr} today.",
    "Everyone teases me about it, but my {attr} has been {v} for years.",
    "Lately I can't get enough of {v}. That's my {attr}, honestly.",
]
IMPLICIT_UPDATE = [
    "I used to say my {attr} was {v1}, but these days it's {v2}.",
    "Funny thing — my {attr} isn't {v1} anymore, I've moved on to {v2}.",
    "I've switched things up: my {attr} is now {v2}, not {v1}.",
    "My {attr} changed — I got tired of {v1} and went with {v2}.",
]
IMPLICIT_FORGET = [
    "I'd rather not talk about my {attr} anymore — let's drop that subject.",
    "Honestly, my {attr} is private now. Don't bring it up again.",
    "Can we stop discussing my {attr}? I'm done with that topic.",
]


def gen_sessions(n_per_op=30):
    sessions = []
    for _ in range(n_per_op):
        attr, vals = rng.choice(ATTRS)
        v = rng.choice(vals)
        sessions.append({"op": "remember",
                         "mem": f"Please remember that my {attr} is {v}.",
                         "query": f"What is my {attr}?", "gold": [v]})
        v1, v2 = rng.sample(vals, 2)
        sessions.append({"op": "update",
                         "mem": f"Please remember that my {attr} is {v1}.",
                         "mem2": f"Update: my {attr} is now {v2}.",
                         "query": f"What is my {attr}?", "gold": [v2]})
        sessions.append({"op": "forget",
                         "mem": f"Please remember that my {attr} is {v}.",
                         "mem2": f"Please forget my {attr}.",
                         "mem3": f"My {attr} is now unset.",
                         "query": f"What is my {attr}?", "gold": None})
    return sessions


def gen_implicit_sessions(n_per_op=30):
    sessions = []
    for _ in range(n_per_op):
        attr, vals = rng.choice(ATTRS)
        v = rng.choice(vals)
        sessions.append({"op": "i-remember",
                         "mem": rng.choice(IMPLICIT_MEM).format(attr=attr, v=v),
                         "query": f"What is my {attr}?", "gold": [v]})
        v1, v2 = rng.sample(vals, 2)
        sessions.append({"op": "i-update",
                         "mem": rng.choice(IMPLICIT_UPDATE).format(attr=attr, v1=v1, v2=v2),
                         "query": f"What is my {attr}?", "gold": [v2]})
        sessions.append({"op": "i-forget",
                         "mem": rng.choice(IMPLICIT_FORGET).format(attr=attr),
                         "query": f"What is my {attr}?", "gold": None})
    return sessions


def find_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request_json(url, payload=None, timeout=600):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def response_text(resp):
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def start_server(srv, gguf, bnmem, port, state_dir=None):
    cmd = [srv, gguf, "--memory-model", bnmem,
           "--host", "127.0.0.1", "--port", str(port)]
    if state_dir is not None:
        cmd += ["--memory-state-dir", state_dir]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    health = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("openai_server exited during startup")
        try:
            request_json(health, timeout=2)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate(); proc.wait()
    raise RuntimeError("openai_server startup timed out")


def main():
    srv, gguf, bnmem = sys.argv[1], sys.argv[2], sys.argv[3]
    protocol = "explicit"
    n_per_op = 30
    persistence = False
    args = sys.argv[4:]
    i = 0
    while i < len(args):
        if args[i] == "--protocol":
            protocol = args[i + 1]; i += 2
        elif args[i] == "--n-per-op":
            n_per_op = int(args[i + 1]); i += 2
        elif args[i] == "--persistence":
            persistence = True; i += 1
        else:
            i += 1

    port = find_port()
    state_dir = tempfile.mkdtemp(prefix="bitnet-memory-state-") if persistence else None
    proc = start_server(srv, gguf, bnmem, port, state_dir)
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    try:
        sessions = (gen_implicit_sessions(n_per_op) if protocol == "implicit"
                    else gen_sessions(n_per_op))
        ops = sorted({s["op"] for s in sessions})
        correct = {op: 0 for op in ops}
        if persistence:
            for si, s in enumerate(sessions):
                sid = f"refeval-{si}"
                for k, m in enumerate(("mem", "mem2", "mem3")):
                    if m not in s:
                        continue
                    request_json(base, {
                        "session_id": sid, "reset_session": k == 0,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": s[m]}],
                    })
                request_json(f"http://127.0.0.1:{port}/v1/memory/export",
                             {"session_id": sid})
                if (si + 1) % 10 == 0:
                    print(f"[export {si+1}/{len(sessions)}] ...", flush=True)
            proc.terminate(); proc.wait()
            proc = start_server(srv, gguf, bnmem, port, state_dir)
        for si, s in enumerate(sessions):
            sid = f"refeval-{si}"
            # their protocol: each mem message committed SOLO (fresh ctx)
            if persistence:
                request_json(f"http://127.0.0.1:{port}/v1/memory/import",
                             {"session_id": sid})
            else:
                for k, m in enumerate(("mem", "mem2", "mem3")):
                    if m not in s:
                        continue
                    request_json(base, {
                        "session_id": sid, "reset_session": k == 0,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": s[m]}],
                    })
            response = request_json(base, {
                "session_id": sid, "max_tokens": 24,
                "messages": [{"role": "user", "content": s["query"]}],
            })
            if persistence:
                session = response.get("session", {})
                cached = session.get("cached_tokens", 0)
                reused = session.get("reused_tokens", 0)
                if cached != 0 or reused != 0:
                    raise RuntimeError(
                        f"session {sid} reused KV cache: "
                        f"cached_tokens={cached}, reused_tokens={reused}")
            ans = response_text(response)
            if s["gold"] is not None:
                ok = any(g.lower() in ans.lower() for g in s["gold"])
            else:
                ok = bool(UNK_PAT.search(ans))
            correct[s["op"]] += int(ok)
            if (si + 1) % 10 == 0:
                print(f"[{si+1}/{len(sessions)}] ...", flush=True)
        mode = "restart persistence" if persistence else protocol
        print(f"\n=== Memory accuracy ({mode}) — OUR model via OUR C runtime ===")
        tc = tn = 0
        for op in ops:
            n = sum(1 for s in sessions if s["op"] == op)
            print(f"{op:11s}: {correct[op]}/{n} = {correct[op]/n:.1%}")
            tc += correct[op]; tn += n
        print(f"{'TOTAL':11s}: {tc}/{tn} = {tc/tn:.1%}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        if state_dir is not None:
            shutil.rmtree(state_dir)


if __name__ == "__main__":
    main()

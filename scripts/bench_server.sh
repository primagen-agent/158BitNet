#!/bin/sh
# Measure the model matrix of a running unified server over OpenAI HTTP.
#
#   scripts/bench_server.sh http://HOST:PORT [models...]   # default: all four CPM
#
# Each CPM model is decoded for 32 greedy tokens; the 27B is measured plain and
# with DSpine at block 2 if it is registered.  Run it on a quiet device: a
# leftover server or a second benchmark on the same cores shifts every number.
U=${1:-http://192.168.70.120:8080}
shift 2>/dev/null || true
MODELS=${*:-cpm05 cpm1b cpm3b cpm8b}
curl -s --max-time 5 "$U/health"; echo
for M in $MODELS; do
    curl -s --max-time 60 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":8,\"temperature\":0}" \
        "$U/v1/chat/completions" >/dev/null 2>&1 || true
    curl -s --max-time 600 -H 'Content-Type: application/json' \
        -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to five.\"}],\"max_tokens\":32,\"temperature\":0}" \
        "$U/v1/chat/completions" > /tmp/bench_$M.json 2>/dev/null || true
    python3 -I - "$M" /tmp/bench_$M.json <<'PY' 2>/dev/null || echo "$M  (no response)"
import json,sys
p=json.load(open(sys.argv[2])).get('perf',{})
print("%-6s %6.2f tok/s  prefill %.0f ms" % (sys.argv[1], p.get('decode_tokens_per_s',0), p.get('prefill_ms',0)))
PY
done
for S in false true; do
    curl -s --max-time 900 -H 'Content-Type: application/json' \
        -d "{\"model\":\"bonsai27b\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0,\"spec\":$S,\"spec_block\":2}" \
        "$U/v1/chat/completions" > /tmp/bench_27.json 2>/dev/null || true
    python3 -I - "$S" /tmp/bench_27.json <<'PY' 2>/dev/null || echo "27B spec=$S (not registered)"
import json,sys
p=json.load(open(sys.argv[2])).get('perf',{})
print("27B spec=%-5s %s %.2f tok/s" % (sys.argv[1], p.get('engine'), p.get('decode_tokens_per_s',0)))
PY
done

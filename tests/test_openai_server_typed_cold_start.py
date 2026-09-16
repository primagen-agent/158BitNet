#!/usr/bin/env python3
"""Real-model regression: first observed changes are facts, not missing targets."""
import json
import sys
import tempfile

from test_openai_server_typed_chat_auto_memory import (
    assert_no_kv, chat, free_port, post, start_server, stop_server,
)


def main():
    if len(sys.argv) not in (7, 8):
        raise SystemExit("usage: test_openai_server_typed_cold_start.py server gguf pair link query writer [controller]")
    paths = sys.argv[1:7]
    controller = sys.argv[7] if len(sys.argv) == 8 else None
    original = "Briar's conference registration is a printed access card."
    changed = ("Briar's conference registration is now a virtual access ticket after work, "
               "replacing a printed access card.")
    unrelated = "Cleo's preferred news source is public radio bulletins near home."
    question = "What is Briar's current conference registration?"
    expected = "a virtual access ticket after work"
    results = []

    def write(base, sid, text, **kwargs):
        response = chat(base, sid, text, **kwargs)
        assert_no_kv(response)
        return response["memory_auto"]

    with tempfile.TemporaryDirectory(prefix="memory-cold-start-") as directory:
        process, base = start_server(*paths, directory, free_port(), controller=controller)
        try:
            result = write(base, "cold", changed)
            assert result["status"] == "stored", result
            assert result["writer_predicted_operation"] == "supersede", result
            assert result["target_resolution"] == "no_predecessor_asserted", result
            assert result["operation"] == "assert" and result["target_event_id"] == "", result
            assert result["entity"] == "Briar" and result["value"] == expected, result
            results.append("cold-start fact extraction")
            result = write(base, "cold", changed)
            assert result["deduplicated"] == 1, result
            exported = post(base, "/v1/memory/export", {"session_id": "cold"})
            assert exported["typed_events"] == 1 and exported["episodic_records"] == 1, exported
            results.append("idempotent repeated observation")

            result = write(base, "strict", changed, memory_action="update")
            assert result["status"] == "failed", result
            empty = post(base, "/v1/memory/export", {"session_id": "strict"})
            assert empty["typed_events"] == empty["episodic_records"] == 0, empty
            results.append("strict explicit update and rollback")

            first = write(base, "known", original)
            assert first["status"] == "stored", first
            result = write(base, "known", changed)
            assert result["operation"] == "supersede", result
            assert result["target_resolution"] == "neural_activation", result
            assert result["target_event_id"] == first["event_id"], result
            assert result["value"] == expected, result
            results.append("known predecessor neural update")

            first = write(base, "unrelated", unrelated)
            assert first["status"] == "stored", first
            result = write(base, "unrelated", changed)
            assert result["status"] == "stored" and result["operation"] == "assert", result
            preserved = post(base, "/v1/memory/event/current", {
                "session_id": "unrelated", "entity": first["entity"], "predicate": first["predicate"]})
            assert preserved["value"] == first["value"] and preserved["active_count"] == 1, preserved
            results.append("unrelated memory preserved")
        finally:
            stop_server(process)
        process, base = start_server(*paths, directory, free_port(), controller=controller)
        try:
            before = chat(base, "cold", question, memory_action="ignore")
            assert_no_kv(before)
            assert not before.get("memory_copy"), before
            post(base, "/v1/memory/import", {"session_id": "cold"})
            after = chat(base, "cold", question, memory_action="ignore")
            assert_no_kv(after)
            assert after.get("memory_copy", {}).get("mode") == "neural_typed_query_activation_then_compiled_pointer", after
            assert after["choices"][0]["message"]["content"] == expected, after
            results.append("restart/import and no-KV neural recall")
        finally:
            stop_server(process)
    print(json.dumps({"passed": len(results), "checks": results}), flush=True)


if __name__ == "__main__":
    main()

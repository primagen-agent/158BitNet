"""Validate opt-in native feature diagnostics from a complete child stderr log.

The caller must also check the child's exit status and all operation results.
KV reuse means positions present before an encoding, excluding causal attention
to tokens freshly evaluated within that encoding. No token text is recorded.
"""

import json


TRACE_MARKER = "[ns-feature-eval]"
TRACE_PREFIX = TRACE_MARKER + " "
_INTEGER_FIELDS = ("pid", "requested_tokens", "kv_before", "kv_after", "eval_status",
                   "hidden_rows", "encode_status")
_FIELDS = frozenset((*_INTEGER_FIELDS, "context_created", "eval_attempted"))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate feature trace field")
        result[key] = value
    return result


def summarize_feature_trace(stderr, *, expected_pid, expected_calls=None):
    """Require complete, successful, fresh-context evidence and return totals.

    Event order is immaterial, including when independent contexts finish in
    different thread orders. ``expected_calls`` additionally detects missing
    records when the caller knows the exact number of encoding attempts.
    """
    if type(expected_pid) is not int or expected_pid <= 0:
        raise ValueError("expected_pid must identify the launched child")
    if expected_calls is not None and (type(expected_calls) is not int or expected_calls < 1):
        raise ValueError("expected_calls must be a positive integer")
    if not isinstance(stderr, str):
        raise ValueError("stderr must be the complete decoded child log")
    records = []
    for number, line in enumerate(stderr.splitlines(keepends=True), 1):
        if TRACE_MARKER not in line:
            continue
        if not line.startswith(TRACE_PREFIX) or not line.endswith("\n"):
            raise ValueError(f"malformed or truncated feature trace at line {number}")
        try:
            record = json.loads(line[len(TRACE_PREFIX):], object_pairs_hook=_unique_object)
        except ValueError as error:
            raise ValueError(f"malformed feature trace at line {number}") from error
        if not isinstance(record, dict) or record.keys() != _FIELDS:
            raise ValueError(f"incomplete feature trace fields at line {number}")
        if any(type(record[key]) is not int for key in _INTEGER_FIELDS):
            raise ValueError(f"unmeasured or invalid feature trace values at line {number}")
        if record["pid"] != expected_pid:
            raise ValueError(f"unexpected feature trace process at line {number}")
        if (record["context_created"] is not True or record["eval_attempted"] is not True or
                record["eval_status"] != 0 or record["encode_status"] != 0):
            raise ValueError(f"unsuccessful feature encoding at line {number}")
        if record["kv_before"] != 0:
            raise ValueError(f"feature encoding has pre-existing KV positions at line {number}")
        if (record["requested_tokens"] < 2 or
                record["hidden_rows"] != record["requested_tokens"] or
                record["kv_after"] - record["kv_before"] != record["hidden_rows"]):
            raise ValueError(f"inconsistent feature evaluation counts at line {number}")
        records.append(record)
    if not records or (expected_calls is not None and len(records) != expected_calls):
        raise ValueError("missing feature evaluation events")
    return dict(native_pid=expected_pid, actual_new_context_calls=len(records),
                fresh_context_calls=sum(record["kv_before"] == 0 for record in records),
                requested_tokens=sum(record["requested_tokens"] for record in records),
                evaluated_tokens=sum(record["hidden_rows"] for record in records),
                reused_kv_tokens=sum(record["kv_before"] for record in records),
                records=records)

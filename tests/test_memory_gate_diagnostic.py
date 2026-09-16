import json
import unittest
from unittest.mock import patch

from eval_memory_chat_holdout import force_writer_diagnostic


class MemoryGateDiagnosticTest(unittest.TestCase):
    @patch("eval_memory_chat_holdout.urllib.request.urlopen")
    def test_bypass_gate_does_not_force_operation_or_supply_gold(self, urlopen):
        result = {"kv_cache_touched": 0, "operation": "supersede"}
        urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(result).encode()
        actual = force_writer_diagnostic("http://localhost:1", "session", "raw statement?")
        request = urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data), {"session_id": "session", "memory_record": "raw statement?"})
        self.assertEqual(actual["operation"], "supersede")
        self.assertEqual(actual["stored"], 1)
        result["kv_cache_touched"] = 1
        urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(result).encode()
        with self.assertRaises(AssertionError):
            force_writer_diagnostic("http://localhost:1", "session", "raw")


if __name__ == "__main__":
    unittest.main()

"""Offline CLI validation tests; model and retrieval dependencies are stubbed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class SearchCliTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("05_search_evaluate.py")
        spec = importlib.util.spec_from_file_location("search_cli_under_test", path)
        self.cli = importlib.util.module_from_spec(spec)
        core = types.SimpleNamespace(normalize=None, rank_clips=None, retrieval_metrics=None)
        dependencies = {
            "numpy": types.ModuleType("numpy"),
            "torch": types.ModuleType("torch"),
            "04_vectors": types.ModuleType("04_vectors"),
            "search_core": core,
        }
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(self.cli)

    def test_blank_query_is_rejected_before_index_access(self):
        for query in ("", "   ", "\t\n"):
            with self.subTest(query=repr(query)):
                stderr = io.StringIO()
                with patch.object(self.cli, "load_index") as load_index:
                    load_index.side_effect = AssertionError("Index accessed before query validation")
                    with patch.object(sys, "argv", ["search", "--query", query]):
                        with contextlib.redirect_stderr(stderr):
                            with self.assertRaises(SystemExit) as error:
                                self.cli.main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("--query must contain non-whitespace text", stderr.getvalue())
                    load_index.assert_not_called()

    def test_valid_query_preserves_text_method_and_result_limit(self):
        for method in ("max", "mean"):
            with self.subTest(method=method):
                query = "  a batter attempting a bunt  "
                vectors, records, feature = object(), [], [1.0, 0.0]
                ranked = [{"clip_id": str(i)} for i in range(6)]
                stdout = io.StringIO()
                with patch.object(self.cli, "load_index", return_value=(vectors, records)):
                    with patch.object(self.cli, "text_vectors", return_value=[feature]) as encode:
                        with patch.object(self.cli, "rank_clips", return_value=ranked) as rank:
                            with patch.object(sys, "argv", ["search", "--query", query, "--method", method]):
                                with contextlib.redirect_stdout(stdout):
                                    self.cli.main()
                encode.assert_called_once_with([query])
                rank.assert_called_once_with(vectors, records, feature, method)
                self.assertEqual(json.loads(stdout.getvalue()), {"query": query, "results": ranked[:5]})

    def test_omitted_query_still_reaches_evaluation_index(self):
        with patch.object(self.cli, "load_index", side_effect=RuntimeError("evaluation index")) as load:
            with patch.object(sys, "argv", ["search"]):
                with self.assertRaisesRegex(RuntimeError, "evaluation index"):
                    self.cli.main()
        load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

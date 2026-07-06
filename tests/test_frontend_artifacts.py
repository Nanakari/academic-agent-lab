"""Tests for exact persisted-result downloads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.frontend.result_artifacts import read_result_json_bytes


class ResultArtifactTests(unittest.TestCase):
    def test_existing_result_json_is_read_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            expected = b'{\n  "canonical": true\n}\n'
            path.write_bytes(expected)

            result = read_result_json_bytes(path)

        self.assertEqual(result, expected)

    def test_missing_result_json_has_no_serialization_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-result.json"

            result = read_result_json_bytes(path)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

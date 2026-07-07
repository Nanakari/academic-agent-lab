"""Tests for project configuration loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.config as config_module


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_falls_back_to_example_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / "config.example.toml"
            example.write_text(
                '[llm]\nmodel = "example-model"\napi_key = ""\n',
                encoding="utf-8",
            )

            with patch.object(config_module, "ROOT_DIR", root), patch.object(
                config_module,
                "CONFIG_PATH",
                root / "config.toml",
            ), patch.object(
                config_module,
                "EXAMPLE_CONFIG_PATH",
                example,
            ), patch.dict(
                "os.environ",
                {},
                clear=True,
            ):
                loaded = config_module.load_config()

            self.assertEqual(loaded["llm"]["model"], "example-model")
            self.assertEqual(loaded["_config_path"], str(example))

    def test_load_config_prefers_local_config_and_env_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "config.toml"
            example = root / "config.example.toml"
            local.write_text(
                '[llm]\nmodel = "local-model"\napi_key = "from-file"\n',
                encoding="utf-8",
            )
            example.write_text(
                '[llm]\nmodel = "example-model"\napi_key = ""\n',
                encoding="utf-8",
            )

            with patch.object(config_module, "ROOT_DIR", root), patch.object(
                config_module,
                "CONFIG_PATH",
                local,
            ), patch.object(
                config_module,
                "EXAMPLE_CONFIG_PATH",
                example,
            ), patch.dict(
                "os.environ",
                {"GEMINI_API_KEY": "from-env"},
                clear=True,
            ):
                loaded = config_module.load_config()

            self.assertEqual(loaded["llm"]["model"], "local-model")
            self.assertEqual(loaded["llm"]["api_key"], "from-env")
            self.assertEqual(loaded["_config_path"], str(local))

    def test_load_config_errors_when_no_config_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(config_module, "ROOT_DIR", root), patch.object(
                config_module,
                "CONFIG_PATH",
                root / "config.toml",
            ), patch.object(
                config_module,
                "EXAMPLE_CONFIG_PATH",
                root / "config.example.toml",
            ):
                with self.assertRaises(FileNotFoundError):
                    config_module.load_config()


if __name__ == "__main__":
    unittest.main()

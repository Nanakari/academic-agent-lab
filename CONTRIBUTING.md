# Contributing

## Setup

1. Create and activate a Python 3.10 or newer virtual environment.
2. Install development dependencies with `python -m pip install -r requirements-dev.txt`.
3. Copy `config.example.toml` to `config.toml` only when local settings are needed.

Do not commit API keys, private papers, runtime outputs, caches, or local research memory.

## Checks

Run the same core checks used by CI:

```bash
python -m pip check
python -m compileall -q app main.py
python -m pytest -q
python app/ai_scientific_demo.py --topic "LVLM hallucination mitigation" --papers-dir tests/fixtures/papers --top-k 5 --offline
python main.py --topic "LVLM hallucination mitigation" --papers-dir tests/fixtures/papers --top-k 5 --offline
```

Keep pull requests focused. Explain changes to evidence handling, evaluation behavior, prompts, or reproducibility metadata in the pull request description.

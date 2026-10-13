# A2R-Decision

Official implementation of the A2R-Decision method described in *Measuring Source-Resolution Distortion in LLM Output Arbitration: A Source-Bounded Evaluation Framework*.

## Installation

```bash
python3 -m pip install -e ".[openai]"
```

## Usage

```bash
python3 -m a2r --input cases.jsonl --output results.jsonl --model MODEL_NAME
```

Each JSONL input line contains `query` and `candidates` (`{"A": "...", "B": "..."}`). The output preserves the input fields and adds an `arbitration` object.

The implementation is in [`a2r/`](a2r/).

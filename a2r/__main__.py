"""python -m a2r --input cases.jsonl --output out.jsonl --model MODEL [--base-url URL]

Each input line needs "query" and "candidates" ({"A": ..., "B": ..., ...}); other fields are kept.
"""
import argparse
import json
from pathlib import Path

from .codec import DPCM
from .llm import openai_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("input and output must be different files")

    codec = DPCM(openai_chat(args.model, args.base_url, temperature=args.temperature))
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            case = json.loads(line)
            try:
                case["dpcm"] = codec(case["query"], case["candidates"]).to_dict()
            except Exception as error:  # record a failed case and continue the batch
                case["dpcm"] = {"error": repr(error)}
            fout.write(json.dumps(case, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()

"""python -m a2r --input cases.jsonl --output out.jsonl --model MODEL [--base-url URL] [--rounds N] [--temperature T]

Each input line needs "query" and "candidates" ({"A": ..., "B": ..., ...}); other fields are kept.
"""
import argparse
import json

from .arbiter import Arbiter
from .llm import openai_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--rounds", type=int, default=1, help="passes over all orders of the candidates")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature of the reader")
    args = parser.parse_args()

    arbiter = Arbiter(openai_chat(args.model, args.base_url, temperature=args.temperature), rounds=args.rounds)
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            case = json.loads(line)
            try:
                case["arbitration"] = arbiter(case["query"], case["candidates"]).to_dict()
            except Exception as error:  # keep the batch going; the failure is recorded with the case
                case["arbitration"] = {"error": repr(error)}
            fout.write(json.dumps(case, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()

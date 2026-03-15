#!/usr/bin/env python3
"""
CLI for LLM Evaluation & Hallucination Detection.
Usage:
  python run_eval.py --input data/samples.jsonl [--config configs/eval_basic.yaml] [--output results.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path so we can import llm_eval when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_eval.runner import run_eval, summarize


def main():
    p = argparse.ArgumentParser(description="Run LLM evaluation and/or hallucination detection on JSONL input.")
    p.add_argument("--input", "-i", required=True, help="Input JSONL file")
    p.add_argument("--output", "-o", default=None, help="Output JSONL file (default: print to stdout)")
    p.add_argument("--config", "-c", default=None, help="Config YAML (optional)")
    p.add_argument("--summary", action="store_true", help="Print summary stats to stderr")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    results = run_eval(
        input_path=input_path,
        output_path=args.output,
        config_path=args.config,
    )

    if args.output:
        print(f"Wrote {len(results)} results to {args.output}", file=sys.stderr)

    if args.summary:
        s = summarize(results)
        print("Summary:", json.dumps(s, indent=2), file=sys.stderr)

    if not args.output:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

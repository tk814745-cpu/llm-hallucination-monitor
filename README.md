# LLM Evaluation & Hallucination Detection Framework

A Python framework for evaluating LLM outputs and detecting hallucinations (unsupported or false claims). Use it to score factuality, coherence, and relevance, and to flag when model responses contradict sources or make up information.

## Features

- **Evaluation metrics**: factuality, coherence, relevance, length, exact match
- **Hallucination detection**: NLI-based (claim vs. source), reference-based, self-consistency
- **Pluggable design**: add custom evaluators and detectors
- **Batch runs**: evaluate many (prompt, response, reference) triples from JSON/JSONL
- **Reports**: structured results (JSON) and optional summary stats

## Project structure

```
llm-eval-hallucination/
├── src/
│   └── llm_eval/
│       ├── __init__.py
│       ├── evaluation/      # Core evaluation logic
│       ├── hallucination/   # Hallucination detection
│       ├── metrics/         # Metric implementations
│       └── runner.py        # Batch evaluation runner
├── configs/                 # Example configs
├── data/                    # Sample inputs/outputs
├── requirements.txt
└── run_eval.py              # CLI entrypoint
```

## Setup

```bash
cd llm-eval-hallucination
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Optional (for NLI-based hallucination detection):

```bash
pip install transformers torch
```

## Quick start

1. **Evaluate with reference** (factuality, relevance, exact match):

   ```bash
   python run_eval.py --config configs/eval_basic.yaml --input data/samples.jsonl
   ```

2. **Hallucination detection** (claim vs. source using NLI, if installed):

   ```bash
   python run_eval.py --config configs/hallucination_nli.yaml --input data/samples.jsonl
   ```

3. **From Python**:

   ```python
   from llm_eval import evaluate_response, detect_hallucination

   result = evaluate_response(
       prompt="What is the capital of France?",
       response="The capital of France is Paris.",
       reference="Paris"
   )
   print(result)  # factuality, relevance, exact_match, etc.

   score, label = detect_hallucination(
       claim="The Eiffel Tower is in London.",
       source="The Eiffel Tower is a landmark in Paris, France."
   )
   print(score, label)  # e.g. contradiction
   ```

## Web application (interactive query)

Run a small web app so a user can submit a query (prompt/response/reference and optional claim/source) and see the evaluation + hallucination output.

1. Install deps:

```bash
pip install -r requirements.txt
```

2. Start the server:

```bash
uvicorn webapp:app --reload
```

3. Open:

- `http://127.0.0.1:8000` (UI)
- `http://127.0.0.1:8000/docs` (API docs)

## Input format

JSONL with one JSON object per line. Each object can have:

- `prompt` (str): the user/instruction prompt
- `response` (str): the model output to evaluate
- `reference` (str, optional): ground-truth or source text
- `claim` (str, optional): single claim to check (for hallucination detection)
- `source` (str, optional): source context for the claim

Example:

```json
{"prompt": "Capital of France?", "response": "Paris.", "reference": "Paris"}
{"claim": "The sky is green.", "source": "The sky is blue during the day."}
```

## Configuration

See `configs/eval_basic.yaml` and `configs/hallucination_nli.yaml` for examples. You can enable/disable metrics, set thresholds, and choose detectors (e.g. NLI model name).

## License

MIT.

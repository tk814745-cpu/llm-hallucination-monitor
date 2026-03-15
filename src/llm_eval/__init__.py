"""
LLM Evaluation & Hallucination Detection Framework.
"""
from llm_eval.evaluation.evaluator import evaluate_response
from llm_eval.hallucination.detector import detect_hallucination

__all__ = ["evaluate_response", "detect_hallucination"]

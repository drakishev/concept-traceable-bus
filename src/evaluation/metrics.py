"""Evaluation metrics for histopathology report generation.

Implements standard NLG metrics: BLEU, METEOR, ROUGE-L.
"""
from __future__ import annotations

from dataclasses import dataclass

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score as _meteor_score
from rouge_score import rouge_scorer


def _ensure_nltk_data() -> None:
    """Download required NLTK data if not present."""
    for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            prefix = "tokenizers" if "punkt" in resource else "corpora"
            nltk.data.find(f"{prefix}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


@dataclass
class ReportMetrics:
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    meteor: float
    rouge_l: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bleu_1": self.bleu_1,
            "bleu_2": self.bleu_2,
            "bleu_3": self.bleu_3,
            "bleu_4": self.bleu_4,
            "meteor": self.meteor,
            "rouge_l": self.rouge_l,
        }

    def __str__(self) -> str:
        lines = [f"  {k}: {v:.4f}" for k, v in self.to_dict().items()]
        return "ReportMetrics:\n" + "\n".join(lines)


def compute_metrics(
    predictions: list[str],
    references: list[str],
) -> ReportMetrics:
    """Compute all report generation metrics.

    Args:
        predictions: list of generated report strings
        references: list of ground-truth report strings

    Returns:
        ReportMetrics with BLEU-1..4, METEOR, ROUGE-L
    """
    _ensure_nltk_data()

    assert len(predictions) == len(references), (
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    # tokenize for BLEU/METEOR
    pred_tokens = [nltk.word_tokenize(p.lower()) for p in predictions]
    ref_tokens = [nltk.word_tokenize(r.lower()) for r in references]

    smoother = SmoothingFunction().method1

    # BLEU scores (corpus-level)
    refs_wrapped = [[r] for r in ref_tokens]  # corpus_bleu expects list of list of refs
    def bleu(weights: tuple[float, ...]) -> float:
        return corpus_bleu(refs_wrapped, pred_tokens, weights=weights, smoothing_function=smoother)

    bleu_1 = bleu((1, 0, 0, 0))
    bleu_2 = bleu((0.5, 0.5, 0, 0))
    bleu_3 = bleu((0.33, 0.33, 0.33, 0))
    bleu_4 = bleu((0.25, 0.25, 0.25, 0.25))

    # METEOR (sentence-level, then average)
    meteor_scores = []
    for pred_tok, ref_tok in zip(pred_tokens, ref_tokens):
        score = _meteor_score([ref_tok], pred_tok)
        meteor_scores.append(score)
    meteor = sum(meteor_scores) / len(meteor_scores)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(score["rougeL"].fmeasure)
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    return ReportMetrics(
        bleu_1=bleu_1,
        bleu_2=bleu_2,
        bleu_3=bleu_3,
        bleu_4=bleu_4,
        meteor=meteor,
        rouge_l=rouge_l,
    )

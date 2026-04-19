import re
import string
import psutil
from collections import Counter
from rouge_score import rouge_scorer


def get_memory_usage_mb():
    """
    Return current process memory usage in MB.
    """
    process = psutil.Process()
    memory_bytes = process.memory_info().rss
    return memory_bytes / (1024 * 1024)


def normalize_text(text: str) -> str:
    """
    Normalize text for simple evaluation.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Simple token-level F1 score.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def rouge_scores(reference: str, generated: str):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference, generated)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


def answer_grounded_in_context(answer: str, contexts, min_overlap_tokens: int = 5) -> bool:
    """
    Simple grounding heuristic:
    checks whether the answer shares enough normalized tokens
    with any retrieved context.
    """
    answer_tokens = set(normalize_text(answer).split())

    for ctx in contexts:
        ctx_tokens = set(normalize_text(ctx).split())
        overlap = answer_tokens & ctx_tokens
        if len(overlap) >= min_overlap_tokens:
            return True

    return False


def retrieval_hit_at_k(results, expected_answer: str) -> int:
    """
    Returns 1 if any retrieved chunk contains enough overlap
    with the expected answer, else 0.
    """
    expected_tokens = set(normalize_text(expected_answer).split())

    for item in results:
        chunk_tokens = set(normalize_text(item["text"]).split())
        overlap = expected_tokens & chunk_tokens
        if len(overlap) >= 3:
            return 1

    return 0
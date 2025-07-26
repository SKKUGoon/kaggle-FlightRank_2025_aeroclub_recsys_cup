import torch

def hitrate_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """
    Compute HitRate@k for one group.
    scores: Tensor [G] predicted scores
    labels: Tensor [G] ground truth (1 for positive, 0 for negatives)
    """
    # Get indices of top-k scores
    topk = torch.topk(scores, k=min(k, scores.shape[0])).indices
    # Check if any of top-k labels is 1
    return 1.0 if torch.any(labels[topk] == 1) else 0.0
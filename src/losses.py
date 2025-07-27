import torch

def pairwise_loss(scores: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """
    A simple RankNet-style pairwise loss:
    - scores: [N] float, model scores
    - labels: [N] float/binary, 1 for selected, 0 for not
    - groups: [N] int, group ids (same group = same session)

    Returns: scalar loss
    """
    total_loss = 0.0
    total_pairs = 0

    # 그룹별로 나누기
    unique_groups = torch.unique(groups)
    for gid in unique_groups:
        mask = (groups == gid)
        s = scores[mask]
        y = labels[mask]

        # positive / negative 인덱스
        pos_idx = (y > 0.5).nonzero(as_tuple=True)[0]
        neg_idx = (y <= 0.5).nonzero(as_tuple=True)[0]

        # 각 pos-neg 쌍에 대해 loss 계산
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue  # 쌍 만들 수 없음
        pos_scores = s[pos_idx].view(-1, 1)  # [P,1]
        neg_scores = s[neg_idx].view(1, -1)  # [1,N]

        # score 차이: pos - neg
        diff = pos_scores - neg_scores  # [P,N]
        # RankNet logistic loss: log(1 + exp(-(pos-neg)))
        loss_mat = torch.log1p(torch.exp(-diff))
        total_loss += loss_mat.sum()
        total_pairs += loss_mat.numel()

    if total_pairs == 0:
        return torch.tensor(0.0, device=scores.device)
    return total_loss / total_pairs


def groupwise_softmax_loss(scores: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """
    Group-wise softmax cross-entropy:
    For each group, softmax over scores and compute CE with one-hot label.

    scores: [N] float
    labels: [N] float/binary (1=selected, 0=not)
    groups: [N] int (group ids)

    Returns: scalar loss
    """
    total_loss = 0.0
    total_groups = 0

    unique_groups = torch.unique(groups)
    for gid in unique_groups:
        mask = (groups == gid)
        s = scores[mask]  # [M]
        y = labels[mask]  # [M]

        if y.sum() == 0:
            # no positive in this group (should not happen in training)
            continue

        # softmax over this group's scores
        log_probs = torch.log_softmax(s, dim=0)  # [M]
        # Cross entropy with one-hot label
        loss = - (log_probs * y).sum()  # only selected contributes
        total_loss += loss
        total_groups += 1

    if total_groups == 0:
        return torch.tensor(0.0, device=scores.device)
    return total_loss / total_groups


# Lambda loss framwork for ranking metric optim
# https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/
def lambda_loss(scores: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, sigma: float = 1.0):
    """
    LambdaRank-style loss for single-positive-per-group ranking.

    1. Iterates group by group
      - If find the one item that is actually chosen
      - It compares the model's score for that chosen item (s_pos) to the scores for every unchosen item (s_neg)
    2. For each negative item in the group
      - It asks. Did the model score the chosen item higher than unchosen item?
      - penalty = log(ep(-sigma * diff)) not sufficient above the unchosen item
        - s_pos > s_neg => Penalty is near 0. Good
        - s_pos ~ s_neg => Penalty is larger
        - s_pos < s_neg => Penalty grows quickly
    3. Accumulate
    4. Returns an average penalty across the whole batch

    Args:
      scores: Tensor: Predicted scores for all items, shape [N].
      labels: Tensor: Binary relevance labels, shape [N], exactly one `1` per group.
      groups: Tensor: Group IDs for each item
      sigma: Slope parameter for logistic loss
    """
    device = scores.device

    # Sort by groups to make gathering easier
    sort_idx = torch.argsort(groups)
    scores_sorted = scores[sort_idx]
    labels_sorted = labels[sort_idx]
    groups_sorted = groups[sort_idx]

    # Identify group boundaries
    unique_groups, group_counts = torch.unique_consecutive(groups_sorted, return_counts=True)
    group_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(group_counts, dim=0)[:-1]])

    # Build index tensors for positives and negatives
    losses = []
    for offset, count in zip(group_offsets, group_counts):
        s = scores_sorted[offset:offset+count]
        y = labels_sorted[offset:offset+count]

        pos_mask = (y == 1)
        if not torch.any(pos_mask):
            continue
        # one positive expected
        s_pos = s[pos_mask][0]  # scalar
        s_neg = s[~pos_mask]    # [n_neg]
        if s_neg.numel() == 0:
            continue

        diff = s_pos - s_neg  # [n_neg]
        losses.append(torch.log1p(torch.exp(-sigma * diff)))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    all_losses = torch.cat(losses)  # [total_negatives]
    return all_losses.mean()
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
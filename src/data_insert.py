from pathlib import Path
import polars as pl
import torch
from torch.nn import parameter
from torch.utils.data import IterableDataset
from typing import Iterator, Tuple, List
import random

class ParquetRankDataset(IterableDataset):
    def __init__(
        self,
        parquet_paths: list[str | Path],
        exclude_feature_cols: list[str],
        label_col: str,
        group_col: str,
        max_rows: int,
        normalization_parquet: str | Path,
    ) -> None:
        super().__init__()
        # Normalize to string list
        self.parquet_paths: list[str] = [str(p) for p in parquet_paths]
        self.exclude_feature_cols: set[str] = set(exclude_feature_cols + [label_col, group_col])
        self.label_col: str = str(label_col)
        self.group_col: str = str(group_col)
        self.max_rows: int = int(max_rows)

        # determine feature columns from first parquet
        # (assume all files have the same schema)
        first_schema = pl.scan_parquet(self.parquet_paths[0])
        self.feature_cols: list[str] = [c for c in first_schema.columns if c not in self.exclude_feature_cols]

        # normalization
        norm_df = pl.read_parquet(str(normalization_parquet)).select(self.feature_cols)
        mean_series = norm_df.select([pl.col(c).mean().alias(c) for c in self.feature_cols])
        std_series = norm_df.select([pl.col(c).std().alias(c) for c in self.feature_cols])

        self.mean_dict = {c: mean_series[0, c] for c in self.feature_cols}
        self.std_dict = {c: (std_series[0, c] if std_series[0, c] is not None else 1.0) for c in self.feature_cols}

        # std 가 0인 경우 방어
        for c in self.feature_cols:
            if self.std_dict[c] == 0 or self.std_dict[c] is None:
                self.std_dict[c] = 1.0

        print(f"[INFO] Normalization stats loaded from {normalization_parquet}")
        print(f"[INFO] Example mean/std: {list(self.mean_dict.items())[:5]}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        current_features: List[List[float]] = []
        current_labels: List[float] = []
        current_groups: List[int] = []
        current_gid: int | None = None

        # Iterate through each parquet file
        for path in self.parquet_paths:
            scan = pl.scan_parquet(path).select(self.feature_cols + [self.label_col, self.group_col])
            for row in scan.collect(streaming=True).iter_rows(named=True):
                gid = abs(hash(row[self.group_col])) % (2**31)
                if current_gid is None:
                    current_gid = gid
                # flush condition
                if len(current_features) >= self.max_rows and gid != current_gid:
                    yield (
                        torch.tensor(current_features, dtype=torch.float32),
                        torch.tensor(current_labels, dtype=torch.float32),
                        torch.tensor(current_groups, dtype=torch.long),
                    )
                    current_features, current_labels, current_groups = [], [], []
                    current_gid = gid


                # normalize
                norm_feat = [
                    (row[c] - self.mean_dict[c]) / (self.std_dict[c] + 1e-6)
                    for c in self.feature_cols
                ]
                current_features.append(norm_feat)
                current_labels.append(row[self.label_col])
                current_groups.append(gid)
                current_gid = gid

        # final flush
        if current_features:
            yield (
                torch.tensor(current_features, dtype=torch.float32),
                torch.tensor(current_labels, dtype=torch.float32),
                torch.tensor(current_groups, dtype=torch.long),
            )

    @property
    def feature_len(self) -> int:
        return len(self.feature_cols)


class ParquetResampleRankDataset(ParquetRankDataset):
    def __init__(
        self,
        parquet_paths: list[str | Path],
        exclude_feature_cols: list[str],
        label_col: str,
        group_col: str,
        max_rows: int,
        normalization_parquet: str | Path,
        k_negative: int = 9,
        k_neg_ratio: float = 0.1, # number of negative samples per positive sample
    ) -> None:
        super().__init__(parquet_paths, exclude_feature_cols, label_col, group_col, max_rows, normalization_parquet)
        if k_neg_ratio > 1:
            raise ValueError("k_negatives must be less than 1")
        self.k_neg_ratio = k_neg_ratio
        self.k_negative = k_negative

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Temporary storage for a single group
        group_features: list[list[float]] = []
        group_labels: list[float] = []
        group_gids: list[int] = []
        current_gid: int | None = None


        def flush_group():
            """Apply resampling on the currently collected group and yield."""
            if not group_features:
                return
            labels_tensor = torch.tensor(group_labels, dtype=torch.float32)
            pos_idx = (labels_tensor == 1).nonzero(as_tuple=False).squeeze(1).tolist()
            if len(pos_idx) == 0:
                return  # skip group with no positive
            pos_idx = pos_idx[0]  # assume exactly one positive

            # get all negatives
            neg_indices = [i for i, lab in enumerate(group_labels) if lab == 0]

            # sample K negatives
            k = max(self.k_negative, int(len(neg_indices) * self.k_neg_ratio))
            if len(neg_indices) > k:
                sampled_negatives = random.sample(neg_indices, k)
            else:
                # Group too small, take all
                sampled_negatives = neg_indices

            keep_indices = [pos_idx] + sampled_negatives
            feats = torch.tensor([group_features[i] for i in keep_indices], dtype=torch.float32)
            labs = torch.tensor([group_labels[i] for i in keep_indices], dtype=torch.float32)
            gids = torch.tensor([group_gids[i] for i in keep_indices], dtype=torch.long)
            return feats, labs, gids

        # Iterate through each parquet
        for path in self.parquet_paths:
            scan = pl.scan_parquet(path).select(self.feature_cols + [self.label_col, self.group_col])
            for row in scan.collect(streaming=True).iter_rows(named=True):
                gid = abs(hash(row[self.group_col])) % (2**31)
                # new group?
                if current_gid is not None and gid != current_gid:
                    result = flush_group()
                    if result is not None:
                        yield result
                    group_features, group_labels, group_gids = [], [], []

                # normalize features
                norm_feat = [
                    (row[c] - self.mean_dict[c]) / (self.std_dict[c] + 1e-6)
                    for c in self.feature_cols
                ]
                group_features.append(norm_feat)
                group_labels.append(row[self.label_col])
                group_gids.append(gid)
                current_gid = gid

        # final flush
        result = flush_group()
        if result is not None:
            yield result
            
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Iterator, Tuple

class ParquetRankDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: Path | str,
        feature_cols: list[str],
        label_col: str,
        group_col: str,
        max_rows: int,
    ) -> None:
        super().__init__()
        self.parquet_path: str = str(parquet_path)
        self.feature_cols: list[str] = feature_cols
        self.label_col: str = str(label_col)
        self.group_col: str = str(group_col)
        self.max_rows: int = int(max_rows)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        torch.IterableDataset implementation.

        - It should yield data samples (or batches) one by one.
        - Each `yield` is what the DataLoader will treat as one item

        Yield batches of features, labels, and group IDs. (ranker id)
        Stream data, so save memory
        """
        scan = pl.scan_parquet(self.parquet_path).select(self.feature_cols + [self.label_col, self.group_col])
        df_iter = scan.collect(streaming=True).iter_rows(named=True) # iterator, one row at a time, lazy
        
        current_features: List[List[float]] = []
        current_labels: List[float] = []
        current_groups: List[int] = []
        current_gid: int | None = None

        # Only flush the batch (yield) when:
        # 1. You've reached your desired size(self.max_rows)
        # 2. The next row belongs to a different group than current_gid
        # 1 && 2.
        for row in df_iter:
            gid = row[self.group_col]
            if current_gid is None:
                current_gid = gid
            # If adding this row would exceed max_rows but group changes? yield now
            if len(current_features) >= self.max_rows and gid != current_gid:
                yield (
                    torch.tensor(current_features, dtype=torch.float32),
                    torch.tensor(current_labels, dtype=torch.float32),
                    torch.tensor(current_groups, dtype=torch.long),
                )
                current_features, current_labels, current_groups = [], [], []
                current_gid = gid
            # Add row
            current_features.append([row[c] for c in self.feature_cols])
            current_labels.append(row[self.label_col])
            current_groups.append(gid)
            current_gid = gid

        # yield last batch
        if current_features:
            yield (
                torch.tensor(current_features, dtype=torch.float32),
                torch.tensor(current_labels, dtype=torch.float32),
                torch.tensor(current_groups, dtype=torch.long),
            )
        
        
# Usage
# ```python
# dataset: ParquetRankDataset = ParquetRankDataset(
#     parquet_path='data/train.parquet',
#     feature_cols=['feature1', 'feature2', 'feature3'],
#     label_col='label',
#     group_col='group',
#     batch_size=4096,
# )
# ```
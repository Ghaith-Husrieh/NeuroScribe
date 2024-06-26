import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import neuroscribe as ns

__all__ = ['DataLoader']


class DataLoader:
    def __init__(self, dataset, batch_size=None, shuffle=True, train=False, num_workers=0):
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("Batch size must be a positive integer or None")
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError(
                "Number of workers must be a non-negative integer")
        if num_workers > os.cpu_count():
            raise ValueError(
                "Number of workers cannot exceed the number of CPU cores")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.indices = ns.arange(len(dataset))
        self.train = train
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.dataset.to(device)

    def __len__(self):
        if self.batch_size is None:
            return 1
        else:
            batch_size = self.batch_size or self.indices.numel()
            return (self.indices.numel() + batch_size - 1) // batch_size

    def __iter__(self):
        if self.shuffle:
            ns.shuffle_(self.indices)

        if self.batch_size is None:
            batch_size = self.indices.numel()
        else:
            batch_size = self.batch_size

        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._load_batch, self.indices.data[i:i + batch_size])
                           for i in range(0, self.indices.numel(), batch_size)]
                for future in as_completed(futures):
                    batch = future.result()
                    if batch:
                        yield self._group_batch(batch)
        else:
            for i in range(0, self.indices.numel(), batch_size):
                batch = self._load_batch(self.indices.data[i:i + batch_size])
                if batch:
                    yield self._group_batch(batch)

    def _group_batch(self, batch):
        if not batch:
            return None

        if isinstance(batch[0], tuple):
            num_elements = len(batch[0])
            grouped = tuple(ns.tensor([item[i].data for item in batch], device=self.device,
                            requires_grad=self.train) for i in range(num_elements))
            return grouped

        elif isinstance(batch[0], dict):
            batch_dict = {}
            for key in batch[0].keys():
                batch_dict[key] = ns.tensor(
                    [item[key] for item in batch], requires_grad=self.train)
            return batch_dict

    def _load_batch(self, batch_indices):
        return [self.dataset[idx] for idx in batch_indices if (item := self._load_item(idx)) is not None]

    def _load_item(self, idx):
        try:
            return self.dataset[idx]
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return None

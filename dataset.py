import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import itertools


class interger_sort_dataset(Dataset):
    def __init__(self, num_sample=10000, low=0, high=100, min_len=1, max_len=10, seed=1):
        self.prng = np.random.RandomState(seed=seed)
        self.input_dim = high

        self.seqs = [list(map(lambda x: [x], self.prng.choice(np.arange(low, high), size=self.prng.randint(min_len, max_len+1)).tolist())) for _ in range(num_sample)]
        self.labels = [sorted(range(len(seq)), key=seq.__getitem__) for seq in self.seqs]

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = self.labels[index]

        len_seq = len(seq)
        row_col_index = list(zip(* [(i, number) for i, numbers in enumerate(seq) for number in numbers]))
        num_values = len(row_col_index[0])

        i = torch.LongTensor(row_col_index)
        v = torch.FloatTensor([1] * num_values)
        data = torch.sparse.FloatTensor(i, v, torch.Size([len_seq, self.input_dim]))

        return data, len_seq, label

    def __len__(self):
        return len(self.seqs)


def sparse_seq_collate_fn(batch):
    batch_size = len(batch)

    sorted_seqs, sorted_lengths, sorted_labels = zip(* sorted(batch, key=lambda x: x[1], reverse=True))
    padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]
    # * the size of seq is changed to be [batch_size, max_seq_len, high_dimension(one-hot)]

    seq_tensor = torch.stack(padded_seqs)

    length_tensor = torch.LongTensor(sorted_lengths)

    padded_labels = list(zip(* (itertools.zip_longest(* sorted_labels, fillvalue=-1))))

    label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)
    # * label tensor: [batch_size, max_seq_len(padding with -1)]

    seq_tensor = seq_tensor.to_dense()
    # * the coo sparse matrix of seq_tensor is change to be a dense matrix
    return seq_tensor, length_tensor, label_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from net import pointer_net
from dataset import interger_sort_dataset, sparse_seq_collate_fn
from torch.utils.data import DataLoader


class average_meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def masked_accuracy(output, target, mask):
    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()

        return accuracy


if __name__ == '__main__':
    batch_size = 256
    train_samples = 100000
    test_samples = 1000
    embedding_dim = 8
    high = 100
    low = 0
    min_len = 5
    max_len = 10
    num_workers = 1
    learning_rate = 1e-3
    weight_decay = 1e-5
    epochs = 100
    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")

    train_set = interger_sort_dataset(num_sample=train_samples, high=high, min_len=min_len, max_len=max_len, seed=1)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=sparse_seq_collate_fn)

    test_set = interger_sort_dataset(num_sample=test_samples, high=high, min_len=min_len, max_len=max_len, seed=2)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=sparse_seq_collate_fn)

    model = pointer_net(input_dim=high,
                        embedding_dim=embedding_dim,
                        hidden_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    train_loss = average_meter()
    train_accuracy = average_meter()
    test_loss = average_meter()
    test_accuracy = average_meter()

    for epoch in range(epochs):
        # Train
        model.train()
        for batch_idx, (seq, length, target) in enumerate(train_loader):
            seq, length, target = seq.to(device), length.to(device), target.to(device)

            optimizer.zero_grad()
            log_pointer_score, argmax_pointer, mask = model(seq, length)

            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            train_accuracy.update(
                masked_accuracy(argmax_pointer, target, mask).item(),
                mask.int().sum().item())

            if batch_idx % 20 == 0:
                print(
                    'Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
                    .format(epoch, batch_idx * len(seq),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            train_loss.avg, train_accuracy.avg))

        # Test
        model.eval()
        for seq, length, target in test_loader:
            seq, length, target = seq.to(device), length.to(device), target.to(
                device)

            log_pointer_score, argmax_pointer, mask = model(seq, length)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            test_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            test_accuracy.update(
                masked_accuracy(argmax_pointer, target, mask).item(),
                mask.int().sum().item())
        print('Epoch {}: Test\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
            epoch, test_loss.avg, test_accuracy.avg))

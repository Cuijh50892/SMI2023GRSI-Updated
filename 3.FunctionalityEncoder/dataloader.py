import torch
from dataset import ManifoldDataset
from dataset import PairDataset
from dataset import VisualizeDataset


def get_dataloader(split='S'):
    dataset = ManifoldDataset(
        split=split,
    )

    batch_size = dataset.__len__()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )

    return dataloader

def get_pair_data_loader(b_size=8):
    dataset = PairDataset()

    batch_size = b_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    return dataloader

def get_visualizeloader(split='S'):
    dataset = VisualizeDataset(
        split=split,
    )

    batch_size = dataset.__len__()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )

    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader(split='S')
    print("dataloader size:", dataloader.dataset.__len__())


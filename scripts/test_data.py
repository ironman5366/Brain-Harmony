from torch.utils.data import DataLoader
from brain_datasets.datasets import MedarcDataset
from tqdm import tqdm
from pathlib import Path


def main():
    ds = MedarcDataset(statistic_dir=Path("statistics-cache"), split="all")
    dl = DataLoader(ds)

    for row in tqdm(dl):
        pass
        # print(row)
        # print(len(row))
        # a, b = row
        # print(a.shape)
        # print(b.shape)
        # break


if __name__ == "__main__":
    main()

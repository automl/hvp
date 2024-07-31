import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.model_selection import cross_val_score, train_test_split


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, x_file, y_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = torch.tensor(pd.read_csv(x_file).to_numpy(), dtype=torch.float32).cuda()
        self.y = torch.tensor(pd.read_csv(y_file).to_numpy().squeeze(), dtype=torch.long).cuda()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(61, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    w_dir = "../../metrics-data"
    x_path = os.path.join(w_dir, "X-simsiam-minsim-normalized-seed0.csv")
    y_path = os.path.join(w_dir, "Y-simsiam-minsim-normalized-seed0.csv")
    dataset = CSVDataset(x_path, y_path)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=0)

    model = TinyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        with tqdm(dataloader) as pbar:
            for x, y in pbar:
                # x = x.cuda()
                # y = y.cuda()
                logit = model(x)
                loss = torch.nn.functional.cross_entropy(logit, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f"Epoch {epoch} loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        acc = []
        for x, y in dataloader:
            logit = model(x)
            acc.append((logit.argmax(dim=1) == y).float().mean().item())
        print(np.mean(acc))


    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #
    # clf = HistGradientBoostingClassifier(verbose=True)
    # # clf.fit(X_train, y_train)
    # # score = clf.score(X_test, y_test)
    # scores = cross_val_score(clf, X, Y, cv=5, verbose=1)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    # 0.41 accuracy with a standard deviation of 0.01
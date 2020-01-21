import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MakeDataset(Dataset):
    def __init__(self, data):
        self.strings = list(data['string'])
        self.valid = list(data['valid'])
        self.len = len(self.valid)
        self.valid_list = [0, 1]

    def __getitem__(self, index):
        return self.strings[index], self.valid[index]

    def __len__(self):
        return self.len


def data_loaders(batch_size):
	train = MakeDataset(train_data)
	test = MakeDataset(test_data)
	train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader


train_data = pd.read_csv('../../dataset/train_data.csv')
test_data = pd.read_csv('../../dataset/test_data.csv')
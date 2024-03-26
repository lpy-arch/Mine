# pylint: disable=E1102

# feature:[batch_size, sequence_length]
# target:[batch_size]

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from encode import encode, SEQUENCE_LENGTH
from parse import SAVE_DIR

FEWER_DATA = 0.01
USE_ONE_HOT = True

# custom dataset settings
class SongsDataset(Dataset):
    # the input of __init__ are the same input as we init the class
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.int_songs = encode(dataset_path=SAVE_DIR)  # the encoded int type song
        self.sequence_length = sequence_length  # as the length of the sliding window
        self.num_sequences = round((len(self.int_songs) - self.sequence_length)*FEWER_DATA)  # as the number of data-sets

    def __len__(self):
        return self.num_sequences   # the pairs of data

    def __getitem__(self, idx):
        # get the songs and targets

        # create the sliding window to generate the training data
        inputs = []
        targets = []
        for i in range(self.num_sequences):
            inputs.append(self.int_songs[i:i+self.sequence_length])
            targets.append(self.int_songs[i+self.sequence_length])

        # songs and targets are in seprate lists, use "idx" as index to make sure they are matched
        song = inputs[idx]
        target = targets[idx]

        # make sure song and target are "tensor" type both
        song = torch.tensor(song)
        target = torch.tensor(target)

        # switch the song into one-hot encoding
        if USE_ONE_HOT == True:
            song = F.one_hot(song, num_classes=38)
        else:
            pass

        # return "song, label" as the return of the dataset class
        return song, target


if __name__ == "__main__":
    # load the dataloader
    train_data = SongsDataset()
    train_dataloader = DataLoader(train_data, batch_size=32)

    # print the size of the data in dataloader
    train_features, train_labels = next(iter(train_dataloader))

    train_features = train_features.permute(1, 0, 2)

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    data = [[1,2],
            [3,4],
            [5,6]]
    tensor = torch.tensor(data)
    print(tensor.size())

    tensor = tensor.permute(1, 0)
    print(tensor.size())
    print(tensor)

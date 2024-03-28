import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from traindata import SongsDataset
from encode import SEQUENCE_LENGTH

EPOCHS = 3
BATCH_SIZES = 32
LEARNING_RATE = 1e-3
INPUT_SIZE = 38
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 38

SAVE_MODEL_PATH = "./models/LSTM_model.pth"


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        # init the h0 and c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))

        # dropout layer
        out = nn.Dropout(0.2)(out)
        
        # pass the Linear_Relu stack
        out = self.linear_relu_stack(out)

        # get only the last time-step of the sequence of the output
        out = out[:, -1, :]
        
        return out
    

def train(dataloader, model, writer):
    # Get the total number of samples in the dataset
    size = len(dataloader.dataset)

    # Set the model to training mode
    model.train()

    # Iterate over the batches in the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # init loss_fn and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Compute prediction error
        X = X.float()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # TensorBoard log gradient
        for gradient, param in model.named_parameters():
            writer.add_histogram(gradient, param.grad, global_step=0)

        optimizer.step()
        optimizer.zero_grad()

        # TensorBoard log loss
        writer.add_scalar("Loss/train", loss, batch)
        writer.flush()  # refresh the writer

        # Check if the current batch number is divisible by 100
        # If true, print the training loss and the number of processed samples
        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            


def train_test_save_model(model, train_dataloader):
    # init TensorBoard writer
    writer = SummaryWriter('TensorBoard_Logs')

    # train and test the model
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, writer)
    print("Done!")

    # close TensorBoard writer
    writer.close()

    # save the model
    # .pt/.pth/.pkl all the same
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Saved PyTorch Model State to: ", SAVE_MODEL_PATH)


if __name__ == "__main__":
    # load the dataloader
    train_data = SongsDataset()
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZES)

    # print the size of the data in dataloader
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # init the model
    model = LSTM()
    print(model)

    # train、save the model
    # train_test_save_model(model, train_dataloader)

    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    with torch.no_grad():
        pred = model(train_features.float())
        print(train_features.size())
        print(pred.size())



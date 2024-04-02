# pylint: disable=E1102

import torch
import json
from datetime import datetime
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from traindata import SongsDataset
from encode import SEQUENCE_LENGTH, MAPPING_PATH

EPOCHS = 3
BATCH_SIZES = 64
LEARNING_RATE = 1e-3
INPUT_SIZE = 38
HIDDEN_SIZE = 256
NUM_LAYERS = 2
OUTPUT_SIZE = 38
DROP_OUT = 0.2

TRAIN_MODEL = True

SAVE_MODEL_PATH = "./models/LSTM_model.pth"


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=DROP_OUT if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # init the h0 and c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # pass the Linear_Relu stack
        out = self.fc(out[:, -1, :])

        # get only the last time-step of the sequence of the output
        # the out's shape is [batch_size, sequence_length, hidden_size]
        # so we only use the last element of "sequence_length" (as the prediction), and keep other dimensions
        # out = out[:, -1, :]
        
        return out
    

def train(dataloader, model, writer, loss_fn, optimizer):
    # Get the total number of samples in the dataset
    size = len(dataloader.dataset)

    # Set the model to training mode
    model.train()

    # Iterate over the batches in the dataloader
    for batch, (X, y) in enumerate(dataloader):
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

        # Check if the current batch number is divisible by 100
        # If true, print the training loss and the number of processed samples

        # if batch % 100 == 0:
        # loss, current = loss.item(), (batch + 1) * len(X)
        # print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        loss, current = loss.item(), (batch + 1)
        mother = int(28969/BATCH_SIZES)
        print(f"Train loss: {loss:>7f}  [{current:>5d}/{mother:>5d}]")

        # TensorBoard log loss
        writer.add_scalar("Loss/train", loss, batch)
        writer.flush()  # refresh the writer

            
def test(dataloader, model, writer, loss_fn, epochs):
    # Get the total number of samples in the dataset
    size = len(dataloader.dataset)

    # Get the total number of batches in the dataloader
    num_batches = len(dataloader)

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for test loss and correct predictions
    test_loss, correct = 0, 0
    
    # Turn off gradients during testing to save computational resources
    with torch.no_grad():
        for X, y in dataloader:
            # Perform forward pass
            pred = model(X)

            # Compute the test loss
            test_loss += loss_fn(pred, y).item()

            # Count the number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Calculate average test loss
    test_loss /= num_batches

    # Calculate accuracy
    correct /= size

    # TensorBoard log loss
    writer.add_scalar("Loss/test", test_loss, epochs)
    writer.add_scalar("Accuracy", (100*correct), epochs)
    writer.flush()  # refresh the writer

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def train_test_save_model(model, train_dataloader, test_dataloader):
    # set the loss_func and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # init TensorBoard writer
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d-%H%M")
    writer = SummaryWriter(f'TensorBoard_Logs/{folder_name}')

    # train and test the model
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, writer, loss_fn, optimizer)
        test(test_dataloader, model, writer, loss_fn, t)
    print("Done!")

    # close TensorBoard writer
    writer.close()

    # save the model
    # .pt/.pth/.pkl all the same
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Saved PyTorch Model State to: ", SAVE_MODEL_PATH)

def test_model_inoutput(model, train_features):
    with torch.no_grad():
        pred = model(train_features)

        lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        print("LSTM model: ", lstm)

        print("Input shape: ", train_features.shape)
        print("Model Output shape: ", pred.shape)

        out_from_lstm, (hn, cn) = lstm(train_features)
        print("hn shape: ", hn.shape)
        print("cn shape: ", cn.shape)
        print("Out_from_lstm shape: ", out_from_lstm.shape)

        linear_relu_stack = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, INPUT_SIZE),
        nn.ReLU())
        out_from_stack = linear_relu_stack(out_from_lstm)
        print("out_from_stack shape: ", out_from_stack.shape)

def test_interference_inoutput(model):
    seed = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    seed = seed.split()
    melody = seed

    _start_symbols = ["/"] * SEQUENCE_LENGTH
    seed = _start_symbols + seed

    with open(MAPPING_PATH, "r") as fp:
        _mappings = json.load(fp)
    seed = [_mappings[symbol] for symbol in seed]

    max_seed_length = 64
    seed = seed[-max_seed_length:]
    print(seed)

    seed = torch.tensor(seed)
    onehot_seed = F.one_hot(seed, num_classes=38)
    print(onehot_seed.size())
    onehot_seed = onehot_seed.unsqueeze(0)
    print(onehot_seed.size())

    onehot_seed = onehot_seed.float()
    probabilities = model(onehot_seed)[0]
    probabilities = probabilities.detach().numpy()
    print(probabilities.shape)


if __name__ == "__main__":
    # load the dataloader
    train_data = SongsDataset(train=True)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZES, shuffle=True, num_workers=4)

    test_data = SongsDataset(train=False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZES, shuffle=True, num_workers=4)

    # print the size of the data in dataloader
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # init the model
    model = LSTM()
    print(model)


    if TRAIN_MODEL:
        # train、save the model
        train_test_save_model(model, train_dataloader, test_dataloader)
    else:
        # load the model
        model.load_state_dict(torch.load(SAVE_MODEL_PATH))
        model.eval()

        # test the input and output of model and each layers
        # test_model_inoutput(model, train_features)

        # test the interference step
        # test_interference_inoutput(model)
        
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from traindata import SongsDataset
from encode import SEQUENCE_LENGTH

EPOCHS = 10
BATCH_SIZES = 64
LEARNING_RATE = 1e-3
INPUT_SIZE = 38
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 38

TRAIN_MODEL = True

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
        # the out's shape is [batch_size, sequence_length, hidden_size]
        # so we only use the last element of "sequence_length" (as the prediction), and keep other dimensions
        out = out[:, -1, :]
        
        return out
    

def train(dataloader, model, writer, loss_fn, optimizer):
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

        # Check if the current batch number is divisible by 100
        # If true, print the training loss and the number of processed samples
        if batch % 30 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    
    # init loss_fn
    loss_fn = nn.CrossEntropyLoss()
    
    # Turn off gradients during testing to save computational resources
    with torch.no_grad():
        for X, y in dataloader:
            # Perform forward pass
            X = X.float()
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
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # init TensorBoard writer
    writer = SummaryWriter('TensorBoard_Logs')

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


if __name__ == "__main__":
    # load the dataloader
    train_data = SongsDataset(train=True)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZES)

    test_data = SongsDataset(train=False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZES)

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
        # test the input and output of model and each layers
        model.load_state_dict(torch.load(SAVE_MODEL_PATH))
        model.eval()

        with torch.no_grad():
            pred = model(train_features.float())

            lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
            print("LSTM model: ", lstm)

            print("Input shape: ", train_features.shape)
            print("Model Output shape: ", pred.shape)

            out_from_lstm, (hn, cn) = lstm(train_features.float())
            print("hn shape: ", hn.shape)
            print("cn shape: ", cn.shape)
            print("Out_from_lstm shape: ", out_from_lstm.shape)

            linear_relu_stack = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, INPUT_SIZE),
            nn.ReLU())
            out_from_stack = linear_relu_stack(out_from_lstm)
            print("out_from_stack shape: ", out_from_stack.shape)


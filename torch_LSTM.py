import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from traindata import SongsDataset
from encode import SEQUENCE_LENGTH

BATCH_SIZES = 64
LEARNING_RATE = 1e-3
EPOCHS = 30
SAVE_MODEL_PATH = "./models/LSTM_model.pth"


# VAE model
class VAE(nn.Module):
    def __init__(self, sequence_length=SEQUENCE_LENGTH, h_dim=32, z_dim=16):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(sequence_length, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, sequence_length)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    def forward(self, x):
        # the input and the network must have the same dtype, for instance:float
        x = x.to(torch.float)

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
    

def train(dataloader, model):
    # init TensorBoard writer
    writer = SummaryWriter()

    # Get the total number of samples in the dataset
    size = len(dataloader.dataset)

    # Set the model to training mode
    model.train()

    # Iterate over the batches in the dataloader
    for batch, (X, _) in enumerate(dataloader):
        # Compute prediction error
        x_reconst, mu, log_var = model(X)

        # the input and the network must have the same dtype, for instance:float
        X = X.to(torch.float)

        # calculate the reconst_loss and kl_div
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        reconst_loss = nn.CrossEntropyLoss()(x_reconst, X)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TensorBoard log scale
        writer.add_scalar("Reconst_Loss/train", loss, batch)
        writer.add_scalar("KL_div/train", kl_div, batch)

        # Check if the current batch number is divisible by 100
        # If true, print the training loss and the number of processed samples
        if (batch+1) % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print ("Reconst Loss: {:.4f}, KL Div: {:.4f}, [{:>5d}/{:>5d}]" 
                   .format(reconst_loss.item(), kl_div.item(), current, size))
            
    # flush TensorBoard writer
    writer.flush()


def train_test_save_model(model, train_dataloader):
    # train and test the model
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model)
    print("Done!")

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
    model = VAE()
    print(model)

    # train、save the model
    train_test_save_model(model, train_dataloader)

    # loading models
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # rand a tensor as the "z" as the input of decoder
        shape = (BATCH_SIZES, 16)
        x = torch.rand(shape)
        # put z in decoder to generate a result
        pred = model.decode(x)
        print(x)
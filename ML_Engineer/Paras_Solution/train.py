import torch
import config
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import SobelOperator
from dataset import SobelDataset


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = SobelDataset("train", device)
    test_data = SobelDataset("test", device)
    
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = SobelOperator().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.lr)


    for epoch in tqdm(range(config.EPOCHS)):
        print(f"Epoch: {epoch}\n-------")
        ### Training
        train_loss = 0
        for X, y in train_dataloader:
            model.train() 
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)

        test_loss = 0 
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y) 
            test_loss /= len(test_dataloader)

        ## Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

    kernels = list(model.parameters())
    print("Learned kernels")
    print("Gx: ", kernels[0])
    print("Gy: ", kernels[1])

if __name__ == "__main__":
    train()
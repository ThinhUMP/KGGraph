import math
import numpy
import torch
from torch import nn
from tqdm import tqdm

import torch
import numpy
import math
from torch import nn
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
def training(train_loader, model, criterion, optimizer, device=device):
    model.train()
    model.to(device)
    current_loss = 0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        data.x = data.x.float()

        output = model(data)

        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
<<<<<<< HEAD
        current_loss += loss / len(train_loader)
=======
        current_loss += loss.item() / len(train_loader)
>>>>>>> 27b41571b57fea2c7cbe59ae8d2c8bc72b85497f
        loss.backward()
        optimizer.step()
    return current_loss, model

def validation(val_loader, model, criterion, device=device):
    model.eval()
    val_loss = 0
<<<<<<< HEAD
    for data in val_loader:
        out = model(data)
        loss = criterion(out, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        val_loss += loss / len(val_loader)
    return val_loss

@torch.no_grad()
def testing(test_loader, model):
    """Testing

    Args:
        test_loader (DataLoader): test dataset
        model (nn.Module): trained model

    Returns:
        float: test loss
    """
=======
    for data in tqdm(val_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        val_loss += loss.item() / len(val_loader)
    return val_loss

@torch.no_grad()
def testing(test_loader, model, device=device):
    model.to(device)
>>>>>>> 27b41571b57fea2c7cbe59ae8d2c8bc72b85497f
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss = 0
    test_target = numpy.empty((0))
    test_y_target = numpy.empty((0))
<<<<<<< HEAD
    for data in test_loader:
        out = model(data)
        # NOTE
        # out = out.view(d.y.size())
        loss = criterion(out, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        test_loss += loss / len(test_loader)
=======
    for data in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        test_loss += loss.item() / len(test_loader)
>>>>>>> 27b41571b57fea2c7cbe59ae8d2c8bc72b85497f

        test_target = numpy.concatenate((test_target, output.cpu().detach().numpy()[:, 0]))
        test_y_target = numpy.concatenate((test_y_target, data.y.cpu().detach().numpy()))

    return test_loss, test_target, test_y_target

def train_epochs(epochs, model, train_loader, val_loader, path, device=device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

<<<<<<< HEAD
    Args:
        epochs (int): number of epochs to train for
        model (nn.Module): the current model
        train_loader (DataLoader): training data in batches
        val_loader (DataLoader): validation data in batches
        path (string): path to save the best model

    Returns:
        array: returning train and validation losses over all epochs, prediction and ground truth values for training data in the last epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_target = numpy.empty((0))
    train_y_target = numpy.empty((0))
=======
>>>>>>> 27b41571b57fea2c7cbe59ae8d2c8bc72b85497f
    train_loss = numpy.empty(epochs)
    val_loss = numpy.empty(epochs)
    best_loss = math.inf

    for epoch in range(epochs):
        epoch_loss, model = training(train_loader, model, criterion, optimizer)
        v_loss = validation(val_loader, model, criterion)
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), path)

        train_loss[epoch] = epoch_loss
        val_loss[epoch] = v_loss

        if epoch % 2 == 0:
            print(f"Epoch: {epoch}, Train loss: {epoch_loss}, Val loss: {v_loss}")
    return train_loss, val_loss


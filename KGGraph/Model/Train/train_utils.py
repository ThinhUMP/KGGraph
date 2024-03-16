import math
import numpy
import torch
from torch import nn

def training(train_loader, model, criterion, optimizer):
    """Training one epoch

    Args:
        train_loader (DataLoader): loader (DataLoader): training data divided into batches
        model (nn.Module): GNN model to train on
        criterion (nn.functional): loss function to use during training
        optimizer (torch.optim): optimizer during training

    Returns:
        float: training loss
    """
    model.train()

    current_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data.x = data.x.float()

        output = model(data)

        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        current_loss += loss / len(train_loader)
        loss.backward()
        optimizer.step()
    return current_loss, model

def validation(val_loader, model, criterion):
    """Validation

    Args:
        loader (DataLoader): validation set in batches
        model (nn.Module): current trained model
        criterion (nn.functional): loss function

    Returns:
        float: validation loss
    """
    model.eval()
    val_loss = 0
    for data in val_loader:
        out = model(data)
        loss = criterion(out, torch.reshape(data.y, (len(data.y), 1)))
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
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss = 0
    test_target = numpy.empty((0))
    test_y_target = numpy.empty((0))
    for data in test_loader:
        out = model(data)
        # NOTE
        # out = out.view(d.y.size())
        loss = criterion(out, torch.reshape(data.y, (len(data.y), 1)))
        test_loss += loss / len(test_loader)

        # save prediction vs ground truth values for plotting
        test_target = numpy.concatenate((test_target, out.detach().numpy()[:, 0]))
        test_y_target = numpy.concatenate((test_y_target, data.y.detach().numpy()))

    return test_loss, test_target, test_y_target

def train_epochs(epochs, model, train_loader, val_loader, path):
    """Training over all epochs

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
    train_loss = numpy.empty(epochs)
    val_loss = numpy.empty(epochs)
    best_loss = math.inf

    for epoch in range(epochs):
        epoch_loss, model = training(train_loader, model, criterion, optimizer)
        v_loss = validation(val_loader, model, criterion)
        if v_loss < best_loss:
            torch.save(model.state_dict(), path)
        for data in train_loader:
            out = model(data)
            if epoch == epochs - 1:
                # record truly vs predicted values for training data from last epoch
                train_target = numpy.concatenate((train_target, out.detach().numpy()[:, 0]))
                train_y_target = numpy.concatenate((train_y_target, data.y.detach().numpy()))

        train_loss[epoch] = epoch_loss.detach().numpy()
        val_loss[epoch] = v_loss.detach().numpy()

        # print current train and val loss
        if epoch % 2 == 0:
            print(
                "Epoch: "
                + str(epoch)
                + ", Train loss: "
                + str(epoch_loss.item())
                + ", Val loss: "
                + str(v_loss.item())
            )
    return train_loss, val_loss, train_target, train_y_target


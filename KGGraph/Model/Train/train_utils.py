import torch
import numpy
import math
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def training(train_loader, model, criterion, optimizer):
    device = torch.device("cpu" if torch.cuda.is_available() else torch.device("cpu"))

    model.train()
    model.to(device)
    current_loss = 0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        output = model(data)

        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        current_loss += loss.item() / len(train_loader)
        loss.backward()
        optimizer.step()
    return current_loss, model

def validation(val_loader, model, criterion):
    device = torch.device("cpu" if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    for data in tqdm(val_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        val_loss += loss.item() / len(val_loader)

        all_targets.extend(data.y.detach().cpu().numpy())
        all_outputs.extend(output.detach().cpu().numpy())

    val_auc = roc_auc_score(all_targets, all_outputs)
    return val_loss, val_auc

@torch.no_grad()
def testing(test_loader, model):
    device = torch.device("cpu" if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss = 0
    all_targets = []
    all_outputs = []
    for data in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, torch.reshape((data.y+1)/2, (len(data.y), 1)))
        test_loss += loss.item() / len(test_loader)

        all_targets.extend(data.y.cpu().detach().numpy())
        all_outputs.extend(output.cpu().detach().numpy())

    test_auc = roc_auc_score(all_targets, all_outputs)
    return test_loss, test_auc, all_outputs, all_targets

def train_epochs(epochs, model, train_loader, val_loader, path):
    device = torch.device("cpu" if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loss = numpy.empty(epochs)
    val_loss = numpy.empty(epochs)
    val_aucs = numpy.empty(epochs)
    best_loss = math.inf

    for epoch in range(epochs):
        _, model = training(train_loader, model, criterion, optimizer)
        v_loss, v_auc = validation(val_loader, model, criterion)
        val_loss[epoch] = v_loss
        val_aucs[epoch] = v_auc

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), path)

        if epoch % 2 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss[epoch]}, Val loss: {v_loss}, Val AUC: {v_auc}")

    return train_loss, val_loss, val_aucs


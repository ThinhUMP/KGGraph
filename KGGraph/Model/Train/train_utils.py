import torch
import numpy as np
import math
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy
def training(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        data.y = data.y.float()
        output = model(data)

        loss = criterion(output, torch.reshape(data.y, (len(data.y), 1)))
        train_loss += loss/len(train_loader)
        loss.backward()
        optimizer.step()
    return train_loss

def validation(val_loader, model, criterion, device):
    
    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    all_predictions = []
    for data in tqdm(val_loader):
        data = data.to(device)
        data.y = data.y.float()
        output = model(data)
        loss = criterion(output, torch.reshape(data.y, (len(data.y), 1)))
        val_loss += loss/len(val_loader)
        
        all_targets.extend(data.y.detach().cpu().numpy())
        all_outputs.extend(output.detach().cpu().numpy())
        
        predictions = (output.detach().cpu().numpy() > 0.5).astype(int)
        all_predictions.extend(predictions)
    val_f1 = f1_score(all_targets, all_predictions)
    val_ap = average_precision_score(all_targets, all_outputs)
    val_auc = roc_auc_score(all_targets, all_outputs)
    return val_loss, val_auc, val_f1, val_ap

@torch.no_grad()
def testing(test_loader, model, criterion, device):
    test_loss = 0
    all_targets = []
    all_outputs = []
    for data in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, torch.reshape(data.y, (len(data.y), 1)))
        test_loss += loss/len(test_loader)

        all_targets.extend(data.y.cpu().detach().numpy())
        all_outputs.extend(output.cpu().detach().numpy())

    test_auc = roc_auc_score(all_targets, all_outputs)
    return test_loss, test_auc, all_outputs, all_targets

def train_epochs(epochs, model, train_loader, val_loader, path):
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.BCELoss()
    best_loss = math.inf

    for epoch in range(epochs):
        train_loss = training(train_loader, model, criterion, optimizer, device)
        v_loss, v_auc, v_f1, v_ap = validation(val_loader, model, criterion, device)
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), path)

        # if epoch % 2 == 0:
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {v_loss},\n Val AUC: {v_auc}, Val F1: {v_f1}, Val AP: {v_ap}")

    # return train_loss, val_loss, val_aucs, val_f1, v_ap


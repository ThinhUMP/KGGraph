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
    all_targets = []
    all_outputs = []
    all_predictions = []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        data.y = data.y.float()
        output = model(data)

        loss = criterion(output, torch.reshape(data.y, (len(data.y), 1)))
        train_loss += loss.detach().cpu().item()/len(train_loader)
        loss.backward()
        optimizer.step()
        
        all_targets.extend(data.y.detach().cpu().numpy())
        all_outputs.extend(output.detach().cpu().numpy())
        
        predictions = (output.detach().cpu().numpy() >= 0.5).astype(float)
        all_predictions.extend(predictions)
    train_auc = roc_auc_score(all_targets, all_outputs)
    train_f1 = f1_score(all_targets, all_predictions)
    train_ap = average_precision_score(all_targets, all_outputs)
    
    return train_loss, train_auc, train_f1, train_ap

def validation(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    all_predictions = []
    for data in tqdm(val_loader):
        data = data.to(device)
        data.y = data.y.float()
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, torch.reshape(data.y, (len(data.y), 1)))
        val_loss += loss.detach().cpu().item()/len(val_loader)
        
        all_targets.extend(data.y.detach().cpu().numpy())
        all_outputs.extend(output.detach().cpu().numpy())
        
        predictions = (output.detach().cpu().numpy() >= 0.5).astype(float)
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
    all_predictions = []
    for data in tqdm(test_loader):
        data = data.to(device)
        data.y = data.y.float()
        output = model(data)
        loss = criterion(output, torch.reshape(data.y, (len(data.y), 1)))
        test_loss += loss/len(test_loader)

        all_targets.extend(data.y.detach().cpu().numpy())
        all_outputs.extend(output.detach().cpu().numpy())
        predictions = (output.detach().cpu().numpy() >= 0.5).astype(float)
        all_predictions.extend(predictions)
    test_auc = roc_auc_score(all_targets, all_outputs)
    test_f1 = f1_score(all_targets, all_predictions)
    test_ap = average_precision_score(all_targets, all_outputs)
    
    return test_loss, test_auc, test_f1, test_ap

def train_epochs(epochs, model, train_loader, val_loader, path):
    device = torch.device("cpu" if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = nn.BCELoss()
    best_loss = math.inf
    train_loss_list = []
    train_auc_list = []
    train_f1_list = []
    train_ap_list = []
    val_loss_list = []
    val_auc_list = []
    val_f1_list = []
    val_ap_list = []
    for epoch in range(epochs):
        train_loss, train_auc, train_f1, train_ap = training(train_loader, model, criterion, optimizer, device)
        val_loss, val_auc, val_f1, val_ap = validation(val_loader, model, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path)
        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)
        train_f1_list.append(train_f1)
        train_ap_list.append(train_ap)
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc)
        val_f1_list.append(val_f1)
        val_ap_list.append(val_ap)
        print(f"Epoch: {epoch}")
        print(f"Train loss: {train_loss}, Train auc: {train_auc}, Train F1: {train_f1}, Train AP: {train_ap}")
        print(f"Val loss: {val_loss}, Val auc: {val_auc}, Val F1: {val_f1}, Val AP: {val_ap}")
    return train_loss_list, train_auc_list, train_f1_list, train_ap_list, val_loss_list, val_auc_list, val_f1_list, val_ap_list
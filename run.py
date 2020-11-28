import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TCPDataset
from model import GraphSAGE
from utils import collate, plot_values


def train(model, train_loader, val_loader, n_epochs, device):

    f = open("results.log", "w")
    train_accuracies, val_accuracies, train_losses, val_losses = [], [], [], []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        train_acc, val_acc, train_loss, val_loss = 0.0, 0.0, 0.0, 0.0
        # TRAIN MODE #
        model.train()
        for batch_id, (batched_graph, labels) in enumerate(tqdm(train_loader, desc='Training')):
            num_batches = batch_id + 1
            labels = labels.to(device)
            feats = batched_graph.ndata['attr']
            logits = model(batched_graph.to(device), feats.to(device))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # element-wise sigmoid to logits & transform values to either 0 or 1
            sigvals = torch.sigmoid(logits.data)
            sigvals[sigvals >= threshold] = 1.
            sigvals[sigvals < threshold] = 0.

            # compare sigval predictions to ground-truth labels (torch.equal() compares matrix to matrix)
            num_correct = sum([torch.equal(pred, label) for pred, label in zip(sigvals, labels)])
            acc = num_correct / len(labels)
            train_loss += loss.item()
            train_acc += acc
            del loss
        
        avg_train_loss = train_loss / num_batches
        avg_train_acc = train_acc / num_batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        # VALIDATION MODE #
        model.eval()
        with torch.no_grad():
            for batch_id, (batched_graph, labels) in enumerate(tqdm(val_loader, desc='Validating')):
                num_batches = batch_id + 1
                labels = labels.to(device)
                feats = batched_graph.ndata['attr']
                logits = model(batched_graph.to(device), feats.to(device))
                loss = criterion(logits, labels)

                sigvals = torch.sigmoid(logits.data)
                sigvals[sigvals >= threshold] = 1.
                sigvals[sigvals < threshold] = 0.
                
                num_correct = sum([torch.equal(pred, label) for pred, label in zip(sigvals, labels)])
                acc = num_correct / len(labels)
                val_loss += loss.item()
                val_acc += acc
                del loss

        avg_val_loss = val_loss / num_batches
        avg_val_acc = val_acc / num_batches
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        print(f"Epoch: {epoch+1}\tAvg Train Loss: {avg_train_loss}\tAvg Train Acc: {avg_train_acc}")
        print(f"Epoch: {epoch+1}\tAvg Val Loss: {avg_val_loss}\tAvg Val Acc: {avg_val_acc}")
        f.write(f"Epoch: {epoch+1}\tAvg Train Loss: {avg_train_loss}\tAvg Train Acc: {avg_train_acc}\n")
        f.write(f"Epoch: {epoch+1}\tAvg Val Loss: {avg_val_loss}\tAvg Val Acc: {avg_val_acc}\n")

    f.close()
    return train_accuracies, val_accuracies, train_losses, val_losses


if __name__ == "__main__":
    # Reproduce same results
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ### TRAINING HYPERPARAMETERS ###
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)
    multi_gpu = False
    n_epochs = 10
    learning_rate = 1e-2
    weight_decay = 5e-4
    batch_size = 16
    threshold = 0.8                 # if sigmoid prob >= 0.8, predict that class

    ### GraphSAGE HYPERPARAMETERS ###
    in_feats = 5                    # number of node features
    n_hidden = 16
    out_feats = 16
    n_classes = 25                  # number of classes must be FIXED @ 25
    n_layers = 2
    activation = F.relu
    dropout = 0.5
    aggregator_type = "gcn"         # OPTIONS: mean, gcn, pool, lstm
    graph_pooling_type = "mean"     # OPTIONS: mean, max, sum

    # Load & store data into memory
    train_data = TCPDataset(raw_dir="./train")
    val_data = TCPDataset(raw_dir='./valid_query')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Load model
    model_name = "graphsage"
    model = GraphSAGE(in_feats, n_hidden, out_feats, n_classes, n_layers, activation, dropout, aggregator_type, graph_pooling_type)
    if multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    # Train
    train_accuracies, val_accuracies, train_losses, val_losses = train(model, train_loader, val_loader, n_epochs, device)

    # Plot accuracies & losses
    plot_values(train_losses, val_losses, title=model_name+"_losses")
    plot_values(train_accuracies, val_accuracies, title=model_name+"_accuracies")

import os
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from utils import *


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int=25):      
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([1]*input_dim))

    def forward(self, x):
        return self.weight*x


if __name__ == "__main__":
    ### HYPERPARAMETERS ###
    K = 2
    sim_threshold = 0.6
    num_epochs = 100
    learning_rate = 1e-2

    # Shingling
    if os.path.exists(f"{K}-shingles.pt"):
        print("Loading cached shingles...")
        shingles = torch.load(f"{K}-shingles.pt")
        train_doc2shingles, train_doc2labels, valid_doc2shingles, valid_doc2labels, test_doc2shingles, label2shingles = shingles
    else:
        shingles = get_shingles(K)
        train_doc2shingles, train_doc2labels, valid_doc2shingles, valid_doc2labels, test_doc2shingles, label2shingles = build_doc2shingles(shingles, K)
        _save = [train_doc2shingles, train_doc2labels, valid_doc2shingles, valid_doc2labels, test_doc2shingles, label2shingles]
        torch.save(_save, f"{K}-shingles.pt")

    # Build union & intersection matrices for jaccard
    if os.path.exists(f"{K}-union_intersection_matrices.pt"):
        print("Loading cached union and intersection matrices...")
        matrices = torch.load(f"{K}-union_intersection_matrices.pt")
        train_union_matrices, train_intersection_matrices, valid_union_matrices, valid_intersection_matrices, test_union_matrices, test_intersection_matrices = matrices
    else:
        train_union_matrices, train_intersection_matrices = build_union_intersection_matrices(train_doc2shingles, train_doc2shingles, label2shingles)
        valid_union_matrices, valid_intersection_matrices = build_union_intersection_matrices(train_doc2shingles, valid_doc2shingles, label2shingles)
        test_union_matrices, test_intersection_matrices = build_union_intersection_matrices(train_doc2shingles, test_doc2shingles, label2shingles)
        _save = [train_union_matrices, train_intersection_matrices, valid_union_matrices, valid_intersection_matrices, test_union_matrices, test_intersection_matrices]
        torch.save(_save, f"{K}-union_intersection_matrices.pt")

    # Load model & optimizer
    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, approx_F1_scores, train_F1_scores, valid_F1_scores = [], [], [], []
    f = open(f"weighted_jaccard_{sim_threshold}_{learning_rate}_F1.log", "w")
    f2 = open(f"weighted_jaccard_{sim_threshold}_{learning_rate}.log", "w")
    highest_epoch, highest_F1 = -1, 0

    # Train jaccard weights
    for epoch in trange(num_epochs, desc="Epoch"):
        train_doc2sim, train_doc2score = {}, {}
        train_A, train_B = 0, 0
        train_scores_dict = defaultdict(float)

        for p in model.parameters():
            p.data.clamp_(0.01)
        
        ### TRAIN MODE ###
        model.train()
        # Get weighted similarities for each document pair
        for t_idx in range(len(train_doc2labels)):
            union = model(train_union_matrices[t_idx].float())
            intersection = model(train_intersection_matrices[t_idx].float())
            weighted_sim = intersection / union.sum(dim=1, keepdim=True)
            train_doc2sim[t_idx] = weighted_sim
            train_doc2score[t_idx] = weighted_sim.sum(dim=1)
        
        # 
        for t_idx in train_doc2score.keys():
            # Find all document pairs with t_idx whose similarity > threshold and whose not
            mask = train_doc2score[t_idx] >= sim_threshold
            mask[t_idx] = False     # should not look at itself
            x = train_doc2sim[t_idx] * (mask.unsqueeze(1))
            x_not = train_doc2sim[t_idx] * ~mask.unsqueeze(1)

            # All document pairs with t_idx with low similarity should only predict '-'
            _padding = torch.zeros(x.shape)
            _padding[:, 0] = 1
            padding = _padding * ~mask.unsqueeze(1)
            x_not = x_not * padding

            # Predict scores and labels
            scores = x + x_not                      # (num_documents, num_classes)
            scores, _ = torch.max(scores, dim=0)    # (num_classes,)
            pred_indices = [x.item() for x in torch.where(mask == True)[0]]
            preds = set()
            for idx in pred_indices:
                pred = train_doc2labels[idx]
                if '-' not in pred:     # only predict attacks, not '-'
                    preds.update(pred)

            if len(preds) == 0:
                preds.add('-')

            ground_truth = torch.tensor(convert_labels_to_ids(train_doc2labels[t_idx])).unsqueeze(0)
            
            # Calculate F1
            if set(train_doc2labels[t_idx]) == set('-'):
                train_scores_dict['Approx_F1_B'] += calculate_approx_F1(ground_truth, scores)
                train_scores_dict['Real_F1_B'] += calculate_F1(train_doc2labels[t_idx], preds)
                train_B +=1
            else:
                train_scores_dict['Approx_F1_A'] += calculate_approx_F1(ground_truth, scores)
                train_scores_dict['Real_F1_A'] += calculate_F1(train_doc2labels[t_idx], preds)
                train_A +=1

        approx_F1 = 0.5 * (train_scores_dict['Approx_F1_B']/train_B + train_scores_dict['Approx_F1_A']/train_A)
        train_weighted_F1 = 0.5 * (train_scores_dict['Real_F1_B']/train_B + train_scores_dict['Real_F1_A']/train_A)
        approx_F1_scores.append(approx_F1)
        train_F1_scores.append(train_weighted_F1)
        print('Approx:', approx_F1, 'Real:', train_weighted_F1)
        f.write(f"Approx F1: {approx_F1}\tReal F1: {train_weighted_F1}\n")
        # print(list(model.parameters()))

        optimizer.zero_grad()
        loss = (1 - approx_F1)**2
        train_losses.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()

        ### VALIDATION MODE ###
        valid_doc2sim, valid_doc2score = {}, {}
        valid_A, valid_B = 0, 0
        valid_scores_dict = defaultdict(float)
        model.eval()
        with torch.no_grad():
            # Get weighted similarities for each document pair
            for v_idx in range(len(valid_doc2labels)):
                union = model(valid_union_matrices[v_idx].float())
                intersection = model(valid_intersection_matrices[v_idx].float())
                weighted_sim = intersection / union.sum(dim=1, keepdim=True)
                valid_doc2sim[v_idx] = weighted_sim
                valid_doc2score[v_idx] = weighted_sim.sum(dim=1)

            f2.write(f"Epoch: {epoch}\n")
            for v_idx in valid_doc2score.keys():
                # Find all document pairs with t_idx whose similarity > threshold
                mask = valid_doc2score[v_idx] > sim_threshold
                pred_indices = [x.item() for x in torch.where(mask == True)[0]]
                preds = set()
                for idx in pred_indices:
                    pred = train_doc2labels[idx]
                    if '-' not in pred:     # only predict attacks, not '-'
                        preds.update(pred)

                if len(preds) == 0:
                    preds.add('-')

                # Calculate F1
                if set(valid_doc2labels[v_idx]) == set('-'):
                    valid_scores_dict['Real_F1_B'] += calculate_F1(valid_doc2labels[v_idx], preds)
                    valid_B += 1
                else:
                    valid_scores_dict['Real_F1_A'] += calculate_F1(valid_doc2labels[v_idx], preds)
                    valid_A += 1
                
                f2.write(f"\tGround Truth: {set(valid_doc2labels[v_idx])}\tPreds: {set(preds)}\n")
        
        valid_weighted_F1 = 0.5 * (valid_scores_dict['Real_F1_B']/valid_B + valid_scores_dict['Real_F1_A']/valid_A)
        if valid_weighted_F1 > highest_F1:
            highest_F1 = valid_weighted_F1
            highest_epoch = epoch
        valid_F1_scores.append(valid_weighted_F1)
        print("Validation F1:", valid_weighted_F1)
        f.write(f"Validation F1: {valid_weighted_F1}\n")

    f.write(f"Epoch: {highest_epoch}\tHighest F1: {highest_F1}\n")
    f.close()
    f2.close()
    plot_values(train_F1_scores, valid_F1_scores, f"f1_scores_{sim_threshold}_{learning_rate}")

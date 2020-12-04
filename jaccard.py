import os
from collections import defaultdict
import torch

from utils import *

if __name__ == "__main__":
    ### HYPERPARAMETERS ###
    K = 4
    sim_threshold = 0.3

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

    sim_dict = defaultdict(list)
    F1_A, F1_B = 0, 0
    A, B = 0, 0

    # Compute Jaccard similarity betwee pairs of valid files and train files
    for v_idx in trange(len(valid_doc2shingles), desc="Computing Jaccard similarity"):
        for t_idx in range(len(train_doc2shingles)):
            jaccard_sim = jaccard_similarity(set(valid_doc2shingles[v_idx]), set(train_doc2shingles[t_idx]))
            sim_dict[v_idx].append((t_idx, jaccard_sim))

    f = open(f"jaccard_{sim_threshold}.log", "w")
    for v_idx, sim_list in sim_dict.items():
        preds = []
        ground_truth = valid_doc2labels[v_idx]
        for t_idx, jaccard_sim in sorted(sim_list, key=lambda tup: tup[1], reverse=True):
            pred = train_doc2labels[t_idx]
            if jaccard_sim >= sim_threshold and '-' not in pred:    # only predict attacks, not '-'
                preds += pred
            else:
                break
        
        # if nothing was predicted, predict '-' (no attack)
        if len(preds) == 0:
            preds.append('-')

        # Calculate weighted F1
        if set(ground_truth) == set('-'):
            F1_B += calculate_F1(ground_truth, preds)
            B += 1
        else:
            F1_A += calculate_F1(ground_truth, preds)
            A += 1
        
        f.write(f"Ground Truth: {set(ground_truth)}\tPreds: {set(preds)}\n")

    
    weighted_F1 = 0.5 * (F1_B/B + F1_A/A)
    print(F1_B/B, F1_A/A)
    print("Weighted F1:", weighted_F1)
    f.write(f"Weighted F1: {weighted_F1}\n")
    f.close()

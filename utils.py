import os
import csv
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def jaccard_similarity(s1, s2):
    """Compute Jaccard Similarity between two sets.

    Args:
        s1 (set): Set 1
        s2 (set): Set 2

    Returns:
        (float): Jaccard Similarity between Set 1 and Set 2
    """
    return len(s1.intersection(s2)) / len(s1.union(s2))


def get_shingles(K):
    """Get all K-shingles in the training & validation files.

    Args:
        K (int): length of shingle

    Returns:
        shingles (set): set of all shingles in documents
    """
    shingles = set()

    for f in tqdm(os.listdir("./train"), desc='Shingling training documents'):        
        with open(os.path.join("./train", f), "r") as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                # Get first K items from (src_id, dst_id, port_num, timestamp, con_type)
                shingles.add(tuple(row[:K]))

    for f in tqdm(os.listdir("./valid_query"), desc='Shingling validation documents'):
        with open(os.path.join("./valid_query", f), "r") as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                shingles.add(tuple(row[:K]))

    for f in tqdm(os.listdir("./test_query"), desc='Shingling test documents'):
        with open(os.path.join("./test_query", f), "r") as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                shingles.add(tuple(row[:K]))
                
    return shingles


def build_doc2shingles(shingles, K):
    """Convert document to a list of shingles.

    Args:
        shingles (set): set of all shingles in documents
        K (int): length of shingle

    Returns:
        train_doc2shingles (dict): dictionary mapping each training document to a list of shingles
        train_doc2labels   (dict): dictionary mapping each training document to a set of labels
        valid_doc2shingles (dict): dictionary mapping each validation document to a list of shingles
        valid_doc2labels   (dict): dictionary mapping each validation document to a set of labels
        test_doc2shingles  (dict): dictionary mapping each test document to a list of shingles
        label2shingles     (dict): dictionary mapping each label to a set of shingles
    """
    train_doc2shingles = defaultdict(list)
    valid_doc2shingles = defaultdict(list)
    test_doc2shingles = defaultdict(list)
    train_doc2labels = {}
    valid_doc2labels = {}
    shingle2idx = {}
    label2shingles = defaultdict(set)

    # shingle2idx = {shingle0: 0, shingle1: 1, ...}
    for idx, shingle in enumerate(shingles):
        shingle2idx[shingle] = idx
    
    # doc2shingles = {doc_idx0: list of shingles, ...}
    for idx, f in enumerate(tqdm(os.listdir("./train"), desc='Building training doc2shingles')):
        labels = set()
        with open(os.path.join("./train", f), "r") as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                train_doc2shingles[idx].append(shingle2idx[tuple(row[:K])])
                labels.add(row[-1])
                label2shingles[row[-1]].add(shingle2idx[tuple(row[:K])])
        
        # doc2labels = {doc_idx0: list of labels, ...}
        if len(labels) > 1:     # if bad connection exists, we don't care about benign connection
            labels.remove('-')
        train_doc2labels[idx] = list(labels)
    
    for idx, f in enumerate(tqdm(os.listdir("./valid_query"), desc='Building validation doc2shingles')):
        with open(os.path.join("./valid_query", f), "r") as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                valid_doc2shingles[idx].append(shingle2idx[tuple(row[:K])])
    
    for idx, f in enumerate(os.listdir("./valid_answer")):
        if os.stat(os.path.join('./valid_answer', f)).st_size == 0: # if file is empty, label is 'benign' connection.
            valid_doc2labels[idx] = ['-']
        else:
            with open(os.path.join("./valid_answer", f), "r") as tsv:
                tsv_reader = csv.reader(tsv, delimiter='\t')
                for row in tsv_reader:
                    valid_doc2labels[idx] = row
    
    for idx, f in enumerate(tqdm(os.listdir("./test_query"), desc='Building test doc2shingles')):
        with open(os.path.join("./test_query", f), "r") as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                test_doc2shingles[idx].append(shingle2idx[tuple(row[:K])])

    return train_doc2shingles, train_doc2labels, valid_doc2shingles, valid_doc2labels, test_doc2shingles, label2shingles


def build_union_intersection_matrices(doc2shingles1, doc2shingles2, label2shingles):
    """Build union and intersection matrices between inferring and training shingles.

    Args:
        doc2shingles1 (dict): training shingles 
        doc2shingles2 (dict): inferring shingles
        label2shingles (dict): dictionary mapping each label to a set of shingles

    Returns:
        union_matrices (torch.Tensor): union matrix of size (#infer_shingles, #train_shingles, #classes)
        intersection_matrices (torch.Tensor): intersection matrix of size (#infer_shingles, #train_shingles, #classes)
    """
    union_matrices = []
    intersection_matrices = []
    train_matrix = []
    infer_matrix = []

    for shingles in doc2shingles1.values():
        train_matrix.append(set(shingles))

    for shingles in doc2shingles2.values():
        infer_matrix.append(set(shingles))
    
    for i in trange(len(infer_matrix), desc="Building union and intersection matrices"):
        union_matrix = []
        intersection_matrix = []
        for j in range(len(train_matrix)):
            unions, intersections = [], []
            union = infer_matrix[i].union(train_matrix[j])
            intersection = infer_matrix[i].intersection(train_matrix[j])
            
            for label_shingles in label2shingles.values():
                unions.append(len(label_shingles.intersection(union)))
                intersections.append(len(label_shingles.intersection(intersection)))

            union_matrix.append(torch.tensor(unions).unsqueeze(0))
            intersection_matrix.append(torch.tensor(intersections).unsqueeze(0))

        union_matrix = torch.cat(union_matrix, dim=0)
        intersection_matrix = torch.cat(intersection_matrix, dim=0)
        union_matrices.append(union_matrix.unsqueeze(0))
        intersection_matrices.append(intersection_matrix.unsqueeze(0))
    
    union_matrices = torch.cat(union_matrices, dim=0)
    intersection_matrices = torch.cat(intersection_matrices, dim=0)

    return union_matrices, intersection_matrices


def convert_labels_to_ids(con_types):
    """ Maps names of the connection types to corresponding IDs
        and returns a one-hot vector to be used in computation.

    Args:
        con_types (set): set of connection types

    Returns:
        ids (list): one-hot list of ids corresponding to connection types
    """
    label_dict = {
        '-': 0, 'smurf': 1, 'neptune': 2, 'teardrop': 3, 'portsweep': 4,
        'ipsweep': 5, 'back': 6, 'satan': 7, 'nmap': 8, 'warezclient': 9,
        'pod': 10, 'rootkit': 11, 'apache2': 12, 'snmpguess': 13, 'snmpgetattack': 14,
        'processtable': 15, 'smurfttl': 16, 'warez': 17, 'guest': 18, 'dict': 19,
        'mscan': 20, 'httptunnel-e': 21, 'mailbomb': 22, 'ignore': 23, 'saint': 24
    }
    ids = [0] * len(label_dict)

    # Check for empty valid_answer file => which means label is just '-'
    if not con_types:
        ids[0] = 1
        return ids

    # Assign value 1 to all existing labels
    for con_type in con_types:
        ids[label_dict[con_type]] = 1

    # if there are more than 1 label, exclude the label '-'
    if len(con_types) > 1:
        ids[0] = 0

    return ids


def calculate_approx_F1(ground_truth, prediction):
    prec = (1+(prediction*ground_truth).sum())/(1+ground_truth.sum())
    recall = (1+(prediction*ground_truth).sum())/(1+prediction.sum())
    return 2*prec*recall/(prec+recall)


def calculate_F1(ground_truth, prediction):
    prec = (1+len(set(prediction)&set(ground_truth)))/(1+len(set(prediction)))
    recall = (1+len(set(prediction)&set(ground_truth)))/(1+len(set(ground_truth)))
    return 2*prec*recall/(prec+recall)


def plot_values(train_values, val_values, title):
    x = list(range(1, len(train_values)+1))
    plt.figure()
    plt.title(title)
    plt.plot(x, train_values, marker='o', label='Training')
    plt.plot(x, val_values, marker='x', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend()
    plt.savefig(title + '.png')

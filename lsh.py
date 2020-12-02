import os
import csv
from collections import defaultdict
import numpy as np
from tqdm import tqdm, trange


class Hash():
    def __init__(self, M, N, p):
        """Generate M universal hash functions. 

        Args:
            M (int): number of hash functions
            N (int): number of shingles
            p (int): prime number (p > N)
        """
        self.M = M
        self.N = N
        self.p = p
        
        self.a = np.random.randint(9999)
        self.b = np.random.randint(9999)
    
    def __call__(self, x):
        return np.mod(np.mod((self.a * x + self.b), self.p), self.N)


def is_prime(n):
    """Check if the number is prime.

    Args:
        n (int): number to check

    Returns:
        (bool): prime number or not
    """
    for i in range(2,int(np.sqrt(n))+1):
        if not n % i:
            return False
    return True


def generate_prime_numbers(M, N):
    """Generate M prime numbers where each prime number is greater than N.

    Args:
        M (int): number of prime numbers to generate
        N (int): number in which prime number should be greater

    Returns:
        primes (list): list of prime numbers
    """
    primes = []
    cnt = 0
    n = N + 1
    
    while cnt < M:
        if is_prime(n):
            primes.append(n)
            cnt += 1
        n += 1
    return primes


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

    for f in tqdm(os.listdir("./valid_query"), desc='Shingling valiation documents'):
        with open(os.path.join("./valid_query", f), "r") as tsv:
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
        doc2shingles (dict): dictionary mapping each document to a list of shingles
    """
    train_doc2shingles = defaultdict(list)
    valid_doc2shingles = defaultdict(list)
    train_doc2labels = {}
    valid_doc2labels = {}
    shingle2idx = {}

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

    return train_doc2shingles, train_doc2labels, valid_doc2shingles, valid_doc2labels


def min_hash(doc2shingles, hash_functions):
    """Compute Min-Hashing to create signatures for documents.

    Args:
        doc2shingles (dict): dictionay mapping each document to a list of shinglesr
        hash_functions (list): list of hash functions

    Returns:
        signatures (np.array): numpy array of size (M=number of hash functions, C=number of documents)
    """
    C = len(doc2shingles)
    M = len(hash_functions)
    signatures = np.array(np.ones((M, C)) * np.inf, dtype=np.int)

    for doc, shingles in tqdm(doc2shingles.items(), desc="Min-hashing"):
        signatures[:, doc] = [np.min(hash_func(np.array(shingles))) for hash_func in hash_functions]

    return signatures


def lsh(signatures, b, r):
    """Compute Locality-Sensitive Hashing to find candidate pairs of similar documents.

    Args:
        signatures (np.array): numpy array of size (M=number of hash functions, C=number of documents)
        b (int): number of bands
        r (int): number of rows per each band

    Returns:
        candidate_pairs (Set[Tuple[int, int]]): set of candidate document pairs
    """
    M, C = signatures.shape
    assert M == b * r
    candidate_pairs = set()

    # For each band, create a hash table to put similar documents into the same bucket
    for i in trange(0, M, r, desc="Computing LSH"):
        hash_table = defaultdict(list)
        # Hash portions of columns into buckets
        for j in range(C):
            strip = tuple(signatures[i:i+r, j])
            # If bucket not empty, add them as candidate pairs
            if len(hash_table[strip]) > 0:
                for similar_doc in hash_table[strip]:
                    candidate_pairs.add((j, similar_doc))
            hash_table[strip].append(j)

    return candidate_pairs


if __name__ == "__main__":
    # Shingling
    K = 2
    shingles = get_shingles(K=K)
    train_doc2shingles, train_doc2labels, valid_doc2shingles, valid_doc2labels = build_doc2shingles(shingles, K=K)

    # Compare all pairs of train files and valid files
    sim_dict = defaultdict(list)
    for t_idx in trange(len(train_doc2shingles)):
        for v_idx in range(len(valid_doc2shingles)):
            true_sim = jaccard_similarity(set(train_doc2shingles[t_idx]), set(valid_doc2shingles[v_idx]))
            sim_dict[v_idx].append((t_idx, true_sim))

    f = open("sim_results.log", "w")
    num_correct = 0
    for v_idx, sim_list in sim_dict.items():
        ground_truth = valid_doc2labels[v_idx]
        prediction = []
        f.write(f"Valid Label: {ground_truth}\n")
        for t_idx, true_sim in sorted(sim_list, key=lambda tup: tup[1], reverse=True)[:5]:
            if true_sim > 0.5:
            f.write(f"Pred: {train_doc2labels[t_idx]}\tSim: {true_sim}\n")
                prediction += train_doc2labels[t_idx]

        if len(prediction) == 0:
            prediction.append('-')
            
        if set(prediction) == set(ground_truth):
            num_correct += 1

    print(num_correct)
    print(num_correct / len(valid_doc2shingles))
    f.close()

    # # 2: Min-Hashing
    # M = 100
    # N = len(shingles)
    # primes = generate_prime_numbers(M, N)
    # hash_functions = [Hash(M, N, p) for p in primes]
    # signatures = min_hash(doc2shingles, hash_functions)
    
    # # 3: Locality-Sensitive Hashing
    # b = 10
    # candidate_pairs = lsh(signatures, b, M // b)
    
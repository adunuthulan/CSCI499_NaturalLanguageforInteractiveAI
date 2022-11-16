import re
import torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    ep_lens = []
    for episode in train:
        ep_lens.append(len(episode))
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
        int(np.average(ep_lens) + np.std(ep_lens) * 2 + 0.5),
    )


# def build_output_tables(train):
#     actions = set()
#     targets = set()

#     # Add BOS and EOS tokens
#     actions.add("<BOS>")
#     actions.add("<EOS>")

#     for episode in train:
#         for _, outseq in episode:
#             a, t = outseq
#             actions.add(a)
#             targets.add(t)
#     actions_to_index = {a: i for i, a in enumerate(actions)}
#     targets_to_index = {t: i for i, t in enumerate(targets)}
#     index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
#     index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
#     return actions_to_index, index_to_actions, targets_to_index, index_to_targets

def build_output_tables(train):
    actions = set()
    targets = set()
    actions_to_index = {"<pad>": 0, "<BOS>": 1, "<EOS>": 2}
    targets_to_index = {"<pad>": 0, "<BOS>": 1, "<EOS>": 2}
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index.update({a: i+3 for i, a in enumerate(actions)})
    targets_to_index.update({t: i+3 for i, t in enumerate(targets)})
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

# # adapted from lecture assignment
# def encode_data(data, v2i, seq_len, o2i):
#     n_samples = len(data)
#     x = np.zeros((n_samples, seq_len), dtype=np.int32)
#     y = np.zeros((n_samples, 4), dtype=np.int32)

#     idx = 0
#     n_early_cutoff = 0
#     n_unks = 0
#     n_tks = 0
#     for inst, outseq in data:
#         a, t = outseq
#         inst = preprocess_string(inst)
#         x[idx][0] = v2i["<start>"]
#         jdx = 1
#         for word in inst.split():
#             if len(word) > 0:
#                 x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
#                 n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
#                 n_tks += 1
#                 jdx += 1
#                 if jdx == seq_len - 1:
#                     n_early_cutoff += 1
#                     break
#         x[idx][jdx] = v2i["<end>"]

#         y[idx][0] = o2i["<BOS>"]
#         y[idx][1] = o2i[a]
#         y[idx][2] = o2i[t]
#         y[idx][3] = o2i["<EOS>"]
#         idx += 1
#     print(
#         "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
#         % (n_unks, n_tks, n_unks / n_tks, len(v2i))
#     )
#     print(
#         "INFO: cut off %d instances at len %d before true ending"
#         % (n_early_cutoff, seq_len)
#     )
#     print("INFO: encoded %d instances without regard to order" % idx)
#     return x, y

# ======================================================================
# ============== inputs = [batch_size, ep_len * seq_len] ===============
# def encode_data(data, v2i, ep_len, seq_len, a2i, t2i):
#     n_episodes = len(data)
#     x = np.zeros((n_episodes, ep_len*seq_len), dtype=np.int32)
#     y = np.zeros((n_episodes, ep_len*2+2), dtype=np.int32)

  
#     n_early_cutoff = 0
#     n_unks = 0
#     n_tks = 0

#     for i, ep in enumerate(data):
#         idx = 0
#         n_instr = 0
#         y[i, 0] = a2i["<BOS"]
#         for inst, outseq in ep:
#             if n_instr == ep_len: 
#                 break # cut off the ending instructions

#             a, t = outseq
#             inst = preprocess_string(inst)
#             x[i, idx] = v2i["<start>"]
#             idx += 1
#             for word in inst.split():
#                 if len(word) > 0:
#                     x[i, idx] = v2i[word] if word in v2i else v2i["<unk>"]
#                     n_unks += 1 if x[i, idx, jdx] == v2i["<unk>"] else 0
#                     n_tks += 1
#                     idx += 1
#                     if idx == ep_len*seq_len - 1:
#                         n_early_cutoff += 1
#                         break
#             x[i, idx] = v2i["<end>"]

#             y[i, 2*n_instr+1] = a2i[a]
#             y[i, 2*n_instr+2] = t2i[t]
            
#             n_instr += 1
#         y[i, 2*n_instr+1] = a2i["<EOS>"]

#     print(
#         "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
#         % (n_unks, n_tks, n_unks / n_tks, len(v2i))
#     )
#     print(
#         "INFO: cut off %d instances at len %d before true ending"
#         % (n_early_cutoff, seq_len)
#     )
#     print("INFO: encoded %d instances without regard to order" % n_instr)
#     return x, y

def encode_data(data, v2i, ep_len, seq_len, a2i, t2i):
    n_episodes = len(data)
    x = np.zeros((n_episodes, ep_len, seq_len), dtype=np.int32)
    x_flat = np.zeros((n_episodes, ep_len*seq_len), dtype=np.int32)
    y = np.zeros((n_episodes, ep_len+2, 2), dtype=np.int32)

  
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for i, episodes in enumerate(data):
        y[i, 0, 0] = a2i["<BOS>"]
        y[i, 0, 1] = t2i["<BOS>"]

        idx = 0
        for inst, outseq in episodes:
            if idx == ep_len: 
                break # cut off the ending instructions

            a, t = outseq
            inst = preprocess_string(inst)
            x[i, idx, 0] = v2i["<start>"]
            jdx = 1
            for word in inst.split():
                if len(word) > 0:
                    x[i, idx, jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                    n_unks += 1 if x[i, idx, jdx] == v2i["<unk>"] else 0
                    n_tks += 1
                    jdx += 1
                    if jdx == seq_len - 1:
                        n_early_cutoff += 1
                        break
            x[i, idx, jdx] = v2i["<end>"]
            y[i, idx+1, 0] = a2i[a]
            y[i, idx+1, 1] = t2i[t]
            
            idx += 1
        y[i, idx+1, 0] = a2i["<EOS>"]
        y[i, idx+1, 1] = t2i["<EOS>"]

    for i, episodes in enumerate(data):
        x_flat[i, 0] = v2i["<start>"]
        idx = 1
        for inst, outseq in episodes:
            for word in inst.split():
                if len(word) > 0:
                    x_flat[i, idx] = v2i[word] if word in v2i else v2i["<unk>"]
                    idx += 1
                    if idx == ep_len*seq_len - 1: break
            if idx == ep_len*seq_len - 1: break
        x_flat[i, idx] = v2i["<end>"]

    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    return x, x_flat, y



def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    seq_length = len(gt_labels)
    
    for i in range(len(gt_labels)):
        if predicted_labels[i, 0] != gt_labels[i, 0] or predicted_labels[i, 1] != gt_labels[i, 1]:
            break
    
    pm = (1.0 / seq_length) * i

    return pm

# from G2G implementation
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n] / len(Y)
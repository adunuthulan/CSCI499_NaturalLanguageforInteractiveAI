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
    inst_count = 0
    for episode in train:
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
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

# adapted from lecture assignment
def encode_data(data, v2i, seq_len, a2i, t2i):
    n_samples = len(data)
    x = np.zeros((n_samples, seq_len), dtype=np.int32)
    y = np.zeros((n_samples, 2), dtype=np.int32)

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for inst, outseq in data:
        a, t = outseq
        inst = preprocess_string(inst)
        x[idx][0] = v2i["<start>"]
        jdx = 1
        for word in inst.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        y[idx] = [a2i[a], t2i[t]]
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    return x, y
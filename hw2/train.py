import argparse
import os
import tqdm
from model import SkipGram
import torch
from sklearn.metrics import accuracy_score

from eval_utils import downstream_validation
import utils
import data_utils

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
import matplotlib.pyplot as plt

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
        corpus
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    # create input/output pairs using window size for Skip-Gram
    print("Creating input/output pairs...")

    window = args.window
    input_words = []
    contexts = []

    for sent in encoded_sentences:
        for i, w in enumerate(sent):
            if w == 0: break # if you reach the end of the sentence stop creating pairs
            ctx = []
            for c in range(1, window+1):
                if (i+c < len(sent)):
                    ctx.append(sent[i+c])
                if (i-c > 0):
                    ctx.append(sent[i-c])
            input_words.append(w)
            contexts.append(ctx)

    print("Generated ", len(input_words), " samples....")

    # train/val split
    prop_train = 0.8
    idxs = set(range(len(input_words)))
    train_idxs = set(random.sample(idxs, int(len(input_words)*prop_train + 0.5)))
    val_idxs = idxs - train_idxs
    
    train_np_x = np.zeros(len(train_idxs))
    train_np_y = np.zeros((len(train_idxs), 2*window))
    val_np_x = np.zeros(len(val_idxs))
    val_np_y = np.zeros((len(val_idxs), 2*window))
    
    for i, idx in enumerate(train_idxs):
        train_np_x[i] = input_words[idx]
        for j, ctx in enumerate(contexts[idx]):
            train_np_y[i][j] = ctx
    for i, idx in enumerate(val_idxs):
        val_np_x[i] = input_words[idx]
        for j, ctx in enumerate(contexts[idx]):
            val_np_y[i][j] = ctx

    print("split samples into", len(train_np_x), "training and", len(val_np_x),"validation...")
    
    # create dataloaders
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)
    
    print("Finished input/output pairs!")

    return train_loader, val_loader, len(train_np_y), index_to_vocab


def setup_model(args, num_labels):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = SkipGram(args.vocab_size, args.emb_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    learning_rate = 0.0001

    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    correct_preds = 0
    total_preds = 0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        
        # put model inputs to device
        inputs = inputs.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # turn labels into multihot encoding
        l = np.zeros((inputs.size(0), args.vocab_size))
        for i, batch in enumerate(labels):
            for ctx in batch:
                l[i, int(ctx.item())] = 1

        multihot_labels = torch.from_numpy(l) 
        multihot_labels = multihot_labels.to(device).float()

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze().float(), multihot_labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # avg metric using top 2*C
        w = args.window * -2
        ind = np.argpartition(pred_logits.detach().numpy(), w)[:, w:]
        labels = labels.numpy()
        for i in range(len(labels)): 
            correct_preds += len(np.intersect1d(ind[i], labels[i]))
            total_preds += len(ind[i])
        
    acc = correct_preds / total_preds
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, num_labels, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, num_labels)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    metrics = {'tl':[], 'vl':[], 'ta':[], 'va':[]}

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")
        metrics["tl"].append(train_loss)
        metrics["ta"].append(train_acc)

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")
            metrics["vl"].append(val_loss)
            metrics["va"].append(val_acc)

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

    gen_figs(metrics)

def gen_figs(metrics):
    x = [x for x in range(len(metrics["tl"]))]

    # training loss
    y = metrics["tl"]
    plt.scatter(x, y, c='lightblue', label='target loss')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f'training_loss.png')
    plt.show()

    # training accuracy
    y = metrics["ta"]
    plt.scatter(x, y, c='lightblue', label='target accuracy')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig(f'training_acc.png')
    plt.show()

    x = [(epoch+1) * args.val_every for epoch in range(len(metrics["vl"]))]

    # val loss
    y = metrics["vl"]
    plt.scatter(x, y, c='lightblue', label='validation loss')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Validation Loss')
    plt.legend()
    plt.savefig(f'validation_loss.png')
    plt.show()

    # val accuracy
    y = metrics["va"]
    plt.scatter(x, y, c='lightblue', label='validation accuracy')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(f'validation_acc.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", required=True)
    parser.add_argument("--window", type=int, help="window size", required=True)

    args = parser.parse_args()
    main(args)

from cProfile import label
import tqdm
import torch
import argparse
import json
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from model import ActionTargetPredictor
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    encode_data,
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    print("Setting up dataloaders")

    # Read in data
    f = open(args.in_data_fn)
    json_data = json.load(f)
    train_samples = json_data['train']
    val_samples = json_data['valid_seen']
    
    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_samples)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_samples)

    # flatten the samples
    train_samples = [x for idx in range(len(train_samples)) for x in train_samples[idx]]
    val_samples = [x for idx in range(len(val_samples)) for x in val_samples[idx]]

    # Encode the training and validation set inputs/outputs.
    train_np_x, train_np_y = encode_data(train_samples, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    val_np_x, val_np_y = encode_data(val_samples, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    maps = {'v2i': vocab_to_index, 'i2v': index_to_vocab, 'len_cutoff': len_cutoff, 'a2i': actions_to_index, 'i2a':index_to_actions, 't2i': targets_to_index, 'i2t': index_to_targets}
    return train_loader, val_loader, maps


def setup_model(args, maps, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    print("Initializing model")

    model = ActionTargetPredictor(device, len(maps['v2i']), args.emb_dim, args.hidden_size, len(maps['a2i']), len(maps['t2i']), 1, maps['len_cutoff'])
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    learning_rate = 0.0001
    print(f"Setting up optimizer with lr=${learning_rate}")

    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.squeeze().cpu().numpy())
        target_preds.extend(target_preds_.squeeze().cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    train_metrics = {'al':[], 'tl':[], 'aa':[], 'ta':[]}
    val_metrics = {'al':[], 'tl':[], 'aa':[], 'ta':[]}

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        train_metrics['al'].append(train_action_loss)
        train_metrics['tl'].append(train_target_loss)
        train_metrics['aa'].append(train_action_acc)
        train_metrics['ta'].append(train_target_acc)

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            val_metrics['al'].append(val_action_loss)
            val_metrics['tl'].append(val_target_loss)
            val_metrics['aa'].append(val_action_acc)
            val_metrics['ta'].append(val_target_acc)

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )
        


    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #

    # training loss
    x = [x for x in range(len(train_metrics["al"]))]
    plt.subplot(2, 2, 1)
    y = train_metrics["al"]
    plt.scatter(x, y, c='coral', label='action loss')
    plt.plot(x, y, c='coral')
    y = train_metrics["tl"]
    plt.scatter(x, y, c='lightblue', label='target loss')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Training Loss')
    plt.legend()

    # training accuracy
    plt.subplot(2, 2, 2)
    y = train_metrics["aa"]
    plt.scatter(x, y, c='coral', label='action accuracy')
    plt.plot(x, y, c='coral')
    y = train_metrics["ta"]
    plt.scatter(x, y, c='lightblue', label='target accuracy')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title('Training Accuracy')
    plt.legend()

    # val loss
    x = [(epoch+1) * args.val_every for epoch in range(len(val_metrics["al"]))]

    plt.subplot(2, 2, 3)
    y = val_metrics["al"]
    plt.scatter(x, y, c='coral', label='action loss')
    plt.plot(x, y, c='coral')
    y = val_metrics["tl"]
    plt.scatter(x, y, c='lightblue', label='target loss')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Validation Loss')
    plt.legend()

    # val accuracy
    plt.subplot(2, 2, 4)
    y = val_metrics["aa"]
    plt.scatter(x, y, c='coral', label='action accuracy')
    plt.plot(x, y, c='coral')
    y = val_metrics["ta"]
    plt.scatter(x, y, c='lightblue', label='target accuracy')
    plt.plot(x, y, c='lightblue')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title('Validation Accuracy')
    plt.legend()

    plt.savefig(f'results-{args.num_epochs}-{args.emb_dim}-{args.hidden_size}.png')


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )
        if not os.path.exists(args.model_output_dir):
            os.makedirs(args.model_output_dir)
        modelName = datetime.datetime.now().strftime("hw1_%m-%d-%Y_%H:%M:%S.pt")
        torch.save(model, os.path.join(os.getcwd(), args.model_output_dir, modelName))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", required=True)
    parser.add_argument("--hidden_size", type=int, help="hidden dimension", required=True)
    # parser.add_argument("--glove", action="store_true", help="initialize embedding layer with pretrained embeddings")
    # parser.add_argument("--glove_path", type=str, help="initialize embedding layer with pretrained embeddings from the given path")

    args = parser.parse_args()

    main(args)

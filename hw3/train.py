import json
import os
import datetime
import tqdm
from model_transformer import EncoderDecoderWithTransformer
from model_attention import EncoderDecoderWithAttention
from model import EncoderDecoder
import torch
import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    encode_data,
    prefix_match,
    lcs
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
    # Read in data
    f = open(args.in_data_fn)
    json_data = json.load(f)
    train_samples = json_data['train']
    val_samples = json_data['valid_seen']

    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff, instr_cutoff = build_tokenizer_table(train_samples)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_samples)

    # flatten the samples
    # train_samples = [x for idx in range(len(train_samples)) for x in train_samples[idx]]
    # val_samples = [x for idx in range(len(val_samples)) for x in val_samples[idx]]

    # TODO: Don't flatten the samples and keep the heirarchical order. Two LSTMs 
    # encode the <EOS> as an action (since there are fewer)
    # hard stop at 8 because the model doesn't learn to predict stop

    # Encode the training and validation set inputs/outputs.
    train_np_x, train_np_x_flat, train_np_y = encode_data(train_samples[:800], vocab_to_index, instr_cutoff, len_cutoff, actions_to_index, targets_to_index)
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    val_np_x, val_np_x_flat, val_np_y = encode_data(val_samples[:200], vocab_to_index, instr_cutoff, len_cutoff, actions_to_index, targets_to_index)
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    maps = {'v2i': vocab_to_index, 'i2v': index_to_vocab, 'len_cutoff': len_cutoff, 'a2i': actions_to_index, 'i2a': index_to_actions, 't2i': targets_to_index, 'i2t': index_to_targets}
    return train_loader, val_loader, maps, instr_cutoff


def setup_model(args, maps, instr_cutoff, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    input_size = len(maps["v2i"])
    emb_dim = args.emb_dim
    action_enc_size = len(maps["a2i"])
    target_enc_size = len(maps["t2i"])
    hidden_size = args.hidden_size
    hierarchical = args.hierarchical

    if args.mode == 'attention':
        model = EncoderDecoderWithAttention(input_size, emb_dim, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, hierarchical=hierarchical)
    elif args.mode == 'transformer':
        model = EncoderDecoderWithTransformer(input_size, emb_dim, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, hierarchical=hierarchical)
    else:
        model = EncoderDecoder(input_size, emb_dim, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, hierarchical=hierarchical)
    
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    learning_rate = 0.001

    action_criterion = torch.nn.CrossEntropyLoss()#ignore_index = 0) # for pad
    target_criterion = torch.nn.CrossEntropyLoss()#ignore_index = 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    maps=None
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    # epoch_acc = 0.0
    epoch_acc = {'pair': {'exact':0, 'lcs':0}, 'action': {'exact':0, 'lcs':0}, 'target': {'exact':0, 'lcs':0}}
    examples = 0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        action_out, target_out = model(inputs, labels)

        action_loss = action_criterion(action_out.squeeze(), labels[:, :, 0].long())
        target_loss = target_criterion(target_out.squeeze(), labels[:, :, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("Computing grad")
            # print(model.decoder.lstm.weight_ih_l0.grad)
    

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # take the prediction with the highest probability
        preds = torch.zeros(labels.shape)
        preds[:, :, 0] = action_out.argmax(1)
        preds[:, :, 1] = target_out.argmax(1)

        # Metrics for Validation
        if not training: 
            ## Pairs
            ### Exact 
            epoch_acc['pair']['exact'] += accuracy_score(preds.flatten(), labels.flatten())

            ### LCS
            epoch_acc['pair']['lcs'] += lcs(preds[:, :, 0].flatten(), labels.flatten())

            ## Individual
            ### Exact
            epoch_acc['action']['exact'] += accuracy_score(preds[:, :, 0].flatten(), labels[:, :, 0].flatten()) 
            epoch_acc['target']['exact'] += accuracy_score(preds[:, :, 1].flatten(), labels[:, :, 1].flatten()) 

            ### LCS
            epoch_acc['action']['lcs'] += lcs(preds[:, :, 0].flatten(), labels[:, :, 0].flatten())
            epoch_acc['target']['lcs'] += lcs(preds[:, :, 1].flatten(), labels[:, :, 1].flatten())

        if not training and examples < 5:
            print("Pred: "),
            for i in range(preds.shape[1]):
                print('{message:<30}'.format(
                    message='({action} {target})'.format(
                        action=maps["i2a"][preds[0, i, 0].item()],
                        target=maps["i2t"][preds[0, i, 1].item()],
                    )) + " -> " + '({action} {target})'.format(
                        action=maps["i2a"][labels[0, i, 0].item()],
                        target=maps["i2t"][labels[0, i, 1].item()],
                    )
                )
            examples += 1   

        # TODO logging
        epoch_loss += loss.item()
        # epoch_acc += acc

    epoch_loss /= len(loader)
    # epoch_acc /= len(loader)
    for key1 in epoch_acc.keys():
        for key2 in epoch_acc[key1].keys():
            epoch_acc[key1][key2] /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, action_criterion, target_criterion, maps, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
            maps=maps
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, maps, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    train_metrics = {'loss':[], 'acc':[]}
    val_metrics = {'loss':[], 'acc':{'pair': {'exact': [], 'lcs':[]}, 'action': {'exact': [], 'lcs':[]}, 'target': {'exact': [], 'lcs':[]}}}

    for epoch in tqdm.tqdm(range(int(args.num_epochs))):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        train_metrics["loss"].append(train_loss)

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % int(args.val_every) == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                maps,
                device,
            )

            val_metrics["loss"].append(val_loss)
            # val_metrics["acc"].append(val_acc)

            ## Pair
            val_metrics["acc"]['pair']['exact'].append(val_acc['pair']['exact'])
            val_metrics["acc"]['pair']['lcs'].append(val_acc['pair']['lcs'])

            ## Individual
            val_metrics["acc"]['action']['exact'].append(val_acc['action']['exact']) 
            val_metrics["acc"]['target']['exact'].append(val_acc['target']['exact']) 
            val_metrics["acc"]['action']['lcs'].append(val_acc['action']['lcs'])
            val_metrics["acc"]['target']['lcs'].append(val_acc['target']['lcs'])


            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #
    x = [x for x in list(range(len(train_metrics["loss"])))]
    
    # training loss
    y = train_metrics["loss"]
    plt.scatter(x, y, c='coral')
    plt.plot(x, y, c='coral')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Training Loss')
    plt.savefig(f'training_loss.png')
    plt.show()

    x = [((epoch+1) * int(args.val_every)) for epoch in list(range(len(val_metrics["loss"])))]

    # val loss
    y = val_metrics["loss"]
    plt.scatter(x, y, c='coral')
    plt.plot(x, y, c='coral')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Validation Loss')
    plt.savefig(f'validation_loss.png')
    plt.show()

    # val accuracy
    plt.scatter(x, val_metrics["acc"]['pair']['exact'])
    plt.plot(x, val_metrics["acc"]['pair']['exact'], label = 'pair - exact')

    plt.scatter(x, val_metrics["acc"]['pair']['lcs'])
    plt.plot(x, val_metrics["acc"]['pair']['lcs'], label = 'pair - lcs')

    plt.scatter(x, val_metrics["acc"]['action']['exact'])
    plt.plot(x, val_metrics["acc"]['action']['exact'], label = 'action - exact')

    plt.scatter(x, val_metrics["acc"]['target']['exact'])
    plt.plot(x, val_metrics["acc"]['target']['exact'], label = 'target - exact')

    plt.scatter(x, val_metrics["acc"]['action']['lcs'])
    plt.plot(x, val_metrics["acc"]['action']['lcs'], label = 'action - lcs')

    plt.scatter(x, val_metrics["acc"]['target']['lcs'])
    plt.plot(x, val_metrics["acc"]['target']['lcs'], label = 'target - lcs')

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(f'validation_acc.png')
    plt.show()

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps, instr_cutoff = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, instr_cutoff, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        # model = torch.load(args.model_output_dir)
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion, 
            target_criterion,
            maps,
            device,
        )
    else:
        train(args, model, loaders, optimizer, action_criterion, target_criterion, maps, device)
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
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", required=True)
    parser.add_argument("--hidden_size", type=int, help="hidden dimension", required=True)
    parser.add_argument("--hierarchical", action="store_false", help="whether to use the low level/high level encoding bonus")
    parser.add_argument(
        "--mode", type=str, default='standard', help="type of model to run"
    )
    args = parser.parse_args()

    main(args)

# IMPLEMENT YOUR MODEL CLASS HERE
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_size, emb_dim, hidden_size, device, hierarchical=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.nlayers = 1
        self.nhead = 2
        self.dropout = 0.3
        self.device = device
        self.hierarchical = hierarchical

        self.embedding = nn.Embedding(input_size, emb_dim)

        self.pos_encoder = PositionalEncoding(emb_dim, self.dropout)
        encoder_layers = TransformerEncoderLayer(emb_dim, self.nhead, hidden_size, self.dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

    def forward(self, src):
        # x = [batch_size, ep_len, seq_len]
        src = self.embedding(src) * math.sqrt(self.emb_dim)
        src =  src.flatten(start_dim=1, end_dim=2) # [batch_size, ep_len*seq_len, emb_dim]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        print("output shape: ", output.shape)

        return output


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = 1
        self.nhead = 2
        self.dropout = 0.3
        self.device = device

        self.action_enc_size = action_enc_size
        self.target_enc_size = target_enc_size
        self.action_emb_dim = 5
        self.target_emb_dim = 17
        self.instr_cutoff = instr_cutoff+2

        self.action_embedding = nn.Embedding(action_enc_size, self.action_emb_dim)
        self.target_embedding = nn.Embedding(target_enc_size, self.target_emb_dim)

        self.fc = nn.Linear(self.action_emb_dim+self.target_emb_dim, hidden_size)

        self.pos_encoder = PositionalEncoding(hidden_size, self.dropout)
        decoder_layers = TransformerDecoderLayer(hidden_size, self.nhead, hidden_size, self.dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, self.nlayers)

        self.action_fc = nn.Linear(hidden_size, action_enc_size)
        self.target_fc = nn.Linear(hidden_size, target_enc_size)

       

    def forward(self, a, t, memory):
        action_embeds = self.action_embedding(a)
        target_embeds = self.target_embedding(t)
        a_and_t = self.fc(torch.cat((action_embeds, target_embeds), -1))

        output = self.transformer_decoder(a_and_t, memory)
        action = self.action_fc(output).squeeze()
        target = self.target_fc(output).squeeze()

        return action, target


class EncoderDecoderWithTransformer(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_size, emb_dim, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, hierarchical=False, teacher_forcing=True):
        super(EncoderDecoderWithTransformer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.instr_cutoff = instr_cutoff+2
        self.action_enc_size = action_enc_size
        self.target_enc_size = target_enc_size
        self.teacher_forcing = teacher_forcing
        self.device = device

        self.encoder = Encoder(input_size, emb_dim, hidden_size, device, hierarchical=hierarchical)
        self.decoder = Decoder(action_enc_size, target_enc_size, instr_cutoff, hidden_size, device)

    def forward(self, inputs, labels):
        memory = self.encoder(inputs) 
        actions, targets = self.decoder(labels[:, :, 0], labels[:, :, 1], memory) 

        return actions.transpose(1, 2), targets.transpose(1, 2)

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
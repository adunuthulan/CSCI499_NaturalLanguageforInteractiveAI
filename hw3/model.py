# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_size, emb_dim, hidden_size, device, hierarchical=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.device = device
        self.hierarchical = hierarchical

        self.embedding = nn.Embedding(input_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, self.num_layers, batch_first=True)

        self.low_level_lstm = nn.LSTM(emb_dim, hidden_size, self.num_layers, batch_first=True)
        self.high_level_lstm = nn.LSTM(emb_dim, hidden_size, self.num_layers, batch_first=True)

    def forward(self, x):
        # x = [batch_size, ep_len, seq_len]
        batch_size = x.shape[0]
        ep_len = x.shape[1]
        seq_len = x.shape[2]

        embeds = self.embedding(x) # [batch_size, ep_len, seq_len, emb_dim]

        embeds =  embeds.flatten(start_dim=1, end_dim=2) # [batch_size, ep_len*seq_len, emb_dim]
        _, (hn, cn) = self.lstm(embeds) # [batch_size, ep_len*seq_len, hidden_size]

        # straighforward implementation
        # if not self.hierarchical: 
        #     embeds =  embeds.flatten(start_dim=1, end_dim=2) # [batch_size, ep_len*seq_len, emb_dim]
        #     _, (hn, cn) = self.lstm(embeds) # [batch_size, ep_len*seq_len, hidden_size]
        # bonus implementation
        # else:
        #     latent_states = torch.zeros(batch_size, ep_len, self.hidden_size).to(self.device) # [batch_size, ep_len, hidden_size]
        #     for instr in range(ep_len):
        #         _, (hn, cn) = self.low_level_lstm(embeds[:, instr, :, :])
        #         latent_states[:, instr, :] = hn 
        #     _, (hn, cn) = self.high_level_lstm(latent_states)

        return (hn, cn)


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, teacher_forcing=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.device = device

        self.action_enc_size = action_enc_size
        self.target_enc_size = target_enc_size
        self.action_emb_dim = 6
        self.target_emb_dim = 18
        self.instr_cutoff = instr_cutoff+2
        self.teacher_forcing = teacher_forcing

        self.action_embedding = nn.Embedding(action_enc_size, self.action_emb_dim)
        self.target_embedding = nn.Embedding(target_enc_size, self.target_emb_dim)

        self.lstm = nn.LSTM(self.action_emb_dim+self.target_emb_dim, hidden_size, self.num_layers, batch_first=True)

        self.action_fc = nn.Linear(hidden_size, action_enc_size)
        self.target_fc = nn.Linear(hidden_size, target_enc_size)

        # learn an embedding layer for both actions and targets
        # concatenate then

    def forward(self, labels, h0, c0):
        # targets = [batch_size, ep_len+2, 2] add two to length for <BOS> and <EOS>
        # h0 is the hidden state from the encoder = [1, batch_size, hidden_size]
        # instr_cutoff is the max tokens to be generated

        batch_size = labels.shape[0]

        actions = torch.zeros(batch_size, self.action_enc_size, self.instr_cutoff).to(self.device)
        targets = torch.zeros(batch_size, self.target_enc_size, self.instr_cutoff).to(self.device)
        actions[:, 1, 0] = 1 # <BOS>
        targets[:, 1, 0] = 1 # <BOS>
        h = h0
        c = c0

        # Set to a smaller amount for debugging
        # timesteps = 3

        timesteps = self.instr_cutoff
        for i in range(1, timesteps):
            if self.teacher_forcing: 
                action_embeds = self.action_embedding(labels[:, i-1, 0])
                target_embeds = self.target_embedding(labels[:, i-1, 1])
            else:
                action_pred = actions.argmax(1)
                target_pred = targets.argmax(1)

                action_embeds = self.action_embedding(action_pred[:, i-1])
                target_embeds = self.target_embedding(target_pred[:, i-1])

            a_and_t = torch.cat((action_embeds, target_embeds), -1).unsqueeze(1)

            output, (h, c) = self.lstm(a_and_t, (h, c))

            actions[:, :, i] = self.action_fc(output).squeeze()
            targets[:, :, i] = self.target_fc(output).squeeze()

        return actions, targets 


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_size, emb_dim, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, hierarchical=False):
        super(EncoderDecoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.encoder = Encoder(input_size, emb_dim, hidden_size, device, hierarchical=hierarchical)
        self.decoder = Decoder(action_enc_size, target_enc_size, instr_cutoff, hidden_size, device)

    def forward(self, inputs, targets):
        # inputs = [batch_size, ep_len, seq_len]
        # targets = [batch_size, 2*ep_len+2]

        # TODO: flip input
        (h, c) = self.encoder(inputs)
        actions, targets = self.decoder(targets, h, c)

        return actions, targets

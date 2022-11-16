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

        # self.low_level_lstm = nn.LSTM(emb_dim, hidden_size, self.num_layers, batch_first=True)
        # self.high_level_lstm = nn.LSTM(emb_dim, hidden_size, self.num_layers, batch_first=True)

    def forward(self, x):
        # x = [batch_size, ep_len, seq_len]
        batch_size = x.shape[0]
        ep_len = x.shape[1]
        seq_len = x.shape[2]

        embeds = self.embedding(x) # [batch_size, ep_len, seq_len, emb_dim]

        embeds =  embeds.flatten(start_dim=1, end_dim=2) # [batch_size, ep_len*seq_len, emb_dim]
        output, (hn, cn) = self.lstm(embeds) # output: [batch_size, ep_len*seq_len, hidden_size], hn: [1, batch_size, hidden_size]

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

        return output, (hn, cn)


class DecoderWithAttention(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.device = device

        self.action_enc_size = action_enc_size
        self.target_enc_size = target_enc_size
        self.action_emb_dim = 6
        self.target_emb_dim = 18
        self.instr_cutoff = instr_cutoff+2


        self.action_embedding = nn.Embedding(action_enc_size, self.action_emb_dim)
        self.target_embedding = nn.Embedding(target_enc_size, self.target_emb_dim)

        self.lstm = nn.LSTM(self.action_emb_dim+self.target_emb_dim, hidden_size, self.num_layers, batch_first=True)

        self.action_fc = nn.Linear(hidden_size, action_enc_size)
        self.target_fc = nn.Linear(hidden_size, target_enc_size)

    def forward(self, a, t, h, c):
        # h0 is the hidden state from the encoder = [1, batch_size, hidden_size]

        action_embeds = self.action_embedding(a)
        target_embeds = self.target_embedding(t)
        a_and_t = torch.cat((action_embeds, target_embeds), -1).unsqueeze(1)

        output, (h, c) = self.lstm(a_and_t, (h, c))
        action = self.action_fc(output).squeeze()
        target = self.target_fc(output).squeeze()

        return action, target, h, c


class EncoderDecoderWithAttention(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_size, emb_dim, action_enc_size, target_enc_size, instr_cutoff, hidden_size, device, hierarchical=False, teacher_forcing=True):
        super(EncoderDecoderWithAttention, self).__init__()
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
        self.decoder = DecoderWithAttention(action_enc_size, target_enc_size, instr_cutoff, hidden_size, device)

        self.attention = nn.Linear(hidden_size*2, 1)
        self.attention_combined = nn.Linear(hidden_size*2, hidden_size)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, labels):
        # inputs = [batch_size, instr_cutoff, seq_len]
        # targets = [batch_size, 2*instr_cutoff+2]

        # TODO: flip input
        enc_out, (h_enc, c_enc) = self.encoder(inputs) # enc_out: [batch_size, instr_cutoff*seq_len, hidden_size], hn: [1, batch_size, hidden_size]

        batch_size = labels.shape[0]
        enc_len = enc_out.shape[1]

        actions = torch.zeros(batch_size, self.action_enc_size, self.instr_cutoff).to(self.device)
        targets = torch.zeros(batch_size, self.target_enc_size, self.instr_cutoff).to(self.device)
        actions[:, 1, 0] = 1 # <BOS>
        targets[:, 1, 0] = 1 # <BOS>

        h = h_enc
        c = c_enc
        decoder_timesteps = self.instr_cutoff
        for i in range(1, decoder_timesteps):
            if self.teacher_forcing: 
                a, t = labels[:, i-1, 0], labels[:, i-1, 1]
            else:
                a, t = actions.argmax(1)[:, i-1], targets.argmax(1)[:, i-1]
            pred_action, pred_target, h, c = self.decoder(a, t, h, c) # pred: [batch_size, instr_cutoff*seq_len, action_enc_size], h: [1, batch_size, hidden_size]
            
            h = h.transpose(0,1).expand(-1, enc_len, -1) # h: [batch_size, instr_cutoff*seq_len, hidden_size]

            weights = self.softmax(self.attention(torch.cat((h, enc_out), -1))).squeeze() # cat: [batch_size, instr_cutoff*seq_len, 2*hidden_out] -> weights: [B, instr_cutoff*seq_len, 1]
            context = torch.bmm(weights.unsqueeze(1), enc_out) # context: [batch_size, 1, hidden_size]
            h = context.transpose(0, 1) # h: [1, batch_size, hidden_size]

            actions[:, :, i] = pred_action
            targets[:, :, i] = pred_target

        return actions, targets

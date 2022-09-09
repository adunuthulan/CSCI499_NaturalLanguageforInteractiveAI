# IMPLEMENT YOUR MODEL CLASS HERE
import torch
class ActionTargetPredictor(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        emb_dim,
        hidden_size,
        num_actions,
        num_targets,
        num_layers,
        len_cutoff,
    ):
        super(ActionTargetPredictor, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.num_layers = num_layers
        self.len_cutoff = len_cutoff

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.emb_dim*self.len_cutoff, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True)
        
        # self.fc_1 = torch.nn.Linear(self.hidden_size, 128)
        # self.relu = torch.nn.ReLU()

        # self.action_fc = torch.nn.Linear(128, self.num_actions)
        # self.target_fc = torch.nn.Linear(128, self.num_targets)

        self.action_fc = torch.nn.Linear(self.hidden_size, self.num_actions)
        self.target_fc = torch.nn.Linear(self.hidden_size, self.num_targets)

    def forward(self, x):

        embeds = self.embedding(x)
        # print("embeds:", embeds.size())
        embeds = embeds.view(-1, 1, self.emb_dim*self.len_cutoff) 
        # print("embeds:", embeds.size())

        output, (hn, cn) = self.lstm(embeds) 
        # print("output:", output.size(), " hn:", hn.size())

        # hn = hn.view(-1, self.hidden_size) 
        # out = self.fc_1(out)
        # out = self.relu(out)
        action_out = self.action_fc(hn)
        target_out = self.target_fc(hn)

        # print(" action_out:", action_out.size(), " target_out:", target_out.size())

        return action_out, target_out
# IMPLEMENT YOUR MODEL CLASS HERE
import torch

class SkipGram(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
    ):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.hidden = torch.nn.Linear(self.embedding_dim, self.vocab_size)


    def forward(self, x):
        input_emb = self.embed(x) # BxV -> BxN
        out = self.hidden(input_emb) # BxN -> BxV

        return out
    
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):

        embeds = self.embeddings(inputs)  
        embeds_avg = torch.mean(embeds, dim=1)
        output = self.linear(embeds_avg)  
        
        return output

    def load_embedding_weight(self,embeddings_path):
        self.embeddings.load_state_dict(torch.load(embeddings_path))
        


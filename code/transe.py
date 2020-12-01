import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def convert_dict_numpy( size, dim, dict_):
    embeddings = np.zeros((size, dim))
    for key, value in dict_.items():
        embeddings[int(key)] = np.array(value)
    return embeddings

# gpu_ids = [0, 1]
class TransE(nn.Module):
    def __init__(self, entity_vocab_size, relation_vocab_size ,hidden_size, p_norm=1, margin=1, ent_embeddings=None):
        super(TransE, self).__init__()
        self.p_norm = p_norm
        self.num_entities = entity_vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.ent_embeddings = nn.Embedding(entity_vocab_size, hidden_size)

        if ent_embeddings is not None:
            if isinstance(ent_embeddings, nn.Embedding):
                weight_shape = ent_embeddings.weight.data.shape
                self.pre_ent_embeddings = nn.Embedding(*weight_shape)

            self.ent_embeddings = nn.Sequential(self.pre_ent_embeddings, nn.Linear(300, hidden_size))

        self.rel_embeddings = nn.Embedding(relation_vocab_size, hidden_size)

    def forward(self, h, r, t):
        h = self.ent_embeddings(h)
        t = self.ent_embeddings(t)
        r = self.rel_embeddings(r)
        return h, t, r

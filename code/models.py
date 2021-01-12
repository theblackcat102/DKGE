import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import json
import numpy as np
import random
import math
import os

path = os.path.join(__file__.replace('models.py',''), 'tokenizer.json')

EMBED_MAX_SIZE = 753
EMBED_MAX_LENGTH = 30

class Mapping():

    def __init__(self, datapath):
        self.training = True
        with open(os.path.join(datapath, 'entity2wikidata.json'), 'r') as f:
            self.entity2wiki = json.load(f)

        with open(os.path.join(datapath, 'relation2wikidata.json'), 'r') as f:
            self.relation2wiki = json.load(f)

        self.relation2wiki['[UNK]'] = {
            'label': '[UNK]',
            'alternatives':[],
        }
        self.entity2wiki['[UNK]'] = {
            'label': '[UNK]',
            'alternatives':[],
        }

        self.tokenizer = Tokenizer.from_file(path)
        self.pad_token_id = self.tokenizer.token_to_id('[PAD]')

        self.id2entity = { value: key  for key, value in torch.load(os.path.join(datapath,'entity2id.pt')).items() }
        self.id2entity[ len(self.id2entity) ] = '[UNK]'

        self.entity2tokens = torch.from_numpy(np.array([  self.get_entity(idx)  for idx in range(len(self.id2entity)) ]))

        self.id2relation = { value: key  for key, value in torch.load(os.path.join(datapath,'rel2id.pt')).items() }
        self.id2relation[ len(self.id2relation) ] = '[UNK]'

        self.relations2tokens = torch.from_numpy(np.array([  self.get_relation(idx)  for idx in range(len(self.id2relation)) ]))

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def get_entity(self, idx):
        entity_id = self.id2entity[idx]
        name = '[UNK]'
        if entity_id not in self.entity2wiki:
            name = '[UNK]'
        elif not self.training and 'label' in self.entity2wiki[entity_id]:
            name = self.entity2wiki[entity_id]['label']
        elif 'label' in self.entity2wiki[entity_id] and 'alternatives' in self.entity2wiki[entity_id]:
            name = random.choice([self.entity2wiki[entity_id]['label']]+\
                self.entity2wiki[entity_id]['alternatives'])

        token_ids = self.tokenizer.encode(name).ids[:EMBED_MAX_LENGTH]
        token_ids += [self.pad_token_id]*max( EMBED_MAX_LENGTH-len(token_ids),  0)
        return token_ids

    def get_relation(self, idx):
        rel_id = self.id2relation[idx]
        if rel_id not in self.relation2wiki:
            name = '[UNK]'
        elif not self.training:
            name = self.relation2wiki[rel_id]['label']
        else:
            name = random.choice([self.relation2wiki[rel_id]['label']]+\
                self.relation2wiki[rel_id]['alternatives'])
        
        token_ids = self.tokenizer.encode(name).ids[:EMBED_MAX_LENGTH]
        token_ids += [self.pad_token_id]*max( EMBED_MAX_LENGTH-len(token_ids),  0)
        return token_ids

class DynamicKGE(nn.Module):

    def __init__(self, config, entity_id2tokens, rel_id2tokens):
        super(DynamicKGE, self).__init__()
        self.entity_id2tokens = entity_id2tokens
        self.rel_id2tokens = rel_id2tokens
        self.config = config
        # sub token embedding
        self.use_embedding = config.use_embedding
        if self.use_embedding:
            self.ent_embed = nn.Embedding(config.entity_total+1, config.dim)
            self.context_ent_embed = nn.Embedding(config.entity_total+1, config.dim)

            self.rel_embed = nn.Embedding(config.relation_total+1, config.dim)
            self.context_rel_embed = nn.Embedding(config.relation_total+1, config.dim)
        else:
            self.word_token_emb = nn.Embedding(EMBED_MAX_SIZE, config.dim)
            self.ent_offset = nn.Linear(config.dim, config.dim)
            self.rel_offset = nn.Linear(config.dim, config.dim)
            # neighbour token embedding
            self.context_emb = nn.Embedding(EMBED_MAX_SIZE, config.dim)

        self.entity_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))
        self.relation_gcn_weight = nn.Parameter(torch.Tensor(config.dim, config.dim))

        self.gate_entity = nn.Parameter(torch.Tensor(config.dim))
        self.gate_relation = nn.Parameter(torch.Tensor(config.dim))

        self.v_ent = nn.Parameter(torch.Tensor(config.dim))
        self.v_rel = nn.Parameter(torch.Tensor(config.dim))

        self.pht_o = dict()
        self.pr_o = dict()

        self._init_parameters()

    def _init_parameters(self):
        if self.use_embedding:
            nn.init.xavier_uniform_(self.ent_embed.weight)
            nn.init.xavier_uniform_(self.rel_embed.weight)
            nn.init.xavier_uniform_(self.context_ent_embed.weight)
            nn.init.xavier_uniform_(self.context_rel_embed.weight)
        else:
            nn.init.xavier_uniform_(self.word_token_emb.weight)
            nn.init.xavier_uniform_(self.context_emb.weight)

        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)
        nn.init.uniform_(self.v_ent.data)
        nn.init.uniform_(self.v_rel.data)

        stdv = 1. / math.sqrt(self.entity_gcn_weight.size(1))
        self.entity_gcn_weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.relation_gcn_weight.size(1))
        self.relation_gcn_weight.data.uniform_(-stdv, stdv)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=self.config.norm, dim=1)

    def get_entity_context(self, entities):
        if entities.device != 'cpu':
            entities_index = entities.cpu().numpy()
        else:
            entities_index = entities.numpy()
        entities_context = torch.LongTensor(self.config.entity_adj_table[entities_index])
        return entities_context.to(entities.device)

    def get_relation_context(self, relations):
        if relations.device != 'cpu':
            relations_index = relations.cpu().numpy()
        else:
            relations_index = relations.numpy()
        relations_context = torch.LongTensor(self.config.relation_adj_table[relations_index])
        return relations_context.to(relations.device)

    def get_adj_entity_vec(self, entity_vec_list, adj_entity_list, device):
        # adj node
        if self.use_embedding:
            adj_entity_vec_list = self.context_ent_embed(adj_entity_list.to(device))
            entity_vec_list = self.context_ent_embed(entity_vec_list.to(device))
        else:
            adj_entity_tokens = self.entity_id2tokens[adj_entity_list]
            adj_entity_tokens = adj_entity_tokens.to(device)
            adj_entity_vec_list = self.context_emb(adj_entity_tokens)
            adj_entity_vec_list = self.ent_offset(adj_entity_vec_list.mean(-2))

            # self node            
            entity_vec_list = self.entity_id2tokens[entity_vec_list.long()].to(device)
            entity_vec_list = self.word_token_emb(entity_vec_list)
            entity_vec_list = self.ent_offset(entity_vec_list.mean(1))

        return torch.cat((entity_vec_list.unsqueeze(1), adj_entity_vec_list), dim=1)

    def get_adj_relation_vec(self, relation_vec_list, adj_relation_list, device):
        # adj node
        if self.use_embedding:
            adj_relation_vec_list = self.context_rel_embed(adj_relation_list.to(device))
            relation_vec_list = self.rel_embed(relation_vec_list.to(device))
        else:
            adj_relation_tokens = self.rel_id2tokens[adj_relation_list].to(device)
            adj_relation_vec_list = self.context_emb(adj_relation_tokens)
            adj_relation_vec_list = self.rel_offset(adj_relation_vec_list.mean(-2))
            # self node
            relation_vec_list = self.rel_id2tokens[relation_vec_list.long()].to(device)
            relation_vec_list = self.word_token_emb(relation_vec_list)
            relation_vec_list = self.rel_offset(relation_vec_list.mean(1))
        return torch.cat((relation_vec_list.unsqueeze(1), adj_relation_vec_list), dim=1)

    def score(self, o, adj_vec_list, target='entity'):
        os = torch.cat(tuple([o] * (self.config.max_context_num+1)), dim=1).reshape(-1, self.config.max_context_num+1, self.config.dim)
        tmp = F.relu(torch.mul(adj_vec_list, os), inplace=False)  # batch x max x 2dim
        if target == 'entity':
            score = torch.matmul(tmp, self.v_ent)  # batch x max
        else:
            score = torch.matmul(tmp, self.v_rel)
        return score

    def calc_subgraph_vec(self, o, adj_vec_list, target="entity"):
        alpha = self.score(o, adj_vec_list, target)
        alpha = F.softmax(alpha)

        sg = torch.sum(torch.mul(torch.unsqueeze(alpha, dim=2), adj_vec_list), dim=1)  # batch x dim
        return sg

    def gcn(self, A, H, target='entity'):
        support = torch.bmm(A, H)

        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output

    def save_parameters(self, parameter_path):
        from transe import TransE, convert_dict_numpy

        if not os.path.exists(parameter_path):
            os.makedirs(parameter_path)

        with torch.no_grad():
            for idx in range(self.config.entity_total):
                ent_id = torch.LongTensor([idx])
                A = self.config.entity_A[ent_id.cpu().numpy()]
                if self.config.use_gpu:
                    A = A.cuda()
                    ent_id = ent_id.cuda()
                output = self.compute_entity(ent_id, A)
                self.pht_o[idx] = list(output.cpu().numpy().flatten().astype(float))

            for idx in range(self.config.relation_total):
                ent_id = torch.LongTensor([idx])
                A = self.config.entity_A[ent_id.cpu().numpy()]
                if self.config.use_gpu:
                    A = A.cuda()
                    ent_id = ent_id.cuda()

                output = self.compute_relation(rel_id, A)
                self.pr_o[idx] = list(output.cpu().numpy().flatten().astype(float))


        model = TransE(self.config.entity_total, relation_vocab_size= self.config.relation_total, 
            hidden_size=self.config.dim)

        model.ent_embeddings.weight.data.copy_(
            torch.from_numpy(convert_dict_numpy( self.config.entity_total,
                self.config.dim, self.pht_o)))

        model.rel_embeddings.weight.data.copy_(
            torch.from_numpy(convert_dict_numpy( self.config.relation_total,
                self.config.dim, self.pr_o)))

        torch.save( {'state_dict': model.state_dict()}, os.path.join(parameter_path, 'transe.ckpt'))

        with open(os.path.join(parameter_path, 'entity_o'), "w") as ent_f:
            ent_f.write(json.dumps(self.pht_o))

        with open(os.path.join(parameter_path, 'relation_o'), "w") as rel_f:
            rel_f.write(json.dumps(self.pr_o))
        torch.save(self.state_dict(), os.path.join(parameter_path, 'all_parameters.ckpt'))

    def compute_entity(self, ent_id, A):
        ent_id = ent_id.long()

        if self.use_embedding:
            embedding = self.ent_embed(ent_id.to(A.device))
        else:
            ent_tokens = self.entity_id2tokens[ent_id.to(A.device)]
            ent_tokens = ent_tokens.to(A.device)
            embedding = self.ent_offset(self.word_token_emb(ent_tokens).mean(1))

        adj_entity_list = self.get_entity_context(ent_id)
        adj_entity_vec_list = self.get_adj_entity_vec(ent_id, adj_entity_list, device=A.device)
        gcn_adj_entity_vec_list = self.gcn(A, adj_entity_vec_list, target='entity')

        ent_sg = self.calc_subgraph_vec(embedding, gcn_adj_entity_vec_list)
        o = torch.mul(F.sigmoid(self.gate_entity), embedding) + torch.mul(1 - F.sigmoid(self.gate_entity), ent_sg)
        return o

    def compute_relation(self, rel_id, A):
        rel_id = rel_id.long()

        if self.use_embedding:
            embedding = self.rel_embed(rel_id.to(A.device))

        else:
            ent_tokens = self.rel_id2tokens[rel_id].to(A.device)
            embedding = self.rel_offset(self.word_token_emb(ent_tokens).mean(1))

        adj_rel_list = self.get_relation_context(rel_id)
        adj_rel_vec_list = self.get_adj_relation_vec(rel_id, adj_rel_list, A.device)
        gcn_adj_rel_vec_list = self.gcn(A, adj_rel_vec_list, target='relation')

        ent_sg = self.calc_subgraph_vec(embedding, gcn_adj_rel_vec_list)
        o = torch.mul(F.sigmoid(self.gate_relation), embedding) + torch.mul(1 - F.sigmoid(self.gate_relation), ent_sg)
        return o


    def forward(self, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A):
        # multi golden and multi negative
        pos_h, pos_r, pos_t = golden_triples
        neg_h, neg_r, neg_t = negative_triples

        ph_o = self.compute_entity(pos_h, ph_A)
        pt_o = self.compute_entity(pos_t, pt_A)
        nh_o = self.compute_entity(neg_h, nh_A)
        nt_o = self.compute_entity(neg_t, nt_A)

        pr_o = self.compute_relation(pos_r, pr_A)
        nr_o = self.compute_relation(neg_r, nt_A)

        # score for loss
        p_score = self._calc(ph_o, pt_o, pr_o)
        n_score = self._calc(nh_o, nt_o, nr_o)

        return p_score, n_score


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    from util.train_util import get_batch_A, GraphDataSet
    entity_adj_table, relation_adj_table, max_context_num, entity_A, relation_A = torch.load('cache/adj_table_FB15K-237-2/snapshot1_14054_299_30.pt')

    dataset = GraphDataSet('cache/training_data_FB15K-237-2snapshot1_14054_299_30.pt', 'data/FB15K-237-2/snapshot1')
    dataloader = DataLoader(dataset, batch_size=32)
    from argparse import Namespace

    max_context = 30
    entity_total = 14054
    relation_total = 299
    entity_adj_matrix = [[ entity_total ]*max_context  ]*entity_total
    for key, adj_entities in entity_adj_table.items():
        entity_adj_matrix[key][:len(adj_entities)] = adj_entities
    entity_adj_table = np.array(entity_adj_matrix)

    relation_adj_matrix = [[ relation_total ]*max_context  ]*relation_total
    for key, adj_entities in relation_adj_table.items():
        relation_adj_matrix[key] = adj_entities[:max_context]

    relation_adj_table = np.array(relation_adj_matrix)
    entity_id2tokens = dataset.mapping.entity2tokens
    rel_id2tokens = dataset.mapping.relations2tokens
    config = Namespace(dim=256, entity_total=14055, 
        relation_total=299,
        max_context_num=30, 
        entity_adj_table=entity_adj_table, 
        relation_adj_table=relation_adj_table, 
        norm=6, use_embedding=True)
    model = DynamicKGE(config, entity_id2tokens, rel_id2tokens)
    model = model.cuda()
    for batch in dataloader:

        golden_triples = ( batch['h'], batch['r'], batch['t'])
        golden_triples = [t.cuda() for t in golden_triples]
        negative_triples = ( batch['nh'], batch['nr'], batch['nt'] )
        negative_triples = [t.cuda() for t in negative_triples]

        ph_A, pr_A, pt_A = get_batch_A(golden_triples, entity_A, relation_A)
        nh_A, nr_A, nt_A = get_batch_A(negative_triples, entity_A, relation_A)
        ph_A, pr_A, pt_A = ph_A.cuda(), pr_A.cuda(), pt_A.cuda()
        nh_A, nr_A, nt_A = nh_A.cuda(), nr_A.cuda(), nt_A.cuda()

        p_score, n_score = model(golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)

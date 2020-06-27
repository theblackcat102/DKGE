import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import test
from config import config
from util.parameter_util import *


# gpu_ids = [0, 1]


class DynamicKGE(nn.Module):
    def __init__(self, config):
        super(DynamicKGE, self).__init__()

        self.entity_emb = nn.Parameter(torch.Tensor(config.entity_total, config.dim))
        self.relation_emb = nn.Parameter(torch.Tensor(config.relation_total, config.dim))

        # self.entity_context = nn.Parameter(torch.Tensor(config.entity_total + 1, config.dim), )
        # self.relation_context = nn.Parameter(torch.Tensor(config.relation_total + 1, config.dim))
        self.entity_context = nn.Embedding(config.entity_total + 1, config.dim, padding_idx=config.entity_total)
        self.relation_context = nn.Embedding(config.relation_total + 1, config.dim, padding_idx=config.relation_total)

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
        nn.init.xavier_uniform_(self.entity_emb.data)
        nn.init.xavier_uniform_(self.relation_emb.data)
        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)
        nn.init.uniform_(self.v_ent.data)
        nn.init.uniform_(self.v_rel.data)

        stdv = 1. / math.sqrt(self.entity_gcn_weight.size(1))
        self.entity_gcn_weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.relation_gcn_weight.size(1))
        self.relation_gcn_weight.data.uniform_(-stdv, stdv)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=config.norm, dim=1)

    def get_entity_context(self, entities):
        entities_context = []
        for e in entities:
            entities_context.extend(config.entity_adj_table.get(int(e), [config.entity_total] * config.max_context_num))
        # return entities_context
        return torch.LongTensor(entities_context).cuda()

    def get_relation_context(self, relations):
        relations_context = []
        for r in relations:
            relations_context.extend(
                config.relation_adj_table.get(int(r), [config.relation_total] * 2 * config.max_context_num))
        # return relations_context
        return torch.LongTensor(relations_context).cuda()

    def get_adj_entity_vec(self, entity_vec_list, adj_entity_list):
        # adj_entity_vec_list = self.entity_context[adj_entity_list]
        adj_entity_vec_list = self.entity_context(adj_entity_list)
        adj_entity_vec_list = adj_entity_vec_list.view(-1, config.max_context_num, config.dim)

        return torch.cat((entity_vec_list.unsqueeze(1), adj_entity_vec_list), dim=1)

    def get_adj_relation_vec(self, relation_vec_list, adj_relation_list):
        # adj_relation_vec_list = self.relation_context[adj_relation_list]
        adj_relation_vec_list = self.relation_context(adj_relation_list)
        adj_relation_vec_list = adj_relation_vec_list.view(-1, config.max_context_num, 2,
                                                           config.dim).cuda()
        adj_relation_vec_list = torch.sum(adj_relation_vec_list, dim=2)

        return torch.cat((relation_vec_list.unsqueeze(1), adj_relation_vec_list), dim=1)

    def score(self, o, adj_vec_list, target='entity'):
        os = torch.cat(tuple([o] * (config.max_context_num+1)), dim=1).reshape(-1, config.max_context_num+1, config.dim)
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
        support = torch.matmul(A, H)
        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output

    def save_parameters(self, parameter_path):
        if not os.path.exists(parameter_path):
            os.makedirs(parameter_path)

        ent_f = open(os.path.join(parameter_path, 'entity_o'), "w")
        ent_f.write(json.dumps(self.pht_o))
        ent_f.close()

        rel_f = open(os.path.join(parameter_path, 'relation_o'), "w")
        rel_f.write(json.dumps(self.pr_o))
        rel_f.close()

        para2vec = {}
        lists = self.state_dict()
        for var_name in lists:
            para2vec[var_name] = lists[var_name].cpu().numpy().tolist()

        f = open(os.path.join(parameter_path, 'all_parameters'), "w")
        f.write(json.dumps(para2vec))
        f.close()

    def save_phrt_o(self, pos_h, pos_r, pos_t, ph_o, pr_o, pt_o):
        for i in range(len(pos_h)):
            h = str(int(pos_h[i]))
            self.pht_o[h] = ph_o[i].detach().cpu().numpy().tolist()

            t = str(int(pos_t[i]))
            self.pht_o[t] = pt_o[i].detach().cpu().numpy().tolist()

            r = str(int(pos_r[i]))
            self.pr_o[r] = pr_o[i].detach().cpu().numpy().tolist()

    def forward(self, epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A):
        # multi golden and multi negative
        pos_h, pos_r, pos_t = golden_triples
        neg_h, neg_r, neg_t = negative_triples

        p_h = self.entity_emb[pos_h.cpu().numpy()]
        p_t = self.entity_emb[pos_t.cpu().numpy()]
        p_r = self.relation_emb[pos_r.cpu().numpy()]
        n_h = self.entity_emb[neg_h.cpu().numpy()]
        n_t = self.entity_emb[neg_t.cpu().numpy()]
        n_r = self.relation_emb[neg_r.cpu().numpy()]

        ph_adj_entity_list = self.get_entity_context(pos_h)
        pt_adj_entity_list = self.get_entity_context(pos_t)
        nh_adj_entity_list = self.get_entity_context(neg_h)
        nt_adj_entity_list = self.get_entity_context(neg_t)
        pr_adj_relation_list = self.get_relation_context(pos_r)
        nr_adj_relation_list = self.get_relation_context(neg_r)

        ph_adj_entity_vec_list = self.get_adj_entity_vec(p_h, ph_adj_entity_list)
        pt_adj_entity_vec_list = self.get_adj_entity_vec(p_t, pt_adj_entity_list)
        nh_adj_entity_vec_list = self.get_adj_entity_vec(n_h, nh_adj_entity_list)
        nt_adj_entity_vec_list = self.get_adj_entity_vec(n_t, nt_adj_entity_list)
        pr_adj_relation_vec_list = self.get_adj_relation_vec(p_r, pr_adj_relation_list)
        nr_adj_relation_vec_list = self.get_adj_relation_vec(n_r, nr_adj_relation_list)

        # gcn
        ph_adj_entity_vec_list = self.gcn(ph_A, ph_adj_entity_vec_list, target='entity')
        pt_adj_entity_vec_list = self.gcn(pt_A, pt_adj_entity_vec_list, target='entity')
        nh_adj_entity_vec_list = self.gcn(nh_A, nh_adj_entity_vec_list, target='entity')
        nt_adj_entity_vec_list = self.gcn(nt_A, nt_adj_entity_vec_list, target='entity')
        pr_adj_relation_vec_list = self.gcn(pr_A, pr_adj_relation_vec_list, target='relation')
        nr_adj_relation_vec_list = self.gcn(nr_A, nr_adj_relation_vec_list, target='relation')

        ph_sg = self.calc_subgraph_vec(p_h, ph_adj_entity_vec_list, target='entity')
        pt_sg = self.calc_subgraph_vec(p_t, pt_adj_entity_vec_list, target='entity')
        nh_sg = self.calc_subgraph_vec(n_h, nh_adj_entity_vec_list, target='entity')
        nt_sg = self.calc_subgraph_vec(n_t, nt_adj_entity_vec_list, target='entity')
        pr_sg = self.calc_subgraph_vec(p_r, pr_adj_relation_vec_list, target='relation')
        nr_sg = self.calc_subgraph_vec(n_r, nr_adj_relation_vec_list, target='relation')

        ph_o = torch.mul(F.sigmoid(self.gate_entity), p_h) + torch.mul(1 - F.sigmoid(self.gate_entity), ph_sg)
        pt_o = torch.mul(F.sigmoid(self.gate_entity), p_t) + torch.mul(1 - F.sigmoid(self.gate_entity), pt_sg)
        nh_o = torch.mul(F.sigmoid(self.gate_entity), n_h) + torch.mul(1 - F.sigmoid(self.gate_entity), nh_sg)
        nt_o = torch.mul(F.sigmoid(self.gate_entity), n_t) + torch.mul(1 - F.sigmoid(self.gate_entity), nt_sg)
        pr_o = torch.mul(F.sigmoid(self.gate_relation), p_r) + torch.mul(1 - F.sigmoid(self.gate_relation), pr_sg)
        nr_o = torch.mul(F.sigmoid(self.gate_relation), n_r) + torch.mul(1 - F.sigmoid(self.gate_relation), nr_sg)

        # score for loss
        p_score = self._calc(ph_o, pt_o, pr_o)
        n_score = self._calc(nh_o, nt_o, nr_o)

        if epoch == config.train_times-1:
            self.save_phrt_o(pos_h, pos_r, pos_t, ph_o, pr_o, pt_o)

        return p_score, n_score


def main():
    print('preparing data...')
    phs, prs, pts, nhs, nrs, nts = config.prepare_data()
    print('preparing data complete')

    print('train starting...')
    dynamicKGE = DynamicKGE(config).cuda()

    if config.optimizer == "SGD":
        optimizer = optim.SGD(dynamicKGE.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(dynamicKGE.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adagrad":
        optimizer = optim.Adagrad(dynamicKGE.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(dynamicKGE.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(dynamicKGE.parameters(), lr=config.learning_rate)

    criterion = nn.MarginRankingLoss(config.margin, False).cuda()

    for epoch in range(config.train_times):
        # print(dynamicKGE.entity_context(torch.LongTensor([0]).cuda()))
        # print(dynamicKGE.entity_context(torch.LongTensor([config.entity_total-1]).cuda()))
        # print(dynamicKGE.entity_context(torch.LongTensor([config.entity_total]).cuda()))
        start_time = time.time()
        print('----------training the ' + str(epoch) + ' epoch----------')
        epoch_avg_loss = 0.0
        for batch in range(config.nbatchs):
            optimizer.zero_grad()
            golden_triples, negative_triples = config.get_batch(config.batch_size, batch, epoch, phs, prs, pts, nhs, nrs, nts)
            ph_A, pr_A, pt_A = config.get_batch_A(golden_triples, config.entity_A, config.relation_A)
            nh_A, nr_A, nt_A = config.get_batch_A(negative_triples, config.entity_A, config.relation_A)

            p_scores, n_scores = dynamicKGE(epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)
            y = torch.Tensor([-1]).cuda()
            loss = criterion(p_scores, n_scores, y)

            loss.backward()
            optimizer.step()

            epoch_avg_loss += (float(loss.item()) / config.nbatchs)
            torch.cuda.empty_cache()
        end_time = time.time()

        print('----------epoch avg loss: ' + str(epoch_avg_loss) + ' ----------')
        print('----------epoch training time: ' + str(end_time-start_time) + ' s --------\n')

    print('train ending...')

    dynamicKGE.save_parameters(config.res_dir)

    entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)
    print('test link prediction starting...')
    test.test_link_prediction(config.test_list, set(config.train_list), entity_emb, relation_emb, config.norm)
    print('test link prediction ending...')


main()
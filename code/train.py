import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import test
from config import config
from transe import TransE, convert_dict_numpy
from util.parameter_util import *
from torch.utils.data import DataLoader
import time
from models import DynamicKGE
from tqdm import tqdm
from util.train_util import get_batch_A, GraphDataSet
from tensorboardX import SummaryWriter

def evaluate(config):

    entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)
    print('test link prediction starting...')
    test.test_link_prediction(config.test_list, set(config.train_list), entity_emb, relation_emb, config.norm)
    print('test link prediction ending...')


def main():
    print('preparing data...')
    cache_file = os.path.join('cache', 'training_data_{}_{}_{}_{}.pt'.format(config.dataset_v1.replace('/', ''), 
        config.entity_total, config.relation_total, config.max_context ))
    if os.path.exists(cache_file):
        phs, prs, pts, nhs, nrs, nts = torch.load(cache_file)
    else:
        phs, prs, pts, nhs, nrs, nts = config.prepare_data()
        torch.save((phs, prs, pts, nhs, nrs, nts), cache_file)

    print('preparing data complete')

    print('train starting...')
    print(config.entity_total, config.relation_total)
    dataset = GraphDataSet(cache_file, os.path.join('data',  config.dataset_v1))
    dataloader = DataLoader(dataset, 
        batch_size=config.batch_size, 
        num_workers=10)
    
    entity_id2tokens = dataset.mapping.entity2tokens
    rel_id2tokens = dataset.mapping.relations2tokens
    dynamicKGE = DynamicKGE(config, entity_id2tokens, rel_id2tokens)
 
    if config.test_mode:
        evaluate(config)
        exit(0)

    dynamicKGE = dynamicKGE.cuda()

    if config.optimizer == "SGD":
        optimizer = optim.SGD(dynamicKGE.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adam":
        print('Use Adam')
        optimizer = optim.Adam(dynamicKGE.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adagrad":
        optimizer = optim.Adagrad(dynamicKGE.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(dynamicKGE.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(dynamicKGE.parameters(), lr=config.learning_rate)

    criterion = nn.MarginRankingLoss(config.margin, False).cuda()

    os.makedirs(os.path.join('logging', config.dataset_v1), exist_ok=True)
    writer = SummaryWriter(os.path.join('logging', config.args.name))
    step = 0

    for epoch in range(config.train_times):
        start_time = time.time()
        print('----------training the ' + str(epoch) + ' epoch----------')
        epoch_avg_loss = 0.0
        with tqdm(total=len(dataloader), dynamic_ncols=True) as pbar:
            for batch in dataloader:

                golden_triples = ( batch['h'], batch['r'], batch['t'])
                golden_triples = [t.cuda() for t in golden_triples]
                negative_triples = ( batch['nh'], batch['nr'], batch['nt'] )
                negative_triples = [t.cuda() for t in negative_triples]

                optimizer.zero_grad()

                ph_A, pr_A, pt_A = config.get_batch_A(golden_triples, config.entity_A, config.relation_A)
                nh_A, nr_A, nt_A = config.get_batch_A(negative_triples, config.entity_A, config.relation_A)

                ph_A, pr_A, pt_A = ph_A.cuda(), pr_A.cuda(), pt_A.cuda()
                nh_A, nr_A, nt_A = nh_A.cuda(), nr_A.cuda(), nt_A.cuda()
                p_scores, n_scores = dynamicKGE(golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)
                y = torch.Tensor([-1]).cuda()
                loss = criterion(p_scores, n_scores, y)

                loss.backward()
                optimizer.step()

                epoch_avg_loss += (float(loss.item()) / config.nbatchs)
                writer.add_scalar('loss', loss.item(), step)
                step += 1
                pbar.set_description("loss={:.2f}".format(loss.item()/ config.nbatchs ))
                pbar.update(1)

        end_time = time.time()

        if epoch % 10 == 0:
            dynamicKGE.save_parameters(config.res_dir)

        print('----------epoch avg loss: ' + str(epoch_avg_loss) + ' ----------')
        print('----------epoch training time: ' + str(end_time-start_time) + ' s --------\n')

    print('train ending...')
    dynamicKGE.save_parameters(config.res_dir)

    evaluate(config)
main()

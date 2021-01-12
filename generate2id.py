import torch


if __name__ == '__main__':

    with open('code/data/DBpedia-3SPv2/snapshot1/entity2id.txt', 'r') as f:
        entity2id = {}
        for line in f.readlines()[1:]:
            key, index = line.strip().split()
            entity2id[key] = int(index)
        torch.save(entity2id, 'code/data/DBpedia-3SPv2/snapshot1/entity2id.pt')

    with open('code/data/DBpedia-3SPv2/snapshot1/relation2id.txt', 'r') as f:
        relation2id = {}
        for line in f.readlines()[1:]:
            key, index = line.strip().split()
            relation2id[key] = int(index)
        torch.save(relation2id, 'code/data/DBpedia-3SPv2/snapshot1/rel2id.pt')

    # with open('code/data/FB15K-237-2/snapshot2/entity2id.txt', 'r') as f:
    #     entity2id = {}
    #     for line in f.readlines()[1:]:
    #         key, index = line.strip().split()
    #         entity2id[key] = int(index)
    #     torch.save(entity2id, 'code/data/FB15K-237-2/snapshot2/entity2id.pt')

    # with open('code/data/FB15K-237-2/snapshot2/relation2id.txt', 'r') as f:
    #     relation2id = {}
    #     for line in f.readlines()[1:]:
    #         key, index = line.strip().split()
    #         relation2id[key] = int(index)
    #     torch.save(relation2id, 'code/data/FB15K-237-2/snapshot2/rel2id.pt')
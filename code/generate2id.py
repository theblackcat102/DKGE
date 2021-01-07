import torch


if __name__ == '__main__':

    with open('data/DBpedia-3SP/snapshot1/entity2id.txt', 'r') as f:
        entity2id = {}
        for line in f.readlines()[1:]:
            if len(line.strip().split()) > 1:
                key, index = line.strip().split()
                entity2id[key] = int(index)
        torch.save(entity2id, 'data/DBpedia-3SP/snapshot1/entity2id.pt')

    with open('data/DBpedia-3SP/snapshot1/relation2id.txt', 'r') as f:
        relation2id = {}
        for line in f.readlines()[1:]:
            if len(line.strip().split()) > 1:
                key, index = line.strip().split()
                relation2id[key] = int(index)
        torch.save(relation2id, 'data/DBpedia-3SP/snapshot1/rel2id.pt')

    with open('data/DBpedia-3SP/snapshot2/entity2id.txt', 'r') as f:
        entity2id = {}
        for line in f.readlines()[1:]:
            if len(line.strip().split()) > 1:
                key, index = line.strip().split()
                entity2id[key] = int(index)
        torch.save(entity2id, 'data/DBpedia-3SP/snapshot2/entity2id.pt')

    with open('data/DBpedia-3SP/snapshot2/relation2id.txt', 'r') as f:
        relation2id = {}
        for line in f.readlines()[1:]:
            if len(line.strip().split()) > 1:

                key, index = line.strip().split()
                relation2id[key] = int(index)
        torch.save(relation2id, 'data/DBpedia-3SP/snapshot2/rel2id.pt')
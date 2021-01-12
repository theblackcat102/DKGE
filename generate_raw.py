import os

def generate_train(datapath='code/data/DBpedia-3SPv2/snapshot1'):
    id2entity = {}
    id2relation = {}
    with open(os.path.join(datapath, 'entity2id.txt')) as f:
        for line in f.readlines():
            if len(line.split('\t')) > 1:
                name, id_ = line.strip().split('\t')
                id2entity[id_] = name
    print(len(id2entity))
    with open(os.path.join(datapath, 'relation2id.txt')) as f:
        for line in f.readlines():
            if len(line.split('\t')) > 1:
                name, id_ = line.strip().split('\t')
                id2relation[id_] = name
    print(len(id2relation))
    with open(os.path.join(datapath, 'train2id.txt'), 'r') as f, \
        open(os.path.join(datapath, 'train.txt'), 'w') as g:
        for line in f.readlines():
            print(line.split(' '))
            if len(line.split(' ')) > 1:
                h, t, r = line.strip().split(' ')
                g.write('{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t]))

    with open(os.path.join(datapath, 'test2id.txt'), 'r') as f, \
        open(os.path.join(datapath, 'test.txt'), 'w') as g:
        for line in f.readlines():
            print(line.split(' '))
            if len(line.split(' ')) > 1:
                h, t, r = line.strip().split(' ')
                g.write('{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t]))
    with open(os.path.join(datapath, 'valid2id.txt'), 'r') as f, \
        open(os.path.join(datapath, 'valid.txt'), 'w') as g:
        for line in f.readlines():
            print(line.split(' '))
            if len(line.split(' ')) > 1:
                h, t, r = line.strip().split(' ')
                g.write('{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t]))

if __name__ == '__main__':
    generate_train()
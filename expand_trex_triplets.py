import os
import glob


def rebuild_entity2id(path):
    entity2id_filename = os.path.join(path, 'entity2id.txt')
    entity2id = {}
    with open(entity2id_filename, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            name, idx = line.strip().split()
            idx = int(idx)
            entity2id[name] = idx

    for data_file in ['valid.txt', 'test.txt', 'train.txt']:
        valid_filename = os.path.join(path, data_file)
        with open(valid_filename, 'r') as f:
            for idx, line in enumerate(f):
                split = line.strip().split()
                if len(split) == 3:
                    h, r, t = split
                    if h not in entity2id:
                        print(h)
                        entity2id[h] = len(entity2id)
                    if t not in entity2id:
                        print(t)
                        entity2id[t] = len(entity2id)

    with open(entity2id_filename, 'w') as f:
        f.write(str(len(entity2id))+'\n')
        for key, idx in entity2id.items():
            f.write('{}\t{}\n'.format(key, idx))



def rebuild_relation2id(path):
    relation2id_filename = os.path.join(path, 'relation2id.txt')
    relation2id = {}
    with open(relation2id_filename, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            name, idx = line.strip().split()
            idx = int(idx)
            relation2id[name] = idx

    for data_file in ['valid.txt', 'test.txt', 'train.txt']:
        valid_filename = os.path.join(path, data_file)
        with open(valid_filename, 'r') as f:
            for idx, line in enumerate(f):
                split = line.strip().split()
                if len(split) == 3:
                    h, r, t = split
                    if r not in relation2id:
                        print(r)
                        relation2id[h] = len(relation2id)

    with open(relation2id_filename, 'w') as f:
        f.write(str(len(relation2id))+'\n')
        for key, idx in relation2id.items():
            f.write('{}\t{}\n'.format(key, idx))


def load_trex_mapping():
    wiki2dbpedia = {}
    property2dbpedia = {}
    with open('all_wiki2dbpedia/wiki_id2dbpedia.txt') as f:
        for line in f:
            wiki_id, dbpedia_url = line.strip().split(',', 1)
            wiki2dbpedia[wiki_id] = '<'+dbpedia_url+'>'
    with open('wiki_prop2dbpedia.txt') as f:
        for line in f:
            wiki_id, dbpedia_url = line.strip().split(',', 1)
            property2dbpedia[wiki_id]= '<'+dbpedia_url+'>'

    return wiki2dbpedia, property2dbpedia


def load_trex_triplets():
    triplets = []
    with open('trex_triplets.txt') as f:
        for triplet in f:
            h, r, t = triplet.strip().split(',')
            triplets.append((h, r, t))
    return triplets

if __name__ == "__main__":

    wiki2dbpedia, property2dbpedia = load_trex_mapping()
    triplets = load_trex_triplets()
    with open('trex_dbpedia_triplets.txt', 'w') as f:
        for t in triplets:
            h, r, t = t
            if h in wiki2dbpedia and r in property2dbpedia and t in wiki2dbpedia:
                f.write('{}\t{}\t{}\n'.format( wiki2dbpedia[h],property2dbpedia[r], wiki2dbpedia[t]  ))


    rebuild_entity2id('code/data/DBpedia-3SPv2/snapshot1')
    rebuild_relation2id('code/data/DBpedia-3SPv2/snapshot1')

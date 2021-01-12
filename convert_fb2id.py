import os


if __name__ == '__main__':
    entity2id = {}
    relation2id = {}
    triplets = []
    with open('../kgs/FB15k-237/train.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '/':
                h, r, t = line.strip().split('\t')
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                triplets.append( (entity2id[h], relation2id[r], entity2id[t])  )

    with open('../kgs/FB15k-237/test.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '/':
                h, r, t = line.strip().split('\t')
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)

    with open('../kgs/FB15k-237/valid.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '/':
                h, r, t = line.strip().split('\t')
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)

    with open('code/data/FB15K-237-2/snapshot1-baseline/train2id.txt', 'w') as f:
        f.write(str(len(triplets))+'\n')
        for triplet in triplets:
            f.write('{}\t{}\t{}\n'.format(*triplet))

    with open('code/data/FB15K-237-2/snapshot1-baseline/entity2id.txt', 'w') as f:
        f.write(str(len(entity2id))+'\n')
        for key, idx in entity2id.items():
            f.write('{}\t{}\n'.format(key, idx))

    with open('code/data/FB15K-237-2/snapshot1-baseline/relation2id.txt', 'w') as f:
        f.write(str(len(relation2id))+'\n')
        for key, idx in relation2id.items():
            f.write('{}\t{}\n'.format(key, idx))

    test_triplets = []
    with open('../kgs/FB15k-237/test.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '/':
                h, r, t = line.strip().split('\t')
                test_triplets.append( (entity2id[h], relation2id[r], entity2id[t])  )

    with open('code/data/FB15K-237-2/snapshot1-baseline/test2id.txt', 'w') as f:
        f.write(str(len(test_triplets))+'\n')
        for triplet in test_triplets:
            f.write('{}\t{}\t{}\n'.format(*triplet))

    test_triplets = []
    with open('../kgs/FB15k-237/valid.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '/':
                h, r, t = line.strip().split('\t')
                test_triplets.append( (entity2id[h], relation2id[r], entity2id[t])  )

    with open('code/data/FB15K-237-2/snapshot1-baseline/valid2id.txt', 'w') as f:
        f.write(str(len(test_triplets))+'\n')
        for triplet in test_triplets:
            f.write('{}\t{}\t{}\n'.format(*triplet))

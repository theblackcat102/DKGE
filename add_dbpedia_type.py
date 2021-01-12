import os, glob
import json
import random


valid_entity_type = []
with open('valid_entities_list.txt', 'r') as f:
    for line in f.readlines():
        valid_entity_type.append(line.strip())
valid_entity_type = set(valid_entity_type)

data_path = 'code/data/DBpedia-3SPv2/snapshot1/'

entity2id_filename = 'entity2id.txt'
relation2id_filename = 'relation2id.txt'

train2id_filename = 'train2id.txt'
valid2id_filename = 'valid2id.txt'
test2id_filename = 'test2id.txt'


train_filename = 'train.txt'
valid_filename = 'valid.txt'
test_filename = 'test.txt'


RDF_TYPE_NAME = 'http://purl.org/dc/terms/subject'
RDF_TYPE_NAME = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
print(RDF_TYPE_NAME)

entity2wikipedia = 'entity2wikidata.json'

BLACKLIST_RELATION = [
    'http://dbpedia.org/property/wikiPageUsesTemplate',
    'http://dbpedia.org/ontology/wikiPageExternalLink',
    'http://dbpedia.org/ontology/wikiPageID',
]

def find_dbpedia_json(name, dbpedia_path = 'dbpedia'):
    if '<' == name[0]:
        filename = name.replace('http://dbpedia.org/','').replace('<', '').replace('resource/', '').replace('>', '').replace('/','')
    else:
        filename = name.replace('http://dbpedia.org/','').replace('<', '').replace('resource/', '').replace('>', '').replace('/','')
    file_path = os.path.join(dbpedia_path, filename+'.json')
    all_triplets = []

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                dbpedia_payload = json.load(f)
            except json.decoder.JSONDecodeError:
                print(file_path)
                dbpedia_payload = {}

        if len(dbpedia_payload) == 0:
            return all_triplets
        all_keys = list(dbpedia_payload.keys())

        for key in all_keys:
            if 'http://dbpedia.org/resource/' in key:
                break
        if RDF_TYPE_NAME in dbpedia_payload[key]:
            for entity_type in dbpedia_payload[key][RDF_TYPE_NAME]:
                predicate = entity_type['value']
                if 'http://www.wikidata.org/entity/' not in predicate:
                    all_triplets.append( (  name, RDF_TYPE_NAME, '<'+ predicate +'>'  ) )
                    all_triplets.append( (  '<'+ predicate +'>', RDF_TYPE_NAME, name  ) )

        # do only once
        root_name = name.replace('<', '').replace('>', '')
        if root_name in dbpedia_payload and RDF_TYPE_NAME == 'http://purl.org/dc/terms/subject':
            for key, values in dbpedia_payload[root_name].items():
                if ('/property/' in key or '/ontology/' in key ) and key not in BLACKLIST_RELATION:
                    for value in values:
                        if 'value' in value and isinstance(value['value'], str) and '/resource/' in value['value']:
                            triplet = (  '<'+ root_name +'>', '<'+key+'>', '<'+value['value'] +'>' )
                            if triplet not in all_triplets and triplet[0] in valid_entity_type and triplet[2] in valid_entity_type:
                                all_triplets.append(triplet )
            for key, value in dbpedia_payload.items():
                if 'http://dbpedia.org/resource/' in key and isinstance(value, dict):
                    root = key
                    for relation, values in dbpedia_payload[root].items():
                        if isinstance(values, list) and ('/property/' in relation or '/ontology/' in relation ) and relation not in BLACKLIST_RELATION:
                            for value in values:
                                if 'value' in value and isinstance(value['value'], str) and 'http://dbpedia.org/resource/' in value['value']:
                                    triplet = (  '<'+ root +'>', '<'+relation+'>', '<'+value['value'] +'>' )
                                    if triplet not in all_triplets and triplet[0] in valid_entity_type and triplet[2] in valid_entity_type:
                                        all_triplets.append( triplet )
    return all_triplets


def read_file(file_name, to_int=True):
    data = []  # [(h, r, t)]
    is_freebase_dataset = False
    if 'FB15' in file_name:
        is_freebase_dataset = True
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            li = line.split()
            if len(li) == 3:
                if to_int:
                    if is_freebase_dataset:# h, r, t
                        data.append((int(li[0]), int(li[1]), int(li[2])))
                    else: # h, t, r
                        data.append((int(li[0]), int(li[2]), int(li[1])))
                else:
                    data.append((li[0], li[1], li[2]))

    return data


def convert_triplet_name2id(triplets, entity2id, relation2id):
    triplets_int = []
    for (h, r, t) in triplets:
        if h not in entity2id:
            entity2id[h] = len(entity2id)
        if r not in relation2id:
            relation2id[r] = len(relation2id)
        if t not in entity2id:
            entity2id[r] = len(entity2id)

        triplets_int.append(( entity2id[h], relation2id[r], entity2id[t]) )
    return triplets_int, entity2id, relation2id

def write_file(file_name, data, htr_mode=True):
    with open(file_name, 'w') as f:
        f.write(str(len(data)+1)+'\n')
        for (h, r, t) in data:
            # write as h, t, r
            if htr_mode:
                f.write('{}\t{}\t{}\n'.format(h, t, r))
            else:
                f.write('{}\t{}\t{}\n'.format(h, r, t))

if __name__ == '__main__':
    entity2id_path = os.path.join(data_path, entity2id_filename)
    relation2id_path = os.path.join(data_path, relation2id_filename)
    entity2wikipedia_path = os.path.join(data_path, entity2wikipedia)
    with open(entity2wikipedia_path, 'r') as f:
        entity2wikipedia = json.load(f)
            
    entity2wikipedia_map = {}

    for key, value in entity2wikipedia.items():
        if 'wikidata_id' in value:
            entity2wikipedia_map[key] = value['wikidata_id']

    # print(list(entity2wikipedia.keys())[:10])
    # print(entity2wikipedia['<http://dbpedia.org/resource/Argentina_national_football_team>'])
    entity2id_map = {}

    with open(entity2id_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue

            name, idx  = line.strip().split('\t', 1)
            entity2id_map[name] = int(idx)
    assert len(entity2id_map) > 100
    init_entity2id_map_size = len(entity2id_map)
    
    relation2id_map = {}
    with open(relation2id_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue

            name, idx  = line.strip().split('\t', 1)
            relation2id_map[name] = int(idx)
    init_relation2id_map_size = len(relation2id_map)
    # print(relation2id_map)
    assert len(relation2id_map) > 100

    if RDF_TYPE_NAME not in relation2id_map:
        print('write rdf type name to relation2id')
        relation2id_map[RDF_TYPE_NAME] = len(relation2id_map)

        with open(relation2id_path, 'w') as f:
            f.write(str(len(relation2id_map)+1)+'\n')
            for key, idx in relation2id_map.items():
                f.write(key+'\t'+str(idx)+'\n')

    new_triplets = []
    for name, idx in entity2id_map.items():
        new_triplets += find_dbpedia_json(name)


    print('write new triplets to entity2id')

    for (h, r, t) in new_triplets:
        if t not in entity2id_map:
            entity2id_map[t] = len(entity2id_map)

    with open(entity2id_path, 'w') as f:
        f.write(str(len(entity2id_map)+1)+'\n')
        for key, idx in entity2id_map.items():
            f.write(key+'\t'+str(idx)+'\n')

    # DKGE dataset is written as h, t, r
    training_triplets = []

    train2id_files = [ os.path.join(data_path, train2id_filename),
                    os.path.join(data_path, valid2id_filename),
                    os.path.join(data_path, test2id_filename),
                ]
    train_files = [ os.path.join(data_path, train_filename),
                    os.path.join(data_path, valid_filename),
                    os.path.join(data_path, test_filename),
                ]

    # data read as h, r, t
    for filename in train2id_files:
        training_triplets += read_file(filename)

    if new_triplets[0] not in training_triplets:

        random.shuffle(new_triplets)

        split_1 = int(0.9 * len(new_triplets))
        split_2 = int(0.95 * len(new_triplets))

        train_triplets, entity2id_map, relation2id_map = convert_triplet_name2id(new_triplets[:split_1], entity2id_map, relation2id_map)
        dev_triplets, entity2id_map, relation2id_map = convert_triplet_name2id(new_triplets[split_1:split_2], entity2id_map, relation2id_map)
        test_triplets, entity2id_map, relation2id_map = convert_triplet_name2id(new_triplets[split_2:], entity2id_map, relation2id_map)

        print(len(entity2id_map), init_entity2id_map_size)
        print(len(relation2id_map), init_relation2id_map_size)

        data_triplets = read_file(train2id_files[0]) # train
        data_triplets += train_triplets
        write_file(train2id_files[0], data_triplets)

        data_triplets = read_file(train2id_files[1]) # valid
        data_triplets += dev_triplets
        write_file(train2id_files[1], data_triplets)

        data_triplets = read_file(train2id_files[2]) # test
        data_triplets += test_triplets
        write_file(train2id_files[2], data_triplets)

        print(len(data_triplets))

        data_triplets = read_file(train_files[0], to_int=False) # train
        data_triplets += new_triplets[:split_1]
        write_file(train_files[0], data_triplets, htr_mode=False)

        data_triplets = read_file(train_files[1], to_int=False) # valid
        data_triplets += new_triplets[split_1:split_2]
        write_file(train_files[1], data_triplets, htr_mode=False)

        data_triplets = read_file(train_files[2], to_int=False) # test
        data_triplets += new_triplets[split_2:]
        write_file(train_files[2], data_triplets, htr_mode=False)


        if len(entity2id_map) != init_entity2id_map_size:
            print('update entity2id map')
            with open(entity2id_path, 'w') as f:
                f.write(str(len(entity2id_map))+'\n')
                for key, idx in entity2id_map.items():
                    f.write('{}\t{}\n'.format(key, idx))

        if len(relation2id_map) != init_relation2id_map_size:
            print('update relation2id map')

            with open(relation2id_path, 'w') as f:
                f.write(str(len(relation2id_map))+'\n')
                for key, idx in relation2id_map.items():
                    f.write('{}\t{}\n'.format(key, idx))

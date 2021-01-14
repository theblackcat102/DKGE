import json
import os
import pandas as pd
import requests
from time import sleep
import random
from tqdm import tqdm
from multiprocessing import Pool


CRAWL = False
print('CRAWL ', CRAWL)
def read_csv(filename_path):
    WD_id = []
    context = []
    with open(filename_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx > 0:
                tokens = line.split('|')
                if 'defs.csv' in filename_path:
                    WD_id.append(tokens[1].strip())
                    context.append(tokens[0].strip())
                else:
                    WD_id.append(tokens[0])
                    context.append(tokens[-1].strip())

    return pd.DataFrame({
        'WD_id': WD_id,
        'context': context
    })


def build_entity2wikidata(
        entity2id_file = 'code/data/FB15K-237-2/snapshot3/entity2id.txt',
        output_file = 'code/data/FB15K-237-2/snapshot3/entity2wikidata.json'):
    entity_description_path = '/mnt/storage/wiki_data/en_large_output/entity_descriptions.csv'

    entity_alias_path = '/mnt/storage/wiki_data/en_large_output/entity_alias.csv'
    entity_definitions_path = '/mnt/storage/wiki_data/en_large_output/entity_defs.csv'


    entity_description_df = read_csv(entity_description_path)
    print(entity_description_df[entity_description_df['WD_id'] == 'Q4916'])

    entity_alias_df = read_csv(entity_alias_path)
    entity_definitions_df = read_csv(entity_definitions_path)


    original_id = 'code/data/FB15K-237-2/snapshot1/entity2wikidata.json'
    with open(original_id, 'r') as f:
        fb15k_entity2wikidata = json.load(f)
    
    entity2wikidata = {}

    with open(entity2id_file, 'r') as f:
        for idx, line in enumerate(f):
            if len(line.split('\t')) > 1:
                entity_name = line.split('\t')[0]
                print(idx, entity_name)
                if entity_name in fb15k_entity2wikidata:
                    entity2wikidata[entity_name] = fb15k_entity2wikidata[entity_name]
                elif len(entity_alias_df['WD_id'] == entity_name) > 0:
                    alias = [ row['context'] for _, row in entity_alias_df[entity_alias_df['WD_id'] == entity_name].iterrows() ]
                    description = entity_description_df[entity_description_df['WD_id'] == entity_name].iloc[0]['context']
                    definition = entity_definitions_df[entity_definitions_df['WD_id'] == entity_name].iloc[0]['context']
                    entity2wikidata[entity_name] = {
                        'alternatives': alias,
                        'description': description,
                        'label': definition,
                        'wikidata_id': entity_name
                    }
                else:
                    print(entity_name, ' not found')

            if idx % 1000 == 0:
                with open(output_file, 'w') as f:
                    json.dump(entity2wikidata, f)

    with open(output_file, 'w') as f:
        json.dump(entity2wikidata, f)



def parse_dbpedia(inputs):
    if isinstance(inputs, tuple):
        url = inputs[1]
    else:
        url = inputs

    data = None
    output_filename = 'dbpedia/'+inputs[0].replace('http://dbpedia.org/','').replace('<', '').replace('resource/', '').replace('>', '').replace('/','') +'.json'
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                data = {}
    else:
        if not CRAWL:
            return ( inputs[0], {} )
        else:
            target_url = url.replace('resource', 'data') + '.json'
            for idx in range(10):
                sleep(0.1)
                try:
                    data = requests.get(target_url).json()
                except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
                    continue
                if data is None or (isinstance(data,dict) and len(data) == 0):
                    continue

            if data is not None and len(data) > 0:
                with open(output_filename, 'w') as f:
                    json.dump(data, f)

    if data is None:
        with open('failed_dbpedia_url.txt', 'a') as f:
            f.write(url+'\n')
        if isinstance(inputs, tuple):
            return ( inputs[0], {} )
        return {}


    results = {}
    for key, details in data.items():
        for detail_key, data in details.items():
            if 'owl#sameAs' in detail_key:
                for same_as in data:
                    if 'wikidata.org' in same_as['value']:
                        results['wikidata_id'] = same_as['value'].split('/')[-1]
                        break
            if 'http://xmlns.com/foaf/0.1/name' == detail_key:
                results['label'] = data[0]['value']
            if 'http://purl.org/dc/terms/description' == detail_key:
                results['description'] = data[0]['value']
            if 'http://www.w3.org/2000/01/rdf-schema#label' == detail_key:
                results['alternatives'] = [ d['value'] for d in data ]
            if 'http://xmlns.com/foaf/0.1/isPrimaryTopicOf' == detail_key:
                for prim_topic in data:
                    if 'wikipedia.org' in prim_topic['value']:
                        results['wikipedia'] = prim_topic['value']
                        break
    if isinstance(inputs, tuple):
        return ( inputs[0], results )
    return results


def build_dbpedia_entity2wikidata(
        entity2id_file = 'code/data/DBpedia-3SP/snapshot3/entity2id.txt',
        output_file = 'code/data/DBpedia-3SP/snapshot3/entity2wikidata.json'):
    params = []

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)

    with open(entity2id_file, 'r') as f:
        for line in f.readlines():
            if '<' not in line:
                continue
            key, idx = line.split('\t')
            url = key.replace('<','').replace('>','')
            if key not in results:
                params.append((key, url))
    random.shuffle(params)

    # with open('params_data.json', 'r') as f:
    #     params = json.load(f)
    # params = params[:len(params)//2]
    
    cnt = 0
    with Pool(6) as pool:
        for result in tqdm(pool.imap(parse_dbpedia,params), dynamic_ncols=True, total=len(params)):
            key, result = result
            if len(result) < 3:
                with open('failed_dbpedia_url.txt', 'a') as f:
                    f.write(url+'\n')
            else:
                results[ key ] = result
                cnt += 1

            if cnt % 1000 == 0 and not CRAWL:
                with open(output_file, 'w') as f:
                    json.dump(results, f)
    if not CRAWL:
        with open(output_file, 'w') as f:
            json.dump(results, f)

def build_dbpedia_relation2wikidata(
        entity2id_file = 'code/data/DBpedia-3SP/snapshot1/relation2id.txt',
        output_file = 'code/data/DBpedia-3SP/snapshot1/relation2wikidata.json'):
    params = []

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)

    with open(entity2id_file, 'r') as f:
        for line in f.readlines():
            if '<' not in line:
                continue


            name = line.strip().split('\t')
            key = name[0]

            results[key] = {
                'label': name,
                'alternatives': []
            }
            # break
    with open(output_file, 'w') as f:
        json.dump(results, f)


def build_relation2wikidata(
    relation2id_file='code/data/FB15K-237-2/snapshot2/relation2id.txt',
    output_file = 'code/data/FB15K-237-2/snapshot2/relation2wikidata.json'
    ):
    relation2wikidata = {}
    with open(relation2id_file, 'r') as f:
        for idx, line in enumerate(f):
            if len(line.split('\t')) > 1:
                tokens = line.split('\t')
                if line[0] == '/' and line[0] != 'P':
                    relation2wikidata[tokens[0]] = {
                        'label': tokens[0].replace('/', ' '),
                        'alternatives': []
                    }
                else:
                    for idx in range(10):
                        sleep(0.1)
                        try:
                            raw_html = requests.get('https://www.wikidata.org/wiki/Property:'+tokens[0]).text
                        except (requests.exceptions.ConnectionError):
                            continue
                    label_idx = raw_html.find('<span class="wikibase-title-label">')
                    label_idx += len('<span class="wikibase-title-label">')

                    label_idx_end = raw_html.find('</span>', label_idx)
                    label = raw_html[label_idx:label_idx_end][:501]

                    alias_start = raw_html.find('<li class="wikibase-entitytermsview-aliases-alias" data-aliases-separator="|">')
                    alias_start += len('<li class="wikibase-entitytermsview-aliases-alias" data-aliases-separator="|">')

                    alias_end = raw_html.find('</li>', alias_start)
                    if (alias_end-alias_start) > 1000:
                        alias = []
                    else:
                        alias = [ raw_html[alias_start:alias_end][:501]]

                    relation2wikidata[tokens[0]] = {
                        'label': label,
                        'alternatives': alias
                    }
    with open(output_file, 'w') as f:
        json.dump(relation2wikidata, f)

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


def download_dbpedia_json():
    params = []
    with open('wiki_id2dbpedia.txt', 'r') as f:
        for line in f:
            wikidata_id, dbpedia_url = line.strip().split(',', 1)
            params.append((wikidata_id, dbpedia_url))

    results = {}
    with Pool(4) as pool:
        for result in tqdm(pool.imap(parse_dbpedia,params), dynamic_ncols=True, total=len(params)):
            key, result = result
            if len(result) < 3:
                with open('failed_dbpedia_url.txt', 'a') as f:
                    f.write(key+'\n')
            else:
                results[ key ] = result
                cnt += 1

if __name__ == '__main__':
    download_dbpedia_json()
    # with open('valid_entities_list.txt', 'r') as f:
    #     for line in tqdm(f.readlines()):
    #         parse_dbpedia(line.strip())

    # rebuild_entity2id('code/data/DBpedia-3SPv2/snapshot1')
    # rebuild_relation2id('code/data/DBpedia-3SPv2/snapshot1')

    # build_dbpedia_entity2wikidata(
    #     entity2id_file = 'code/data/DBpedia-3SPv2/snapshot1/entity2id.txt',
    #     output_file = 'code/data/DBpedia-3SPv2/snapshot1/entity2wikidata.json'
    # )
    # build_dbpedia_relation2wikidata(
    #         entity2id_file = 'code/data/DBpedia-3SPv2/snapshot1/relation2id.txt',
    #     output_file = 'code/data/DBpedia-3SPv2/snapshot1/relation2wikidata.json'
    # )
    # build_entity2wikidata(
    #     entity2id_file = 'code/data/FB15K-237-2/snapshot2/entity2id.txt',
    #     output_file = 'code/data/FB15K-237-2/snapshot2/entity2wikidata.json'
    # )
    # build_relation2wikidata(
    #     relation2id_file='code/data/FB15K-237-2/snapshot2/relation2id.txt',
    #     output_file = 'code/data/FB15K-237-2/snapshot2/relation2wikidata.json'
    # )
    # build_relation2wikidata(
    #     relation2id_file='code/data/FB15K-237-2/snapshot1/relation2id.txt',
    #     output_file = 'code/data/FB15K-237-2/snapshot1/relation2wikidata.json'
    # )

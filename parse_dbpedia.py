from SPARQLWrapper import SPARQLWrapper, JSON
import os
from tqdm import tqdm
import urllib
import time
from multiprocessing import Pool


sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

exist_wikidatas = []
if os.path.exists('wiki_id2dbpedia.txt'):
    with open('wiki_id2dbpedia.txt', 'r') as f:
        for line in f:
            wiki_data = line.split(',')[0]
            exist_wikidatas.append(wiki_data)


def parse_results(wiki_data):
    statement = """PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX wd: <http://www.wikidata.org/entity/>

        SELECT DISTINCT ?lang ?name ?dbpedia_id WHERE {{
        ?article schema:about wd:{} ;
                    schema:inLanguage ?lang ;
                    schema:name ?name ;
                    schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
        FILTER(?lang in ('en')) .
        FILTER (!CONTAINS(?name, ':')) .
        OPTIONAL {{ ?article schema:about ?Wikidata_id . 
                    ?article schema:isPartOf <https://en.wikipedia.org/> . }}
        SERVICE <http://dbpedia.org/sparql> {{?dbpedia_id owl:sameAs ?Wikidata_id .?dbpedia_id dbo:wikiPageID ?wikipedia_id.}}
        }}""".format(wiki_data)
    time.sleep(1)
    outputs = []
    try:
        sparql.setQuery(statement)
        results = sparql.query().convert() 
        for result in results["results"]["bindings"][:3]:
            outputs.append((wiki_data , result['dbpedia_id']['value']))
    except KeyboardInterrupt:
        return outputs
    except BaseException as e:
        print(wiki_data, 'failed')
        return outputs

    return outputs


def parse_parallel():
    query_files = [ 'xab','xaa']
    params = []
    for query_file in query_files:
        with open(query_file, 'r') as f:
            for line in f:
                wiki_data, freq = line.strip().split(',',1)
                if wiki_data not in exist_wikidatas:
                    params.append(wiki_data)

    with open('wiki_id2dbpedia.txt', 'a') as g, Pool(3) as pool:
        for outputs in tqdm(pool.imap_unordered(parse_results, params), dynamic_ncols=True, total=len(params)):
            for output in outputs:
                g.write('{},{}\n'.format(output[0], output[1]))


def parse_property(property_id):
    statement = """
    PREFIX       wdt:  <http://www.wikidata.org/prop/direct/>
    PREFIX  wikibase:  <http://wikiba.se/ontology#>
    PREFIX        bd:  <http://www.bigdata.com/rdf#>

    SELECT ?WikidataProp ?itemLabel ?DBpediaProp
    WHERE
    {{
        wdt:{}  wdt:P1628  ?DBpediaProp .
        FILTER ( CONTAINS ( str(?DBpediaProp) , 'dbpedia' ) ) .
        SERVICE wikibase:label
        {{ bd:serviceParam  wikibase:language  "en" }} .
    }}
    """.format(property_id)
    time.sleep(1)
    outputs = []
    try:
        sparql.setQuery(statement)
        results = sparql.query().convert() 
        for result in results["results"]["bindings"][:3]:
            outputs.append((property_id , result['DBpediaProp']['value']))
    except KeyboardInterrupt:
        return outputs
    except BaseException as e:
        print(wiki_data, 'failed')
        return outputs
    return outputs


def parse_prop_parallel():
    query_files = [ 'all_properties.txt']
    params = []
    for query_file in query_files:
        with open(query_file, 'r') as f:
            for line in f:
                wiki_data = line.strip()
                if wiki_data not in exist_wikidatas:
                    params.append(wiki_data)

    with open('wiki_prop2dbpedia.txt', 'a') as g, Pool(3) as pool:
        for outputs in tqdm(pool.imap_unordered(parse_property, params), dynamic_ncols=True, total=len(params)):
            for output in outputs:
                g.write('{},{}\n'.format(output[0], output[1].replace('ontology', 'property')))



if __name__ == "__main__":
    parse_prop_parallel()
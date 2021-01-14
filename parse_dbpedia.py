from SPARQLWrapper import SPARQLWrapper, JSON
import os
from tqdm import tqdm
import urllib
import time

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

exist_wikidatas = []
if os.path.exists('wiki_id2dbpedia.txt'):
    with open('wiki_id2dbpedia.txt', 'r') as f:
        for line in f:
            wiki_data = line.split(',')[0]
            exist_wikidatas.append(wiki_data)


if __name__ == "__main__":
    query_file = 'xac'
    print(query_file)
    with open(query_file, 'r') as f, open('wiki_id2dbpedia.txt', 'a') as g:
        for line in tqdm(f, dynamic_ncols=True):
            wiki_data, freq = line.strip().split(',',1)

            if wiki_data not in exist_wikidatas:
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
                time.sleep(0.5)
                sparql.setQuery(statement)
                try:
                    results = sparql.query().convert() 
                    for result in results["results"]["bindings"][:3]:
                        exist_wikidatas.append(wiki_data)
                        g.write('{},{}\n'.format(wiki_data , result['dbpedia_id']['value']))
                except urllib.error.HTTPError:
                    print(wiki_data, 'failed')
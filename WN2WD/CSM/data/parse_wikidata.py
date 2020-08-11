"""labels和descriptions可能为None，props可能为空"""
import jsonlines
import pickle
from tqdm import tqdm

read_path = 'part-'    # wikidata文件的路径
save_path = 'wiki.pkl'   # 生成的pkl文件的路径

dict_keys=['labels', 'qnode', 'title', 'statement_count',
           'text', 'sitelink_count', 'namespace', 'namespace_text', 
           'descriptions', 'incoming_links', 'type', 'id',
           'wd_descriptions', 'wd_labels', 'wd_aliases', 'person_abbr', 
           'wikipedia_urls', 'dbpedia_urls', 'wd_properties', 'wd_prop_vals', 
           'db_short_abstracts', 'db_instance_types']


def parse_nodes(read_file):
    data = []
    with open(read_file, 'r', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            if 'en' not in item['labels'] or len(item['labels']['en']) == 0:
                labels = ['None']
            else:
                labels = item['labels']['en']

            if 'en' not in item['descriptions'] or len(item['descriptions']['en']) == 0:
                desc = 'None'
            else:
                desc = item['descriptions']['en'][0]

            qnode = item['qnode']
            props = item['wd_prop_vals']
            data.append([qnode, desc, labels, props])
    return data


def read_of_nodes(path, read):
    all_data = []
    for i in tqdm(range(10000), desc='parsing files'):
        read_file = read + "%05d" % i
        data = parse_nodes(read_file)
        all_data += data
    print('writing data to pkl file...')
    with open(path, 'wb') as f:
        pickle.dump(all_data, f)
    print('write to file done')


read_of_nodes(save_path, read_path)

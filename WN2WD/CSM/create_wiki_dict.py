"""生成的字典格式：key为word(短语的话就不变，单词的话就提取词干)，value为对应的qnodes"""
import pickle
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm


class Config:
    wiki_pkl_path = '/nas/home/chengj/anaconda3/envs/huggingface/codes/datasets/wiki.pkl'

    dict_path = 'word2qnodes.pkl'    # 生成的字典路径


def read_wikidata():
    # 读取wikidata数据
    print('loading wikidata pkl file...')
    with open(Config.wiki_pkl_path, 'rb') as f:
        wiki_data = pickle.load(f)
    print('wikidata number:', len(wiki_data))
    return wiki_data


def create_dict(wiki):
    word2qnodes = {}
    porter_stemmer = PorterStemmer()

    for one_wiki in tqdm(wiki, desc='creating word2qnodes dict'):
        if one_wiki[0][0] == 'Q' and one_wiki[2] != ['None']:
            labels = list(set(one_wiki[2]))
            labels = [i if i.find(' ') >= 0 else porter_stemmer.stem(i) for i in labels]
            labels = list(set(labels))
            for one_label in labels:
                if one_label not in word2qnodes.keys():
                    word2qnodes[one_label] = [one_wiki[0]]
                else:
                    word2qnodes[one_label].append(one_wiki[0])
    print('dict length:', len(word2qnodes))

    with open(Config.dict_path, 'wb') as f:
        pickle.dump(word2qnodes, f)


if __name__ == '__main__':
    wiki = read_wikidata()
    create_dict(wiki)

import csv
import pickle
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm


class Config:
    wordnet_path = 'data/original_data/wordnet_data.csv'
    word2qnodes_path = 'word2qnodes.pkl'     # 字典路径

    candidate_path = 'candidate_lab.pkl'      # 生成的结果路径


def read_wordnet():    # 读取wordnet数据，只要名词
    print('loading wordnet data...')
    wn_data, noun_idx = [], []
    with open(Config.wordnet_path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)
        for idx, row in enumerate(f_csv):
            synset_id = row[0]
            synset_des = row[1]
            synset_words = row[2].split(',')
            synset_words = tuple([k.replace('_', ' ') for k in synset_words])
            if synset_id.split('.')[1] == 'n':
                wn_data.append((synset_id, synset_des, synset_words))
                noun_idx.append(idx)
    print('wordnet data number(only noun):', len(wn_data))
    return wn_data, noun_idx


def create_candidates(wn):
    porter_stemmer = PorterStemmer()
    with open(Config.word2qnodes_path, 'rb') as f:
        word2qnodes = pickle.load(f)
    print('dict length:', len(word2qnodes))

    cands = {}
    for one_wn in tqdm(wn):
        one_cand = []
        for one_word in one_wn[2]:
            if one_word.find(' ') >= 0:
                one_cand += word2qnodes.get(one_word, [])
            else:
                one_cand += word2qnodes.get(porter_stemmer.stem(one_word), [])
        cands[one_wn] = one_cand

    with open(Config.candidate_path, 'wb') as f:
        pickle.dump(cands, f)
    print('done!')


if __name__ == '__main__':
    wn, noun_idx = read_wordnet()
    create_candidates(wn)

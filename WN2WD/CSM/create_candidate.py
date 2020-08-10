"""注意wikidata中P开头的，描述为None的"""
import csv
import pickle
from tqdm import tqdm
import itertools
import sys


class Config:
    wordnet_path = 'data/original_data/wordnet_data.csv'
    wiki_pkl_path = '/nas/home/chengj/anaconda3/envs/huggingface/codes/datasets/wiki.pkl'   # wikidata pkl文件路径
    preprocess_path = 'preprocess_result.pkl'      # 预处理结果文件路径
    # 以下是结果的路径
    candidate_result_path = 'candidates.pkl'      # 候选集的并集的路径
    wn_candidate_idx_path = 'wn_candidate_idx.pkl'    # wordnet对应的候选集的索引结果路径
    log_path = 'candidate.log'


def make_print_to_file():
    class Logger(object):
        def __init__(self, fileN='Default.log'):
            self.terminal = sys.stdout
            self.log = open(fileN, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(Config.log_path)


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
            synset_words = [k.replace('_', ' ') for k in synset_words]
            if synset_id.split('.')[1] == 'n':
                wn_data.append((synset_id, synset_des, synset_words))
                noun_idx.append(idx)
    print('wordnet data number(only noun):', len(wn_data))
    return wn_data, noun_idx


def read_wikidata():
    # 读取wikidata数据
    print('loading wikidata pkl file...')
    with open(Config.wiki_pkl_path, 'rb') as f:
        wiki_data = pickle.load(f)
    print('wikidata number:', len(wiki_data))
    return wiki_data


def load_preprocess_data(noun_idx):
    print('loading preprocess data...')
    with open(Config.preprocess_path, 'rb') as f:
        pre_data = pickle.load(f)
    print('preprocess data number:', len(pre_data))
    wn_pre_data = pre_data[:117659]
    wiki_pre_data = pre_data[117659:]
    wn_pre_data = [wn_pre_data[i] for i in noun_idx]
    print('wordnet preprocess data number:', len(wn_pre_data))
    print('wikidata preprocess data number:', len(wiki_pre_data))
    return wn_pre_data, wiki_pre_data


if __name__ == '__main__':
    make_print_to_file()
    wn, noun_idx = read_wordnet()
    wiki = read_wikidata()
    wn_pre, wiki_pre = load_preprocess_data(noun_idx=noun_idx)
    assert len(wn) == len(wn_pre) and len(wiki) == len(wiki_pre)
    candidates = []
    for one_wn, one_wn_pre in tqdm(zip(wn, wn_pre), desc='calculating candidates'):
        one_candidate = []
        for idx, one_wiki, one_wiki_pre in zip(range(len(wiki)), wiki, wiki_pre):
            if one_wiki[0][0] == 'Q':
                if len(set(one_wn_pre).intersection(set(one_wiki_pre))) > 0 or len(set(one_wn[2]).intersection(one_wiki[2])) > 0:
                    one_candidate.append(idx)
        candidates.append(one_candidate)
    candidates_flat = list(itertools.chain(*candidates))
    candidates_flat = list(set(candidates_flat))
    print('all candidate number:', len(candidates_flat))

    print('saving candidates as pkl file...')
    with open(Config.candidate_result_path, 'wb') as f:
        pickle.dump([wiki[i] for i in candidates_flat], f)

    wn_dict = {}
    for one_wn, one_cand in zip(wn, candidates):
        one_wn_new = (one_wn[0], one_wn[1], tuple(one_wn[2]))
        one_cand_new = [candidates_flat.index(i) for i in one_cand]
        wn_dict[one_wn_new] = one_cand_new

    print('saving wordnet candidate index as pkl file...')
    with open(Config.wn_candidate_idx_path, 'wb') as f:
        pickle.dump(wn_dict, f)

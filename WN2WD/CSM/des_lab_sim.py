import pickle
import sys
import csv
from tqdm import tqdm
from multiprocessing import Pool
import itertools


class Config:
    wordnet_data_path = 'data/original_data/wordnet_data.csv'
    wiki_pkl_path = '/nas/home/chengj/anaconda3/envs/huggingface/codes/datasets/wiki.pkl'  # wiki.pkl路径
    preprocess_path = 'preprocess_result.pkl'    # 预处理生成的文件路径
    result_path = 'result.pkl'    # 生成的文件路径
    log_path = 'cal.log'
    wordnet_num = 117659
    wordnet_run_number = 200      # 运行的wordnet个数
    process_num = 3         # 进程个数
    des_weight = 0.35
    lab_weight = 0.65
    tol_num = 5    # 如果结果多于5条就认为匹配失败


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


def read_wordnet():
    # 读取wordnet数据
    print('loading wordnet data...')
    wn_data = []
    with open(Config.wordnet_data_path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)
        for row in f_csv:
            synset_id = row[0]
            synset_des = row[1]
            synset_words = row[2].split(',')
            synset_words = [k.replace('_', ' ') for k in synset_words]
            wn_data.append((synset_id, synset_des, synset_words))
    print('wordnet data number:', len(wn_data))
    return wn_data


def read_wikidata():
    # 读取wikidata数据
    print('loading wikidata pkl file...')
    with open(Config.wiki_pkl_path, 'rb') as f:
        wiki_data = pickle.load(f)
    print('wikidata number:', len(wiki_data))
    return wiki_data


def load_preprocess_data():
    print('loading preprocess data...')
    with open(Config.preprocess_path, 'rb') as f:
        pre_data = pickle.load(f)
    print('preprocess data number:', len(pre_data))
    wn_pre_data = pre_data[:Config.wordnet_num]
    wiki_pre_data = pre_data[Config.wordnet_num:]
    print('wordnet preprocess data number:', len(wn_pre_data))
    print('wikidata preprocess data number:', len(wiki_pre_data))
    print('\n')
    return wn_pre_data, wiki_pre_data


def cal(pars):
    run_wn_pre_data, run_wn_data = pars
    wiki_data = read_wikidata()
    no_use, wiki_pre_data = load_preprocess_data()

    rr = []
    for one_wn_pre, one_wn in tqdm(zip(run_wn_pre_data, run_wn_data), desc='calculating sim'):
        sim = []
        for one_wiki_pre, one_wiki in zip(wiki_pre_data, wiki_data):
            des_sim = len(set(one_wn_pre).intersection(set(one_wiki_pre))) / len(set(one_wn_pre))
            if len(set(one_wn[2]).intersection(set(one_wiki[2]))) > 0:
                lab_sim = 1
            else:
                lab_sim = 0
            sim.append(des_sim * Config.des_weight + lab_sim * Config.lab_weight)
        max_sim = max(sim)
        max_arg = []
        for index, i in enumerate(sim):
            if i == max_sim:
                max_arg.append(index)
        if len(max_arg) > 5:
            rr.append((one_wn, 'Fail', max_sim))
        else:
            rr.append((one_wn, tuple([wiki_data[k] for k in max_arg]), max_sim))
    return rr


def cut_list(lst, number):
    rr = []
    length = len(lst)
    step = int(length/number)+1
    for i in range(0, length, step):
        rr.append(lst[i:i+step])
    return rr


if __name__ == '__main__':
    make_print_to_file()
    wn_data = read_wordnet()
    wn_pre_data, wiki_pre_data = load_preprocess_data()
    assert len(wn_data) == len(wn_pre_data)
    wn_pre_data = wn_pre_data[:Config.wordnet_run_number]
    wn_data = wn_data[:Config.wordnet_run_number]
    pre_cuts = cut_list(wn_pre_data, Config.process_num)
    cuts = cut_list(wn_data, Config.process_num)
    params = []
    for i, j in zip(pre_cuts, cuts):
        params.append((i, j))

    with Pool(Config.process_num) as p:
        result = p.map(cal, params)
    result = list(itertools.chain(*result))

    with open(Config.result_path, 'wb') as f:
        pickle.dump(result, f)
    print('运行完成!')

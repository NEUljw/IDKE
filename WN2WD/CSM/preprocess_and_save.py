import sys
import csv
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from multiprocessing import Pool
import itertools


class Config:
    wordnet_data_path = 'data/original_data/wordnet_data.csv'
    wiki_pkl_path = '/nas/home/chengj/anaconda3/envs/huggingface/codes/datasets/wiki.pkl'     # wiki.pkl路径
    result_path = 'preprocess_result.pkl'    # 生成的文件路径
    log_path = 'preprocess.log'
    process_num = 3      # 进程数（暂时不可修改）
    # wordnet_run_number = 200      # 运行的wordnet个数
    # des_weight = 0.4
    # lab_weight = 0.6
    # tol_num = 5    # 如果结果多于5条就认为匹配失败
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']


def make_print_to_file(log_path):
    class Logger(object):
        def __init__(self, fileN='Default.log'):
            self.terminal = sys.stdout
            self.log = open(fileN, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(log_path)


def read_wordnet(wordnet_data_path):
    # 读取wordnet数据
    print('loading wordnet data...')
    wn_data = []
    with open(wordnet_data_path, 'r', encoding='utf-8') as f:
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


def read_wikidata(wiki_pkl_path):
    # 读取wikidata数据
    print('loading wikidata pkl file...')
    with open(wiki_pkl_path, 'rb') as f:
        wiki_data = pickle.load(f)
    print('wikidata number:', len(wiki_data))
    print('\n')
    return wiki_data


def del_stopwords(part):
    wn = read_wordnet(Config.wordnet_data_path)
    wiki = read_wikidata(Config.wiki_pkl_path)
    all_data = wn + wiki
    if part == 1:
        all_data = all_data[:19000000]
    if part == 2:
        all_data = all_data[19000000:38000000]
    if part == 3:
        all_data = all_data[38000000:]

    rr = []
    for i in tqdm(all_data):
        words = word_tokenize(i[1])
        words = [k for k in words if k not in stopwords.words('english') and k not in Config.english_punctuations]
        rr.append(words)
    return rr


def cut_list(lst, number):
    rr = []
    length = len(lst)
    step = int(length/number)+1
    for i in range(0, length, step):
        rr.append(lst[i:i+step])
    return rr


if __name__ == '__main__':
    # make_print_to_file(Config.log_path)
    # wn = read_wordnet(Config.wordnet_data_path)
    # wiki = read_wikidata(Config.wiki_pkl_path)
    # with Pool(Config.process_num) as p:
    #     corpus = p.map(del_stopwords, [wn]+cut_list(wiki, Config.process_num-1))
    # corpus = list(itertools.chain(*corpus))
    #
    # print('saving preprocess data to pkl file...')
    # with open(Config.result_path, 'wb') as f:
    #     pickle.dump(corpus, f)
    # print('save to file done!')

    make_print_to_file(Config.log_path)

    with Pool(Config.process_num) as p:
        corpus = p.map(del_stopwords, [1, 2, 3])
    corpus = list(itertools.chain(*corpus))

    print('saving preprocess data to pkl file...')
    with open(Config.result_path, 'wb') as f:
        pickle.dump(corpus, f)
    print('save to file done!')

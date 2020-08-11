import pickle
import random
import sys
from nltk.corpus import wordnet as wn
from collections import Counter
import math


def make_print_to_file(log_path):
    class Logger(object):
        def __init__(self, fileN='Default.log'):
            self.terminal = sys.stdout
            self.log = open(fileN, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(log_path)


if __name__ == '__main__':
    # with open('merged_candidate.pkl', 'rb') as f:
    #     m_candi = pickle.load(f)
    # print(len(m_candi))
    # all_len = []
    # for key, value in m_candi.items():
    #     len_value = len(value)
    #     all_len.append(len_value)
    # print(Counter(all_len))
    #
    # c = [0, 0, 0, 0, 0]
    # for i in all_len:
    #     if 0 < i <= 25:
    #         c[0] += 1
    #     if 25 < i <= 50:
    #         c[1] += 1
    #     if 50 < i <= 75:
    #         c[2] += 1
    #     if 75 < i <= 100:
    #         c[3] += 1
    #     if i > 100:
    #         c[4] += 1
    # print(c)


    # lab_sim, des_sim = [], []
    # with open('mapping results/distilbert_result3.pkl', 'rb') as f:
    #     maps = pickle.load(f)
    # for key, value in maps.items():
    #     modf_r = math.modf(value[-1])
    #     des_sim.append(round(modf_r[0], 1))
    #     lab_sim.append(modf_r[1])
    # print(des_sim[:5])
    # print(lab_sim[:5])
    # # print(Counter(lab_sim))
    # print(Counter(des_sim))


    make_print_to_file('mapping results/r.txt')
    random.seed(1111)     # 设置随机种子
    with open('mapping results/distilbert_result0721.pkl', 'rb') as f:       # file name
        maps = pickle.load(f)
    sample_keys = random.sample(maps.keys(), 500)

    n = 1
    for key, value in maps.items():
        if key in sample_keys:
            print(str(n)+':')
            n += 1
            print(key)
            print(value[:3])
            print(value[3])
            print(value[4:7])
            print('candi number:', value[7])
            print('lab rank|des rank:', value[8:])
            print('--'*50)

import csv
from gensim.models import word2vec
import gensim
from nltk.tokenize import word_tokenize
import numpy as np


def read_one_wiki_candidate(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for j in range(int((len(row)-1)/3)):
                if row[3*j+2] != 'None':
                    data.append(row[3*j+2])
    return data


def create_corpus():
    data = []
    with open('../../data/wordnet_data.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        n = 0
        for row in f_csv:
            n += 1
            if n > 1:
                data.append(row[1])

    for file_num in range(1, 14):
        one_data = read_one_wiki_candidate(file_path='../../data/wikidata_candidate/candidate_part_'+str(file_num)+'.csv')
        data += one_data
        print('read number', file_num, 'file done.')

    with open('sentences.txt', 'w', encoding='utf-8-sig') as f:
        for i in data:
            f.write(i+'\n')


# 训练模型
def train_model():
    sentences = word2vec.LineSentence('sentences.txt')
    model = word2vec.Word2Vec(sentences=sentences, size=100, min_count=3)
    model.wv.save_word2vec_format('word2vec.bin', binary=True)


# 计算两句话的相似度
def vector_similarity(s1, s2, model):
    def sentence_vector(s):
        words = word_tokenize(s)
        v = np.zeros(100)
        for word in words:
            try:
                v += model[word]
            except KeyError:    # 未登录词
                v += np.zeros(100)
        v /= len(words)
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    try:
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    except ValueError:
        sim = 0
    return sim


# 计算wordnet描述和所有候选wikidata描述的相似度
def cal_sim_word2vec(all_wordnet_des, all_wikidata_des_list, default_sim):
    model = gensim.models.KeyedVectors.load_word2vec_format('models/word2vec/word2vec.bin', binary=True)
    all_sim_list = []
    for wordnet_des, wikidata_des_list in zip(all_wordnet_des, all_wikidata_des_list):
        sim_list = []
        for wikidata_des in wikidata_des_list:
            if wikidata_des == 'None':
                sim_list.append(default_sim)
            else:
                sim = vector_similarity(s1=wordnet_des, s2=wikidata_des, model=model)
                sim_list.append(round(sim, 4))
        all_sim_list.append(sim_list)
    return all_sim_list


if __name__ == '__main__':
    # create_corpus()
    train_model()

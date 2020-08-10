import csv
from gensim.models import word2vec
import gensim
from nltk.tokenize import word_tokenize
import numpy as np


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
    return round(sim, 6)


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
    train_model()

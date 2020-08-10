import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import models
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv


def get_dict():
    train = []
    fp = codecs.open('sentences.txt', 'r', encoding='utf-8-sig')  # 文本文件，输入需要提取主题的文档
    for line in fp:
        line = word_tokenize(line)
        train.append([w for w in line if w not in stopwords.words('english')])
    dictionary = Dictionary(train)
    dictionary.save('train_data.dict')
    return train


def train_model(num_topic):
    train = get_dict()
    dictionary = Dictionary.load('train_data.dict')
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic)
    # 模型的保存
    lda.save('LDA_trained_model/lda.model')


# 计算两个文档的相似度
def lda_sim(s1, s2, dictionary, model):
    s1 = word_tokenize(s1)  # 新文档进行分词
    doc_bow = dictionary.doc2bow(s1)  # 文档转换成bow
    doc_lda = model[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    list_doc1 = [i[1] for i in doc_lda]

    s2 = word_tokenize(s2)  # 新文档进行分词
    doc_bow2 = dictionary.doc2bow(s2)  # 文档转换成bow
    doc_lda2 = model[doc_bow2]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    list_doc2 = [i[1] for i in doc_lda2]

    # 得到文档之间的相似度
    try:
        sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
    except ValueError:
        sim = 0
    return round(sim, 6)


# 分别计算两个list元素的相似度
def cal_sim_LDA(all_wordnet_desc, all_wikidata_desc_list, default_sim):
    dic = Dictionary.load('models/LDA/train_data.dict')
    model = models.ldamodel.LdaModel.load('models/LDA/LDA_trained_model/lda.model')
    all_sim_list = []
    for wordnet_desc, wikidata_desc_list in zip(all_wordnet_desc, all_wikidata_desc_list):
        sim_list = []
        for wikidata_desc in wikidata_desc_list:
            if wikidata_desc == 'None':
                sim_list.append(default_sim)
            else:
                sim = lda_sim(s1=wordnet_desc, s2=wikidata_desc, dictionary=dic, model=model)
                sim_list.append(sim)
        all_sim_list.append(sim_list)
    return all_sim_list


if __name__ == '__main__':
    # 训练模型
    train_model(num_topic=10)

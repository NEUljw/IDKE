import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import models
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv


def read_one_wiki_candidate(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            for j in range(int((len(row)-1)/3)):
                if row[3*j+2] != 'None':
                    data.append(row[3*j+2])
    return data


# 一行中的内容是相同主题（语义），第一轮时使用这个
def create_corpus1():
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


# 其他轮使用这个
def create_corpus2():
    RESULT_FILE_PATH = 'map of 6models.csv'
    data = []
    with open(RESULT_FILE_PATH, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)  # 跳过表头
        for row in f_csv:
            if row[6] == '60|60' and row[1] != 'None' and row[4] != 'None':
                data.append([row[1], row[4]])

    result = [i[0]+'. '+i[1] for i in data]

    data_b = []
    fp = codecs.open('sentences_before.txt', 'r', encoding='utf-8-sig')
    for line in fp:
        data_b.append(line.strip())

    print('上一轮的训练集数量:', len(data_b))
    data_b = list(set(data_b) - set(sum(data, [])))

    result += data_b
    len_a = len(result)
    result = list(set(result))
    len_b = len(result)
    print('最终训练集去重后:', len_b)
    with open('sentences.txt', 'w', encoding='utf-8-sig') as f:
        for i in result:
            f.write(i+'\n')


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
    return sim


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
                sim_list.append(round(sim, 4))
        all_sim_list.append(sim_list)
    return all_sim_list


if __name__ == '__main__':
    # 建立句子组成的语料库
    # create_corpus2()

    # 训练模型
    train_model(num_topic=10)

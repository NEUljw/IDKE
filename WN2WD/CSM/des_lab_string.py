import sys
import csv
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from tqdm import tqdm

wordnet_data_path = 'data/original_data/wordnet_data.csv'
wiki_pkl_path = 'data/wiki.pkl'     # wiki.pkl路径
result_path = 'result.pkl'
log_path = 'des_lab_string.log'
wordnet_run_number = 200      # 运行的wordnet个数
des_weight = 0.5
lab_weight = 0.5
tol_num = 5    # 如果结果多于5条就认为匹配失败
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']


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


def cosine_sim(a, b):    # 余弦相似度，结果在-1到1之间，保留4位小数
    try:
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except ValueError:
        sim = 0
    return round(sim, 4)


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


# 读取wikidata数据
print('loading wikidata pkl file...')
with open(wiki_pkl_path, 'rb') as f:
    wiki_data = pickle.load(f)
print('wikidata number:', len(wiki_data))
print('\n')


corpus = []
for i in tqdm(wn_data, desc='deleting stopwords from wordnet'):      # !!!!!!!!训练语料库应该用全部数据
    wn_keywords = word_tokenize(i[1])
    wn_keywords = [k for k in wn_keywords if k not in stopwords.words('english') and k not in english_punctuations]
    corpus.append(' '.join(wn_keywords))
for i in tqdm(wiki_data, desc='deleting stopwords from wikidata'):
    wk_keywords = word_tokenize(i[1])
    wk_keywords = [k for k in wk_keywords if k not in stopwords.words('english') and k not in english_punctuations]
    corpus.append(' '.join(wk_keywords))
print('训练语料库大小:', len(corpus))

print('calculating tf-idf matrix...')
# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X).toarray()
print('tf-idf矩阵大小:', tfidf.shape)


final_result = []
for wn_index, wn_row in tqdm(enumerate(tfidf[:wordnet_run_number]), desc='calculating sim'):
    wn_labels = wn_data[wn_index][2]
    one_result = []
    for wiki_index, wiki_row in enumerate(tfidf[len(wn_data):]):
        des_sim = cosine_sim(wn_row, wiki_row)
        wiki_labels = wiki_data[wiki_index][2]
        if len(set(wn_labels).intersection(set(wiki_labels))) > 0:
            lab_sim = 1
        else:
            lab_sim = 0
        one_result.append(des_sim * des_weight + lab_sim * lab_weight)

    max_arg = []
    max_sim = max(one_result)
    for index, i in enumerate(one_result):
        if i == max_sim:
            max_arg.append(index)
    if len(max_arg) > tol_num:
        final_result.append((wn_data[wn_index], 'Fail', max_sim))
    else:
        final_result.append((wn_data[wn_index], tuple([wiki_data[kk] for kk in max_arg]), max_sim))

with open(result_path, 'wb') as f:
    pickle.dump(final_result, f)
print('运行完毕!')

# keywords_args = np.argsort(-row)[:keywords_count].tolist()
# print(keywords_args)
# keywords_args = [k for k in keywords_args if row[k] != 0]
# print(keywords_args)
# keywords = [word[k] for k in keywords_args]
# print(keywords)

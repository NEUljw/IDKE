from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

MODEL_PATH = 'models/FastText/FastText_model.bin'


def text_to_word_list(text):
    # des为none处理
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', 'none']
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords.words('english') and w not in english_punctuations]
    return text


def sen_sim(text1, text2, ft_model):
    text1 = text_to_word_list(text1)
    text2 = text_to_word_list(text2)
    if len(text1) == 0 or len(text2) == 0:
        return -100
    v = []
    for text in [text1, text2]:
        sen_embedding = []
        for word in text:
            try:
                sen_embedding.append(ft_model[word])
            except KeyError:    # 未登录词
                sen_embedding.append(np.random.uniform(-0.1, 0.1, size=100))
        sen_embedding = np.mean(sen_embedding, axis=0)    # 句向量
        v.append(sen_embedding)

    try:
        sim = np.dot(v[0], v[1]) / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
    except ValueError:
        sim = 0
    return round(sim, 6)


def cal_sim_FastText(all_wn_des, all_wiki_list):
    model = FastText.load(MODEL_PATH)

    all_sim_list = []
    for wn_des, wiki_list in zip(all_wn_des, all_wiki_list):
        sim_list = []
        for wiki in wiki_list:
            sim_list.append(sen_sim(wn_des, wiki, model))
        all_sim_list.append(sim_list)
    return all_sim_list

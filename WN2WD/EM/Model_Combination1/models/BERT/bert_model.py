from keras_bert import load_trained_model_from_checkpoint, load_vocabulary, Tokenizer
from keras.utils import multi_gpu_model
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf_keras


# gpu配置与设置
def gpu_option(gpu_name, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
    config = tf.ConfigProto(device_count={'GPU': gpu_num})
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    ktf_keras.set_session(session)

    print('--' * 30)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print('--' * 30)


def data_iter(data, batch_size):
    """生成器"""
    batch_num = len(data) // batch_size
    if len(data) % batch_size != 0:
        batch_num += 1
    X1, X2 = [], []
    for i in range(len(data)):
        X1.append(data[i][0])
        X2.append(data[i][1])
        if len(X1) == batch_size or i == (len(data)-1):
            yield X1, X2
            X1, X2 = [], []


class KerasBERT:
    def __init__(self, batch_size, gpu_num, gpu_name):
        gpu_option(gpu_name, gpu_num)
        self.batch_size = batch_size
        print("##### load KerasBERT start #####")
        model_path = 'models/BERT/pretrained_model/uncased_L-24_H-1024_A-16'
        config_path = os.path.join(model_path, 'bert_config.json')
        checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
        vocab_path = os.path.join(model_path, 'vocab.txt')
        token_dict = load_vocabulary(vocab_path)
        model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        if gpu_num >= 2:
            self.par_model = multi_gpu_model(model, gpus=gpu_num)
        else:
            self.par_model = model
        self.tokenizer = Tokenizer(token_dict)
        print("##### load KerasBERT end #####")

    def bert_encode(self, texts):
        predicts = []

        def create_array():
            data = []
            for text in texts:
                indices, segments = self.tokenizer.encode(first=text, max_len=512)
                data.append([indices, segments])
            return data

        array = create_array()
        my_iter = data_iter(array, batch_size=self.batch_size)
        for w1, w2 in my_iter:
            m_indices = np.array(w1)
            m_segments = np.array(w2)
            predict = self.par_model.predict([m_indices, m_segments])
            batch_predict = predict[:, 0].tolist()    # 每句话取第一个word([CLS])的编码
            predicts += batch_predict
        return predicts


def cosine_distance(v1, v2):     # 余弦距离
    if type(v1) == str or type(v2) == str:
        return -100
    if type(v1) == list:
        v1 = np.array(v1)
    if type(v2) == list:
        v2 = np.array(v2)
    if v1.all() and v2.all():
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0


def cal_sim_bert(all_wn_des, all_wiki_des_list, bert_model):
    texts = all_wn_des+sum(all_wiki_des_list, [])
    # description为'None'的地方做个记录
    none_index = [k for k in range(len(texts)) if texts[k] == 'None']

    all_sen_embed = bert_model.bert_encode(texts)

    for i in none_index:
        all_sen_embed[i] = 'None des'

    wordnet_sen_re = all_sen_embed[:len(all_wn_des)]   # wordnet句向量
    wiki_sen = all_sen_embed[len(all_wn_des):]
    wiki_sen_re = []     # wikidata句向量
    count = 0
    for i in all_wiki_des_list:
        one_wiki_re = []
        while len(one_wiki_re) < len(i):
            one_wiki_re.append(wiki_sen[count])
            count += 1
        wiki_sen_re.append(one_wiki_re)

    all_sim_list = []
    for wordnet, wikidata in zip(wordnet_sen_re, wiki_sen_re):
        sim_list = []
        for wiki in wikidata:
            sim = cosine_distance(wordnet, wiki)
            sim_list.append(sim)
        all_sim_list.append(sim_list)
    return all_sim_list

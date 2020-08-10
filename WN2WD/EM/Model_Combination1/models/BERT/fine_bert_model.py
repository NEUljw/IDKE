from keras.models import load_model
from keras_bert import get_custom_objects, load_vocabulary, Tokenizer
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf_keras
import numpy as np
import os


# gpu配置与设置
def gpu_option(gpu_name, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
    config = tf.ConfigProto(device_count={'GPU': gpu_num})
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)
    ktf_keras.set_session(session)


class FineTuneBert:
    def __init__(self, gpu_name, gpu_num, seq_max_len, batch_size):
        print('--'*10+' Load BERT model start '+'--'*10)
        gpu_option(gpu_name, gpu_num)
        self.seq_max_len = seq_max_len  # 与训练时相同
        self.batch_size = batch_size
        model_path = 'models/BERT/pretrained_model/uncased_L-24_H-1024_A-16'
        vocab_path = os.path.join(model_path, 'vocab.txt')
        # 加载Tokenizer
        token_dict = load_vocabulary(vocab_path)
        self.tokenizer = Tokenizer(token_dict)
        MODEL_SAVE_PATH = 'models/BERT/fine_tune_model/bert_fine_tune.hdf5'
        model = load_model(MODEL_SAVE_PATH, custom_objects=get_custom_objects(), compile=False)
        if gpu_num >= 2:
            self.par_model = multi_gpu_model(model, gpus=gpu_num)
        else:
            self.par_model = model
        print('--' * 10 + ' Load BERT model end ' + '--' * 10)

    # 数据的生成器
    def data_generator(self, data):
        steps = len(data) // self.batch_size
        if len(data) % self.batch_size != 0:
            steps += 1
        X1, X2 = [], []
        for i in range(len(data)):
            d = data[i]
            text1 = d[0]
            text2 = d[1]
            x1, x2 = self.tokenizer.encode(first=text1, second=text2, max_len=self.seq_max_len)  # 512
            X1.append(x1)
            X2.append(x2)
            if len(X1) == self.batch_size or i == (len(data)-1):
                yield np.array(X1), np.array(X2)
                X1, X2 = [], []

    def classify(self, texts):
        pred = []
        my_iter = self.data_generator(texts)
        for indices, segments in my_iter:
            p = self.par_model.predict([indices, segments])
            pred += sum(p.tolist(), [])
        return pred


def cal_sim_fine_bert(all_wn_des, all_wiki_des_list, gpu_name, gpu_num, batch_size):
    wiki_len = [len(i) for i in all_wiki_des_list]
    texts = []
    for wn_des, wiki_list in zip(all_wn_des, all_wiki_des_list):
        for wiki in wiki_list:
            texts.append([wn_des, wiki])
    a = FineTuneBert(gpu_name, gpu_num, 50, batch_size)
    pred = a.classify(texts)

    all_sim_list = []
    for index, i in enumerate(wiki_len):
        all_sim_list.append(pred[sum(wiki_len[:index]):sum(wiki_len[:index+1])])
    return all_sim_list

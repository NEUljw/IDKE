from keras.models import load_model
from keras_xlnet import Tokenizer, get_custom_objects
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf_keras
import numpy as np
import os
import copy


# gpu配置与设置
def gpu_option(gpu_name, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
    config = tf.ConfigProto(device_count={'GPU': gpu_num}, allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)
    ktf_keras.set_session(session)


def create_seg_array(tk, mask_arr):
    try:
        d = tk.index(7505)
    except ValueError:
        for index, i in enumerate(mask_arr):
            mask_arr[index] = 0
        return np.array(mask_arr)
    for index, i in enumerate(mask_arr[:d+1]):    # ||||||
        mask_arr[index] = 0
    return np.array(mask_arr)


class FineTuneXlnet:
    def __init__(self, gpu_name, gpu_num, seq_max_len, batch_size):
        print('--' * 10 + ' Load xlnet model start ' + '--' * 10)
        gpu_option(gpu_name, gpu_num)
        self.seq_max_len = seq_max_len  # 与训练时相同
        self.batch_size = batch_size
        spiece_model = 'models/Xlnet/xlnet_model/spiece.model'
        self.tokenizer = Tokenizer(spiece_model)
        MODEL_SAVE_PATH = 'models/Xlnet/fine_tune_model/xlnet_fine_tune.hdf5'
        model = load_model(MODEL_SAVE_PATH, custom_objects=get_custom_objects(), compile=False)
        if gpu_num >= 2:
            self.par_model = multi_gpu_model(model, gpus=gpu_num)
        else:
            self.par_model = model
        print('--' * 10 + ' Load xlnet model end ' + '--' * 10)

    # 数据的生成器
    def data_generator(self, data):
        steps = len(data) // self.batch_size
        if len(data) % self.batch_size != 0:
            steps += 1
        X1, X2, X3, X4 = [], [], [], []
        for i in range(len(data)):
            d = data[i]
            text1 = d[0]
            text2 = d[1]

            tokens = self.tokenizer.encode(text1 + '|' + text2)
            tokens = tokens + [0] * (self.seq_max_len - len(tokens)) if len(tokens) < self.seq_max_len else tokens[
                                                                                                  0:self.seq_max_len]  # padding
            token_input = np.array(tokens)

            mask_input = [0 if ids == 0 else 1 for ids in tokens]
            mask_input_ = copy.deepcopy(mask_input)
            segment_input = create_seg_array(tokens, mask_input_)
            memory_length_input = np.zeros(1)

            X1.append(token_input)
            X2.append(segment_input)
            X3.append(memory_length_input)
            X4.append(mask_input)
            if len(X1) == self.batch_size or i == (len(data) - 1):
                yield np.array(X1), np.array(X2), np.array(X3), np.array(X4)
                X1, X2, X3, X4 = [], [], [], []

    def classify(self, texts):
        pred = []
        my_iter = self.data_generator(texts)
        for m_token, m_segment, m_memory, m_mask in my_iter:
            p = self.par_model.predict([m_token, m_segment, m_memory, m_mask])
            pred += sum(p.tolist(), [])
        return pred


def cal_sim_fine_xlnet(all_wn_des, all_wiki_des_list, gpu_name, gpu_num, batch_size):
    wiki_len = [len(i) for i in all_wiki_des_list]
    texts = []
    for wn_des, wiki_list in zip(all_wn_des, all_wiki_des_list):
        for wiki in wiki_list:
            texts.append([wn_des, wiki])
    a = FineTuneXlnet(gpu_name, gpu_num, 100, batch_size)
    pred = a.classify(texts)

    all_sim_list = []
    for index, i in enumerate(wiki_len):
        all_sim_list.append(pred[sum(wiki_len[:index]):sum(wiki_len[:index+1])])
    return all_sim_list

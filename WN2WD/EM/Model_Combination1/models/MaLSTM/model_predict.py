import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras.backend as K
import keras.backend.tensorflow_backend as ktf_keras
import tensorflow as tf
import csv
import os


def data_to_csv(all_wn_des, all_wiki_des_list):
    with open('models/MaLSTM/data/data.csv', 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['text1', 'text2'])
        for wn_des, wiki_list in zip(all_wn_des, all_wiki_des_list):
            for wiki in wiki_list:
                f_csv.writerow([wn_des, wiki])


def text_to_word_list(text):
    stops = stopwords.words('english')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    text = str(text)
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w not in stops and w not in english_punctuations]
    return text


# 数据的生成器
def data_generator(data1, data2, batch_size):
    steps = len(data1) // batch_size
    if len(data1) % batch_size != 0:
        steps += 1
    X1, X2 = [], []
    for i in range(len(data1)):
        text1 = data1[i]
        text2 = data2[i]
        X1.append(text1)
        X2.append(text2)
        if len(X1) == batch_size or i == (len(data1)-1):
            yield np.array(X1), np.array(X2)
            X1, X2 = [], []


def exponent_neg_manhattan_distance(left, right):
    """Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


def cal_sim_lstm(wiki_len, gpu_name, gpu_num, batch_size=128):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
    config = tf.ConfigProto(device_count={'GPU': gpu_num}, allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)
    ktf_keras.set_session(session)

    TEST_CSV = 'models/MaLSTM/data/data.csv'
    MODEL_SAVE_PATH = 'models/MaLSTM/my_model.hdf5'
    DICTIONARY_PATH = 'models/MaLSTM/word2id.pkl'

    test_df = pd.read_csv(TEST_CSV)
    with open(DICTIONARY_PATH, 'rb') as f:
        data = pickle.load(f)
        word2id = data['word2id']
        max_seq_length = data['max_seq_length']

    texts_cols = ['text1', 'text2']

    # 将word转换为词典的ID
    for index, row in test_df.iterrows():
        for text in texts_cols:
            t2n = []
            for word in text_to_word_list(row[text]):
                t2n.append(word2id.get(word, 0))
            test_df._set_value(index, text, t2n)

    X_test = {'left': test_df.text1, 'right': test_df.text2}

    X_test['left'] = pad_sequences(X_test['left'], maxlen=max_seq_length)
    X_test['right'] = pad_sequences(X_test['right'], maxlen=max_seq_length)

    # Make sure everything is ok
    assert X_test['left'].shape == X_test['right'].shape
    X_test['left'] = X_test['left'].tolist()
    X_test['right'] = X_test['right'].tolist()

    dg = data_generator(X_test['left'], X_test['right'], batch_size=batch_size)
    malstm = load_model(MODEL_SAVE_PATH,
                        custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance},
                        compile=False)
    print('转换成功! 模型运行开始...')
    all_pred = []
    for i, j in dg:
        pred = malstm.predict([i, j], batch_size=batch_size)
        pred = pred.tolist()
        pred = sum(pred, [])
        all_pred += pred

    all_sim_list = []
    for index, i in enumerate(wiki_len):
        all_sim_list.append(all_pred[sum(wiki_len[:index]):sum(wiki_len[:index+1])])
    return all_sim_list

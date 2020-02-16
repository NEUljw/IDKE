from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Input, Dropout, Reshape, BatchNormalization, TimeDistributed, Lambda, Layer, LSTM, Bidirectional, Average, concatenate
import matplotlib.pyplot as plt
import os
import jsonlines
import jieba

import keras.backend as K
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# epoch 10, accuracy 0.3312

class SiameseNetwork:
    def __init__(self):
        self.class_dict = {'true': 1, 'false': 0}
        # self.embedding_file = 'model/token_vec_300.bin'
        # self.model_path = 'tokenvec_bilstm2_model.h5'
        self.EMBEDDING_DIM = 128
        self.EPOCHS = 10
        self.BATCH_SIZE = 25

        self.vocab = []
        self.VOCAB_SIZE = 0
        self.max_length = 80
        # string格式
        self.train_text_left = []
        self.train_text_right = []
        self.train_label = []
        self.test_text_left = []
        self.test_text_right = []
        self.test_label = []
        # int格式
        self.train_left = []
        self.train_right = []
        self.test_left = []
        self.test_right = []

        self.read_train_data()
        self.read_dev_data()
        self.one_hot()
        self.select_max_length_and_padding()

    def read_train_data(self):
        with open('hellaswag-train-dev/train.jsonl', 'r+', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                pre_text = item['ctx']
                end_text = item['ending_options']
                self.train_text_left.append(pre_text)
                self.train_text_left.append(pre_text)
                self.train_text_left.append(pre_text)
                self.train_text_left.append(pre_text)
                self.train_text_right.append(end_text[0])
                self.train_text_right.append(end_text[1])
                self.train_text_right.append(end_text[2])
                self.train_text_right.append(end_text[3])

        with open('hellaswag-train-dev/train-labels.lst') as f:
            answer = [i.strip() for i in f.readlines()]
            for j in answer:
                one_line_label = [0, 0, 0, 0]
                one_line_label[int(j)] = 1
                self.train_label += one_line_label

    def read_dev_data(self):
        with open('hellaswag-train-dev/valid.jsonl', 'r+', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                pre_text = item['ctx']
                end_text = item['ending_options']
                self.test_text_left.append(pre_text)
                self.test_text_left.append(pre_text)
                self.test_text_left.append(pre_text)
                self.test_text_left.append(pre_text)
                self.test_text_right.append(end_text[0])
                self.test_text_right.append(end_text[1])
                self.test_text_right.append(end_text[2])
                self.test_text_right.append(end_text[3])

        with open('hellaswag-train-dev/valid-labels.lst') as f:
            answer = [i.strip() for i in f.readlines()]
            for j in answer:
                one_line_label = [0, 0, 0, 0]
                one_line_label[int(j)] = 1
                self.test_label += one_line_label

    def remove_blank(self, text):
        after = []
        for i in text:
            if i != ' ':
                after.append(i)
        return after

    def one_hot(self):
        for text in self.train_text_left[:5000]:
            text_seg = self.remove_blank(jieba.lcut(text))
            self.vocab += text_seg
        print('train text left done.')
        for text in self.train_text_right[:5000]:
            text_seg = self.remove_blank(jieba.lcut(text))
            self.vocab += text_seg
        print('train text right done.')
        for text in self.test_text_left[:5000]:
            text_seg = self.remove_blank(jieba.lcut(text))
            self.vocab += text_seg
        print('test text left done.')
        for text in self.test_text_right[:5000]:
            text_seg = self.remove_blank(jieba.lcut(text))
            self.vocab += text_seg
        print('test text right done.')

        self.vocab = list(set(self.vocab))
        self.VOCAB_SIZE = len(self.vocab)
        print('vocab size:', len(self.vocab))

        print(len(self.train_text_left))
        print(len(self.train_text_right))
        print(len(self.train_label))
        print(len(self.test_text_left))
        print(len(self.test_text_right))
        print(len(self.test_label))

        n = 0
        for text in self.train_text_left[:5000]:
            n += 1
            if n % 500 == 0:
                print(n, '/', len(self.train_text_left), 'done')
            text_int = []
            text_seg = self.remove_blank(jieba.lcut(text))
            for i in text_seg:
                text_int.append(self.vocab.index(i))
            self.train_left.append(text_int)
        print('train text left one hot done.')
        n = 0
        for text in self.train_text_right[:5000]:
            n += 1
            if n % 500 == 0:
                print(n, '/', len(self.train_text_right), 'done')
            text_int = []
            text_seg = self.remove_blank(jieba.lcut(text))
            for i in text_seg:
                text_int.append(self.vocab.index(i))
            self.train_right.append(text_int)
        print('train text right one hot done.')
        n = 0
        for text in self.test_text_left[:5000]:
            n += 1
            if n % 500 == 0:
                print(n, '/', len(self.test_text_left), 'done')
            text_int = []
            text_seg = self.remove_blank(jieba.lcut(text))
            for i in text_seg:
                text_int.append(self.vocab.index(i))
            self.test_left.append(text_int)
        print('test text left one hot done.')
        n = 0
        for text in self.test_text_right[:5000]:
            n += 1
            if n % 500 == 0:
                print(n, '/', len(self.test_text_right), 'done')
            text_int = []
            text_seg = self.remove_blank(jieba.lcut(text))
            for i in text_seg:
                text_int.append(self.vocab.index(i))
            self.test_right.append(text_int)
        print('test text right one hot done.')

        print(len(self.train_left))
        print(len(self.train_right))
        print(len(self.test_left))
        print(len(self.test_right))

    '''根据样本长度,选择最佳的样本max-length'''
    def select_max_length_and_padding(self):
        print('calculate max length start.')
        max_len = 0
        all_text = self.train_left+self.train_right+self.test_left+self.test_right
        for i in all_text:
            if len(i) > max_len:
                max_len = len(i)
        print('max length:', max_len)

        self.train_left = pad_sequences(self.train_left, self.max_length, padding='post')
        self.train_right = pad_sequences(self.train_right, self.max_length, padding='post')
        self.test_left = pad_sequences(self.test_left, self.max_length, padding='post')
        self.test_right = pad_sequences(self.test_right, self.max_length, padding='post')

    '''搭建编码层网络,用于权重共享'''
    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(64))(lstm1)
        lstm2 = Dropout(0.5)(lstm2)
        return Model(input, lstm2)

    '''搭建网络'''
    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    input_length=self.max_length,
                                    mask_zero=True)
        left_input = Input(shape=(self.max_length,), dtype='float32')
        right_input = Input(shape=(self.max_length,), dtype='float32')
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        shared_lstm = self.create_base_network(input_shape=(self.max_length, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        print('left:', left_output)
        print('right:', right_output)
        merged = concatenate([left_output, right_output], axis=1)
        print('merged:', merged)
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)
        pred = Dense(1, activation='sigmoid', name='sigmoid_prediction')(merged)
        optimizer = SGD(lr=0.001, momentum=0.9)
        model = Model(inputs=[left_input, right_input], outputs=pred)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        print(model.summary())
        return model

    '''训练模型'''
    def train_model_and_test(self):
        model = self.bilstm_siamese_model()
        for i in range(self.EPOCHS):
            model.fit(
                      x=[self.train_left, self.train_right],
                      y=self.train_label[:5000],
                      batch_size=self.BATCH_SIZE,
                      epochs=1,
                      verbose=0
                    )
            # evaluate the model
            print('epoch', i + 1, ':')
            loss_train, train_acc = model.evaluate([self.train_left[:100], self.train_right[:100]], self.train_label[:100], verbose=0)
            loss, accuracy = model.evaluate([self.test_left[:100], self.test_right[:100]], self.test_label[:100], verbose=0)
            print('Train Accuracy: %f' % (train_acc * 100))
            print('Test  Accuracy: %f' % (accuracy * 100))

        re = model.predict([self.test_left[:5000], self.test_right[:5000]])
        test_list = self.test_label[:5000]
        re_list = re.tolist()

        answer_list = []
        for i in range(int(len(test_list)/4)):
            one_part = test_list[4*i:4*(i+1)]
            answer_list.append(one_part.index(1))

        int_result = []
        for i in range(int(len(re_list)/4)):
            one_part_before = re_list[4*i:4*(i+1)]
            one_part = []
            for j in one_part_before:
                one_part.append(j[0])
            max_value = -1
            for j in one_part:
                if j > max_value:
                    max_value = j
            int_result.append(one_part.index(max_value))

        correct_count = 0
        for i in range(len(answer_list)):
            if answer_list[i] == int_result[i]:
                correct_count += 1
        print('acc:', correct_count/len(answer_list))

        # model.save(self.model_path)


handler = SiameseNetwork()
handler.train_model_and_test()

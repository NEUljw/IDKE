import numpy as np
from keras_xlnet import load_trained_model_from_checkpoint, Tokenizer, ATTENTION_TYPE_BI, get_custom_objects
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf_keras
from keras.callbacks import ModelCheckpoint
import os
import csv
import copy
import time

################
# Parameters
################
sign_num = 3       # 迭代的轮数，为2时在预训练模型基础上微调，为其它时在上一个迭代的模型基础上微调
gpu_name = "0"     # GPU编号
gpu_num = 1        # GPU数量
epoch_num = 5
batch_size = 12
valid_data_ratio = 0.1
seq_max_len = 100

# File path
MODEL_SAVE_PATH = 'fine_tune_model/xlnet_fine_tune.hdf5'   # 微调后的MODEL保存路径
data_path = 'train.csv'     # 训练集路径
ckpt_name = 'xlnet_model/xlnet_model.ckpt'
config_name = 'xlnet_model/xlnet_config.json'
spiece_model = 'xlnet_model/spiece.model'

# GPU设置
K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
config = tf.ConfigProto(device_count={'GPU': gpu_num}, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
ktf_keras.set_session(session)


def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)     # 跳过表头
        for row in f_csv:
            data.append([row[0], row[1], int(row[2])])
    return data


# 读数据并划分为训练集和验证集
all_data = read_data(data_path)
valid_num = int(len(all_data) * valid_data_ratio)
train_num = len(all_data)-valid_num
train_data = all_data[:train_num]
valid_data = all_data[train_num:]
print('data number:', len(all_data))
print('train data number:', len(train_data))
print('valid data number:', len(valid_data))

# 加载Tokenizer
tokenizer = Tokenizer(spiece_model)


def create_seg_array(tk, mask_arr):
    for index, i in enumerate(mask_arr[:tk.index(7505)+1]):    # ||||||
        mask_arr[index] = 0
    return np.array(mask_arr)


# 数据的生成器
class data_generator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, X3, X4, Y = [], [], [], [], []
            for i in range(len(self.data)):
                d = self.data[i]
                text1 = d[0]
                text2 = d[1]

                tokens = tokenizer.encode(text1+'|'+text2)
                tokens = tokens + [0] * (seq_max_len - len(tokens)) if len(tokens) < seq_max_len else tokens[0:seq_max_len]  # padding
                token_input = np.array(tokens)

                mask_input = [0 if ids == 0 else 1 for ids in tokens]
                mask_input_ = copy.deepcopy(mask_input)
                segment_input = create_seg_array(tokens, mask_input_)
                memory_length_input = np.zeros(1)

                y = d[2]
                X1.append(token_input)
                X2.append(segment_input)
                X3.append(memory_length_input)
                X4.append(mask_input)
                Y.append([y])
                if len(X1) == self.batch_size or i == (len(self.data)-1):
                    yield [np.array(X1), np.array(X2), np.array(X3), np.array(X4)], np.array(Y)
                    X1, X2, X3, X4, Y = [], [], [], [], []


if sign_num == 2:
    # 模型加载
    xlnet_model = load_trained_model_from_checkpoint(checkpoint_path=ckpt_name,
                                                     attention_type=ATTENTION_TYPE_BI,
                                                     in_train_phase=True,
                                                     config_path=config_name,
                                                     memory_len=0,
                                                     target_len=seq_max_len,
                                                     batch_size=batch_size,
                                                     mask_index=0)
    # MODEL结构
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))
    x4_in = Input(shape=(None,))
    x = xlnet_model([x1_in, x2_in, x3_in, x4_in])
    x = Lambda(function=lambda x: x[:, 0])(x)   # !!!!!!
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in, x3_in, x4_in], p)
else:
    model = load_model(MODEL_SAVE_PATH, custom_objects=get_custom_objects(), compile=False)

par_model = multi_gpu_model(model, gpus=gpu_num)
par_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),     # 用足够小的学习率
    metrics=['accuracy']
)


train_D = data_generator(train_data, batch_size)
valid_D = data_generator(valid_data, batch_size)


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


checkpoint = ParallelModelCheckpoint(model, filepath=MODEL_SAVE_PATH, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=False)
# MODEL训练和储存
par_model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=epoch_num,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=[checkpoint],
    verbose=2
)

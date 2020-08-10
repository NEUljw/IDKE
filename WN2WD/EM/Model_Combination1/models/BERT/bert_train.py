import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, load_vocabulary, get_custom_objects
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf_keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os
import csv

################
# Parameters
################
sign_num = 3       # 迭代的轮数，为2时在预训练模型基础上微调，为其它时在上一个迭代的模型基础上微调
gpu_name = "1,2,3"     # GPU编号
gpu_num = 3        # GPU数量
epoch_num = 5
batch_size = 3
valid_data_ratio = 0.1
seq_max_len = 50

# File path
MODEL_SAVE_PATH = 'fine_tune_model/bert_fine_tune.hdf5'   # 微调后的MODEL保存路径
data_path = 'train.csv'
model_path = 'pretrained_model/uncased_L-24_H-1024_A-16'
config_path = os.path.join(model_path, 'bert_config.json')
checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
vocab_path = os.path.join(model_path, 'vocab.txt')

# GPU设置
K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
config = tf.ConfigProto(device_count={'GPU': gpu_num})
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
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
token_dict = load_vocabulary(vocab_path)
tokenizer = Tokenizer(token_dict)


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
            X1, X2, Y = [], [], []
            for i in range(len(self.data)):
                d = self.data[i]
                text1 = d[0]
                text2 = d[1]
                x1, x2 = tokenizer.encode(first=text1, second=text2, max_len=seq_max_len)  # 512
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == (len(self.data)-1):
                    yield [np.array(X1), np.array(X2)], np.array(Y)
                    X1, X2, Y = [], [], []


if sign_num == 2:
    # 加载预训练模型并设置为可训练
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    # MODEL结构
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
else:
    with tf.device('/cpu:0'):      # CPU里构建模型
        model = load_model(MODEL_SAVE_PATH, custom_objects=get_custom_objects(), compile=False)
par_model = multi_gpu_model(model, gpus=gpu_num)
par_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),     # 用足够小的学习率
    metrics=['accuracy']
)
par_model.summary()


train_D = data_generator(train_data, batch_size)
valid_D = data_generator(valid_data, batch_size)


# 修改checkpoint类以便多GPU时使用
'''class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


checkpoint = ParallelModelCheckpoint(model, filepath=MODEL_SAVE_PATH, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=False)'''
# MODEL训练和储存
for i in range(epoch_num):
    par_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=1,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        verbose=2
    )
    model.save(MODEL_SAVE_PATH)
    print('--' * 20 + '1个epoch训练结束' + '--' * 20)

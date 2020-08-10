import pandas as pd
import numpy as np
import itertools
import pickle
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping

################
# File paths
################
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin'
MODEL_SAVE_PATH = 'my_model.hdf5'
DICTIONARY_PATH = 'word2id.pkl'

################
# Parameters
################
validation_ratio = 0.1       # 训练集中验证集占的比例
n_hidden = 50
gradient_clip_norm = 1.25
batch_size = 64  # 64
n_epoch = 30

################
# Load train and test data
################
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

stops = stopwords.words('english')


def text_to_word_list(text):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    text = str(text)
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w not in stops and w not in english_punctuations]
    return text


vocabulary = dict()
# '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
inverse_vocabulary = ['<unk>']
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

texts_cols = ['text1', 'text2']

# 将word转换为词典的ID
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        for text in texts_cols:
            t2n = []
            for word in text_to_word_list(row[text]):
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    t2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    t2n.append(vocabulary[word])
            dataset._set_value(index, text, t2n)

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary)+1, embedding_dim)   # 词嵌入矩阵
embeddings[0] = 0      # padding位置的词向量

# Build词嵌入矩阵
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)
del word2vec

################
# Prepare train and validation data
################
max_seq_length = max(train_df.text1.map(lambda x: len(x)).max(),
                     train_df.text2.map(lambda x: len(x)).max(),
                     test_df.text1.map(lambda x: len(x)).max(),
                     test_df.text2.map(lambda x: len(x)).max())
print('max seq length:', max_seq_length)

validation_size = int(len(train_df) * validation_ratio)
training_size = len(train_df) - validation_size

X = train_df[texts_cols]
Y = train_df['is_same']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.text1, 'right': X_train.text2}
X_validation = {'left': X_validation.text1, 'right': X_validation.text2}
X_test = {'left': test_df.text1, 'right': test_df.text2}
# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

################
# 定义model
################

def exponent_neg_manhattan_distance(left, right):
    """Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings],
                            input_length=max_seq_length, trainable=False)
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)
# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

malstm = Model([left_input, right_input], [malstm_distance])
# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clip_norm)
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
malstm.summary()
callbacks_list = [
  EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1),
  ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1,
                  save_weights_only=False)
]

################
# Train model and save
################
# malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
#                             validation_data=([X_validation['left'], X_validation['right']], Y_validation),
#                             callbacks=callbacks_list, verbose=2)
# with open(DICTIONARY_PATH, 'wb') as f:
#     pickle.dump({'word2id': vocabulary, 'max_seq_length': max_seq_length}, f)

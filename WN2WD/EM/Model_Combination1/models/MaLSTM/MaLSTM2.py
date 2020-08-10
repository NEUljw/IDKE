import pandas as pd
import itertools
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping

################
# File paths
################
TRAIN_CSV = 'data/train.csv'
MODEL_SAVE_PATH = 'my_model.hdf5'
DICTIONARY_PATH = 'word2id.pkl'

################
# Parameters
################
validation_ratio = 0.1       # 训练集中验证集占的比例
gradient_clip_norm = 1.25
batch_size = 64  # 64
n_epoch = 10

################
# Load train and test data
################
train_df = pd.read_csv(TRAIN_CSV)

stops = stopwords.words('english')

with open(DICTIONARY_PATH, 'rb') as f:
    data = pickle.load(f)
    word2id = data['word2id']
    max_seq_length = data['max_seq_length']


def text_to_word_list(text):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    text = str(text)
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w not in stops and w not in english_punctuations]
    return text


texts_cols = ['text1', 'text2']

# 将word转换为词典的ID
for index, row in train_df.iterrows():
    for text in texts_cols:
        t2n = []
        for word in text_to_word_list(row[text]):
            t2n.append(word2id.get(word, 0))
        train_df._set_value(index, text, t2n)


################
# Prepare train and validation data
################
print('max seq length:', max_seq_length)

validation_size = int(len(train_df) * validation_ratio)
training_size = len(train_df) - validation_size

X = train_df[texts_cols]
Y = train_df['is_same']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Split to dicts
X_train = {'left': X_train.text1, 'right': X_train.text2}
X_validation = {'left': X_validation.text1, 'right': X_validation.text2}
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


malstm = load_model(MODEL_SAVE_PATH,
                    custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance},
                    compile=False)
# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clip_norm)
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
print(malstm.summary())

callbacks_list = [
  EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1),
  ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1,
                  save_weights_only=False)
]

################
# Train model and save
################
malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                            callbacks=callbacks_list, verbose=2)

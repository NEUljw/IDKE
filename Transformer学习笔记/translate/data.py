import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random

# 设置随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 加载spacy的德语和英语模型
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


# 德语分词
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


# 英语分词
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


# 数据集的迭代器
def build_data_iterator(batch_size):
    # 为数据集的每个句子增加起始符和结尾符
    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>',
                lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>',
                lower=True, batch_first=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

    print("train date number: ", len(train_data.examples))
    print("validation date number: ", len(valid_data.examples))
    print("test data number: ", len(test_data.examples))

    # 建立源语言和目标语言的词典（只对训练集建立）
    # vocab中pad序号为1，句子起始符为2，结尾符为3
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # 生成迭代器，每个batch自动padding为相同长度
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size)
    return train_iterator, valid_iterator, test_iterator, SRC, TRG, train_data, valid_data, test_data

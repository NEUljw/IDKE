import torch
import torch.nn as nn
import torch.optim as optim
import math
import time

from data import build_data_iterator
from model_def import Encoder, Decoder, Seq2Seq


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def del_trg_eos_idx(trg_input, eos_idx):
    trg_input = trg_input.tolist()
    for i in trg_input:
        i.remove(eos_idx)
    return torch.tensor(trg_input)


def train(model, iterator, optimizer, criterion, clip, trg_eos_idx):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        # DECODERS的输入删掉结尾符
        output, _ = model(src, del_trg_eos_idx(trg, trg_eos_idx))
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if (i+1) % 10 == 0:
            print('{}/{}batch done'.format(i+1, len(iterator)))
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, trg_eos_idx):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            # DECODERS的输入删掉结尾符
            output, _ = model(src, del_trg_eos_idx(trg, trg_eos_idx))
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    batch_size = 128  # 128

    # 加载数据集
    train_iter, valid_iter, test_iter, SRC, TRG, train_data, valid_data, test_data = \
        build_data_iterator(batch_size=batch_size)
    print('train batch:', len(train_iter))
    print('valid batch:', len(valid_iter))
    print('test  batch:', len(test_iter))

    # 模型参数
    src_vocab_len = len(SRC.vocab)    # 源语言词典长度
    tgt_vocab_len = len(TRG.vocab)    # 目标语言词典长度
    embed_size = 256      # 词向量维度
    n_layers = 3          # ENCODER和DECODER的个数
    n_heads = 8           # Multi-Head Attention中head的个数
    fnn_hide_dim = 512    # Feed Forward Neural Network隐藏层维度
    dropout = 0.1
    lr = 0.0005     # 学习率
    epochs = 10     # epoch个数
    clip = 1        # 梯度裁剪的最大范数

    ###################
    # 定义模型
    ###################
    enc = Encoder(src_vocab_len, embed_size, n_layers, n_heads, fnn_hide_dim, dropout)
    dec = Decoder(tgt_vocab_len, embed_size, n_layers, n_heads, fnn_hide_dim, dropout)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    TRG_END_IDX = TRG.vocab.stoi[TRG.eos_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX)
    # 模型参数初始化
    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # padding位置不计算loss
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    ###################
    # train
    ###################
    print('##'*30, 'train start', '##'*30)
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip, TRG_END_IDX)
        valid_loss = evaluate(model, valid_iter, criterion, TRG_END_IDX)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # 验证集上的loss小于历史最小loss时，保存模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, 'transformer-trained-model.pt')
        print('---'*40)
        print('Epoch: {} | Time: {}m {}s'.format(epoch+1, epoch_mins, epoch_secs))
        print('Train Loss: {} | Train PPL: {}'.format(train_loss, math.exp(train_loss)))
        print('Val. Loss: {} |  Val. PPL: {}'.format(valid_loss, math.exp(valid_loss)))
    print('##'*30, 'train end', '##'*30)

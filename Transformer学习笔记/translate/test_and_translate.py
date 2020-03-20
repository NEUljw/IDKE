import torch
import torch.nn as nn
import math
import spacy

from data import build_data_iterator
from train_and_eval import evaluate


# max_len为译文的最大长度
def translate_sentence(sentence, src_field, trg_field, model, max_len=50):
    model.eval()
    # 输入未分词，则分词
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    # 输入句子加上起始符和结尾符
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    # 输入的句子用词典id表示
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)
    # 对输入做padding mask
    src_mask = model.make_src_mask(src_tensor)
    # 得到ENCODERS的输出
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # 存储译文的list，初始化为句子的起始符
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    # 依次输出译文的单词
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)
        # 对DECODERS的输入做两个mask
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        # 输出添加到译文list中
        trg_indexes.append(pred_token)
        # 输出为结尾符，则翻译结束
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


if __name__ == '__main__':
    # 加载模型
    model = torch.load('transformer-trained-model.pt')

    batch_size = 128    # 128

    train_iter, valid_iter, test_iter, SRC, TRG, train_data, valid_data, test_data = \
        build_data_iterator(batch_size=batch_size)
    print('train batch:', len(train_iter))
    print('valid batch:', len(valid_iter))
    print('test  batch:', len(test_iter))

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    TRG_END_IDX = TRG.vocab.stoi[TRG.eos_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    ##############
    # test
    ##############
    test_loss = evaluate(model, test_iter, criterion, TRG_END_IDX)
    print('| Test Loss: {} | Test PPL: {} |'.format(test_loss, math.exp(test_loss)))

    ##############
    # translate
    ##############
    print('--'*40)
    example_idx = 8
    src = vars(train_data.examples[example_idx])['src']
    trg = vars(train_data.examples[example_idx])['trg']
    print('src = {}'.format(src))
    print('trg = {}'.format(trg))
    translation, attention = translate_sentence(src, SRC, TRG, model)
    print('predicted trg = {}'.format(translation))

    print('--' * 40)
    example_idx = 6
    src = vars(valid_data.examples[example_idx])['src']
    trg = vars(valid_data.examples[example_idx])['trg']
    print('src = {}'.format(src))
    print('trg = {}'.format(trg))
    translation, attention = translate_sentence(src, SRC, TRG, model)
    print('predicted trg = {}'.format(translation))

    print('--' * 40)
    example_idx = 10
    src = vars(test_data.examples[example_idx])['src']
    trg = vars(test_data.examples[example_idx])['trg']
    print('src = {}'.format(src))
    print('trg = {}'.format(trg))
    translation, attention = translate_sentence(src, SRC, TRG, model)
    print('predicted trg = {}'.format(translation))

import pickle
import itertools


def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        mid = data['model results']
    return mid


# 读取中间结果
bert_re = read_pkl_file('bert1.pkl')
roberta_re = read_pkl_file('roberta1.pkl')
dis_re = read_pkl_file('distilbert1.pkl')
LDA_re = read_pkl_file('LDA1.pkl')
word2vec_re = read_pkl_file('word2vec1.pkl')
FT_re = read_pkl_file('FastText1.pkl')
LSTM_re = read_pkl_file('LSTM1.pkl')
xlnet_re = read_pkl_file('xlnet1.pkl')


assert len(bert_re) == len(roberta_re) == len(dis_re) == len(LDA_re) == \
       len(word2vec_re) == len(FT_re) == len(LSTM_re) == len(xlnet_re)
print('pkl文件中的synset数量：', len(bert_re))

# 嵌套list展开
bert_re = list(itertools.chain(*bert_re))
print('1')
roberta_re = list(itertools.chain(*roberta_re))
print('2')
dis_re = list(itertools.chain(*dis_re))
print('3')
LDA_re = list(itertools.chain(*LDA_re))
print('4')
word2vec_re = list(itertools.chain(*word2vec_re))
print('5')
FT_re = list(itertools.chain(*FT_re))
print('6')
LSTM_re = list(itertools.chain(*LSTM_re))
print('7')
xlnet_re = list(itertools.chain(*xlnet_re))
print('8')

assert len(bert_re) == len(roberta_re) == len(dis_re) == len(LDA_re) == \
       len(word2vec_re) == len(FT_re) == len(LSTM_re) == len(xlnet_re)
print('展开后的候选项总数：', len(bert_re))

# 去掉候选项description为None的
normal_idx = [idx for idx, sim in enumerate(LDA_re) if sim != -100]

bert_ref = [bert_re[i] for i in normal_idx]
print('1')
roberta_ref = [roberta_re[i] for i in normal_idx]
print('2')
dis_ref = [dis_re[i] for i in normal_idx]
print('3')
LDA_ref = [LDA_re[i] for i in normal_idx]
print('4')
word2vec_ref = [word2vec_re[i] for i in normal_idx]
print('5')
FT_ref = [FT_re[i] for i in normal_idx]
print('6')
LSTM_ref = [LSTM_re[i] for i in normal_idx]
print('7')
xlnet_ref = [xlnet_re[i] for i in normal_idx]
print('8')

assert len(bert_ref) == len(roberta_ref) == len(dis_ref) == len(LDA_ref) == \
       len(word2vec_ref) == len(FT_ref) == len(LSTM_ref) == len(xlnet_ref)
print('展开后的候选项总数（除去None）：', len(bert_ref))


# 计算平均分歧
'''subs0 = subs1 = subs2 = subs3 = subs4 = subs5 = subs6 = 0
for a, b, c, d, e, f, g, h in zip(bert_ref, dis_ref, roberta_ref, xlnet_ref, LSTM_ref, word2vec_ref, FT_ref, LDA_ref):     #
    lst0 = [a, b]
    lst1 = [a, b, c]
    lst2 = [a, b, c, d]
    lst3 = [a, b, c, d, e]
    lst4 = [a, b, c, d, e, f]
    lst5 = [a, b, c, d, e, f, g]
    lst6 = [a, b, c, d, e, f, g, h]
    sub0 = max(lst0)-min(lst0)
    sub1 = max(lst1)-min(lst1)
    sub2 = max(lst2)-min(lst2)
    sub3 = max(lst3)-min(lst3)
    sub4 = max(lst4)-min(lst4)
    sub5 = max(lst5)-min(lst5)
    sub6 = max(lst6)-min(lst6)
    subs0 += sub0
    subs1 += sub1
    subs2 += sub2
    subs3 += sub3
    subs4 += sub4
    subs5 += sub5
    subs6 += sub6
print(subs0/len(bert_ref))
print(subs1/len(bert_ref))
print(subs2/len(bert_ref))
print(subs3/len(bert_ref))
print(subs4/len(bert_ref))
print(subs5/len(bert_ref))
print(subs6/len(bert_ref))'''

print('\n开始计算结果的区间分布...')
# 计算结果的区间分布
for r_result in [word2vec_ref, FT_ref, bert_ref, roberta_ref, dis_ref, xlnet_ref, LSTM_ref, LDA_ref]:
    print(len(r_result))
    count = [0 for _ in range(10)]
    print(count)
    for i in r_result:
        if -1 <= i < -0.8:
            count[0] += 1
        if -0.8 <= i < -0.6:
            count[1] += 1
        if -0.6 <= i < -0.4:
            count[2] += 1
        if -0.4 <= i < -0.2:
            count[3] += 1
        if -0.2 <= i < 0:
            count[4] += 1
        if 0 <= i < 0.2:
            count[5] += 1
        if 0.2 <= i < 0.4:
            count[6] += 1
        if 0.4 <= i < 0.6:
            count[7] += 1
        if 0.6 <= i < 0.8:
            count[8] += 1
        if 0.8 <= i <= 1:
            count[9] += 1

    print(count)
    count_sum = sum(count)
    print(count_sum)
    count_percent = [cc/count_sum for cc in count]
    print(count_percent)
    print('--'*20)

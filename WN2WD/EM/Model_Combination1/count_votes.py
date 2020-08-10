"""根据模型的中间结果统计出最终结果，并生成训练用的neg数据"""
import pickle
import csv
from collections import Counter

from run_models import query_candidate


# 读取中间结果，即pkl文件
def read_mid_pkl(model_name, pkl_num):
    all_mid = []
    for i in range(pkl_num):
        with open('models_mid_result/{}{}.pkl'.format(model_name, str(i+1)), 'rb') as f:
            data = pickle.load(f)
            mid = data['model results']
            all_mid += mid
    print(len(all_mid))
    return all_mid


# 和run_models的数字一致
def count_votes(model_weight):
    LDA_sim = read_mid_pkl('LDA', pkl_num=1)
    word2vec_sim = read_mid_pkl('word2vec', pkl_num=1)
    xlnet_sim = read_mid_pkl('xlnet', pkl_num=1)
    FT_sim = read_mid_pkl('FastText', pkl_num=1)
    bert_sim = read_mid_pkl('bert', pkl_num=1)
    LSTM_sim = read_mid_pkl('LSTM', pkl_num=1)

    assert len(LDA_sim) == len(word2vec_sim) == len(FT_sim) == len(xlnet_sim) == len(bert_sim) == len(LSTM_sim)
    query_result = query_candidate(start_num=1, end_num=117660, for_count_votes=True)

    empty_count = 0
    all_most_counter = list()
    for i in range(len(LDA_sim)):
        if len(LDA_sim[i]) == 0:   # 候选集为空就跳过
            empty_count += 1
            continue
        max1 = LDA_sim[i].index(max(LDA_sim[i]))
        max2 = word2vec_sim[i].index(max(word2vec_sim[i]))
        max3 = xlnet_sim[i].index(max(xlnet_sim[i]))
        max4 = FT_sim[i].index(max(FT_sim[i]))
        all_max = [max1, max2, max3, max4]
        max5 = bert_sim[i].index(max(bert_sim[i]))
        all_max.append(max5)
        max6 = LSTM_sim[i].index(max(LSTM_sim[i]))
        all_max.append(max6)

        wiki_des = [j[1] for j in query_result[i][1]]
        all_max_des = []
        for j, k in zip(all_max, model_weight):
            all_max_des += [wiki_des[j] for _ in range(k)]

        counter = Counter(all_max_des)
        most_counter = counter.most_common(1)[0]
        all_most_counter.append((wiki_des.index(most_counter[0]), str(most_counter[1])+'|'+str(len(all_max_des))))

    print('synset number:', len(LDA_sim))
    print('其中wikidata候选集为空的数量:{}'.format(empty_count))
    print('最终储存的synset数量:', len(LDA_sim)-empty_count)

    query_result = [i for i in query_result if len(i[1]) != 0]

    # 某一轮的总结果
    new_file_path = 'result_files/map_.csv'
    # 注意编码
    with open(new_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['synset_id', 'synset_description', 'synset_words',
                        'wikidata_qnode', 'wikidata_description', 'wikidata_labels', 'gained_votes'])
        for query_data, one_counter in zip(query_result, all_most_counter):
            f_csv.writerow([query_data[0][0], query_data[0][1], ','.join(query_data[0][2])]
                           + query_data[1][one_counter[0]] + [one_counter[1]])

    # 训练集中的neg数据
    new_file_path = 'result_files/map_neg_.csv'
    # 注意编码
    with open(new_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['text1', 'text2', 'is_same'])
        for query_data, one_counter in zip(query_result, all_most_counter):
            if one_counter[1] == '60|60':     # 投票为60|60时取neg数据
                neg_seq = [i[1] for i in query_data[1] if i[1] not in
                           [query_data[1][one_counter[0]][1], 'None']]    # 不和正确答案相同，不是None
                if len(neg_seq) == 0:
                    neg_seq = ['None']
                for k in neg_seq:
                    f_csv.writerow([query_data[0][1], k, '0'])


if __name__ == "__main__":
    # weight和为60
    model_weight = [10, 10, 10, 10, 11, 9]
    assert sum(model_weight) == 60
    count_votes(model_weight=model_weight)

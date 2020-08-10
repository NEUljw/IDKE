import pickle
import csv
from collections import Counter
from run_models import query_candidate

from test2 import read_marked_file


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
def count_votes():
    LDA_sim = read_mid_pkl('LDA', pkl_num=1)
    word2vec_sim = read_mid_pkl('word2vec', pkl_num=1)
    roberta_sim = read_mid_pkl('roberta', pkl_num=1)
    FT_sim = read_mid_pkl('FastText', pkl_num=1)
    bert_sim = read_mid_pkl('bert', pkl_num=1)
    LSTM_sim = read_mid_pkl('LSTM', pkl_num=1)
    distilbert_sim = read_mid_pkl('distilbert', pkl_num=1)
    xlnet_sim = read_mid_pkl('xlnet', pkl_num=1)

    assert len(LDA_sim) == len(word2vec_sim) == len(FT_sim) == len(roberta_sim) == len(bert_sim) == \
           len(LSTM_sim) == len(distilbert_sim) == len(xlnet_sim)

    query_result = query_candidate(start_num=1, end_num=117660, for_count_votes=True)
    print(len(LDA_sim), len(query_result))
    eee = 0
    c = []
    empty_count = 0
    ww = 0
    all_most_counter, model_score = [], [0 for _ in range(8)]
    bad_model = [0 for _ in range(8)]
    _, marked_lst = read_marked_file()
    for wrong_map in marked_lst:
        for i in range(len(LDA_sim)):
            if len(LDA_sim[i]) == 0:   # 候选集为空就跳过
                empty_count += 1
                continue
            max1 = LDA_sim[i].index(max(LDA_sim[i]))
            max2 = word2vec_sim[i].index(max(word2vec_sim[i]))
            max3 = roberta_sim[i].index(max(roberta_sim[i]))
            max4 = FT_sim[i].index(max(FT_sim[i]))
            all_max = [max1, max2, max3, max4]
            max5 = bert_sim[i].index(max(bert_sim[i]))
            all_max.append(max5)
            max6 = LSTM_sim[i].index(max(LSTM_sim[i]))
            all_max.append(max6)
            max7 = distilbert_sim[i].index(max(distilbert_sim[i]))
            all_max.append(max7)
            max8 = xlnet_sim[i].index(max(xlnet_sim[i]))
            all_max.append(max8)

            wiki_des = [j[1] for j in query_result[i][1]]
            all_max_des = [wiki_des[j] for j in all_max]


            if query_result[i][0][0] == wrong_map and len([kk for kk in wiki_des if kk != 'None']) in [3,4,5, 6]:
                print('--'*30)
                print(query_result[i][0][0])
                print(query_result[i][0][1])
                print(query_result[i][0][2])
                print(wiki_des)
                print(query_result[i][1])

                # print('LDA:', LDA_sim[i])
                # print('word2vec:', word2vec_sim[i])
                # print('FastText:', FT_sim[i])
                # print('BERT:', bert_sim[i])
                # print('RoBERTa:', roberta_sim[i])
                # print('DistilBERT:', distilbert_sim[i])
                # print('XLNet:', xlnet_sim[i])
                # print('LSTM:', LSTM_sim[i])
                print('--'*30)
                break


        '''counter = Counter(all_max_des)
        most_counter = counter.most_common(1)[0]
        c.append(most_counter[1])


        if wiki_des.count(most_counter[0]) > 1:
            ww += 1
            if ww < 20:
                print(wiki_des)
                print(query_result[i][0])
                print(query_result[i][1])
                print('--'*50)
            else:
                break


        if len(set(wiki_des)) != 1:
            if most_counter[1] == 3:
                for index, k in enumerate(all_max_des):
                    if k == most_counter[0]:
                        model_score[index] += 0.2
            if most_counter[1] == 4:
                for index, k in enumerate(all_max_des):
                    if k == most_counter[0]:
                        model_score[index] += 0.4
            if most_counter[1] == 5:
                for index, k in enumerate(all_max_des):
                    if k == most_counter[0]:
                        model_score[index] += 0.6
            if most_counter[1] == 6:
                for index, k in enumerate(all_max_des):
                    if k == most_counter[0]:
                        model_score[index] += 0.8
            if most_counter[1] == 7:
                for index, k in enumerate(all_max_des):
                    if k == most_counter[0]:
                        model_score[index] += 1
                    else:
                        bad_model[index] += 1
            if most_counter[1] == 8:
                for index, k in enumerate(all_max_des):
                    if k == most_counter[0]:
                        model_score[index] += 1
        else:
            eee += 1

        all_most_counter.append((wiki_des.index(most_counter[0]), str(most_counter[1])+'|'+str(len(all_max_des))))
    counter = Counter(c)
    print(counter)
    print(eee)
    print(model_score)
    print(bad_model)
    print('synset number:', len(LDA_sim))
    print('其中wikidata候选集为空的数量:{}'.format(empty_count))
    print('最终储存的synset数量:', len(all_most_counter))'''

    # query_result = [i for i in query_result if len(i[1]) != 0]
    # print(len(query_result))
    #
    # # 某一轮的总结果
    # new_file_path = 'result_files/map_.csv'
    # # 注意编码
    # with open(new_file_path, 'w', encoding='utf-8-sig', newline='') as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(['synset_id', 'synset_description', 'synset_words',
    #                     'wikidata_qnode', 'wikidata_description', 'wikidata_labels', 'gained_votes'])
    #     for query_data, one_counter in zip(query_result, all_most_counter):
    #         if one_counter[1] == '7|8':
    #             f_csv.writerow([query_data[0][0], query_data[0][1], ','.join(query_data[0][2])]
    #                            + query_data[1][one_counter[0]] + [one_counter[1]])


if __name__ == "__main__":
    count_votes()

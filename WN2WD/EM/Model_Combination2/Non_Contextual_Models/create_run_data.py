import csv
import pickle


def read_wordnet_data(syn_start, syn_end):
    """
    1-10, 10-20
    """
    data = []
    with open('data/wordnet_data.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)
        n = 0
        for row in f_csv:
            n += 1
            if syn_start <= n < syn_end:
                synset_id = row[0]
                synset_des = row[1]
                synset_words = row[2].split(',')
                data.append([synset_id, synset_des, synset_words])
    return data


def read_one_wiki_candidate(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)
    return data


def read_wiki_candidate():
    data = []
    for file_num in range(1, 14):
        one_data = read_one_wiki_candidate(
            file_path='data/wikidata_candidate/candidate_part_' + str(file_num) + '.csv')
        data += one_data
    return data


def create_run_data():
    wordnet_data = read_wordnet_data(syn_start=1, syn_end=117660)
    candidate = read_wiki_candidate()

    # 对数据进行处理
    sign_lst = []
    query_result, unsolvable_result = [], []
    uu = 0
    for one_synset in wordnet_data:
        sign = []
        uu += 1
        if uu % 1000 == 0:
            print(uu)
        one_synset_result = []
        for one_word in one_synset[2]:
            for row in candidate:
                if row[0] == one_word:
                    if len(row) > 1:
                        row = row[1:]
                        row_triple = []
                        for one_triple in range(int(len(row) / 3)):
                            row_triple.append(row[3 * one_triple:3 * (one_triple + 1)])
                        one_synset_result += row_triple
                    break
        # 候选项去重
        one_synset_result_set = []
        for i in one_synset_result:
            if i not in one_synset_result_set:
                one_synset_result_set.append(i)
                sign.append(1)
            else:
                sign.append(0)
        # 将无法判断的候选集储存
        wiki_des = [i[1] for i in one_synset_result_set]
        if set(wiki_des) != {'None'}:
            query_result.append([one_synset, one_synset_result_set])
        else:
            unsolvable_result.append([one_synset, one_synset_result_set])
            sign = 'all is None'
        sign_lst.append(sign)
    print(len(query_result))
    query_result = [i for i in query_result if len(i[1]) != 0]
    print(len(query_result))

    with open('sign_lst.pkl', 'wb') as f:
        pickle.dump({'data': sign_lst}, f)

    # with open('data/run_data.pkl', 'wb') as f:
    #     pickle.dump({'data': query_result}, f)       # 109430
    # with open('data/unsolvable_data.pkl', 'wb') as f:
    #     pickle.dump({'data': unsolvable_result}, f)   # 2857


if __name__ == "__main__":
    create_run_data()

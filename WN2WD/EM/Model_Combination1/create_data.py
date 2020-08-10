"""
生成的结构化的wikidata候选集文件中，第一列是查询的word，然后依次是wikidata的qnode、description、labels（每行top_n个候选）.
注意：某个单词的候选集可能为空，最多有top_n个，且有极少数word查询失败
"""
import csv

# 设置字段最大值
csv.field_size_limit(500 * 1024 * 1024)


def create_wordnet_data():
    data = []
    with open('original_data_all/WN2WD_Mapping.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append([row[0], row[5], row[6]])

    with open('data/wordnet_data.csv', 'w', encoding='utf-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(data)


# list转换为str
def list_to_str(data, replace_blank=True):
    def replace_blanks(word):
        # 空格转换为下划线
        after_word = word.replace(' ', '_')
        return after_word

    data_str = ''
    if replace_blank is True:
        for i in data:
            data_str = data_str+replace_blanks(i)+','
    else:
        for i in data:
            data_str = data_str+i+','
    data_str = data_str[:-1]
    return data_str


# 将wikidata候选集的数据结构化
def structure_candidate(file_path, new_file_path, top_n):
    all_count, empty_count, error_count, right_count = 0, 0, 0, 0
    candidate_dict = dict()
    with open(file_path, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            all_count += 1
            row_results = []
            query_word = row[0]
            row_dict = eval(row[1])

            # 候选集为空
            try:
                if row_dict['hits']['total'] == 0:
                    candidate_dict[query_word] = []
                    empty_count += 1
                    continue
            # 返回的查询结果error
            except KeyError:
                candidate_dict[query_word] = []
                error_count += 1
                continue

            for one_r in row_dict['hits']['hits']:
                one_source = one_r['_source']
                one_qnode = one_source['title']
                if 'descriptions' not in one_source.keys():
                    one_desc = 'None'
                else:
                    if 'en' not in one_source['descriptions'].keys():
                        one_desc = 'None'
                    else:
                        one_desc = one_source['descriptions']['en'][0]

                if 'labels' not in one_source.keys():
                    one_labels = 'None'
                else:
                    if 'en' not in one_source['labels'].keys():
                        one_labels = 'None'
                    else:
                        one_labels = one_source['labels']['en']
                        if len(one_labels) == 0:
                            one_labels = 'None'
                        else:
                            one_labels = list_to_str(one_labels, replace_blank=True)
                row_results.append([one_qnode, one_desc, one_labels])
            # 取候选集的前top_n个结果
            if len(row_results) > top_n:
                row_results = row_results[:top_n]
            candidate_dict[query_word] = row_results
            right_count += 1
    data = []
    for key, value in candidate_dict.items():
        all_value = [key]
        for k in value:
            all_value += k
        data.append(all_value)

    # 注意编码
    with open(new_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(data)
    print('all count:{} | empty count:{} | error count:{} | right count:{}'.format(
        all_count, empty_count, error_count, right_count))


def create_candidate_wikidata(top_n):
    for file_num in range(1, 14):
        structure_candidate(file_path='original_data_all/top50/all_synsets_'+str(file_num)+'.csv',
                            new_file_path='data/wikidata_candidate/candidate_part_'+str(file_num)+'.csv',
                            top_n=top_n)
        print('number '+str(file_num)+' file done.')


if __name__ == '__main__':
    create_candidate_wikidata(top_n=15)

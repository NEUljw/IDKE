"""注意生成的字典和另一个字典形式一致，word为短语或单词的词干，注意最后去重"""
import csv
import pickle
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm

# 设置字段最大值
csv.field_size_limit(500 * 1024 * 1024)
porter_stemmer = PorterStemmer()
candi_dict = {}


def read_one_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in tqdm(f_csv):
            word = row[0].replace('_', ' ')
            web_result = eval(row[1])

            if word.find(' ') < 0:
                word = porter_stemmer.stem(word)

            try:
                web_result = web_result['hits']['hits']
            except KeyError:
                continue
            cands = [i['_id'] for i in web_result]

            if len(cands) > 0:
                if word not in candi_dict.keys():
                    candi_dict[word] = cands
                else:
                    candi_dict[word] += cands


for file_n in range(1, 14):
    read_one_file(file_path='data/original_data/top50/all_synsets_'+str(file_n)+'.csv')
print(len(candi_dict))

cand_final = {}
for key, value in candi_dict.items():
    cand_final[key] = list(set(value))

with open('web_result_word2qnodes.pkl', 'wb') as f:
    pickle.dump(cand_final, f)
print('done!')

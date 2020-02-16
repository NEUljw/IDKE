import networkx as nx
from tqdm import tqdm
import csv
import jsonlines
import jieba

test_data_words = []
stopwords = []


def read_source():
    global stopwords
    with open('stopwords.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            stopwords.append(line)


def remove_blank(text):
    after = []
    for i in text:
        if i != ' ':
            after.append(i)
    return after


def extract_words(text):
    global stopwords
    word_list = jieba.lcut(text)
    word_list = remove_blank(word_list)
    word_set_all = list(set(word_list))
    word_set = []
    # 去除停用词(考虑了大写的情况)
    for i in word_set_all:
        if i.lower() not in stopwords:
            word_set.append(i)
    return word_set


def read_dev_data():
    global test_data_words
    with open('../hellaswag-train-dev/valid.jsonl', 'r+', encoding='utf-8') as f:
        n = 0
        for item in jsonlines.Reader(f):
            n += 1
            if n <= 300:    # 300
                pre_text = item['ctx']
                end_text = item['ending_options']
                test_data_words += extract_words(pre_text)
                test_data_words += extract_words(end_text[0])
                test_data_words += extract_words(end_text[1])
                test_data_words += extract_words(end_text[2])
                test_data_words += extract_words(end_text[3])
            else:
                break
    test_data_words = list(set(test_data_words))


def save_cpnet():
    global test_data_words
    print(len(test_data_words))
    graph = nx.MultiDiGraph()

    with open('total.csv') as f:
        f_csv = csv.reader(f)
        n = 0
        for line in tqdm(f_csv, desc="saving to graph"):
            n += 1
            if n > 1:
                rel = line[1]
                start = line[0]
                end = line[2]
                if rel == "HasContext":
                    continue
                # if not_save(start) or not_save(end):
                #     continue
                if start == end:  # delete loops
                    continue
                if start in test_data_words or end in test_data_words:
                    graph.add_edge(start, end, rel=rel)
                    graph.add_edge(end, start, rel='*'+rel)      # 关系相反前面加*
                if n % 10000 == 0:
                    print(n, '/4920000')

    nx.write_gpickle(graph, 'cpnet.graph')


if __name__ == "__main__":
    read_source()
    read_dev_data()
    save_cpnet()

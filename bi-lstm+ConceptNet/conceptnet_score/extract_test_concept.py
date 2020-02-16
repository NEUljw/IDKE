import jsonlines
import jieba
from openpyxl import Workbook

concept_vocab = []
stopwords = []


def read_source():
    global concept_vocab, stopwords
    with open('concept_vocab.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            concept_vocab.append(line)
    with open('stopwords.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            stopwords.append(line)


def list_to_str(con):
    con_str = ''
    for i in con:
        con_str = con_str+i+'|'
    con_str = con_str[:-1]
    return con_str


def remove_blank(text):
    after = []
    for i in text:
        if i != ' ':
            after.append(i)
    return after


def query_concept(text):
    global concept_vocab, stopwords
    concept = []
    word_list = jieba.lcut(text)
    word_list = remove_blank(word_list)
    word_set_all = list(set(word_list))
    word_set = []
    # 去除停用词(考虑了大写的情况)
    for i in word_set_all:
        if i.lower() not in stopwords:
            word_set.append(i)
    for i in word_set:
        if i in concept_vocab:
            concept.append(i)
    return concept


def read_dev_data():
    q_con, a1_con, a2_con, a3_con, a4_con = [], [], [], [], []
    with open('../hellaswag-train-dev/valid.jsonl', 'r+', encoding='utf-8') as f:
        n = 0
        for item in jsonlines.Reader(f):
            n += 1
            if n <= 300:    # 300
                pre_text = item['ctx']
                end_text = item['ending_options']
                q1 = query_concept(pre_text)
                q2 = query_concept(end_text[0])
                q3 = query_concept(end_text[1])
                q4 = query_concept(end_text[2])
                q5 = query_concept(end_text[3])
                q_con.append(list_to_str(q1))
                a1_con.append(list_to_str(q2))
                a2_con.append(list_to_str(q3))
                a3_con.append(list_to_str(q4))
                a4_con.append(list_to_str(q5))
                if n % 10 == 0:
                    print(n, 'done...')
            else:
                break
    # print(q_con)
    # print(a1_con)
    # print(a2_con)
    # print(a3_con)
    # print(a4_con)

    wb = Workbook()
    ws = wb.active
    for i in range(len(q_con)):
        ws.append([q_con[i], a1_con[i], a2_con[i], a3_con[i], a4_con[i]])
    wb.save('test_data_concept.xlsx')


if __name__ == "__main__":
    read_source()
    read_dev_data()

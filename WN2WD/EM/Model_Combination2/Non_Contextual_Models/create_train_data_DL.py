"""从每一轮的结果中提取出positive训练数据，合并neg、pos数据"""
import csv
from random import shuffle

# File path
RESULT_FILE_PATH = 'result_files/map of 6models.csv'
NEG_FILE_PATH = 'result_files/train_neg.csv'
TRAIN_FILE_PATH = 'result_files/train_.csv'


def create_r1_train_data():
    data = []
    with open('original_data_all/WN2WD_Mapping.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)
        for row in f_csv:
            if len(row[21]) != 0 and row[11] != 'None':
                data.append([row[5], row[11], '1'])
    print('去重前：', len(data))
    data_set = []
    for i in data:
        if i not in data_set:
            data_set.append(i)
    print('去重后：', len(data_set))
    print(data_set[0])

    with open(TRAIN_FILE_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['text1', 'text2', 'similarity'])
        f_csv.writerows(data_set)


def create_r2_train_data():
    pos_votes = [str(i)+'|60' for i in range(50, 61)]
    print(pos_votes)
    data = []
    with open(RESULT_FILE_PATH, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)     # 跳过表头
        for row in f_csv:
            if row[6] in pos_votes and row[1] != 'None' and row[4] != 'None':
                data.append([row[1], row[4], '1'])
    # print(len(data))
    data_new = []
    for i in data:
        if i not in data_new:
            data_new.append(i)
    print('pos num:', len(data_new))

    data_neg = []
    with open(NEG_FILE_PATH, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)     # 跳过表头
        for row in f_csv:
            if row[0] != 'None' and row[1] != 'None':
                data_neg.append([row[0], row[1], row[2]])
    # print(len(data_neg))
    data_new2 = []
    for i in data_neg:
        if i not in data_new2:
            data_new2.append(i)
    data_new2 = data_new2[:len(data_new)]       # 保证pos数据和neg数据数量相等
    print('neg num:', len(data_new2))

    data = data_new + data_new2
    print('train number:', len(data))
    shuffle(data)

    with open(TRAIN_FILE_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['text1', 'text2', 'is_same'])
        f_csv.writerows(data)


if __name__ == "__main__":
    create_r1_train_data()

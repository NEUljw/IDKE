import csv
from collections import Counter

# data = []
# has_common = 0
# a = b = 0
# with open('C:/Users/MyPC/Desktop/map300_marked.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     head_row = next(f_csv)  # 跳过表头
#     for row in f_csv:
#         wn_words = row[2].lower().split(',')
#         wiki_labels = row[5].lower().split(',')
#         wn_words = [i for i in wn_words if len(i) != 0]
#         wiki_labels = [i for i in wiki_labels if len(i) != 0]
#         intersection = list(set(wn_words).intersection(set(wiki_labels)))
#         if len(intersection) > 0:
#             has_common += 1
#             if row[7] == '1':
#                 a += 1
#             if row[7] == '0':
#                 b += 1
#         data.append(row)
# print(len(data))
# print(has_common)
# print(a, b)


# 194, 106
def read_marked_file():
    data = []
    a = b = 0
    e = []
    not_noun_count = 0
    with open('C:/Users/MyPC/Desktop/map300_marked.csv', 'r') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)  # 跳过表头
        for row in f_csv:
            if row[0].split('.')[1] != 'n':
                not_noun_count += 1
            else:
                if len(row[8]) > 0:
                    e.append(row[8])
                if row[7] == '1':
                    a += 1
                if row[7] == '0':
                    b += 1
                    data.append(row[0])
    print(a, b)
    print(len(e))
    print(Counter(e))
    print('非动词的个数：', not_noun_count)
    return data


read_marked_file()

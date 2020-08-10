import numpy as np
import csv
import pickle

'''def norm_score(scores):
    sc_sum = sum(scores)
    print(sc_sum)
    print(scores)
    sc_norm = [i/sc_sum*60 for i in scores]
    print(sc_norm)


norm_score([26760, 32275, 31016.6, 32341.2, 29037.2, 28088.4])'''

# def read_file(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8-sig') as f:
#         f_csv = csv.reader(f)
#         head_row = next(f_csv)  # 跳过表头
#         for row in f_csv:
#             if row[6] in ['6|6', '5|5', '60|60'] and row[1] != 'None' and row[4] != 'None':
#                 data.append([row[1], row[4]])
#     data_new = []
#     for i in data:
#         if i not in data_new:
#             data_new.append(i)
#     print('本轮去重后的有效结果数量:', len(data_new))
#     return data_new
#
#
# tune_r = ['r1.csv', 'r2.csv', 'r3.csv', 'r4.csv', 'r5.csv', 'r6.csv', 'r7.csv']
# r = []
# for one_r in tune_r:
#     r_ = read_file(one_r)
#     r += r_
# print('--'*20)
# r_set = []
# for i in r:
#     if i not in r_set:
#         r_set.append(i)
# print('所有结果去重之后:', len(r_set))


# def read_data(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8-sig') as f:
#         f_csv = csv.reader(f)
#         head_row = next(f_csv)  # 跳过表头
#         for row in f_csv:
#             data.append(row)
#     return data
#
#
# a = read_data('train_all.csv')
# b = read_data('train.csv')
# all_data = a + b
# print(len(a))
# print(len(b))
# print(len(all_data))
#
# with open('tttttt.csv', 'w', encoding='utf-8-sig', newline='') as f:
#     f_csv = csv.writer(f)
#     f_csv.writerow(['text1', 'text2', 'is_same'])
#     f_csv.writerows(all_data)
#
# pos = neg = 0
# with open('tttttt.csv', 'r', encoding='utf-8-sig') as f:
#     f_csv = csv.reader(f)
#     head_row = next(f_csv)  # 跳过表头
#     for row in f_csv:
#         if row[2] == '1':
#             pos += 1
#         else:
#             neg += 1
# print(pos, neg)

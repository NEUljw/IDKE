import csv
from collections import Counter


def read_one_result(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)     # 跳过表头
        for row in f_csv:
            data.append(row)
    return data


def count_result():
    results = read_one_result(file_path='result_files/map of 6models.csv')
    print(len(results))
    votes = [k[6] for k in results]
    counter = Counter(votes)
    vt_count = {'0-10|60': 0, '10-20|60': 0, '20-30|60': 0,
                '30-40|60': 0, '40-50|60': 0, '50-60|60': 0}
    for key, value in counter.items():
        vt = int(key.split('|')[0])
        if 0 < vt <= 10:
            vt_count['0-10|60'] += value
        if 10 < vt <= 20:
            vt_count['10-20|60'] += value
        if 20 < vt <= 30:
            vt_count['20-30|60'] += value
        if 30 < vt <= 40:
            vt_count['30-40|60'] += value
        if 40 < vt <= 50:
            vt_count['40-50|60'] += value
        if 50 < vt <= 60:
            vt_count['50-60|60'] += value
    for key, value in vt_count.items():
        print(key, value)


if __name__ == '__main__':
    count_result()

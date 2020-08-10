import csv


def create_r1_train_data():
    data = []
    with open('original_data_all/WN2WD_Mapping.csv', 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)
        for row in f_csv:
            if len(row[21]) != 0 and row[11] != 'None':
                data.append(row[5]+'. '+row[11])
    print('去重前：', len(data))
    data = list(set(data))
    print('去重后：', len(data))
    print(data[0])

    with open('sentences.txt', 'w', encoding='utf-8-sig') as f:
        for i in data:
            f.write(i+'\n')


if __name__ == "__main__":
    create_r1_train_data()

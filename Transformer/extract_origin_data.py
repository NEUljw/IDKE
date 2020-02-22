import jsonlines
from openpyxl import Workbook


def read_data(text_path, labels_path, file_name, data_num):
    all_text, all_label = [], []
    with open(text_path, 'r+', encoding='utf-8') as f:
        n = 0
        for item in jsonlines.Reader(f):
            n += 1
            if n <= data_num:
                pre_text = item['ctx']
                end_text = item['ending_options']
                seq1 = pre_text + ' ' + end_text[0]
                seq2 = pre_text + ' ' + end_text[1]
                seq3 = pre_text + ' ' + end_text[2]
                seq4 = pre_text + ' ' + end_text[3]
                all_text.append(seq1)
                all_text.append(seq2)
                all_text.append(seq3)
                all_text.append(seq4)

    with open(labels_path) as f:
        answer = [i.strip() for i in f.readlines()]
        m = 0
        for j in answer:
            m += 1
            if m <= data_num:
                one_line_label = [0, 0, 0, 0]
                one_line_label[int(j)] = 1
                all_label += one_line_label

    print(len(all_text), len(all_label))

    wb = Workbook()
    ws = wb.active
    for i in range(len(all_label)):
        ws.append([all_text[i], all_label[i]])
    wb.save('data/'+file_name)


if __name__ == '__main__':
    train_text_path = 'hellaswag-train-dev/train.jsonl'
    train_label_path = 'hellaswag-train-dev/train-labels.lst'
    valid_text_path = 'hellaswag-train-dev/valid.jsonl'
    valid_label_path = 'hellaswag-train-dev/valid-labels.lst'
    read_data(train_text_path, train_label_path, file_name='train.xlsx', data_num=2500)
    read_data(valid_text_path, valid_label_path, file_name='valid.xlsx', data_num=1000)
    print('file save done.')

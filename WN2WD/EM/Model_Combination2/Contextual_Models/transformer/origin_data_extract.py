import jsonlines
from openpyxl import Workbook


# 从原始数据集中读取训练数据，只提取正确的ending，保存为xlsx文件
def read_hellaswag_train_data(text_path, labels_path, file_name):
    all_pre_text, all_endings, all_label = [], [], []
    with open(text_path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            pre_text = item['ctx']
            end_text = item['ending_options']
            all_pre_text.append(pre_text)
            all_endings.append(end_text[0])
            all_endings.append(end_text[1])
            all_endings.append(end_text[2])
            all_endings.append(end_text[3])

    with open(labels_path) as f:
        answer = [i.strip() for i in f.readlines()]
        for j in answer:
            one_line_label = [0, 0, 0, 0]
            one_line_label[int(j)] = 1
            all_label += one_line_label

    print(len(all_pre_text), len(all_endings), len(all_label))

    wb = Workbook()
    ws = wb.active
    pre_index = 0
    for i in range(len(all_label)):
        if all_label[i] == 1:
            ws.append([all_pre_text[pre_index], all_endings[i]])
            pre_index += 1
    wb.save('data/' + file_name)


# 从原始数据集中读取验证数据，保存为xlsx文件
def read_hellaswag_dev_data(text_path, labels_path, file_name):
    all_pre_text, all_endings, all_label = [], [], []
    with open(text_path, 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            pre_text = item['ctx']
            end_text = item['ending_options']
            all_pre_text.append(pre_text)
            all_endings.append(end_text)

    with open(labels_path) as f:
        answer = [i.strip() for i in f.readlines()]
        for j in answer:
            all_label.append(int(j))

    print(len(all_pre_text), len(all_endings), len(all_label))

    wb = Workbook()
    ws = wb.active
    for i in range(len(all_label)):
        ws.append([all_pre_text[i], all_endings[i][0], all_endings[i][1], all_endings[i][2], all_endings[i][3],
                   all_label[i]])
    wb.save('data/' + file_name)


train_text_path = 'hellaswag-train-dev/train.jsonl'
train_label_path = 'hellaswag-train-dev/train-labels.lst'
valid_text_path = 'hellaswag-train-dev/valid.jsonl'
valid_label_path = 'hellaswag-train-dev/valid-labels.lst'
read_hellaswag_train_data(train_text_path, train_label_path, 'train.xlsx')
read_hellaswag_dev_data(valid_text_path, valid_label_path, 'dev.xlsx')

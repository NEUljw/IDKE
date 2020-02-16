import jsonlines
import torch
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

batch_of_pairs = []


def read_dev_data():
    # 读取测试数据
    with open('hellaswag-train-dev/valid.jsonl', 'r+', encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            if len(batch_of_pairs) < 800:
                pre_text = item['ctx']
                end_text = item['ending_options']
                batch_of_pairs.append([pre_text, end_text[0]])
                batch_of_pairs.append([pre_text, end_text[1]])
                batch_of_pairs.append([pre_text, end_text[2]])
                batch_of_pairs.append([pre_text, end_text[3]])


def predict(part_of_pairs):
    # 加载roberta模型
    roberta = RobertaModel.from_pretrained('./models/roberta.large.mnli/', checkpoint_file='model.pt')
    roberta.eval()
    # 编码
    batch = collate_tokens(
        [roberta.encode(pair[0], pair[1]) for pair in part_of_pairs], pad_idx=1
        )

    logprobs = roberta.predict('mnli', batch)
    logprobs = logprobs.tolist()
    entail_logprobs = []
    for i in logprobs:
        entail_logprobs.append(i[2])

    int_result = []
    for i in range(int(len(entail_logprobs)/4)):
        part_logprobs = entail_logprobs[4*i:4*(i+1)]
        max_prob = -10
        for j in part_logprobs:
            if j > max_prob:
                max_prob = j
        int_result.append(part_logprobs.index(max_prob))
    return int_result


if __name__ == "__main__":
    read_dev_data()
    final_result = []
    for i in range(int(len(batch_of_pairs)/200)):
        part_result = predict(batch_of_pairs[200*i:200*(i+1)])
        final_result += part_result

    filename = 'labels_200.txt'
    with open(filename, 'w') as f:
        for i in final_result:
            f.write(str(i))
    print('done!')

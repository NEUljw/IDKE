def read_concept_score():
    all_qa_score = []
    with open('conceptnet_score/concept_score1.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(',')
            qa_score = []
            for i in line:
                qa_score.append(float(i))
            all_qa_score.append(qa_score)
    with open('conceptnet_score/concept_score2.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(',')
            qa_score = []
            for i in line:
                qa_score.append(float(i))
            all_qa_score.append(qa_score)
    return all_qa_score


def read_lstm_score():
    n = 0
    scores, lstm_score = [], []
    with open('LSTM_score/lstm_score.txt', 'r') as f:
        for line in f.readlines():
            n += 1
            if n <= 320:
                line = line.strip('\n')
                scores.append(float(line))
    for i in range(int(len(scores)/4)):
        lstm_score.append(scores[4*i:4*(i+1)])
    return lstm_score


def read_test_answer():
    labels = []
    with open('hellaswag-train-dev/valid-labels.lst') as f:
        answer = [i.strip() for i in f.readlines()]
    for k in answer:
        labels.append(int(k))
    return labels[:80]


def normalization(score):
    nor_score = []
    max_score = max(score)
    min_score = min(score)
    for i in score:
        s = (i-min_score)/(max_score-min_score)
        nor_score.append(s)
    return nor_score


if __name__ == "__main__":
    w = 0   # 0.2875
    true_result = read_test_answer()
    pre_result = []
    score1 = read_concept_score()
    score2 = read_lstm_score()
    for i in range(len(score1)):
        fin_score = []
        nor_score1 = normalization(score1[i])
        nor_score2 = normalization(score2[i])
        fin_score.append(w*nor_score1[0] + (1-w)*nor_score2[0])
        fin_score.append(w*nor_score1[1] + (1-w)*nor_score2[1])
        fin_score.append(w*nor_score1[2] + (1-w)*nor_score2[2])
        fin_score.append(w*nor_score1[3] + (1-w)*nor_score2[3])
        max_s = max(fin_score)
        pre_result.append(fin_score.index(max_s))
    correct_count = 0
    for i in range(len(pre_result)):
        if pre_result[i] == true_result[i]:
            correct_count += 1
    print('accuracy:', correct_count/len(pre_result))

import jsonlines
import jieba
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

stopwords = []


def read_source():
    # 读取停用词表
    global stopwords
    with open('stopwords.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            stopwords.append(line)


def read_data(file_path, labels_path, data_num):
    # 读取数据
    all_data, labels_data = [], []
    with open(file_path, 'r+', encoding='utf-8') as f:
        n = 0
        for item in jsonlines.Reader(f):
            n += 1
            if n <= data_num:
                pre_text = item['ctx']
                end_text = item['ending_options']
                all_data.append([pre_text, end_text[0], end_text[1], end_text[2], end_text[3]])

    with open(labels_path) as f:
        answer = [i.strip() for i in f.readlines()]
        m = 0
        for j in answer:
            m += 1
            if m <= data_num:
                labels_data.append(int(j))
    return all_data, labels_data


def remove_blank(text):
    after = []
    for i in text:
        if i != ' ':
            after.append(i)
    return after


def sentence_split(sen):
    # 数据的预处理
    global stopwords
    # 分词
    word_list = jieba.lcut(sen)
    word_list = remove_blank(word_list)
    # 只保留英文单词，并删除停用词
    key_words = []
    for i in word_list:
        if 'a' <= i[0] <= 'z' or 'A' <= i[0] <= 'Z':
            if i.lower() not in stopwords:
                key_words.append(i)
    # 词性标注
    tags = pos_tag(key_words)
    # 词形还原
    origin_key_words = []
    wnl = WordNetLemmatizer()
    for word, tag in tags:
        if tag.startswith('NN'):
            origin_key_words.append(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            origin_key_words.append(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            origin_key_words.append(wnl.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            origin_key_words.append(wnl.lemmatize(word, pos='r'))
        else:
            origin_key_words.append(wnl.lemmatize(word))
    # 去重
    origin_key_words = list(set(origin_key_words))
    return origin_key_words


def list_cut(syn, length):
    if len(syn) > length:
        return syn[:length]
    else:
        return syn


def cal_distance(story, ending):
    s_words = sentence_split(story)
    e_words = sentence_split(ending)

    # 如果至少有一个句子中的概念数为0，则返回前100对的distance平均值
    if len(s_words) == 0 or len(e_words) == 0:
        return 0.13

    distance = 0
    for e in e_words:
        max_score = 0
        for s in s_words:
            # 单词不相同时才计算
            if s != e:
                # 利用wordnet计算单词之间的相似度
                synset1 = list_cut(wn.synsets(s), length=20)
                synset2 = list_cut(wn.synsets(e), length=20)
                score_avg = 0
                count = 0
                for s1 in synset1:
                    for s2 in synset2:
                        score = s1.path_similarity(s2)
                        if score is not None:
                            score_avg += score
                            count += 1
                if count != 0:
                    score_avg = score_avg/count
                if score_avg > max_score:
                    max_score = score_avg
        distance += max_score
    distance = distance/len(e_words)
    return distance


def cal_acc(pred, answer):
    correct_count = 0
    for i in range(len(pred)):
        if pred[i] == answer[i]:
            correct_count += 1
    return correct_count/len(pred)


def main():
    test_data_number = 500
    read_source()
    valid_text_path = '../hellaswag-train-dev/valid.jsonl'
    valid_label_path = '../hellaswag-train-dev/valid-labels.lst'
    data, labels = read_data(file_path=valid_text_path, labels_path=valid_label_path, data_num=test_data_number)
    cal_result = []
    for one_pair in data:
        all_score = []
        all_score.append(cal_distance(one_pair[0], one_pair[1]))
        all_score.append(cal_distance(one_pair[0], one_pair[2]))
        all_score.append(cal_distance(one_pair[0], one_pair[3]))
        all_score.append(cal_distance(one_pair[0], one_pair[4]))
        max_score = max(all_score)
        cal_result.append(all_score.index(max_score))
    acc = cal_acc(pred=cal_result, answer=labels)
    print('test data number:', test_data_number)
    print('accuracy:', acc)


if __name__ == '__main__':
    main()
    # 300, 0.270
    # 500, 0.282

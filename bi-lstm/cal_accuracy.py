import jsonlines

model_predict = []


def read_test_label():
    with open('hellaswag-train-dev/valid-labels.lst') as f:
        answer = [i.strip() for i in f.readlines()]
        return answer


def read_model_result():
    f = open('labels_200.txt', 'r')
    result = f.read()
    f.close()
    for i in result:
        model_predict.append(i)


if __name__ == "__main__":
    test_label = read_test_label()
    read_model_result()

    print(test_label[:200])
    print(model_predict)

    print('test label number:', len(test_label[:200]))
    print('model predict number:', len(model_predict))

    correct_count = 0
    for i in range(len(model_predict)):
        if test_label[i] == model_predict[i]:
            correct_count += 1
    print('accuracy:', correct_count/len(model_predict))

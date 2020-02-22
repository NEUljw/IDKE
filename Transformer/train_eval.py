import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


def train(config, model, train_iter, dev_iter, test_iter, mode='train'):
    if mode == 'train':
        model.train()   # train模式
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        dev_best_loss = float('inf')

        for epoch in range(config.num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, config.num_epochs))

            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

            train_acc, train_loss = evaluate(model, train_iter)
            dev_acc, dev_loss = evaluate(model, dev_iter)
            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(), config.save_path)
                improve = 'up'
            else:
                improve = '--'

            msg = 'train loss:{} | train acc:{} | dev loss:{} | dev acc:{} | {}'
            print(msg.format(train_loss, train_acc, dev_loss, dev_acc, improve))

    if mode == 'predict':
        print('--'*20, 'predict', '--'*20)
        dev_acc = predict(config, model, dev_iter)
        test_acc = predict(config, model, test_iter)
        print('dev accuracy:{} | test accuracy:{}'.format(dev_acc, test_acc))


def predict(config, model, data):
    # predict
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    all_result = torch.Tensor()
    answer = []
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            all_result = torch.cat([all_result, outputs], dim=0)
            labels = labels.data.numpy().tolist()
            answer += labels
    all_result = all_result.tolist()
    label_1_result = []
    for i in all_result:
        label_1_result.append(i[1])
    answer_int = []
    for i in range(int(len(answer)/4)):
        tmp_list = answer[i*4:(i+1)*4]
        answer_int.append(tmp_list.index(1))
    pred_result = []
    for i in range(int(len(label_1_result)/4)):
        tmp_list = label_1_result[i*4:(i+1)*4]
        tmp_max = max(tmp_list)
        pred_result.append(tmp_list.index(tmp_max))
    correct_count = 0
    for i in range(len(pred_result)):
        if pred_result[i] == answer_int[i]:
            correct_count += 1
    acc = correct_count/len(pred_result)
    return acc


def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.numpy()
            predic = torch.max(outputs.data, 1)[1].numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)

    return acc, loss_total / len(data_iter)

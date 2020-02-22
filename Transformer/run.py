from train_eval import train, predict
from utils import build_dataset, build_iterator
import models.Transformer


if __name__ == '__main__':
    config = models.Transformer.Config()

    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    print('train data num:{} | dev data num:{} | test data num:{}'.format(len(train_data), len(dev_data), len(test_data)))
    msg = 'train batch num:{} | dev batch num:{} | test batch num:{}'
    print(msg.format(len(train_iter), len(dev_iter), len(test_iter)))

    # train
    config.n_vocab = len(vocab)
    model = models.Transformer.Model(config)
    # print(model.parameters)
    print('***'*30)
    train(config, model, train_iter, dev_iter, test_iter, mode='predict')

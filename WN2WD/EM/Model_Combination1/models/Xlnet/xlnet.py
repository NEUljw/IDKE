from keras_xlnet import Tokenizer, ATTENTION_TYPE_BI, ATTENTION_TYPE_UNI
from keras_xlnet import load_trained_model_from_checkpoint
import keras.backend.tensorflow_backend as ktf_keras
from keras.models import Model
from keras.layers import Add
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import os

from models.Xlnet.layers_keras import NonMaskingLayer
from models.Xlnet import args


# gpu配置与设置，在加载模型时运行
def set_gpu_option(gpu_name, gpu_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    # sess = tf.Session(config=config)
    # ktf_keras.set_session(sess)
    config = tf.ConfigProto(device_count={'GPU': gpu_num})
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    # config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    ktf_keras.set_session(session)

    print('--'*30)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print('--'*30)


def cosine_distance(v1, v2):     # 余弦距离
    if type(v1) == list:
        v1 = np.array(v1)
    if type(v2) == list:
        v2 = np.array(v2)
    if v1.all() and v2.all():
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        return 0


class KerasXlnetVector:
    def __init__(self, batch_size, gpu_name, gpu_num):
        set_gpu_option(gpu_name, gpu_num)
        self.attention_type = ATTENTION_TYPE_BI if args.attention_type[0] == 'bi' else ATTENTION_TYPE_UNI
        self.memory_len, self.target_len, self.batch_size = args.memory_len, args.target_len, batch_size
        self.checkpoint_path, self.config_path = args.ckpt_name, args.config_name
        self.layer_indexes, self.in_train_phase = args.layer_indexes, False

        print("##### load KerasXlnet start #####")
        self.graph = tf.get_default_graph()
        # 模型加载
        self.model = load_trained_model_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                        attention_type=self.attention_type,
                                                        in_train_phase=self.in_train_phase,
                                                        config_path=self.config_path,
                                                        memory_len=self.memory_len,
                                                        target_len=self.target_len,
                                                        batch_size=self.batch_size,
                                                        mask_index=0)
        # 字典加载
        self.tokenizer = Tokenizer(args.spiece_model)
        # debug时候查看layers
        self.model_layers = self.model.layers
        len_layers = self.model_layers.__len__()
        len_couche = int((len_layers - 6) / 10)
        # 一共126个layer
        # 每层10个layer,第一是7个layer的输入和embedding层
        # 一共12层
        layer_dict = [5]
        layer_0 = 6
        for i in range(len_couche):
            layer_0 = layer_0 + 10
            layer_dict.append(layer_0 - 2)

        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = self.model.output
        # 分类如果只有一层，取得不正确的话就取倒数第二层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(len_couche + 1)]:
                encoder_layer = self.model.get_layer(index=layer_dict[self.layer_indexes[0]]).output
            else:
                encoder_layer = self.model.get_layer(index=layer_dict[-2]).output

        # 否则遍历需要取的层，把所有层的weight取出来并加起来shape:768*层数
        else:
            # layer_indexes must be [0, 1, 2,3,......12]
            all_layers = [self.model.get_layer(index=layer_dict[lay]).output
                          if lay in [i + 1 for i in range(len_couche + 1)]
                          else self.model.get_layer(index=layer_dict[-3]).output  # 如果给出不正确，就默认输出倒数第二层
                          for lay in self.layer_indexes]
            all_layers = all_layers[1:]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)

        output_layer = NonMaskingLayer()(encoder_layer)
        model = Model(self.model.inputs, output_layer)
        if gpu_num >= 2:
            self.par_model = multi_gpu_model(model, gpus=gpu_num)
        else:
            self.par_model = model
        print("##### load KerasXlnet end #####")
        # model.summary()

    def xlnet_encode(self, texts):
        """输入句子的列表，返回句向量列表"""
        predicts = []

        def create_array():    # 将输入的文本转换为词典序号的形式
            data = []
            for text in texts:
                tokens = self.tokenizer.encode(text)
                tokens = tokens + [0] * (self.target_len - len(tokens)) if len(tokens) < self.target_len else tokens[0:self.target_len]    # padding
                token_input = np.array(tokens)
                mask_input = [0 if ids == 0 else 1 for ids in tokens].count(1)
                segment_input = np.zeros_like(token_input)
                memory_length_input = np.zeros(1)
                data.append([token_input, mask_input, segment_input, memory_length_input])
            return data

        array = create_array()
        my_iter = data_iter(array, batch_size=self.batch_size)
        for w1, w2, w3, w4 in my_iter:
            m_token_input = np.array(w1)
            m_mask_input = w2
            m_segment_input = np.array(w3)
            m_memory_length_input = np.array(w4)

            with self.graph.as_default():
                predict = self.par_model.predict([m_token_input, m_segment_input, m_memory_length_input],
                                                 batch_size=self.batch_size)
                for index, prob in enumerate(predict):
                    # pooled为句向量
                    pooled = sen_embed_cal(prob, m_mask_input[index])
                    pooled = pooled.tolist()
                    predicts.append(pooled)
        return predicts


def sen_embed_cal(word_embed, mask_count):
    """模型输出的所有词向量求平均得到句向量"""
    word_embed = word_embed[:mask_count]
    return np.mean(word_embed, axis=0)


def data_iter(data, batch_size):
    """生成器"""
    batch_num = len(data) // batch_size
    if len(data) % batch_size != 0:
        batch_num += 1
    X1, X2, X3, X4 = [], [], [], []
    for i in range(len(data)):
        X1.append(data[i][0])
        X2.append(data[i][1])
        X3.append(data[i][2])
        X4.append(data[i][3])
        if len(X1) == batch_size or i == (len(data)-1):
            yield X1, X2, X3, X4
            X1, X2, X3, X4 = [], [], [], []


def cal_sim_xlnet(all_wordnet_des, all_wiki_des_list, default_sim, xlnet_model):
    all_sim_list = []
    for wordnet_des, wiki_des_list in zip(all_wordnet_des, all_wiki_des_list):
        none_index = []
        for i in range(len(wiki_des_list)):
            if wiki_des_list[i] == 'None':
                none_index.append(i)
        all_des = [wordnet_des]+wiki_des_list
        pooled = xlnet_model.xlnet_encode(all_des)
        sim_list = []
        for wiki in pooled[1:]:
            sim = cosine_distance(pooled[0], wiki)
            sim_list.append(round(sim, 4))
        for i in none_index:
            sim_list[i] = default_sim
        all_sim_list.append(sim_list)
    return all_sim_list

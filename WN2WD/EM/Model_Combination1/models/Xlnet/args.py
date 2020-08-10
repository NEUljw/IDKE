# path of xlnet model
config_name = 'models/Xlnet/xlnet_model/xlnet_config.json'
ckpt_name = 'models/Xlnet/xlnet_model/xlnet_model.ckpt'
spiece_model = 'models/Xlnet/xlnet_model/spiece.model'
attention_type = 'bi'    # or 'uni'
# 历史序列长度
memory_len = 0
# 当前目标序列长度
target_len = 32
# 默认取倒数第二层的输出值作为句向量
layer_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    # 可填 0, 1, 2, 3, 4, 5, 6, 7..., 24,其中0为embedding层
# gpu使用率
gpu_memory_fraction = 0.8    # 0.64

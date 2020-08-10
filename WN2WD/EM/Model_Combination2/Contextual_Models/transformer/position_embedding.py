import numpy as np


# 位置编码公式的实现
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    print(sinusoid_table)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 维度为偶数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 维度为奇数
    return sinusoid_table


print(get_sinusoid_encoding_table(3, 4))

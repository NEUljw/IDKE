"""注意对描述为None的处理，以及labels数量很少的候选项的考虑(候选集都是Q开头的)"""
import pickle
from tqdm import tqdm
import numpy as np


class Config:
    wordnet_path = 'merged_candidate.pkl'
    wiki_path = 'wiki_new.pkl'      # wiki_new.pkl的路径
    embed_path = 'distilbert_id2embed.pkl'

    result_path = 'distilbert_result.pkl'      # 更改模型时也要修改这个，防止覆盖

    lab_add = 100       # 0.15
    lab_add_more = 100     # 0.25
    none_sim = 0
    lab_num_limit = 3


def cosine_sim(a, b):    # 余弦相似度，结果在-1到1之间，保留4位小数
    try:
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except ValueError:
        sim = 0
    return round(sim, 4)


def read_wordnet():
    with open(Config.wordnet_path, 'rb') as f:
        wn_data = pickle.load(f)
    print('synset总数:', len(wn_data))
    return wn_data


def cal():
    wn2result = {}
    wn = read_wordnet()

    print('读取qnode2wiki字典中...')
    with open(Config.wiki_path, 'rb') as f:
        qnode2wiki = pickle.load(f)
    print('字典大小:', len(qnode2wiki))

    print('读取id2embed字典中...')
    with open(Config.embed_path, 'rb') as f:
        id2embed = pickle.load(f)
    print('字典大小:', len(id2embed))

    for one_wn, candi in tqdm(wn.items(), desc='calculating'):
        one_wn_embed = id2embed[one_wn[0]]
        one_wn_labs = one_wn[2]
        one_wn_sim, one_wn_des_sim, one_wn_lab_sim = [], [], []
        for one_candi in candi:
            one_candi_wiki = qnode2wiki[one_candi]
            one_candi_des = one_candi_wiki[1]
            one_candi_labs = one_candi_wiki[2]
            one_candi_embed = id2embed[one_candi]
            # des sim
            if one_candi_des != 'None':
                des_sim = cosine_sim(one_wn_embed, one_candi_embed)
            else:
                des_sim = Config.none_sim
            one_wn_des_sim.append(des_sim)
            # lab sim
            lab_inter_num = len(set(one_wn_labs).intersection(set(one_candi_labs)))
            if lab_inter_num > 0:
                if len(one_candi_labs) > Config.lab_num_limit:
                    lab_sim = Config.lab_add * lab_inter_num
                else:
                    lab_sim = Config.lab_add_more * lab_inter_num
            else:
                lab_sim = 0
            one_wn_lab_sim.append(lab_sim)
            # sim
            sim = des_sim + lab_sim
            one_wn_sim.append(sim)
        max_sim = max(one_wn_sim)
        max_sim_idx = one_wn_sim.index(max_sim)
        one_wn_lab_sim_sorted = sorted(one_wn_lab_sim, reverse=True)
        one_wn_des_sim_sorted = sorted(one_wn_des_sim, reverse=True)
        wn2result[one_wn] = qnode2wiki[candi[max_sim_idx]] + [max_sim,
                                                              one_wn_lab_sim[max_sim_idx],
                                                              one_wn_des_sim[max_sim_idx],
                                                              len(candi),
                                                              one_wn_lab_sim_sorted.index(one_wn_lab_sim[max_sim_idx])+1,
                                                              one_wn_des_sim_sorted.index(one_wn_des_sim[max_sim_idx])+1]

    print('结果保存中...')
    with open(Config.result_path, 'wb') as f:
        pickle.dump(wn2result, f)
    print('运行完毕!')


if __name__ == '__main__':
    cal()

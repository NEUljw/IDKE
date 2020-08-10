import pickle
import random

# wordnet_path = 'merged_candidate.pkl'
# wiki_path = 'wiki_new.pkl'     # wiki_new.pkl的路径
# result_path = 'result7_15.pkl'    # 结果路径
#
# with open(wordnet_path, 'rb') as f:
#     wn_data = pickle.load(f)
# print('synset总数:', len(wn_data))
# random.seed(1234)
# sample_keys = random.sample(wn_data.keys(), 300)
#
# print('读取qnode2wiki字典中...')
# with open(wiki_path, 'rb') as f:
#     qnode2wiki = pickle.load(f)
# print('字典大小:', len(qnode2wiki))
#
# r_dict = {}
# for key, value in wn_data.items():
#     if key in sample_keys:
#         r_dict[key] = [qnode2wiki[i] for i in value]
#
#
# with open(result_path, 'wb') as f:
#     pickle.dump(r_dict, f)
# print('完成！')

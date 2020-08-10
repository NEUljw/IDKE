"""候选集为空的在这里直接删掉"，最后剩78589个synset"""
import pickle
from tqdm import tqdm
from collections import Counter

# with open('candidate_lab_web.pkl', 'rb') as f:  # 只用网站结果的word2qnodes字典，有3553个synset候选集为空
#     web_candi = pickle.load(f)
# with open('candidate_lab.pkl', 'rb') as f:    # 只用labels结果的word2qnodes字典，有8394个synset候选集为空
#     lab_candi = pickle.load(f)
#
# print(len(web_candi), len(lab_candi))
#
# merge_candi = {}
# for key in tqdm(web_candi):
#     if len(set(web_candi[key]+lab_candi[key])) > 0:
#         merge_candi[key] = list(set(web_candi[key]+lab_candi[key]))
#
# with open('merged_candidate.pkl', 'wb') as f:
#     pickle.dump(merge_candi, f)
# print('done!')


with open('merged_candidate.pkl', 'rb') as f:    # 合并后，有3511个synset候选集为空
    m_candi = pickle.load(f)
print(len(m_candi))

'''for key, value in m_candi.items():
    if key[0] == 'manx.n.01' and 'Q12175' in value:
        print('11')
    if key[0] == 'tonka_bean.n.01' and 'Q901484' in value:
        print('11')
    if key[0] == 'relish.n.03' and 'Q766777' in value:
        print('11')
    if key[0] == 'color_guard.n.01' and 'Q1759959' in value:
        print('11')
    if key[0] == 'blastomere.n.01' and 'Q882097' in value:
        print('11')
    if key[0] == 'dumdum.n.01' and 'Q1163838' in value:
        print('11')
    if key[0] == 'pink_elephants.n.01' and 'Q565650' in value:
        print('11')
    if key[0] == 'oast_house.n.01' and 'Q1166574' in value:
        print('11')
    if key[0] == 'pink_bollworm.n.01' and 'Q2168361' in value:
        print('11')
    if key[0] == 'pauli_exclusion_principle.n.01' and 'Q131594' in value:
        print('11')'''


all_candi_qnodes = []
for key, value in m_candi.items():
    all_candi_qnodes += value
print(len(all_candi_qnodes))
all_candi_qnodes = list(set(all_candi_qnodes))
print(len(all_candi_qnodes))

print('saving qnodes...')
with open('all_candidate_qnode.pkl', 'wb') as f:
    pickle.dump(all_candi_qnodes, f)
print('saving qnodes done')

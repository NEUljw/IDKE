import pickle

from model_use import cal_sim

synset_num = 10000
batch_size = 64


print('create data start!')
with open('run_data.pkl', 'rb') as f:
    data = pickle.load(f)
    query_result = data['data']
query_result = query_result[:synset_num]
synset_num = str(synset_num)

all_synset_des, all_wiki_candidate, all_synset_id = [], [], []
# all_synset_des、all_synset_id都是list，all_wiki_candidate是list的list
for i, value in enumerate(query_result):
    synset_des = value[0][1]
    synset_id = value[0][0]
    all_synset_des.append(synset_des)
    all_synset_id.append(synset_id)
    wiki_candidate = [w[1] for w in value[1]]
    all_wiki_candidate.append(wiki_candidate)
print('create data end!')
print('wordnet synsets number:', len(all_synset_des))


LSTM_result = cal_sim(all_synset_des, all_wiki_candidate, model_path='output/training_sts_bilstm-2020-06-08_17-01-37', batch_size=batch_size)
# with open('sim_result/LSTM_{}.pkl'.format(synset_num), 'wb') as f:
#     pickle.dump({'model results': LSTM_result}, f)

# 下面的3个模型下载到了.cache中
# bert_result = cal_sim(all_synset_des, all_wiki_candidate, model_path='bert-base-nli-stsb-mean-tokens', batch_size=batch_size)
# with open('sim_result/bert_{}.pkl'.format(synset_num), 'wb') as f:
#     pickle.dump({'model results': bert_result}, f)
#
# roberta_result = cal_sim(all_synset_des, all_wiki_candidate, model_path='roberta-base-nli-stsb-mean-tokens', batch_size=batch_size)
# with open('sim_result/roberta_{}.pkl'.format(synset_num), 'wb') as f:
#     pickle.dump({'model results': roberta_result}, f)
#
# distilbert_result = cal_sim(all_synset_des, all_wiki_candidate, model_path='distilbert-base-nli-stsb-mean-tokens', batch_size=batch_size)
# with open('sim_result/distilbert_{}.pkl'.format(synset_num), 'wb') as f:
#     pickle.dump({'model results': distilbert_result}, f)
#
#
# xlnet_result = cal_sim(all_synset_des, all_wiki_candidate, model_path='output/training_nli_sts_xlnet', batch_size=batch_size)
# with open('sim_result/xlnet_{}.pkl'.format(synset_num), 'wb') as f:
#     pickle.dump({'model results': xlnet_result}, f)

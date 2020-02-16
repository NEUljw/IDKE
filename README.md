# NEU IDKE

### bi-lstm

roberta.py是调用roberta模型的代码，cal_accuracy.py是计算roberta模型的结果准确度的代码。<br>
bi-lstm是Bi-LSTM模型的代码。

### bi-lstm+ConceptNet

**conceptnet_score是计算QA在conceptnet中的得分，其中：**<br>bulid_concept_vocab.py功能是建立conceptnet数据中的concept和relation的词典。<br>graph_construction.py功能是根据conceptnet的数据建立gragh。<br>extract_test_concept.py功能是提取测试集中的concept。<br>path_finder.py功能是找到concept之间的路径。<br>cal_concept_score.py功能是计算conceptnet中的得分。<br>total.csv是conceptnet数据。<br>

**LSTM_score是计算QA在Bi-LSTM上的得分，其中：**<br>model.py功能是训练模型并预测QA的得分。<br>**cal_final_score.py功能是计算总得分。**
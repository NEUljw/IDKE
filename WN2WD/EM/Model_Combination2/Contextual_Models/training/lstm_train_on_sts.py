"""
This example runs a BiLSTM after the word embedding lookup. The output of the BiLSTM is than pooled,
for example with max-pooling (which gives a system like InferSent) or with mean-pooling.

Note, you can also pass BERT embeddings to the BiLSTM.
"""
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime

# 参数
batch_size = 128
num_epochs = 10
sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark')
embedding_path = 'glove.6B.300d.txt.gz'
model_save_path = '../output/training_sts_bilstm-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Map tokens to traditional word embeddings like GloVe
word_embedding_model = models.WordEmbeddings.from_text_file(embedding_path)

lstm = models.LSTM(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), hidden_dim=1024)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(lstm.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=True)


model = SentenceTransformer(modules=[word_embedding_model, lstm, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size,
                              collate_fn=model.smart_batching_collate)
train_loss = losses.CosineSimilarityLoss(model=model)
n = 0
for i in train_dataloader:
    n += 1
    if n < 3:
        print(i)
        print('--'*30)


logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size,
                            collate_fn=model.smart_batching_collate)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
n = 0
for i in dev_dataloader:
    n += 1
    if n < 3:
        print(i)
        print('--'*30)

# Configure the training
warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.1)   # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

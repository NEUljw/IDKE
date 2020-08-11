# Mapping of WordNet to WikiData



## Introduction

We propose an **Ensemble Method (EM)** and a **Comprehensive Similarity Method (CSM)** to map WordNet into WikiData. Among them, CSM has the highest correct rate, the final mapping results is 'mapping results.pkl'.



## Setup

We recommend Python 3.6 or higher. The models is implemented with Tensorflow (1.13.1) and Pytorch (at least 1.0.1).

The code does **not** work with Python 2.7.



## Ensemble Method (EM)

For the model selection, we use two combinations.

### Model Combination1

There are six models in this model combination, including three Non-Contextual models: LDA, Word2Vec, FastText, and three Contextual models: MaLSTM, BERT, XLNet.

### Model Combination2

There are eight models in this model combination, including three Non-Contextual models: LDA, Word2Vec, FastText, and five Contextual models: Siamese LSTM, Siamese XLNet, Siamese BERT, Siamese RoBERTa, Siamese DistilBERT.



## Comprehensive Similarity Method (CSM) 

We combine the description similarity and the label similarity as the comprehensive similarity.
<<<<<<< HEAD

The operation steps are as follows:

1. Download data from WikiData official website, a total of 10000 parts (needs a lot of disk space).
2. **Download the rest of the relevant data and put it in the corresponding location.**
3. Set the path parameters in `data/parse_wikidata.py` and run it to parse the original WikiData data.
4. Set the path parameters in `data/create_qnode2wiki.py` and run it to process the data format.
6. Set the path parameters in `create_wiki_dict.py` and run it to generate a dictionary.
7. Set the path parameters in `create_dict_from_web_results.py` and run it to generate another dictionary.
8. Set the path parameters in `create_candidate_lab.py` and run to generate the first candidate set.
9. Set the path parameters in `create_candidate_lab.py` and run to generate the second candidate set.
10. Set the path parameters in `merge_candidate.py` and run to merge the two candidate sets.
11. Set the path parameters in `EM/Model_Combination2/Contextual_Models/encode_and_save.py` and run to generate the embeddings of descriptions.
12. Set the path parameters in `embed_lab_sim.py` and run it to generate the final mapping results.
=======
>>>>>>> 8f68dbf854e3ffcb8e10efdbe569759f8798a7f8

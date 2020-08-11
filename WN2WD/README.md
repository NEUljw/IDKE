# Mapping of WordNet to WikiData

We propose an **Ensemble Method (EM)** and a **Comprehensive Similarity Method (CSM)** to map WordNet into WikiData. Among them, CSM has the highest correct rate, the final mapping results is 'mapping results.pkl'.



Ensemble Method (EM)

For the model selection, we use two combinations.

### Model Combination1

There are six models in this model combination, including three Non-Contextual models: LDA, Word2Vec, FastText, and three Contextual models: MaLSTM, BERT, XLNet.

### Model Combination2

There are eight models in this model combination, including three Non-Contextual models: LDA, Word2Vec, FastText, and five Contextual models: Siamese LSTM, Siamese XLNet, Siamese BERT, Siamese RoBERTa, Siamese DistilBERT.



## Comprehensive Similarity Method (CSM) 

We combine the description similarity and the label similarity as the comprehensive similarity.

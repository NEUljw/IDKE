import csv

concept_vocab, relation_vocab = [], []

with open('total.csv') as f:
    f_csv = csv.reader(f)
    n = 0
    for row in f_csv:
        n += 1
        if n > 1:
            concept_vocab.append(row[0])
            concept_vocab.append(row[2])
            relation_vocab.append(row[1])
    print(len(concept_vocab), len(relation_vocab))
    concept_vocab = list(set(concept_vocab))
    relation_vocab = list(set(relation_vocab))
    print(len(concept_vocab), len(relation_vocab))

with open('concept_vocab.txt', 'w') as f:
    for i in concept_vocab:
        f.write(i+'\n')

with open('relation_vocab.txt', 'w') as f:
    for i in relation_vocab:
        f.write(i+'\n')

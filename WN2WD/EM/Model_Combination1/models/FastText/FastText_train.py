from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

CORPUS_PATH = 'sentences.txt'
MODEL_PATH = 'FastText_model.bin'


def text_to_word_list(text):
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    text = text.lower()
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords.words('english') and w not in english_punctuations]
    return text


def read_corpus():
    f = open(CORPUS_PATH, 'r', encoding='utf-8-sig')
    data = f.readlines()
    data = [i.strip() for i in data]
    f.close()
    print(len(data))
    data = [text_to_word_list(i) for i in data]
    return data


def train(sentences):
    # sentences = [['I', 'know', 'you'], ['He', 'can', 'swim']]
    model = FastText(sentences, size=100, window=4, min_count=1, iter=5, min_n=4,
                     max_n=7, word_ngrams=1)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    sentences = read_corpus()
    train(sentences)

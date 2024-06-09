import gensim
from gensim import corpora

def create_dictionary(data):
    dictionary = corpora.Dictionary(data)
    return dictionary

def create_corpus(data, dictionary):
    corpus = [dictionary.doc2bow(text) for text in data]
    return corpus

def train_lda_model(corpus, dictionary, num_topics=10):
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model

def display_topics(model, num_words=10):
    topics = model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)



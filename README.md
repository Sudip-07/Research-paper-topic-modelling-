# Research-paper-topic-modelling- 
In [1]:
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
In [2]:
import nltk
nltk.download('wordnet')
[nltk_data] Downloading package wordnet to /Users/sli/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
Out[2]:
True
In [3]:
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
In [4]:
for w in ['dogs', 'ran', 'discouraged']:
    print(w, get_lemma(w), get_lemma2(w))
dogs dog dog
ran run ran
discouraged discourage discouraged
In [5]:
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
[nltk_data] Downloading package stopwords to /Users/sli/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
In [6]:
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens
In [7]:
import random
text_data = []
with open('dataset.csv') as f:
    for line in f:
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
            print(tokens)
            text_data.append(token)
In [8]:
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
In [9]:
corpus = [dictionary.doc2bow(text) for text in text_data]
In [10]:
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
Try 5 topics
In [11]:
import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
In [12]:
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
(0, '0.034*"processor" + 0.019*"database" + 0.019*"issue" + 0.019*"overview"')
(1, '0.051*"computer" + 0.028*"design" + 0.028*"graphics" + 0.028*"gallery"')
(2, '0.050*"management" + 0.027*"object" + 0.027*"circuit" + 0.027*"efficient"')
(3, '0.019*"cognitive" + 0.019*"radio" + 0.019*"network" + 0.019*"distribute"')
(4, '0.029*"circuit" + 0.029*"system" + 0.029*"rigorous" + 0.029*"integration"')
In [15]:
new_doc = 'Practical Bayesian Optimization of Machine Learning Algorithms'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))
[(38, 1), (117, 1)]
[(0, 0.06669136), (1, 0.40170625), (2, 0.06670282), (3, 0.39819494), (4, 0.066704586)]
In [18]:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
ldamodel.save('model3.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
(0, '0.029*"processor" + 0.016*"management" + 0.016*"aid" + 0.016*"algorithm"')
(1, '0.026*"radio" + 0.026*"network" + 0.026*"cognitive" + 0.026*"efficient"')
(2, '0.029*"circuit" + 0.029*"distribute" + 0.016*"database" + 0.016*"management"')
In [19]:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
ldamodel.save('model10.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
(0, '0.055*"database" + 0.055*"system" + 0.029*"technical" + 0.029*"recursive"')
(1, '0.038*"distribute" + 0.038*"graphics" + 0.038*"regenerate" + 0.038*"exact"')
(2, '0.055*"management" + 0.029*"multiversion" + 0.029*"reference" + 0.029*"document"')
(3, '0.046*"circuit" + 0.046*"object" + 0.046*"generation" + 0.046*"transformation"')
(4, '0.008*"programming" + 0.008*"circuit" + 0.008*"network" + 0.008*"surface"')
(5, '0.061*"radio" + 0.061*"cognitive" + 0.061*"network" + 0.061*"connectivity"')
(6, '0.085*"programming" + 0.008*"circuit" + 0.008*"subdivision" + 0.008*"management"')
(7, '0.041*"circuit" + 0.041*"design" + 0.041*"processor" + 0.041*"instruction"')
(8, '0.055*"computer" + 0.029*"efficient" + 0.029*"channel" + 0.029*"cooperation"')
(9, '0.061*"stimulation" + 0.061*"sensor" + 0.061*"retinal" + 0.061*"pixel"')

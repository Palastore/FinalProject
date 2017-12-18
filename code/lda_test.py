#! python2
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # print 'i = ',i
    # clean and tokenize document string
    raw = i.lower()
    # print 'raw = ',raw
    tokens = tokenizer.tokenize(raw)
    # print 'tokens = ',tokens

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # print 'stopped_tokens = ',stopped_tokens

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # print 'stemmed_tokens = ',stemmed_tokens
    
    # add tokens to list
    texts.append(stemmed_tokens)
    # print 'texts = ',texts

# print 'texts = ',texts
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
# print dictionary
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
# print corpus

# generate LDA model
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=20)

for d in doc_set:
    bow = dictionary.doc2bow(d.split())
    print bow
    t = ldamodel.get_document_topics(bow)
    print 'doc : \'',d,'\''
    print t
    print 'topic : ', ldamodel.show_topic(t[0][0])

print '\n\n'

for topic in ldamodel.print_topics(num_words=5):
	print  'topic : ',topic[0],' -> ', topic [1]

print '\nfor word ' ,dictionary[0]
for topic in ldamodel.get_term_topics(0):
	print '\tprobability to be topic ',topic[0],' is ',topic[1]
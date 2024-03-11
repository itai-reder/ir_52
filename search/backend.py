import re
import math
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
nltk.download('stopwords')
bucket_name = "ir_52"

# Tokenize the text body
def tokenize(text):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links", 
                        "may", "first", "see", "history", "people", "one", "two", 
                        "part", "thumb", "including", "second", "following", 
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    stemmer = PorterStemmer()
    og_tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [stemmer.stem(x) for x in og_tokens if x not in all_stopwords]
    return tokens

# Inverted Index utility functions
def get_postings(inverted_index, query):
    postings = []
    for term in query:
        pl = inverted_index.read_a_posting_list('', term, bucket_name)
        postings.append(pl)
    return postings

# Scoring functions
def word_count_score(query, index):
    postings = get_postings(index, query)
    tf_dict = {}
    for posting in postings:
        for id, tf in posting:
            if id in tf_dict:
                tf_dict[id] += 1
            else:
                tf_dict[id] = 1
    doc_list = [(id, score) for id, score in tf_dict.items()]
    sorted_list = sorted(doc_list, key=lambda x: x[1], reverse=True)
    return sorted_list

def tf_score(query, index):
    postings = get_postings(index, query)
    tf_dict = {}
    for posting in postings:
        for id, tf in posting:
            if id in tf_dict:
                tf_dict[id] += tf
            else:
                tf_dict[id] = tf
    doc_list = [(id, score) for id, score in tf_dict.items()]
    sorted_list = sorted(doc_list, key=lambda x: x[1], reverse=True)
    return sorted_list

def BM25_score(query, index, N, doc_lengths, avg_doc_length):
    k1 = 1.2
    b = 0.75
    bm25 = Counter()
    for term in query:

        # calculate idf
        try:
            df = index.df[term]
        except:
            continue
        idf = math.log(N/df, 10)

        # calculate bm25 score
        pl = index.read_a_posting_list('', term, bucket_name)
        for id, tf in pl:
            try:
                norm = (tf*(k1+1))/(tf+k1*(1-b+b*(doc_lengths[id]/avg_doc_length)))
                bm25[id] += idf*norm
            except:
                pass
    bm25_final = bm25.most_common()
    return bm25_final


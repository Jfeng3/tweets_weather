import pandas as pd
import numpy as np
import nltk
def TFIDF(document,unique_terms,collection):
    word_tfidf = []
    count = 0 
    for word in unique_terms:
        word_tfidf.append(collection.tf_idf(word,document))
        count = count +1
        print count
    return word_tfidf
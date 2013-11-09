# remove stopwords
import nltk
import pandas as pd
import numpy as np
def get_tweet_stopwords():
    stopwords = []
    stopwords1 = nltk.corpus.stopwords.words('english')
    for word in stopwords1:
        word = word.encode('ascii','ignore')
        stopwords.append(word)
    stopwords.append("!")
    stopwords.append("the")
    stopwords.append(",")
    stopwords.append(":")
    stopwords.append("@")
    stopwords.append("mention")
    stopwords.append("#")
    stopwords.append(".")
    stopwords.append("...")
    stopwords.append("link")
    stopwords.append("{")
    stopwords.append("}")
    stopwords.append("'s")
    stopwords.append("?")
    stopwords.append(")")
    stopwords.append("(")
    stopwords.append("!")
    stopwords.append("I")
    stopwords.append("RT")
    stopwords.append("&")
    stopwords.append("It")
    stopwords.append("-")
    stopwords.append(";")
    stopwords.append("This")
    stopwords.append("This")
    
    
    return stopwords


def remove_tweet_stopwords(word_frame, stopwords):
    word_frame['stopwords'] = np.nan
    word_frame['stopwords'] = word_frame['term'].apply( lambda x: True if x in stopwords else False )
    mask = (word_frame['stopwords']==False)
    word_frame = word_frame[mask]
    return word_frame
    
    
    
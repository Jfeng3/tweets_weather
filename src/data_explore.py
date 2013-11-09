import pandas as pd
import numpy as np
import nltk
from collections import Counter
import tweet_stop_words

train_data = pd.read_csv("train.csv")
texts = []
for i in range(train_data.shape[0]):
    tokens = nltk.word_tokenize(train_data.tweet[i])
    text = nltk.Text(tokens)
    texts.append(text)
    
print "Prepared ", len(texts), " documents..."
print "They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]"

#Load the list of texts into a TextCollection object.
collection = nltk.TextCollection(texts)
print "Created a collection of", len(collection), "terms."

#get a list of unique terms
unique_terms = list(set(collection))
print "Unique terms found: ", len(unique_terms)

word_list = [word for word_list in texts for word in word_list]
word_dict = Counter(word_list)
common_word_list = word_dict.most_common(5000)
common_word_frame = pd.DataFrame(common_word_list) 
common_word_frame.columns = [["term","count"]]
common_word_frame.head(20)
stopwords = tweet_stop_words.get_tweet_stopwords()
common_word_frame = tweet_stop_words.remove_tweet_stopwords(common_word_frame, stopwords)

# remove stopwords

    



# def TFIDF(document):
#     word_tfidf = []
#     count = 0 
#     for word in document:
#         word_tfidf.append(collection.tf_idf(word,document))
#         count = count +1
#         print count
#     return word_tfidf
#     
# def vec_words(word_list,terms):
#for i in xrange(len(texts)):
#    terms_frame['i'] = terms_frame['term'].apply(lambda x: 1 if x in texts[i] else 0)
#    print i
    


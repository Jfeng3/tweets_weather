import pandas as pd
import numpy as np
import nltk

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

# Function to create a TF*IDF vector for one document.  For each of
# our unique words, we have a feature which is the td*idf for that word
# in the current document
def TFIDF(document):
    word_tfidf = []
    count = 0 
    for word in unique_terms:
        word_tfidf.append(collection.tf_idf(word,document))
        count = count +1
        print count
    return word_tfidf

### And here we actually call the function and create our array of vectors.
vectors = [np.array(TFIDF(f)) for f in texts]
print "Vectors created."
print "First 10 words are", unique_terms[:10]
print "First 10 stats for first document are", vectors[0][0:10] 
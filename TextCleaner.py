# # Install the latest Tensorflow version.
# !pip3 install --quiet "tensorflow>=1.7"
# # Install TF-Hub.
# !pip3 install --quiet tensorflow-hub
# !pip3 install --quiet seaborn

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

print("Initializing Universal Sentence Encoder")
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

print("Reading data")
df= pd.read_csv('../data/raw_text.csv')

from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

clean_text = []

for index, row in df.iterrows():
    text = row[0]

    # Split into words
    tokens = word_tokenize(text)

    # Convert to lower case
    tokens = [w.lower() for w in tokens]

    # Remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    # Stem words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    stemmed = " ".join(stemmed)
    clean_text.append(stemmed)

# Load list of happiness rankings
happiness_rankings = pd.read_csv('../data/happiness_rankings.csv')
rainbow = cm.rainbow(np.linspace(0,1,len(happiness_rankings)))

# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.0)
X = vectorizer.fit_transform(clean_text)
#print(vectorizer.get_feature_names())
#print(X.shape)
new_messages = []
for message in clean_text:
    words = message.split()
    words = [w for w in words if w in set(vectorizer.get_feature_names())]
    message = " ".join(words)
    new_messages.append(message)

tf.logging.set_verbosity(tf.logging.ERROR)

print(vectorizer.get_feature_names())

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(embed(new_messages))

  for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    print("Message: {}".format(new_messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

from sklearn.decomposition import PCA

pca = PCA(n_components=50)

principalComponents = pca.fit_transform(message_embeddings)
principalDf = pd.DataFrame(data = principalComponents)

from sklearn.manifold import TSNE

# First dim reduction from PCA and then via t-SNE
from tqdm import tqdm
perplexities = range(1, 120, 1)
for i in tqdm(perplexities):
    pca = PCA(n_components=50)
    principalComponents = pca.fit_transform(message_embeddings)
    principalDf = pd.DataFrame(data = principalComponents)
    embedded = TSNE(n_components=2, perplexity=i, learning_rate=10, n_iter=5000).fit_transform(principalDf)
    principalDf = pd.DataFrame(data = embedded)
    plt.figure(figsize=(8, 8), dpi=80)

    x = principalDf[0]
    y = principalDf[1]


    # plt.scatter(x, y)

    for index, row in happiness_rankings.iterrows():
        text = row[0]
        plt.plot(principalDf[0][index], principalDf[1][index], "o", c=rainbow[index])
        plt.annotate(text, (x[index], y[index]))

    plt.savefig('../visualizations/pca-tsne/p{}.png'.format(i))

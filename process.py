import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ct22_task1_base import *
from sklearn.metrics import accuracy_score







# split training data into training set and validation set at 80 vs 20


'''
training_labels_final = tf.keras.utils.to_categorical(training_labels,
                                                      num_classes=2)
val_labels_final = tf.keras.utils.to_categorical(val_labels, num_classes=2)
'''

def model(site, dropout_rate, dense_layer):
    hub_layer = hub.KerasLayer(site,  input_shape=[], dtype=tf.string, trainable=True)
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(dense_layer, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


# compute class weights
def compute_class_weights(labels):
    n_samples = len(labels)
    n_classes = len(labels.unique())
    class_weights = {}
    class_names = labels.value_counts().index.tolist()
    for i in range(len(labels.value_counts())):
        class_weights[class_names[i]] = round(n_samples/(n_classes *
                                        labels.value_counts()[i]), 2)
    return class_weights

#class_weights = compute_class_weights(training_labels)
#print("Class weights computed: ", class_weights)


def save_file(file_name, model):
    test_data = pd.read_csv(file_name)
    labels = model.predict(test_data['text'])

    del test_data['text']
    del test_data['keyword']
    del test_data['location']

    test_data['target'] = np.rint(labels).astype(int)
    test_data.to_csv('sub2.csv', index=False)

def predict(sent, model):
    pred = model.predict(sent)
    return np.rint(pred).astype(int)

# process training data by lowering case and removing stopwords and punctuation


def bag_of_words(sent, sent1):  # bag_of_word vectorizer
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(sent).toarray()
    val_x = vectorizer.transform(sent1).toarray()
    return x, val_x


def tfidf_vectorizer(sent, sent1, sent2):  #tf-idf vectorizer
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(sent).toarray()
    val_x = vectorizer.transform(sent1).toarray()
    test_x = vectorizer.transform(sent2).toarray()
    return x, val_x, test_x

def tokenize_padd(sentences, vocab_size, oov_tok, max_length,
                      padding_type, trunc_type):
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length,
                               padding=padding_type,
                               truncating=trunc_type)
    return padded

# to oversample minority class using SMOTE
def oversample_smote(padded, labels):
    oversample = SMOTE()
    oversample_padded, labels_final = oversample.fit_resample(padded,
                                                                  labels)
    counter = Counter(labels_final)
    print(counter)
    return oversample_padded, labels_final



def define_gru(dropout_rate, vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(embedding_dim)),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(embedding_dim, activation='elu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(
                                  0.01)),
        tf.keras.layers.BatchNormalization(),  # Batch Normalization
        #tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    return model
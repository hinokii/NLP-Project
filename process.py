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




data = pd.read_csv('train1.csv')
print(data.columns)
print(data.head())
print(data.describe())
print(data.nunique())
print(data['target'].value_counts())
test_data = pd.read_csv('test1.csv')
print(test_data.columns)

# split training data into training set and validation set at 80 vs 20
training_sentences, val_sentences, training_labels, val_labels = \
    train_test_split(data['text'], data['target'], test_size=0.3)
print(training_labels.value_counts())

'''
training_labels_final = tf.keras.utils.to_categorical(training_labels,
                                                      num_classes=2)
val_labels_final = tf.keras.utils.to_categorical(val_labels, num_classes=2)
'''
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",  input_shape=[],
    dtype=tf.string, trainable=True)

# build a model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', \
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            #optimizer=tf.keras.optimizers.Adam(learning_rate),
             metrics=['accuracy'])

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

class_weights = compute_class_weights(training_labels)
print("Class weights computed: ", class_weights)


# early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

num_epochs = 50

history = model.fit(training_sentences, training_labels, batch_size=32,
    epochs=num_epochs, class_weight=class_weights, validation_data=(
        val_sentences, val_labels), callbacks=[callback])

# evaluate validation set
loss, test_acc = model.evaluate(val_sentences, val_labels)
print('Test loss: ', loss)
print('Test accuracy: ', test_acc)

# process training data by lowering case and removing stopwords and punctuation
'''

def bag_of_words(sent, sent1):  # bag_of_word vectorizer
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(sent).toarray()
    val_x = vectorizer.transform(sent1).toarray()
    return x, val_x


def tfidf_vectorizer(sent, sent1):  #tf-idf vectorizer
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(sent).toarray()
    val_x = vectorizer.transform(sent1).toarray()
    return x, val_x

# to oversample minority class using SMOTE
def oversample_smote(padded, labels):
    oversample = SMOTE()
    oversample_padded, labels_final = oversample.fit_resample(padded,
                                                                  labels)
    counter = Counter(labels_final)
    print(counter)
    return oversample_padded, labels_final

x, test_x = tfidf_vectorizer(training_sentences, val_sentences)
x, training_labels = oversample_smote(x, training_labels)
best = best_model(models, x, training_labels, test_x, val_labels, 'accuracy')
best.fit(x, training_labels)
pred = best.predict(test_x)
f1 = f1_score(val_labels, pred)
print("F1 score - test: ", f1)








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

# double layer bidirectional GRU


#compute class weight
class_weights = compute_class_weights(training_labels)
print("Class weights computed: ", class_weights)

# early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

'''
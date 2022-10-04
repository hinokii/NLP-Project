import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def create_model(site, dropout_rate, dense_layer):
    hub_layer = hub.KerasLayer(site,  input_shape=[], dtype=tf.string, trainable=True)
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(dense_layer, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

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



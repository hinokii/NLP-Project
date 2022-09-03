
from app.process import *

def test_predict():
    data = pd.read_csv('train1.csv')
    training_sentences, val_sentences, training_labels, val_labels = \
    train_test_split(data['text'], data['target'], test_size=0.3)

    model = create_model("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", 0.6, 16)
    #model = define_gru(0.6, 2000, 128, 400)
    model.compile(loss='binary_crossentropy', \
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025),
            #optimizer=tf.keras.optimizers.Adam(learning_rate),
             metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    #class_weights = compute_class_weights(training_labels)

    num_epochs = 50
    history = model.fit(training_sentences, training_labels, batch_size=128,
        epochs=num_epochs, validation_data=(
        val_sentences, val_labels), callbacks=[callback])

    save_file('test1.csv', model)

    test = []
    test.append(['a turnado hit the area and many people had to evacuate. the Turnado destroyed many houses and people lost homes'])
    test.append(['I went to Hawaii this summer and had a great time. It had been so long since I went there last time, but it seems the same.'])
    assert predict(test, model)[0] == 1
    assert predict(test, model)[1] == 0


# model("https://tfhub.dev/google/universal-sentence-encoder/4", 0.6, 8)
# optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
# model.fit(training_sentences, training_labels, batch_size=256,

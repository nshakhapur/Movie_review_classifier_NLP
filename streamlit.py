import matplotlib.pyplot as plt
import os
import re

import string
import tensorflow as tf
import keras as layers
import keras as losses
import streamlit as st

st.title("Movie Review Classifier")
st.subheader("A Natural Language Processing Model")
movie_name=st.text_input("Movie name: ")
user_input = st.text_area("Text Input", "")
user_input=[user_input]
predict=st.button("How's The Review?!")

if predict:

 batch_size = 32
 seed = 42

 raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

 for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])

 print("Label 0 corresponds to", raw_train_ds.class_names[0])
 print("Label 1 corresponds to", raw_train_ds.class_names[1])




 raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

 raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)

 def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')



 tf.keras.layers.TextVectorization(
    max_tokens=None,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int',
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
  )




 max_features = 10000
 sequence_length = 250

 vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
 train_text = raw_train_ds.map(lambda x, y: x)
 vectorize_layer.adapt(train_text)

 def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label



 text_batch, label_batch = next(iter(raw_train_ds))
 first_review, first_label = text_batch[0], label_batch[0]
 print("Review", first_review)
 print("Label", raw_train_ds.class_names[first_label])
 print("Vectorized review", vectorize_text(first_review, first_label))

 print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
 print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
 print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


 print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
 print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
 print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))



 train_ds = raw_train_ds.map(vectorize_text)
 val_ds = raw_val_ds.map(vectorize_text)
 test_ds = raw_test_ds.map(vectorize_text)


 AUTOTUNE = tf.data.AUTOTUNE

 train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
 val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
 test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


 embedding_dim=16

 model = tf.keras.Sequential([
  tf.keras.layers.Embedding(max_features + 1, embedding_dim),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
 ])
 model.summary()


 model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])



 epochs = 10
 history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


 loss, accuracy = model.evaluate(test_ds)

 print("Loss: ", loss)
 print("Accuracy: ", accuracy)


 
 history_dict = history.history
 history_dict.keys()





 acc = history_dict['accuracy']
 val_acc = history_dict['val_accuracy']
 loss = history_dict['loss']
 val_loss = history_dict['val_loss']

 epochs = range(1, len(acc) + 1)


 export_model = tf.keras.Sequential([
  vectorize_layer,
  model
 ])

 export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy']
 )


 loss, accuracy = export_model.evaluate(raw_test_ds)
 print(accuracy)

 numerical_predictions=export_model.predict(user_input)
 
 def get_string_label(prediction):
    if prediction >= 0:
        return "Positive"
    else:
        return "Negative"

 string_predictions = [get_string_label(pred) for pred in numerical_predictions]
  
 for user_input, prediction in zip(user_input, string_predictions):
       st.markdown("## Results: ")
       
       st.markdown(f"### Sentiment: {prediction}\n")



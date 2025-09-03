import sys
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten, Embedding, Input, BatchNormalization, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, Recall, Precision, AUC
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
from scipy.sparse import csr_matrix
import pickle
import gc
import os

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('float32')

import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ########################################################################################
# DATA PRE-PROCESSING:
  
def kmer_encode(sequences):

  kmer_list = []
  k = 4

  for sequence in sequences:
  
    sequence = sequence.upper()
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    kmer_sequence = ' '.join(kmers)
    kmer_list.append(kmer_sequence)
    
  tokeniser = Tokenizer(lower=False, split=' ')
  tokeniser.fit_on_texts(kmer_list)
  new_sequences = tokeniser.texts_to_sequences(kmer_list)
  
  return new_sequences, tokeniser
    

def integer_encode(sequences):
  
  print("ENCODING...")
  
  new_sequences = []
  base_int = {'A': 0, 'C': 1, 'G':2, 'T': 3, 'N': 4}
  
  for sequence in sequences:
    sequence = sequence.upper()
    encoded = np.array([base_int[base] for base in sequence])
    new_sequences.append(encoded)
  
  print("ENCODING COMPLETE")
  
  return new_sequences
  
def undersample(texts, labels):
  
  combine = list(zip(texts, labels))
  
  cl_0 = [pair for pair in combine if pair[1] == 0]
  cl_1 = [pair for pair in combine if pair[1] == 1]
  
  #if len(cl_0) > len(cl_1):
    #cl_0 = random.sample(cl_0, len(cl_1))
  #else:
    #cl_1 = random.sample(cl_1, len(cl_0))
    
  cl_1 = random.sample(cl_1, 1000000)
  cl_0 = random.sample(cl_0, 1000000)
  
  undersampled = cl_1 + cl_0
  
  random.shuffle(undersampled)
  
  return zip(*undersampled)
  
def focal_loss(gamma=2., alpha=0.25):
  def focal_loss_fixed(y_true, y_pred):
    eps = 1e-8
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
  return focal_loss_fixed
  
# ########################################################################################
# NEURAL NETWORK TRAINING:
  
# CNN MODEL:

def build_cnn(vocab_size): 

  cnn_model = Sequential([
    Embedding(input_dim=vocab_size,output_dim=32, input_length=300),
    
    Conv1D(16, 4, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    MaxPooling1D(),
    
    Bidirectional(LSTM(16, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    GlobalMaxPooling1D(),
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),
    
    # Classify data binarily
    Dense(1, activation='sigmoid', dtype='float32')
  ], name='CNN_model')

  # Compile model, requesting all required performance metrics
  # Given the binary nature of target variable, set loss to 'binary cross-entropy'
  cnn_model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=focal_loss(gamma=2., alpha=0.25), 
    metrics=['BinaryAccuracy', 'Precision', 'Recall', AUC()]
  )
  
  return cnn_model

# FIT AND SAVE MODEL:

def fit_and_save(model, model_name, train_dataset, val_dataset, weights):

  print("FITTING MODEL...")
  
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience = 5, restore_best_weights=True,mode="max")
  
  checkpoint = tf.keras.callbacks.ModelCheckpoint(f'mtb_{model_name}_best.h5', save_best_only=True, monitor="val_recall", mode="max")

  results = model.fit(train_dataset, validation_data=val_dataset, class_weight=weights, epochs=100,verbose=1,callbacks=[early_stopping, checkpoint])
  
  model.save(f'mtb_{model_name}.h5')
  
  print("MODEL FIT AND SAVED")
  
  # Get max validation accuracy and index
  val_accs = results.history.get('val_binary_accuracy') or results.history.get('val_accuracy')
  max_acc = max(val_accs) if val_accs else None
  max_epoch = val_accs.index(max_acc) if max_acc else None

  print(f"\nMax validation accuracy: {max_acc}\n")
  if max_epoch is not None:
    # Safely get other metrics if available
    val_precision = results.history.get('val_precision', [None]*len(val_accs))[max_epoch]
    val_recall = results.history.get('val_recall', [None]*len(val_accs))[max_epoch]
    val_loss = results.history.get('val_loss', [None]*len(val_accs))[max_epoch]
    val_auc = results.history.get('val_auc', [None]*len(val_accs))[max_epoch]
    
    print(f"Epoch with Max Accuracy: {max_epoch}\n")
    print(f"Final validation AUC: {val_auc}\n")
    print(f"Precision score: {val_precision}\n")
    print(f"Recall score: {val_recall}\n")
    print(f"Loss: {val_loss}\n")
    
    with open('mtb_cnn_results.pkl', 'wb') as f:
      pickle.dump(results, f)
      
  return
  
#########################################################################################
#MAIN CODE START:

#Download dataset
dataset = pd.read_csv('MTB_twc_dataset.csv')
dataset = dataset[dataset['sequence'].str.len() >= 3]

# Check dataset for any problems with the data
print("Number of files: " + str(len(dataset)))
print("\nValue counts: " + str(dataset['status'].value_counts()))
print(str(dataset.head()))

encoder = LabelEncoder()
labels = encoder.fit_transform(dataset['status'])

with open('mtb_twc_label_encoder.pkl', 'wb') as f:
  pickle.dump(encoder, f)

dna_sequence, tokeniser = kmer_encode(dataset['sequence'])
vocab_size = len(tokeniser.word_index) + 1

with open('mtb_twc_kmer_tokenizer.pkl', 'wb') as f:
  pickle.dump(tokeniser, f)

del dataset, tokeniser

labels = np.array(labels)
  
# Split into train/test 

sequence_train_sequence, sequence_test_sequence, sequence_train_status, sequence_test_status = train_test_split(dna_sequence, labels, test_size=0.2, random_state=42)

sequence_train_sequence, sequence_train_status = undersample(sequence_train_sequence, sequence_train_status)
sequence_train_status = np.array(sequence_train_status)
sequence_train_sequence = pad_sequences(sequence_train_sequence, padding='post', maxlen=300)
sequence_train_sequence = np.array(sequence_train_sequence)  
sequence_test_sequence = pad_sequences(sequence_test_sequence, padding='post', maxlen=300)
sequence_test_sequence = np.array(sequence_test_sequence)
  
with open('mtb_cnn_test_data.pkl', 'wb') as f:
  pickle.dump((sequence_test_sequence, sequence_test_status), f)

del sequence_test_sequence, sequence_test_status, dna_sequence, labels
gc.collect()

train_sequence, val_sequence, train_status, val_status = train_test_split(sequence_train_sequence, sequence_train_status, test_size=0.1, random_state=42)

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_status), y=train_status)
class_weight_dict = dict(enumerate(class_weights)) 
 
# Check training data
print("Number of TRAIN files: " + str(len(train_sequence)))
print("\nValue counts: " + str(pd.Series(train_status).value_counts()))
 
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequence, train_status))
# train_dataset = train_dataset.shuffle(buffer_size=100_000,reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(512)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  
val_dataset = tf.data.Dataset.from_tensor_slices((val_sequence, val_status))
val_dataset = val_dataset.batch(512)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  
del sequence_train_sequence, sequence_train_status
  
cnn = build_cnn(vocab_size)
fit_and_save(cnn, 'cnn_model', train_dataset, val_dataset, class_weight_dict)
del cnn, train_dataset
gc.collect()

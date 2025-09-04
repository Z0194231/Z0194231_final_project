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

# Import random and ensure that experiment is repeatable 
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ########################################################################################
# DATA PRE-PROCESSING:

# Create a function to encode data
def kmer_encode(sequences):
  
  #Create list to store k-mer encoded sequences 
  kmer_list = []
  # Set k-mer = 4
  k = 4

  # Iterate over all sequences in dataset
  for sequence in sequences:
    
    # Normalise sequence into upper letters
    sequence = sequence.upper()
    # Break sequence into k-mers
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    # Join sequence back into k-mers separated by a white space
    kmer_sequence = ' '.join(kmers)
    # Add encoded sequence into list
    kmer_list.append(kmer_sequence)

  # Initialise tokeniser
  tokeniser = Tokenizer(lower=False, split=' ')
  # Fit tokeniser to the produced k-mer encoded sequence list, and tokenise data 
  tokeniser.fit_on_texts(kmer_list)
  new_sequences = tokeniser.texts_to_sequences(kmer_list)

  # Return encoded sequences
  return new_sequences, tokeniser

# For training, undersampling is applied to the data
def undersample(texts, labels):
  
  # Zip tests and store them in a list
  combine = list(zip(texts, labels))

  # Separate the two classes using their encoded labels
  cl_0 = [pair for pair in combine if pair[1] == 0]
  cl_1 = [pair for pair in combine if pair[1] == 1]

  # Sample each class individually, based on a fixed sample size
  cl_1 = random.sample(cl_1, 3000000)
  cl_0 = random.sample(cl_0, 3000000)

  # Combine the two samples back into one set
  undersampled = cl_1 + cl_0
  
  # Suffle samples into random order to avoid order bias
  random.shuffle(undersampled)

  # Return under-sampled data ready for use
  return zip(*undersampled)
  
# Customise loss for model improvement. Gamma and alpha can be set during model build and before loss is calculated
# Focal loss inspired by: https://medium.com/@saptarshimt/object-as-points-anchor-free-object-detection-from-scratch-2019-tensorflow-6170eb815c07
def focal_loss(gamma, alpha):
  # Nest target requirements so they are invoked after loss is used within model 
  def focal_loss_nested(y_true, y_pred):
    # Prevent probabilities from leaving the defined range 
    eps = 1e-8
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    # Selected true class probability -> pt equals the correct label
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    # Returns focal loss mean after computing loss formula
    return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
  # Returns calculated focal loss after invoked 
  return focal_loss_nested
  
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
  # Given the binary nature of target variable, binary accuracy was selected
  cnn_model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    # Custom focal loss is used 
    loss=focal_loss(gamma=2., alpha=0.25), 
    metrics=['BinaryAccuracy', 'Precision', 'Recall', AUC()]
  )
  
  return cnn_model

# FIT AND SAVE MODEL:

def fit_and_save(model, model_name, train_dataset, val_dataset):

  # Alert user that model fitting is being performed
  print("FITTING MODEL...")

  # Define an early stopping based on validation set accuracy, in case model stops improving performance, with patience 5
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience = 5, restore_best_weights=True,mode="max")

  # Save the best performing model so far each time a new high is achieved in validation recall 
  checkpoint = tf.keras.callbacks.ModelCheckpoint(f'filtered_{model_name}_best.h5', save_best_only=True, monitor="val_recall", mode="max")

  # Start model fit and store history 
  results = model.fit(train_dataset, validation_data=val_dataset, class_weight=None, epochs=100,verbose=1,callbacks=[early_stopping, checkpoint])

  #Save fit model at the end of its training
  model.save(f'filtered_{model_name}.h5')

  # Alert user model fit is completed
  print("MODEL FIT AND SAVED")
  
  # Get max validation accuracy and index
  val_accs = results.history.get('val_binary_accuracy') or results.history.get('val_accuracy')
  max_acc = max(val_accs) if val_accs else None
  max_epoch = val_accs.index(max_acc) if max_acc else None

  # Print maximum validation accuracy achieved by model 
  print(f"\nMax validation accuracy: {max_acc}\n")
  
  # Print further metrics based on epoch with max validation accuracy 
  if max_epoch is not None:
    # Get validation performance metrics from results 
    val_precision = results.history.get('val_precision', [None]*len(val_accs))[max_epoch]
    val_recall = results.history.get('val_recall', [None]*len(val_accs))[max_epoch]
    val_loss = results.history.get('val_loss', [None]*len(val_accs))[max_epoch]
    val_auc = results.history.get('val_auc', [None]*len(val_accs))[max_epoch]

    # Print full performance metrics report for usage in model evaluation 
    print(f"Epoch with Max Accuracy: {max_epoch}\n")
    print(f"Final validation AUC: {val_auc}\n")
    print(f"Precision score: {val_precision}\n")
    print(f"Recall score: {val_recall}\n")
    print(f"Loss: {val_loss}\n")

    # Save results for further work 
    with open('filtered_cnn_results.pkl', 'wb') as f:
      pickle.dump(results, f)
      
  return
  
#########################################################################################
#MAIN CODE START:

#Download dataset
dataset = pd.read_csv('filtered_twc_dataset.csv')
dataset = dataset[dataset['sequence'].str.len() >= 3]

# Check dataset for any problems with the data
print("Number of files: " + str(len(dataset)))
print("\nValue counts: " + str(dataset['status'].value_counts()))
print(str(dataset.head()))

# Initialise encoder and encode categories 
encoder = LabelEncoder()
labels = encoder.fit_transform(dataset['status'])

# Save encoder for debugging needs
with open('filtered_twc_label_encoder.pkl', 'wb') as f:
  pickle.dump(encoder, f)

# Encode sequences to k-mer encoding
dna_sequence, tokeniser = kmer_encode(dataset['sequence'])
# Calculate vocabulary size for CNN dimension use
vocab_size = len(tokeniser.word_index) + 1

# Save tokeniser for debugging needs
with open('filtered_twc_kmer_tokenizer.pkl', 'wb') as f:
  pickle.dump(tokeniser, f)

# Delete saved data to free space
del dataset, tokeniser

# Ensure labels are in the correct shape
labels = np.array(labels)
  
# Split into train/test 
sequence_train_sequence, sequence_test_sequence, sequence_train_status, sequence_test_status = train_test_split(dna_sequence, labels, test_size=0.2, random_state=42)

# Under-sample training sets
sequence_train_sequence, sequence_train_status = undersample(sequence_train_sequence, sequence_train_status)
# Ensure labels are in correct shape again
sequence_train_status = np.array(sequence_train_status)
# Pad training sequences to a max of 300
sequence_train_sequence = pad_sequences(sequence_train_sequence, padding='post', maxlen=300)
# Ensure sequences are in correct shape
sequence_train_sequence = np.array(sequence_train_sequence)  
# Pad testing sequences to a max of 300
sequence_test_sequence = pad_sequences(sequence_test_sequence, padding='post', maxlen=300)
# Ensure sequences are in correct shape
sequence_test_sequence = np.array(sequence_test_sequence)

# Save test data for prediction runs
with open('filtered_cnn_test_data.pkl', 'wb') as f:
  pickle.dump((sequence_test_sequence, sequence_test_status), f)

# Delete saved data to free space
del sequence_test_sequence, sequence_test_status, dna_sequence, labels
gc.collect()

# Further split training dataset into training and validation sets
train_sequence, val_sequence, train_status, val_status = train_test_split(sequence_train_sequence, sequence_train_status, test_size=0.1, random_state=42)
 
# Check training data
print("Number of TRAIN files: " + str(len(train_sequence)))
print("\nValue counts: " + str(pd.Series(train_status).value_counts()))

# Convert training data into a TF Dataset that is compatible with model
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequence, train_status))
# Batch dataset
train_dataset = train_dataset.batch(512)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Convert validation data into a TF Dataset that is compatible with model  
val_dataset = tf.data.Dataset.from_tensor_slices((val_sequence, val_status))
# Batch dataset
val_dataset = val_dataset.batch(512)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Delete saved data to free space
del sequence_train_sequence, sequence_train_status

# Build model
cnn = build_cnn(vocab_size)
# Fit and save model 
fit_and_save(cnn, 'cnn_model', train_dataset, val_dataset)
# Delete model and dataset to free space
del cnn, train_dataset
gc.collect()

import sys
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def focal_loss(gamma=2., alpha=0.25):
  def focal_loss_fixed(y_true, y_pred):
    eps = 1e-8
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
  return focal_loss_fixed
  
def undersample(texts, labels):
  
  combine = list(zip(texts, labels))
  
  cl_0 = [pair for pair in combine if pair[1] == 0]
  cl_1 = [pair for pair in combine if pair[1] == 1]
  
  target = min(len(cl_0), len(cl_1))
    
  cl_1 = random.sample(cl_1, target)
  cl_0 = random.sample(cl_0, target)
  
  undersampled = cl_1 + cl_0
  
  random.shuffle(undersampled)
  
  texts_sam, labels_sam = zip(*undersampled)
  return list(texts_sam), list(labels_sam)

with open("mtb_cnn_test_data.pkl", 'rb') as test_data:
  sequence_test, status_test = pickle.load(test_data)
  
sequence_test, status_test = undersample(sequence_test, status_test)

sequence_test = np.array(sequence_test)  
status_test = np.array(status_test)

classes, counts = np.unique(status_test, return_counts=True)
final_counts = dict(zip(classes, counts))
    
print('VALUE COUNTS:\n' + str(final_counts))
print('VALUES SHAPE:\n', np.unique(status_test))
    
cnn_model = load_model('mtb_cnn_model.h5', custom_objects={'focal_loss_fixed': focal_loss()})
cnn_model_best = load_model('mtb_cnn_model_best.h5', custom_objects={'focal_loss_fixed': focal_loss()})

cnn_pred = cnn_model.predict(sequence_test)

y_score = cnn_pred[:,0]
y_true = status_test
import pickle
with open("mtb_preds.pkl", "wb") as f:
    pickle.dump({"y_true": y_true, "y_score": y_score}, f) 

cnn_pred = (cnn_pred > 0.5).astype(int) 

best_cnn_pred = cnn_model_best.predict(sequence_test)
best_cnn_pred = (best_cnn_pred > 0.5).astype(int) 

print("Confusion matrix CNN MODEL:\n", tf.math.confusion_matrix(status_test, cnn_pred))
print(classification_report(cnn_pred, status_test, target_names=['MTB', 'non-MTB']))

print("Confusion matrix BEST CNN MODEL:\n", tf.math.confusion_matrix(status_test, best_cnn_pred))
print(classification_report(best_cnn_pred, status_test, target_names=['MTB', 'non-MTB']))

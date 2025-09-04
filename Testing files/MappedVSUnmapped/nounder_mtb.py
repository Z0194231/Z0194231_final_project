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

# Import random and ensure that experiment is repeatable 
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Customise loss for model improvement. Gamma and alpha can be set during model build and before loss is calculated
# Focal loss inspired by: https://medium.com/@saptarshimt/object-as-points-anchor-free-object-detection-from-scratch-2019-tensorflow-6170eb815c07
def focal_loss(gamma=2., alpha=0.25):
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

# Load in saved test data
with open("mtb_cnn_test_data.pkl", 'rb') as test_data:
  sequence_test, status_test = pickle.load(test_data)

# Conduct a count for testing labels and save them 
classes, counts = np.unique(status_test, return_counts=True)
final_counts = dict(zip(classes, counts))

# Print counted values for testing data check 
print('VALUE COUNTS:\n' + str(final_counts))
print('VALUES SHAPE:\n', np.unique(status_test))

# Load saved models, first with max accuracy and second with max recall 
cnn_model = load_model('filtered_cnn_model.h5', custom_objects={'focal_loss_fixed': focal_loss})
cnn_model_best = load_model('filtered_cnn_model_best.h5', custom_objects={'focal_loss_fixed': focal_loss})

# Predict on first model 
cnn_pred = cnn_model.predict(sequence_test)

# Store predictions and labels. 
y_score = cnn_pred[:,0]
y_true = status_test
# Save predictions and labels for use in graph plotting 
with open("filtered_preds.pkl", "wb") as f:
    pickle.dump({"y_true": y_true, "y_score": y_score}, f)

# Apply threshold on predicted probabilities to determine final classification 
cnn_pred = (cnn_pred > 0.5).astype(int) 

# Predict on second model and apply threshold 
best_cnn_pred = cnn_model_best.predict(sequence_test)
best_cnn_pred = (best_cnn_pred > 0.5).astype(int) 

# Print confusion matrix and classification report for first model 
print("Confusion matrix CNN MODEL:\n", tf.math.confusion_matrix(status_test, cnn_pred))
print(classification_report(cnn_pred, status_test, target_names=['MTB', 'non-MTB']))

# Print confusion matrix and classification report for second model 
print("Confusion matrix BEST CNN MODEL:\n", tf.math.confusion_matrix(status_test, best_cnn_pred))
print(classification_report(best_cnn_pred, status_test, target_names=['MTB', 'non-MTB']))
